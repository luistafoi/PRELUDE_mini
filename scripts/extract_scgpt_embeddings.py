"""Extract scGPT cell embeddings for DepMap and Sanger cell lines.

Takes bulk RNA-seq expression data, treats each cell line as a pseudo-cell,
and extracts 512-dim embeddings from the pretrained scGPT whole-human model.

Bypasses scGPT's high-level API to avoid torchtext dependency issues.
Directly loads the TransformerModel and vocab, handles tokenization and
binning manually.

Usage:
    python scripts/extract_scgpt_embeddings.py \
        --data_dir data/processed \
        --gpu 0

    # For Sanger:
    python scripts/extract_scgpt_embeddings.py \
        --data_dir data/sanger_processed \
        --output_dir data/sanger_processed/embeddings \
        --gpu 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Mock torchtext BEFORE any scgpt imports ---
import types
import collections

class _MockVocab:
    """Minimal mock of torchtext.vocab.Vocab for scGPT compatibility."""
    def __init__(self, ordered_dict=None, min_freq=1, specials=None, **kw):
        self.stoi = {}
        self.itos = []
        idx = 0
        for s in (specials or []):
            self.stoi[s] = idx
            self.itos.append(s)
            idx += 1
        if ordered_dict:
            for token in ordered_dict:
                if token not in self.stoi:
                    self.stoi[token] = idx
                    self.itos.append(token)
                    idx += 1
        self.vocab = self  # GeneVocab expects .vocab attribute

    def __getitem__(self, token):
        return self.stoi.get(token, -1)
    def __len__(self):
        return len(self.itos)
    def __contains__(self, token):
        return token in self.stoi
    def set_default_index(self, idx):
        self._default_idx = idx
    def get_stoi(self):
        return self.stoi
    def get_itos(self):
        return self.itos
    def append_token(self, token):
        if token not in self.stoi:
            idx = len(self.itos)
            self.stoi[token] = idx
            self.itos.append(token)

    def insert_token(self, token, index):
        """Insert token at a specific index (used by GeneVocab.from_dict)."""
        # Extend itos list if needed
        while len(self.itos) <= index:
            self.itos.append(None)
        self.itos[index] = token
        self.stoi[token] = index

    def set_default_token(self, token):
        """Set default token for unknown lookups."""
        if token in self.stoi:
            self._default_idx = self.stoi[token]

    def get_default_index(self):
        return getattr(self, '_default_idx', -1)

def _mock_vocab_factory(ordered_dict, min_freq=1, **kw):
    return _MockVocab(ordered_dict, min_freq, **kw)

_tv = types.ModuleType('torchtext')
_tvv = types.ModuleType('torchtext.vocab')
_tvv.Vocab = _MockVocab
_tvv.vocab = _mock_vocab_factory
_tv.vocab = _tvv
sys.modules['torchtext'] = _tv
sys.modules['torchtext.vocab'] = _tvv
# --- End torchtext mock ---

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
import anndata
from pathlib import Path

from dataloaders.data_loader import PRELUDEDataset
from scgpt.model.model import TransformerModel
from scgpt.tokenizer import GeneVocab


def load_expression_as_anndata(expression_path, cell_ids):
    """Load DepMap expression CSV into AnnData, filtered to graph cells."""
    print("Loading expression data...")
    df = pd.read_csv(expression_path, index_col=0)
    print(f"  Expression matrix: {df.shape[0]} cells x {df.shape[1]} genes")

    # Strip Entrez IDs: "ACE2 (59272)" -> "ACE2"
    df.columns = [col.split(' (')[0].strip() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"  After dedup: {df.shape[1]} unique genes")

    cell_ids_upper = [c.upper() for c in cell_ids]
    df.index = df.index.str.upper()

    matched = len(set(df.index) & set(cell_ids_upper))
    print(f"  Matched {matched}/{len(cell_ids)} cells")

    # Build matrix in graph cell order
    expr_matrix = np.zeros((len(cell_ids), df.shape[1]), dtype=np.float32)
    for i, cid in enumerate(cell_ids_upper):
        if cid in df.index:
            expr_matrix[i] = df.loc[cid].values.astype(np.float32)

    adata = anndata.AnnData(
        X=expr_matrix,
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=df.columns),
    )
    return adata


def select_hvgs(adata, n_top_genes=3000):
    """Select highly variable genes."""
    print(f"Selecting top {n_top_genes} HVGs...")
    adata_copy = adata.copy()
    try:
        sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes, flavor='seurat')
        adata.var['highly_variable'] = adata_copy.var['highly_variable']
    except Exception:
        print("  HVG selection failed, using top genes by variance")
        variances = np.var(adata.X, axis=0)
        top_idx = np.argsort(variances)[-n_top_genes:]
        hvg_mask = np.zeros(adata.shape[1], dtype=bool)
        hvg_mask[top_idx] = True
        adata.var['highly_variable'] = hvg_mask

    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    print(f"  Selected {adata_hvg.shape[1]} HVGs")
    return adata_hvg


def binning(row, n_bins=51):
    """Quantile-bin non-zero expression values into n_bins bins (scGPT style).

    Zero values stay 0. Non-zero values get binned into [1, n_bins-1].
    """
    non_zero_mask = row != 0
    if non_zero_mask.sum() == 0:
        return np.zeros_like(row, dtype=np.int64)

    non_zero = row[non_zero_mask]
    bins = np.quantile(non_zero, np.linspace(0, 1, n_bins - 1))
    binned = np.digitize(non_zero, bins, right=False)
    binned = np.clip(binned, 1, n_bins - 1)

    result = np.zeros_like(row, dtype=np.int64)
    result[non_zero_mask] = binned
    return result


def extract_embeddings(adata, model_dir, batch_size=64, max_length=1200,
                       device="cuda", n_bins=51):
    """Extract embeddings by directly running the TransformerModel."""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "args.json") as f:
        model_config = json.load(f)

    # Load vocab
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    pad_token = model_config.get("pad_token", "<pad>")
    pad_value = model_config.get("pad_value", -2)

    special_tokens = [pad_token, "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    vocab_size = len(vocab)
    pad_token_id = vocab[pad_token]

    print(f"  Vocab size: {vocab_size}")
    print(f"  Pad token: '{pad_token}' -> id {pad_token_id}")

    # Match genes to vocab
    gene_names = list(adata.var_names)
    gene_ids = np.array([vocab[g] for g in gene_names])
    valid_mask = gene_ids >= 0  # genes found in vocab

    gene_names_valid = [g for g, v in zip(gene_names, valid_mask) if v]
    gene_ids_valid = gene_ids[valid_mask]
    print(f"  Matched {valid_mask.sum()}/{len(gene_names)} genes to vocab")

    # Filter expression to valid genes only
    expr = adata.X[:, valid_mask]

    # Initialize model
    embsize = model_config["embsize"]
    nheads = model_config["nheads"]
    d_hid = model_config["d_hid"]
    nlayers = model_config["nlayers"]

    model = TransformerModel(
        ntoken=vocab_size,
        d_model=embsize,
        nhead=nheads,
        d_hid=d_hid,
        nlayers=nlayers,
        vocab=vocab,
        dropout=0.0,  # no dropout at inference
        pad_token=pad_token,
        pad_value=pad_value,
        n_input_bins=n_bins,
        use_fast_transformer=False,  # no flash-attn
    )

    # Load weights
    state_dict = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"  Model loaded: {nlayers} layers, {embsize}-dim, {nheads} heads")

    # Process cells in batches
    n_cells = expr.shape[0]
    all_embeddings = []

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch_expr = expr[start:end]  # (B, n_genes)
            B = batch_expr.shape[0]

            # Per-cell: select top genes by expression, bin values
            batch_gene_ids = []
            batch_values = []

            for i in range(B):
                row = batch_expr[i]

                # Sort by expression (descending), take top max_length
                nonzero_idx = np.where(row > 0)[0]
                if len(nonzero_idx) > max_length:
                    top_idx = nonzero_idx[np.argsort(row[nonzero_idx])[-max_length:]]
                elif len(nonzero_idx) > 0:
                    top_idx = nonzero_idx
                else:
                    # All zeros - use random genes
                    top_idx = np.arange(min(max_length, len(row)))

                # Get gene token IDs and expression values
                cell_gene_ids = gene_ids_valid[top_idx]
                cell_values = row[top_idx]

                # Bin expression values
                cell_binned = binning(cell_values, n_bins=n_bins)

                batch_gene_ids.append(cell_gene_ids)
                batch_values.append(cell_binned)

            # Pad to same length
            max_len = max(len(g) for g in batch_gene_ids)
            padded_genes = np.full((B, max_len), pad_token_id, dtype=np.int64)
            padded_values = np.full((B, max_len), pad_value, dtype=np.int64)

            for i in range(B):
                n = len(batch_gene_ids[i])
                padded_genes[i, :n] = batch_gene_ids[i]
                padded_values[i, :n] = batch_values[i]

            gene_ids_t = torch.from_numpy(padded_genes).long().to(device)
            values_t = torch.from_numpy(padded_values).float().to(device)

            # Forward pass
            # TransformerModel expects: src, values, src_key_padding_mask
            src_key_padding_mask = gene_ids_t.eq(pad_token_id)

            output = model._encode(
                gene_ids_t,
                values_t,
                src_key_padding_mask=src_key_padding_mask,
            )

            # Mean pool over non-padded positions
            mask = ~src_key_padding_mask  # (B, seq_len)
            mask_expanded = mask.unsqueeze(-1).float()  # (B, seq_len, 1)
            summed = (output * mask_expanded).sum(dim=1)  # (B, embsize)
            counts = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            cell_embeds = summed / counts  # (B, embsize)

            all_embeddings.append(cell_embeds.cpu().numpy())

            if (start // batch_size) % 5 == 0:
                print(f"  Processed {end}/{n_cells} cells...")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Output embeddings: {embeddings.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract scGPT cell embeddings")
    parser.add_argument('--expression_path', type=str,
                        default='data/misc/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv')
    parser.add_argument('--model_dir', type=str,
                        default='models/scGPT_human/scGPT_human')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_hvgs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=1200)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, 'embeddings')
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Load graph to get cell ordering
    print("Loading graph dataset...")
    dataset = PRELUDEDataset(args.data_dir)

    cell_type = dataset.node_name2type['cell']
    num_cells = dataset.nodes['count'][cell_type]
    gid2name = {gid: name for name, gid in dataset.node2id.items()}

    cell_ids = [None] * num_cells
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == cell_type:
            cell_ids[lid] = gid2name[gid]

    print(f"Graph has {num_cells} cells")

    # Load expression
    adata = load_expression_as_anndata(args.expression_path, cell_ids)

    # Select HVGs
    adata_hvg = select_hvgs(adata, n_top_genes=args.n_hvgs)

    # Extract embeddings
    embeddings = extract_embeddings(
        adata_hvg, args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)

    embed_path = os.path.join(output_dir, 'scgpt_cell_embeddings.npy')
    np.save(embed_path, embeddings)
    print(f"\nSaved embeddings to {embed_path}")
    print(f"  Shape: {embeddings.shape}")

    names_path = os.path.join(output_dir, 'scgpt_cell_names.txt')
    with open(names_path, 'w') as f:
        for cid in cell_ids:
            f.write(f"{cid}\n")
    print(f"Saved cell names to {names_path}")

    # Sanity check
    print(f"\n--- Embedding Summary ---")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")

    zero_rows = (np.abs(embeddings).sum(axis=1) == 0).sum()
    if zero_rows > 0:
        print(f"  WARNING: {zero_rows} cells have zero embeddings")


if __name__ == '__main__':
    main()
