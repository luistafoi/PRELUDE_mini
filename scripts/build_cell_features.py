"""Build multi-omic cell feature vectors focused on drug-target genes.

For each cell × each drug-target gene (964 genes), extracts:
  - Expression (rank-normalized within cell)
  - CRISPR dependency (Chronos gene effect)
  - Copy number (relative CN)
  - Mutation pathogenicity (max REVEL/PolyPhen score)

Plus 2 binary flags: has_crispr, has_cn.

Output: cell_features_multiomic.pt  (N_cells, 964*4 + 2) tensor
        cell_feature_config.json    (gene list, channel info)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from dataloaders.data_loader import PRELUDEDataset


def get_drug_target_genes(dataset):
    """Extract drug-target gene names from train.dat structural edges."""
    gene_type = dataset.node_name2type['gene']
    drug_type = dataset.node_name2type['drug']
    gid2name = {gid: name for name, gid in dataset.node2id.items()}

    dg_genes = set()
    with open(os.path.join(dataset.data_dir, 'train.dat')) as f:
        for line in f:
            parts = line.strip().split('\t')
            src, tgt = int(parts[0]), int(parts[1])
            st = dataset.nodes['type_map'].get(src, (-1, -1))[0]
            tt = dataset.nodes['type_map'].get(tgt, (-1, -1))[0]
            if st == drug_type and tt == gene_type:
                dg_genes.add(gid2name[tgt].upper())
            elif st == gene_type and tt == drug_type:
                dg_genes.add(gid2name[src].upper())

    return sorted(dg_genes)


def get_cell_ordering(dataset):
    """Get cell ACH-IDs ordered by local ID."""
    cell_type = dataset.node_name2type['cell']
    num_cells = dataset.nodes['count'][cell_type]
    gid2name = {gid: name for name, gid in dataset.node2id.items()}

    cells = [None] * num_cells
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == cell_type:
            cells[lid] = gid2name[gid].upper()
    return cells


def load_expression(expression_path, cell_ids, target_genes):
    """Load expression for target genes, rank-normalize within each cell.

    Returns:
        (N_cells, N_genes) array, rank-normalized to [0, 1]
        Gene columns are in target_genes order.
    """
    print("Loading expression data...")
    df = pd.read_csv(expression_path, index_col=0)

    # Parse column names: "SYMBOL (ENTREZ)" -> SYMBOL
    col_to_symbol = {}
    for col in df.columns:
        sym = col.split(' (')[0].strip().upper()
        col_to_symbol[col] = sym

    # Find columns for target genes
    symbol_to_col = {}
    for col, sym in col_to_symbol.items():
        if sym in set(target_genes):
            symbol_to_col[sym] = col

    found = [g for g in target_genes if g in symbol_to_col]
    missing = [g for g in target_genes if g not in symbol_to_col]
    print(f"  Expression: {len(found)}/{len(target_genes)} target genes found, {len(missing)} missing")

    # Build matrix (cells × genes) in target_genes order
    result = np.zeros((len(cell_ids), len(target_genes)), dtype=np.float32)
    cell_set = set(df.index)

    for i, cell in enumerate(cell_ids):
        if cell not in cell_set:
            continue
        row = df.loc[cell]
        for j, gene in enumerate(target_genes):
            col = symbol_to_col.get(gene)
            if col is not None:
                result[i, j] = row[col]

    # Rank-normalize within each cell (across genes)
    # Rank from 0 to N-1, then divide by N-1 to get [0, 1]
    print("  Rank-normalizing expression within each cell...")
    for i in range(len(cell_ids)):
        row = result[i]
        ranks = np.argsort(np.argsort(row)).astype(np.float32)  # rank transform
        n = len(row)
        if n > 1:
            result[i] = ranks / (n - 1)  # normalize to [0, 1]

    matched_cells = sum(1 for c in cell_ids if c in cell_set)
    print(f"  Matched {matched_cells}/{len(cell_ids)} cells")
    return result


def load_crispr(crispr_path, cell_ids, target_genes):
    """Load CRISPR gene effect scores for target genes.

    Returns:
        (N_cells, N_genes) array of gene effect scores
        (N_cells,) boolean array: True if cell has CRISPR data
    """
    print("Loading CRISPR gene effect data...")
    df = pd.read_csv(crispr_path, index_col=0)

    col_to_symbol = {}
    for col in df.columns:
        sym = col.split(' (')[0].strip().upper()
        col_to_symbol[col] = sym

    symbol_to_col = {}
    for col, sym in col_to_symbol.items():
        if sym in set(target_genes):
            symbol_to_col[sym] = col

    found = [g for g in target_genes if g in symbol_to_col]
    print(f"  CRISPR: {len(found)}/{len(target_genes)} target genes found")

    result = np.zeros((len(cell_ids), len(target_genes)), dtype=np.float32)
    has_crispr = np.zeros(len(cell_ids), dtype=bool)
    cell_set = set(df.index)

    for i, cell in enumerate(cell_ids):
        if cell not in cell_set:
            continue
        has_crispr[i] = True
        row = df.loc[cell]
        for j, gene in enumerate(target_genes):
            col = symbol_to_col.get(gene)
            if col is not None:
                val = row[col]
                if pd.notna(val):
                    result[i, j] = val

    # Clip extreme outliers
    result = np.clip(result, -3.0, 1.0)

    matched = has_crispr.sum()
    print(f"  Matched {matched}/{len(cell_ids)} cells")
    return result, has_crispr


def load_copy_number(cn_path, cell_ids, target_genes):
    """Load copy number data for target genes.

    Returns:
        (N_cells, N_genes) array of relative copy number
        (N_cells,) boolean array: True if cell has CN data
    """
    print("Loading copy number data...")
    df = pd.read_csv(cn_path)

    # OmicsCNGeneWGS has columns: SequencingID, ModelID, IsDefaultEntry, ..., then gene columns
    # Find the ModelID column and gene columns
    # Gene columns start after the metadata columns
    meta_cols = ['SequencingID', 'ModelID', 'IsDefaultEntryForModel',
                 'ModelConditionID', 'IsDefaultEntryForMC']

    # Find which columns are genes (SYMBOL (ENTREZ) format)
    gene_cols = [c for c in df.columns if '(' in c and c not in meta_cols]
    col_to_symbol = {}
    for col in gene_cols:
        sym = col.split(' (')[0].strip().upper()
        col_to_symbol[col] = sym

    symbol_to_col = {}
    for col, sym in col_to_symbol.items():
        if sym in set(target_genes):
            symbol_to_col[sym] = col

    found = [g for g in target_genes if g in symbol_to_col]
    print(f"  CN: {len(found)}/{len(target_genes)} target genes found")

    # Filter to default entries and index by ModelID
    if 'IsDefaultEntryForModel' in df.columns:
        df = df[df['IsDefaultEntryForModel'].isin([True, 'Yes', 'True', 'yes', 'TRUE'])]

    if 'ModelID' not in df.columns:
        print("  WARNING: No ModelID column in CN file")
        return np.ones((len(cell_ids), len(target_genes)), dtype=np.float32), np.zeros(len(cell_ids), dtype=bool)

    df = df.set_index('ModelID')

    result = np.ones((len(cell_ids), len(target_genes)), dtype=np.float32)  # default: diploid
    has_cn = np.zeros(len(cell_ids), dtype=bool)
    cell_set = set(df.index)

    for i, cell in enumerate(cell_ids):
        if cell not in cell_set:
            continue
        has_cn[i] = True
        row = df.loc[cell]
        # Handle duplicate entries (take first)
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for j, gene in enumerate(target_genes):
            col = symbol_to_col.get(gene)
            if col is not None:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    result[i, j] = val

    # Clip to reasonable range
    result = np.clip(result, 0.0, 5.0)

    matched = has_cn.sum()
    print(f"  Matched {matched}/{len(cell_ids)} cells")
    return result, has_cn


def load_mutations(mutation_path, cell_ids, target_genes):
    """Load mutation pathogenicity scores for target genes.

    For each cell × gene, takes the max pathogenicity score across variants.
    Uses RevelScore first, falls back to PolyPhen.

    Returns:
        (N_cells, N_genes) array of max pathogenicity scores [0, 1]
    """
    print("Loading mutation data...")
    cell_set = set(cell_ids)
    cell_to_idx = {c: i for i, c in enumerate(cell_ids)}
    gene_set = set(target_genes)
    gene_to_idx = {g: j for j, g in enumerate(target_genes)}

    result = np.zeros((len(cell_ids), len(target_genes)), dtype=np.float32)

    # Read in chunks to handle large file
    chunks = pd.read_csv(mutation_path, chunksize=100000,
                         usecols=['ModelID', 'HugoSymbol', 'RevelScore',
                                  'Polyphen', 'VepImpact', 'AMPathogenicity'],
                         dtype={'RevelScore': 'float64', 'Polyphen': 'str',
                                'AMPathogenicity': 'float64'})

    n_variants = 0
    n_matched = 0
    for chunk in chunks:
        for _, row in chunk.iterrows():
            n_variants += 1
            cell = str(row.get('ModelID', '')).upper()
            gene = str(row.get('HugoSymbol', '')).upper()

            if cell not in cell_to_idx or gene not in gene_to_idx:
                continue

            # Get pathogenicity score
            score = 0.0
            revel = row.get('RevelScore')
            if pd.notna(revel):
                score = max(score, float(revel))

            am = row.get('AMPathogenicity')
            if pd.notna(am):
                score = max(score, float(am))

            # Parse PolyPhen (format: "probably_damaging(0.999)")
            pp = row.get('Polyphen')
            if pd.notna(pp) and isinstance(pp, str) and '(' in pp:
                try:
                    pp_score = float(pp.split('(')[1].rstrip(')'))
                    score = max(score, pp_score)
                except (ValueError, IndexError):
                    pass

            # VepImpact as fallback binary
            if score == 0.0:
                vep = str(row.get('VepImpact', ''))
                if vep == 'HIGH':
                    score = 0.8
                elif vep == 'MODERATE':
                    score = 0.4

            i = cell_to_idx[cell]
            j = gene_to_idx[gene]
            result[i, j] = max(result[i, j], score)
            n_matched += 1

    cells_with_mut = (result.max(axis=1) > 0).sum()
    print(f"  Processed {n_variants} variants, {n_matched} matched cell+gene pairs")
    print(f"  {cells_with_mut}/{len(cell_ids)} cells have at least one mutation in target genes")
    return result


def impute_missing(data, has_data):
    """Fill missing rows with per-gene population mean from cells that have data."""
    if has_data.all():
        return data
    col_means = data[has_data].mean(axis=0)
    data[~has_data] = col_means
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build multi-omic cell features for drug-target genes.")
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Graph data directory (contains train.dat, node.dat, etc.)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to data_dir)')
    parser.add_argument('--misc_dir', type=str, default='data/misc',
                        help='Directory with DepMap source files')
    parser.add_argument('--gene_list', type=str, default=None,
                        help='Path to existing cell_feature_config.json to reuse target gene list')
    cli_args = parser.parse_args()

    data_dir = cli_args.data_dir
    misc_dir = cli_args.misc_dir
    output_dir = cli_args.output_dir or data_dir

    # File paths
    expression_path = os.path.join(misc_dir, '24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv')
    crispr_path = os.path.join(misc_dir, 'CRISPRGeneEffect.csv')
    cn_path = os.path.join(misc_dir, 'OmicsCNGeneWGS.csv')
    mutation_path = os.path.join(misc_dir, 'OmicsSomaticMutations_24Q2.csv')

    # Load dataset for graph info
    print("Loading dataset...")
    dataset = PRELUDEDataset(data_dir)

    # Get drug-target genes (reuse existing list if provided) and cell ordering
    if cli_args.gene_list:
        with open(cli_args.gene_list) as f:
            target_genes = json.load(f)['target_genes']
        print(f"Reusing {len(target_genes)} target genes from {cli_args.gene_list}")
    else:
        target_genes = get_drug_target_genes(dataset)
    cell_ids = get_cell_ordering(dataset)
    print(f"\nTarget genes: {len(target_genes)}")
    print(f"Cells: {len(cell_ids)}")

    # Load each modality
    expr = load_expression(expression_path, cell_ids, target_genes)
    crispr, has_crispr = load_crispr(crispr_path, cell_ids, target_genes)
    cn, has_cn = load_copy_number(cn_path, cell_ids, target_genes)
    mutation = load_mutations(mutation_path, cell_ids, target_genes)

    # Impute missing CRISPR and CN with population means
    print(f"\nImputing missing data...")
    print(f"  CRISPR: {has_crispr.sum()}/{len(cell_ids)} cells have data")
    print(f"  CN: {has_cn.sum()}/{len(cell_ids)} cells have data")
    crispr = impute_missing(crispr, has_crispr)
    cn = impute_missing(cn, has_cn)

    # Stack channels: (N_cells, N_genes, 4) then flatten to (N_cells, N_genes*4)
    # Channel order: expression, dependency, copy_number, mutation
    stacked = np.stack([expr, crispr, cn, mutation], axis=-1)  # (N, G, 4)
    flat = stacked.reshape(len(cell_ids), -1)  # (N, G*4)

    # Add binary flags
    flags = np.stack([has_crispr.astype(np.float32), has_cn.astype(np.float32)], axis=-1)  # (N, 2)
    features = np.concatenate([flat, flags], axis=-1)  # (N, G*4 + 2)

    print(f"\nFinal feature shape: {features.shape}")
    print(f"  = {len(target_genes)} genes × 4 channels + 2 flags")

    # Save
    tensor = torch.from_numpy(features)
    output_path = os.path.join(output_dir, 'cell_features_multiomic.pt')
    torch.save(tensor, output_path)
    print(f"Saved features to {output_path}")

    # Save config
    config = {
        'target_genes': target_genes,
        'n_genes': len(target_genes),
        'channels': ['expression_rank', 'crispr_dependency', 'copy_number', 'mutation_pathogenicity'],
        'n_channels': 4,
        'flags': ['has_crispr', 'has_cn'],
        'feature_dim': features.shape[1],
        'n_cells': len(cell_ids),
        'cell_ids': cell_ids,
        'coverage': {
            'expression': int(len(cell_ids)),
            'crispr': int(has_crispr.sum()),
            'copy_number': int(has_cn.sum()),
        }
    }
    config_path = os.path.join(output_dir, 'cell_feature_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Summary stats
    print(f"\n--- Feature Summary ---")
    print(f"Expression (rank-normalized): min={expr.min():.3f}, max={expr.max():.3f}, mean={expr.mean():.3f}")
    print(f"CRISPR dependency: min={crispr.min():.3f}, max={crispr.max():.3f}, mean={crispr.mean():.3f}")
    print(f"Copy number: min={cn.min():.3f}, max={cn.max():.3f}, mean={cn.mean():.3f}")
    print(f"Mutation pathogenicity: min={mutation.min():.3f}, max={mutation.max():.3f}, mean={mutation.mean():.3f}, nonzero={np.count_nonzero(mutation)}")


if __name__ == '__main__':
    main()
