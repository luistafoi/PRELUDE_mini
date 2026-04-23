# scripts/inference_sanger.py
#
# Full GNN inference on Sanger cross-dataset evaluation pairs.
# Uses the real HetAgg model with message passing through Cell-Gene and
# Cell-Cell edges — no lightweight clone, no weight mapping.
#
# Expects data/sanger_processed/ built by prep_sanger_graph.py with
# generate_neighbors.py already run.

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg


# --- LinkPredictionDataset (same as train.py) ---
class LinkPredictionDataset(Dataset):
    def __init__(self, labeled_links, full_dataset: PRELUDEDataset):
        self.full_dataset = full_dataset
        self.links = list(labeled_links)

        n_pos = sum(1 for _, _, l in self.links if l > 0.5)
        n_neg = sum(1 for _, _, l in self.links if l <= 0.5)
        print(f"    Sanger LP Dataset: {len(self.links)} links (pos={n_pos}, neg={n_neg})")

        self.u_lids = np.zeros(len(self.links), dtype=np.int64)
        self.v_lids = np.zeros(len(self.links), dtype=np.int64)
        self.u_types = np.zeros(len(self.links), dtype=np.int64)
        self.u_gids = np.zeros(len(self.links), dtype=np.int64)
        self.v_gids = np.zeros(len(self.links), dtype=np.int64)
        self.labels = np.zeros(len(self.links), dtype=np.float32)

        if len(self.links) == 0:
            return

        for i, (u_gid, v_gid, label) in enumerate(self.links):
            u_type_id, u_lid = self.full_dataset.nodes['type_map'][u_gid]
            v_type_id, v_lid = self.full_dataset.nodes['type_map'][v_gid]
            self.u_lids[i] = u_lid
            self.v_lids[i] = v_lid
            self.u_types[i] = u_type_id
            self.u_gids[i] = u_gid
            self.v_gids[i] = v_gid
            self.labels[i] = float(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.u_lids[idx], dtype=torch.long),
                torch.tensor(self.v_lids[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float),
                torch.tensor(self.u_types[idx], dtype=torch.long),
                torch.tensor(self.u_gids[idx], dtype=torch.long),
                torch.tensor(self.v_gids[idx], dtype=torch.long))


def load_sanger_links(data_dir, scenario):
    """Load sanger_S1_links.dat or sanger_S3_links.dat."""
    filename = f"sanger_{scenario}_links.dat"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"FATAL: {path} not found")
        sys.exit(1)

    links = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src, tgt = int(parts[0]), int(parts[1])
                label = float(parts[2])
                links.append((src, tgt, label))

    print(f"  > Loaded {len(links)} Sanger {scenario} links from {filename}")
    return links


def evaluate(model, data_loader, generator, device, dataset):
    """Run evaluation, return metrics + results DataFrame."""
    model.eval()

    all_pred_probs = []
    all_true_labels = []
    all_cell_gids = []
    all_drug_gids = []

    cell_type_id = dataset.node_name2type.get('cell', -1)
    drug_type_id = dataset.node_name2type.get('drug', -1)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for u_lids_b, v_lids_b, labels_b, u_types_b, u_gids_b, v_gids_b in pbar:
            u_lids_b = u_lids_b.to(device)
            v_lids_b = v_lids_b.to(device)
            u_types_b = u_types_b.to(device)

            if u_lids_b.numel() == 0:
                continue

            u_type_id = u_types_b[0].item()

            if u_type_id == drug_type_id:
                drug_lids, cell_lids = u_lids_b, v_lids_b
                drug_gids, cell_gids = u_gids_b, v_gids_b
            else:
                drug_lids, cell_lids = v_lids_b, u_lids_b
                drug_gids, cell_gids = v_gids_b, u_gids_b

            preds = model.link_prediction_forward(drug_lids, cell_lids, generator)

            all_pred_probs.extend(preds.cpu().numpy())
            all_true_labels.extend(labels_b.numpy())
            all_cell_gids.extend(cell_gids.numpy())
            all_drug_gids.extend(drug_gids.numpy())

    results_df = pd.DataFrame({
        'cell_gid': all_cell_gids,
        'drug_gid': all_drug_gids,
        'true_label': all_true_labels,
        'pred_prob': all_pred_probs,
    })

    # Add readable names
    results_df['cell_name'] = results_df['cell_gid'].map(
        lambda g: dataset.id2node.get(int(g), ''))
    results_df['drug_name'] = results_df['drug_gid'].map(
        lambda g: dataset.id2node.get(int(g), ''))

    # Metrics
    true_np = results_df['true_label'].values
    pred_np = results_df['pred_prob'].values
    binary_labels = (true_np > 0.5).astype(int)

    metrics = {}
    if len(np.unique(binary_labels)) >= 2:
        metrics['ROC-AUC'] = roc_auc_score(binary_labels, pred_np)
        metrics['F1-Score'] = f1_score(binary_labels, pred_np > 0.5)
        metrics['PRC-AUC'] = average_precision_score(binary_labels, pred_np)
    else:
        metrics['ROC-AUC'] = 0.0
        metrics['F1-Score'] = 0.0
        metrics['PRC-AUC'] = 0.0
        print("Warning: Only one class in labels — metrics may be meaningless.")

    # MRR per cell
    reciprocal_ranks = []
    for cell_gid, group in results_df.groupby('cell_gid'):
        positives = group[group['true_label'] > 0.5]
        if len(positives) == 0:
            continue
        ranked = group.sort_values(by='pred_prob', ascending=False).reset_index(drop=True)
        for _, pos_row in positives.iterrows():
            rank_idx = ranked.index[ranked['drug_gid'] == pos_row['drug_gid']]
            if len(rank_idx) > 0:
                reciprocal_ranks.append(1.0 / (rank_idx[0] + 1))
    metrics['MRR'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return metrics, results_df


def main():
    parser = argparse.ArgumentParser(
        description="Sanger cross-dataset evaluation with full GNN inference.")

    # Paths
    parser.add_argument('--data_dir', type=str, default='data/sanger_processed',
                        help='Directory with augmented graph files.')
    parser.add_argument('--embedding_dir', type=str, default='data/sanger_processed/embeddings',
                        help='Directory with augmented embedding files.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth).')
    parser.add_argument('--output_dir', type=str, default='results/sanger_gnn',
                        help='Directory to save prediction results.')

    # Scenario
    parser.add_argument('--scenario', type=str, required=True, choices=['S1', 'S2', 'S3', 'S4'],
                        help='Evaluation scenario: S1 (known/known), S2 (known cell/new drug), '
                             'S3 (new cell/known drug), S4 (new/new).')

    # Model architecture (must match training)
    parser.add_argument('--embed_d', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--max_neighbors', type=int, default=20)
    parser.add_argument('--use_skip_connection', action='store_true')
    parser.add_argument('--use_node_gate', action='store_true',
                        help='Per-node adaptive skip gate MLP (must match training config).')
    parser.add_argument('--dropout', type=float, default=0.2)

    # Cell features
    parser.add_argument('--cell_feature_source', type=str, default='vae',
                        choices=['vae', 'multiomic', 'hybrid', 'scgpt'])
    parser.add_argument('--gene_encoder_dim', type=int, default=0,
                        help='Per-gene MLP hidden dim (0=disabled).')
    parser.add_argument('--use_cross_attention', action='store_true')
    parser.add_argument('--cross_attn_dim', type=int, default=32)

    # Hardware
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Scenario: {args.scenario}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")

    # --- Load augmented graph ---
    print("\n--- Loading augmented graph ---")
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device, embedding_dir=args.embedding_dir,
                                    cell_feature_source=args.cell_feature_source)
    generator = DataGenerator(args.data_dir)

    # --- Initialize model (same architecture as training) ---
    print("\n--- Initializing model ---")
    model_args = argparse.Namespace(
        embed_d=args.embed_d,
        n_layers=args.n_layers,
        max_neighbors=args.max_neighbors,
        use_skip_connection=args.use_skip_connection,
        use_node_gate=args.use_node_gate,
        dropout=args.dropout,
        data_dir=args.data_dir,
        cell_feature_source=args.cell_feature_source,
        gene_encoder_dim=args.gene_encoder_dim,
        use_cross_attention=args.use_cross_attention,
        cross_attn_dim=args.cross_attn_dim,
    )
    model = HetAgg(model_args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")

    # --- Load checkpoint ---
    print(f"\n--- Loading checkpoint ---")
    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # strict=False because augmented graph may have different pos_weight etc.
    # But all learned parameters should match.
    model.load_state_dict(state_dict, strict=False)
    print("  > Checkpoint loaded successfully.")
    model.eval()

    # --- Load Sanger evaluation links ---
    print(f"\n--- Loading Sanger {args.scenario} evaluation links ---")
    sanger_links = load_sanger_links(args.data_dir, args.scenario)

    if not sanger_links:
        print("No evaluation links found. Exiting.")
        return

    eval_dataset = LinkPredictionDataset(sanger_links, dataset)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Evaluate ---
    print(f"\n--- Running evaluation ({args.scenario}) ---")
    metrics, results_df = evaluate(model, eval_loader, generator, device, dataset)

    # --- Report ---
    print(f"\n{'='*50}")
    print(f"Sanger {args.scenario} Results (Full GNN Inference)")
    print(f"{'='*50}")
    print(f"  ROC-AUC:  {metrics['ROC-AUC']:.4f}")
    print(f"  PRC-AUC:  {metrics['PRC-AUC']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print(f"  Pairs:    {len(results_df)}")
    print(f"{'='*50}")

    # Per-drug breakdown
    print(f"\nPer-drug breakdown:")
    for drug_gid, group in results_df.groupby('drug_gid'):
        drug_name = dataset.id2node.get(int(drug_gid), '?')
        binary = (group['true_label'] > 0.5).astype(int)
        if len(binary.unique()) >= 2:
            auc = roc_auc_score(binary, group['pred_prob'])
        else:
            auc = float('nan')
        n_pos = binary.sum()
        n_neg = len(binary) - n_pos
        print(f"  {drug_name:25s}  AUC={auc:.4f}  pos={n_pos}  neg={n_neg}")

    # --- Save predictions ---
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"sanger_{args.scenario}_gnn_preds.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")


if __name__ == "__main__":
    main()
