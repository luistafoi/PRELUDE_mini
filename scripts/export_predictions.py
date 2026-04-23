"""Export all (cell, drug) predictions with scores for downstream analysis.

Produces:
  - test_inductive_predictions.csv — full test inductive predictions (all cells x all drugs)
  - test_transductive_predictions.csv — test transductive predictions
  - per_drug_summary.csv — per-drug AUC, accuracy, score distribution
  - per_cell_summary.csv — per-cell AUC, top drugs, accuracy

Usage:
    python scripts/export_predictions.py --model_name v2_M03_drug_sim --data_dir data/processed_v2 --gpu 0
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg
from scripts.train import LinkPredictionDataset


def load_model(args, dataset, feature_loader, device):
    model_args = argparse.Namespace(
        data_dir=args.data_dir, embed_d=256, n_layers=2, max_neighbors=10,
        dropout=0.2, use_skip_connection=True, use_node_gate=True,
        cell_feature_source='vae', gene_encoder_dim=0, use_cross_attention=False,
        cross_attn_dim=32, freeze_cell_gnn=False, regression=False,
        use_residual_gnn=False, residual_scale=0.0, ema_lambda=0.0, ema_momentum=0.999,
        include_cell_drug=True, dual_head=True, inductive_loss_weight=1.0,
        backbone_lr_scale=1.0,
    )
    model = HetAgg(model_args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction('drug', 'cell')
    model.load_state_dict(torch.load(f'checkpoints/{args.model_name}.pth', map_location=device))
    model.eval()
    return model


def run_inference(model, dataset, links, device, generator, use_inductive_head):
    """Run inference on a set of links."""
    drug_type = dataset.node_name2type['drug']
    cell_type = dataset.node_name2type['cell']

    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()

    ds = LinkPredictionDataset(links, dataset)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)

    gid_to_name = dict(dataset.id2node)
    all_rows = []

    with torch.no_grad():
        for u_lids, v_lids, labels, u_types, u_gids, v_gids in loader:
            u_lids = u_lids.to(device)
            v_lids = v_lids.to(device)
            u_type = u_types[0].item()

            if u_type == drug_type:
                drug_lids, cell_lids = u_lids, v_lids
                drug_gids, cell_gids = u_gids, v_gids
            else:
                drug_lids, cell_lids = v_lids, u_lids
                drug_gids, cell_gids = v_gids, u_gids

            preds = model.link_prediction_forward(
                drug_lids, cell_lids, generator,
                embedding_tables=eval_tables,
                use_inductive_head=use_inductive_head,
            )

            for i in range(len(preds)):
                all_rows.append({
                    'cell_gid': int(cell_gids[i].item()),
                    'cell': gid_to_name.get(int(cell_gids[i].item()), ''),
                    'drug_gid': int(drug_gids[i].item()),
                    'drug': gid_to_name.get(int(drug_gids[i].item()), ''),
                    'pred_score': float(preds[i].item()),
                    'true_label': int(labels[i].item() > 0.5),
                })

    return pd.DataFrame(all_rows)


def compute_summaries(df):
    """Compute per-drug and per-cell summary metrics."""
    df['correct'] = (df['pred_score'] > 0.5) == (df['true_label'] > 0.5)
    df['pred_class'] = (df['pred_score'] > 0.5).astype(int)

    # Per-drug summary
    drug_rows = []
    for drug, group in df.groupby('drug'):
        binary = group['true_label'].values
        preds = group['pred_score'].values
        row = {
            'drug': drug,
            'drug_gid': int(group['drug_gid'].iloc[0]),
            'n_cells': len(group),
            'n_sensitive_true': int((binary > 0.5).sum()),
            'n_resistant_true': int((binary <= 0.5).sum()),
            'n_sensitive_pred': int((preds > 0.5).sum()),
            'accuracy': float(group['correct'].mean()),
            'mean_score': float(preds.mean()),
            'median_score': float(np.median(preds)),
            'std_score': float(preds.std()),
        }
        if len(np.unique(binary)) == 2:
            row['roc_auc'] = float(roc_auc_score(binary, preds))
            row['f1'] = float(f1_score(binary, preds > 0.5))
        else:
            row['roc_auc'] = None
            row['f1'] = None
        drug_rows.append(row)

    drug_df = pd.DataFrame(drug_rows).sort_values('roc_auc', ascending=False, na_position='last')

    # Per-cell summary
    cell_rows = []
    for cell, group in df.groupby('cell'):
        binary = group['true_label'].values
        preds = group['pred_score'].values
        top_drugs = group.nlargest(5, 'pred_score')['drug'].tolist()
        row = {
            'cell': cell,
            'cell_gid': int(group['cell_gid'].iloc[0]),
            'n_drugs': len(group),
            'accuracy': float(group['correct'].mean()),
            'mean_score': float(preds.mean()),
            'top_5_predicted_sensitive': ', '.join(top_drugs),
        }
        if len(np.unique(binary)) == 2:
            row['roc_auc'] = float(roc_auc_score(binary, preds))
        else:
            row['roc_auc'] = None
        cell_rows.append(row)

    cell_df = pd.DataFrame(cell_rows).sort_values('roc_auc', ascending=False, na_position='last')

    return drug_df, cell_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed_v2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'results/{args.model_name}/predictions'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"EXPORT PREDICTIONS: {args.model_name}")
    print("=" * 60)

    # Load everything
    print("\n--- Loading ---")
    emb_dir = os.path.join(args.data_dir, 'embeddings')
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device, embedding_dir=emb_dir)
    generator = DataGenerator(args.data_dir, include_cell_drug=True)
    model = load_model(args, dataset, feature_loader, device)

    # --- Test inductive ---
    print("\n--- Test Inductive ---")
    links = dataset.links.get('test_inductive', [])
    if links:
        df_ind = run_inference(model, dataset, links, device, generator, use_inductive_head=True)
        print(f"  Pairs: {len(df_ind):,}")
        print(f"  Unique cells: {df_ind['cell'].nunique()}, Drugs: {df_ind['drug'].nunique()}")

        overall_auc = roc_auc_score(df_ind['true_label'], df_ind['pred_score'])
        print(f"  Overall AUC: {overall_auc:.4f}")

        df_ind.to_csv(os.path.join(args.output_dir, 'test_inductive_predictions.csv'), index=False)

        drug_summary, cell_summary = compute_summaries(df_ind)
        drug_summary.to_csv(os.path.join(args.output_dir, 'test_inductive_per_drug.csv'), index=False)
        cell_summary.to_csv(os.path.join(args.output_dir, 'test_inductive_per_cell.csv'), index=False)

        print(f"\n  Top 10 drugs by AUC:")
        for _, row in drug_summary.head(10).iterrows():
            if row['roc_auc'] is not None:
                print(f"    {row['drug'][:30]:30s}  AUC={row['roc_auc']:.3f}  "
                      f"(sens={row['n_sensitive_true']}, res={row['n_resistant_true']})")

    # --- Test transductive ---
    print("\n--- Test Transductive ---")
    links_t = dataset.links.get('test_transductive', [])
    if links_t:
        df_trans = run_inference(model, dataset, links_t, device, generator, use_inductive_head=False)
        print(f"  Pairs: {len(df_trans):,}")

        overall_auc_t = roc_auc_score(df_trans['true_label'], df_trans['pred_score'])
        print(f"  Overall AUC: {overall_auc_t:.4f}")

        df_trans.to_csv(os.path.join(args.output_dir, 'test_transductive_predictions.csv'), index=False)

    # --- Sanger scenarios ---
    for scenario in ['S1', 'S2', 'S3', 'S4']:
        path = os.path.join(args.data_dir, f'sanger_{scenario}_links.dat')
        if not os.path.exists(path):
            continue
        print(f"\n--- Sanger {scenario} ---")
        sanger_links = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                sanger_links.append((int(parts[0]), int(parts[1]), float(parts[2])))

        use_ind = scenario != 'S1'
        df_s = run_inference(model, dataset, sanger_links, device, generator, use_inductive_head=use_ind)
        print(f"  Pairs: {len(df_s):,}")

        if len(np.unique(df_s['true_label'])) == 2:
            auc = roc_auc_score(df_s['true_label'], df_s['pred_score'])
            print(f"  AUC: {auc:.4f}")

        df_s.to_csv(os.path.join(args.output_dir, f'sanger_{scenario}_predictions.csv'), index=False)

        drug_summary_s, _ = compute_summaries(df_s)
        drug_summary_s.to_csv(os.path.join(args.output_dir, f'sanger_{scenario}_per_drug.csv'), index=False)

    print(f"\n--- Complete ---")
    print(f"  Output: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f)) / 1024
        print(f"    {f} ({size:.0f} KB)")


if __name__ == '__main__':
    main()
