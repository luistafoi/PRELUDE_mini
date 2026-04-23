"""Master diagnostic plotting script for PRELUDE v2.

Generates comprehensive diagnostic plots from training logs and model checkpoints.
Designed to be run after training or appended to run scripts.

Plots generated:
  1. Training curves (loss, AUC, F1 over epochs)
  2. Train vs validation loss (overfitting detection)
  3. Gate alpha dynamics
  4. ROC and PRC curves (test sets + Sanger S1-S4)
  5. Per-tissue AUC breakdown
  6. Per-drug AUC breakdown
  7. t-SNE of cell embeddings (colored by tissue)
  8. Confusion matrices
  9. Summary comparison panel

Usage:
    python scripts/plot_diagnostics_v2.py --model_name v2_M01_baseline --data_dir data/processed_v2
    python scripts/plot_diagnostics_v2.py --model_name v2_M01_baseline --data_dir data/processed_v2 --skip_tsne
"""

import os
import sys
import argparse
import pickle
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score, confusion_matrix)
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg
from utils.evaluation import evaluate_model

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# ================================================================
# HELPER: Sanger Dataset
# ================================================================
class SangerLPDataset(Dataset):
    def __init__(self, links_path, dataset):
        self.pairs = []
        with open(links_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                src, tgt = int(parts[0]), int(parts[1])
                label = float(parts[2])
                src_type, src_lid = dataset.nodes['type_map'][src]
                tgt_type, tgt_lid = dataset.nodes['type_map'][tgt]
                self.pairs.append((src_lid, tgt_lid, label, src_type, src, tgt))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, t, l, st, sg, tg = self.pairs[i]
        return (torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long),
                torch.tensor(l, dtype=torch.float), torch.tensor(st, dtype=torch.long),
                torch.tensor(sg, dtype=torch.long), torch.tensor(tg, dtype=torch.long))


def predict_sanger(model, dataset, data_dir, scenario, device, eval_tables, generator):
    """Run predictions on a Sanger scenario, return labels + preds + metadata."""
    path = os.path.join(data_dir, f'sanger_{scenario}_links.dat')
    if not os.path.exists(path):
        return None

    ds = SangerLPDataset(path, dataset)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    drug_type = dataset.node_name2type['drug']

    all_preds, all_labels, all_cell_gids, all_drug_gids = [], [], [], []
    with torch.no_grad():
        for s_lids, t_lids, labels, s_types, s_gids, t_gids in loader:
            s_lids, t_lids = s_lids.to(device), t_lids.to(device)
            if s_types[0].item() == drug_type:
                drug_lids, cell_lids = s_lids, t_lids
                drug_gids, cell_gids = s_gids, t_gids
            else:
                drug_lids, cell_lids = t_lids, s_lids
                drug_gids, cell_gids = t_gids, s_gids

            use_ind = scenario != 'S1'
            preds = model.link_prediction_forward(
                drug_lids, cell_lids, generator,
                embedding_tables=eval_tables, use_inductive_head=use_ind)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cell_gids.extend(cell_gids.numpy())
            all_drug_gids.extend(drug_gids.numpy())

    return {
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'cell_gids': np.array(all_cell_gids),
        'drug_gids': np.array(all_drug_gids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed_v2')
    parser.add_argument('--emb_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_tsne', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.emb_dir is None:
        args.emb_dir = os.path.join(args.data_dir, 'embeddings')
    if args.output_dir is None:
        args.output_dir = f'results/{args.model_name}'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    log_path = f'checkpoints/{args.model_name}_log.csv'
    ckpt_path = f'checkpoints/{args.model_name}.pth'

    # ================================================================
    # PLOT 1: Training Curves
    # ================================================================
    print("--- Plot 1: Training Curves ---")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{args.model_name} — Training Diagnostics', fontsize=14, fontweight='bold')

        # 1a. Loss curves
        ax = axes[0, 0]
        ax.plot(log['epoch'], log['lp_loss'], label='LP Loss', linewidth=2)
        if 'triplet_loss' in log.columns:
            ax.plot(log['epoch'], log['triplet_loss'], label='Triplet Loss', linewidth=1.5)
        ax.plot(log['epoch'], log['total_loss'], label='Total Loss', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1b. AUC curves
        ax = axes[0, 1]
        ax.plot(log['epoch'], log['val_ind_auc'], label='Val Inductive', linewidth=2, color='tab:blue')
        if 'val_trans_auc' in log.columns:
            ax.plot(log['epoch'], log['val_trans_auc'], label='Val Transductive', linewidth=2, color='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('Validation AUC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1c. F1 curves
        ax = axes[0, 2]
        ax.plot(log['epoch'], log['val_ind_f1'], label='Val Inductive', linewidth=2, color='tab:blue')
        if 'val_trans_f1' in log.columns:
            ax.plot(log['epoch'], log['val_trans_f1'], label='Val Transductive', linewidth=2, color='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1')
        ax.set_title('Validation F1')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1d. Learning rate
        ax = axes[1, 0]
        ax.plot(log['epoch'], log['lr'], linewidth=2, color='tab:green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 1e. MAD (over-smoothing)
        ax = axes[1, 1]
        if 'mad_cell' in log.columns:
            ax.plot(log['epoch'], log['mad_cell'], label='Cell', linewidth=1.5)
            ax.plot(log['epoch'], log['mad_drug'], label='Drug', linewidth=1.5)
            ax.plot(log['epoch'], log['mad_gene'], label='Gene', linewidth=1.5)
            ax.plot(log['epoch'], log['mad_overall'], label='Overall', linewidth=2, linestyle='--')
            ax.set_title('MAD (Over-smoothing)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAD')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1f. Gate alpha
        ax = axes[1, 2]
        if 'gate_cell' in log.columns:
            ax.plot(log['epoch'], log['gate_cell'], label='Cell', linewidth=2)
            ax.plot(log['epoch'], log['gate_drug'], label='Drug', linewidth=2)
            ax.plot(log['epoch'], log['gate_gene'], label='Gene', linewidth=2)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_title('Gate α (1=proj, 0=GNN)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Alpha')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/01_training_curves.png')
        plt.close()
        print(f"  Saved: {args.output_dir}/01_training_curves.png")

    # ================================================================
    # Load model for remaining plots
    # ================================================================
    print("\n--- Loading model ---")
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device, embedding_dir=args.emb_dir)
    generator = DataGenerator(args.data_dir, include_cell_drug=True)

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

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()

    # Build GID → name lookups
    gid_to_name = {gid: name for gid, name in dataset.id2node.items()}

    # Load metadata
    model_csv = pd.read_csv('data/misc/Model.csv')
    ach_to_tissue = {row['ModelID']: row['OncotreeLineage']
                     for _, row in model_csv.iterrows() if pd.notna(row.get('OncotreeLineage'))}

    # ================================================================
    # PLOT 2: ROC & PRC for DepMap Test Sets
    # ================================================================
    print("\n--- Plot 2: ROC & PRC Curves ---")
    from scripts.train import LinkPredictionDataset

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'{args.model_name} — ROC & PRC Curves', fontsize=14, fontweight='bold')

    # DepMap test sets
    for idx, (split_name, use_ind) in enumerate([
        ('test_inductive', True), ('test_transductive', False)
    ]):
        links = dataset.links.get(split_name, [])
        if not links:
            continue
        ds = LinkPredictionDataset(links, dataset)
        loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
        metrics, results_df = evaluate_model(model, loader, generator, device, dataset,
                                             embedding_tables=eval_tables, use_inductive_head=use_ind)

        binary = (results_df['true_label'].values > 0.5).astype(int)
        preds = results_df['pred_prob'].values

        # ROC
        fpr, tpr, _ = roc_curve(binary, preds)
        axes[0, idx].plot(fpr, tpr, linewidth=2, label=f'AUC={metrics["ROC-AUC"]:.4f}')
        axes[0, idx].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[0, idx].set_title(f'DepMap {split_name.replace("_", " ").title()}')
        axes[0, idx].set_xlabel('FPR')
        axes[0, idx].set_ylabel('TPR')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # PRC
        prec, rec, _ = precision_recall_curve(binary, preds)
        ap = average_precision_score(binary, preds)
        axes[1, idx].plot(rec, prec, linewidth=2, label=f'AP={ap:.4f}')
        axes[1, idx].set_xlabel('Recall')
        axes[1, idx].set_ylabel('Precision')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

    # Sanger S1 and S3 (most important)
    for idx, scenario in enumerate(['S1', 'S3']):
        result = predict_sanger(model, dataset, args.data_dir, scenario, device, eval_tables, generator)
        if result is None:
            continue

        binary = (result['labels'] > 0.5).astype(int)
        if len(np.unique(binary)) < 2:
            continue

        auc = roc_auc_score(binary, result['preds'])
        fpr, tpr, _ = roc_curve(binary, result['preds'])
        axes[0, 2].plot(fpr, tpr, linewidth=2, label=f'{scenario} AUC={auc:.4f}')

        prec, rec, _ = precision_recall_curve(binary, result['preds'])
        ap = average_precision_score(binary, result['preds'])
        axes[1, 2].plot(rec, prec, linewidth=2, label=f'{scenario} AP={ap:.4f}')

    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 2].set_title('Sanger Cross-Dataset')
    axes[0, 2].set_xlabel('FPR')
    axes[0, 2].set_ylabel('TPR')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlabel('Recall')
    axes[1, 2].set_ylabel('Precision')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/02_roc_prc.png')
    plt.close()
    print(f"  Saved: {args.output_dir}/02_roc_prc.png")

    # ================================================================
    # PLOT 3: Per-Tissue and Per-Drug Breakdown
    # ================================================================
    print("\n--- Plot 3: Per-Tissue & Per-Drug AUC ---")

    # Use Sanger S3 for breakdown (most interesting — new cells)
    result_s3 = predict_sanger(model, dataset, args.data_dir, 'S3', device, eval_tables, generator)

    if result_s3 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{args.model_name} — Sanger S3 Breakdown (New Cells + Known Drugs)',
                     fontsize=14, fontweight='bold')

        binary = (result_s3['labels'] > 0.5).astype(int)

        # Per-tissue
        tissue_aucs = {}
        for i in range(len(result_s3['cell_gids'])):
            cell_name = gid_to_name.get(int(result_s3['cell_gids'][i]), '')
            tissue = ach_to_tissue.get(cell_name, 'Unknown')
            if tissue not in tissue_aucs:
                tissue_aucs[tissue] = {'preds': [], 'labels': []}
            tissue_aucs[tissue]['preds'].append(result_s3['preds'][i])
            tissue_aucs[tissue]['labels'].append(binary[i])

        tissue_results = []
        for tissue, data in tissue_aucs.items():
            labels = np.array(data['labels'])
            if len(np.unique(labels)) == 2 and len(labels) >= 10:
                auc = roc_auc_score(labels, data['preds'])
                tissue_results.append((tissue, auc, len(labels)))

        tissue_results.sort(key=lambda x: x[1], reverse=True)
        if tissue_results:
            names = [f"{t[:20]} (n={n})" for t, a, n in tissue_results[:20]]
            aucs = [a for _, a, _ in tissue_results[:20]]
            colors = ['tab:green' if a >= 0.8 else 'tab:blue' if a >= 0.6 else 'tab:red' for a in aucs]
            axes[0].barh(range(len(names)), aucs, color=colors, alpha=0.8)
            axes[0].set_yticks(range(len(names)))
            axes[0].set_yticklabels(names, fontsize=8)
            axes[0].set_xlabel('ROC-AUC')
            axes[0].set_title('Per-Tissue AUC (Top 20)')
            axes[0].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
            axes[0].set_xlim(0, 1)
            axes[0].invert_yaxis()

        # Per-drug
        drug_aucs = {}
        for i in range(len(result_s3['drug_gids'])):
            drug_name = gid_to_name.get(int(result_s3['drug_gids'][i]), '')
            if drug_name not in drug_aucs:
                drug_aucs[drug_name] = {'preds': [], 'labels': []}
            drug_aucs[drug_name]['preds'].append(result_s3['preds'][i])
            drug_aucs[drug_name]['labels'].append(binary[i])

        drug_results = []
        for drug, data in drug_aucs.items():
            labels = np.array(data['labels'])
            if len(np.unique(labels)) == 2 and len(labels) >= 10:
                auc = roc_auc_score(labels, data['preds'])
                drug_results.append((drug, auc, len(labels)))

        drug_results.sort(key=lambda x: x[1], reverse=True)
        if drug_results:
            names = [f"{d[:25]} (n={n})" for d, a, n in drug_results[:20]]
            aucs = [a for _, a, _ in drug_results[:20]]
            colors = ['tab:green' if a >= 0.8 else 'tab:blue' if a >= 0.6 else 'tab:red' for a in aucs]
            axes[1].barh(range(len(names)), aucs, color=colors, alpha=0.8)
            axes[1].set_yticks(range(len(names)))
            axes[1].set_yticklabels(names, fontsize=8)
            axes[1].set_xlabel('ROC-AUC')
            axes[1].set_title('Per-Drug AUC (Top 20)')
            axes[1].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
            axes[1].set_xlim(0, 1)
            axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/03_tissue_drug_breakdown.png')
        plt.close()
        print(f"  Saved: {args.output_dir}/03_tissue_drug_breakdown.png")

    # ================================================================
    # PLOT 4: Sanger S1-S4 Comparison
    # ================================================================
    print("\n--- Plot 4: Sanger S1-S4 Summary ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{args.model_name} — Sanger Cross-Dataset Summary', fontsize=14, fontweight='bold')

    descs = {'S1': 'Known cells\nKnown drugs', 'S2': 'Known cells\nNew drugs',
             'S3': 'New cells\nKnown drugs', 'S4': 'New cells\nNew drugs'}
    sanger_aucs = {}
    sanger_f1s = {}

    for scenario in ['S1', 'S2', 'S3', 'S4']:
        result = predict_sanger(model, dataset, args.data_dir, scenario, device, eval_tables, generator)
        if result is None:
            continue
        binary = (result['labels'] > 0.5).astype(int)
        if len(np.unique(binary)) == 2:
            sanger_aucs[scenario] = roc_auc_score(binary, result['preds'])
            sanger_f1s[scenario] = f1_score(binary, result['preds'] > 0.5)

    if sanger_aucs:
        scenarios = list(sanger_aucs.keys())
        aucs = [sanger_aucs[s] for s in scenarios]
        f1s = [sanger_f1s[s] for s in scenarios]
        x = range(len(scenarios))

        colors = ['tab:green' if a > 0.7 else 'tab:orange' if a > 0.55 else 'tab:red' for a in aucs]
        axes[0].bar(x, aucs, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{s}\n{descs[s]}" for s in scenarios], fontsize=9)
        axes[0].set_ylabel('ROC-AUC')
        axes[0].set_title('AUC by Scenario')
        axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
        axes[0].set_ylim(0, 1)
        for i, v in enumerate(aucs):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        axes[0].legend()

        axes[1].bar(x, f1s, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{s}\n{descs[s]}" for s in scenarios], fontsize=9)
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('F1 by Scenario')
        axes[1].set_ylim(0, 1)
        for i, v in enumerate(f1s):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/04_sanger_summary.png')
    plt.close()
    print(f"  Saved: {args.output_dir}/04_sanger_summary.png")

    # ================================================================
    # PLOT 5: t-SNE of Cell Embeddings
    # ================================================================
    if not args.skip_tsne:
        print("\n--- Plot 5: t-SNE Cell Embeddings ---")
        cell_type = dataset.node_name2type['cell']
        cell_embeds = eval_tables[cell_type].cpu().numpy()
        n_cells = cell_embeds.shape[0]

        # Get tissue labels and train/val/test split
        split_config_path = os.path.join(args.data_dir, 'split_config.json')
        import json
        with open(split_config_path) as f:
            split_config = json.load(f)
        train_set = set(split_config['train_cells'])
        val_set = set(split_config['val_cells'])
        test_set = set(split_config['test_cells'])

        tissues = []
        splits = []
        cell_names = []
        for gid in range(n_cells):
            name = gid_to_name.get(gid, '')
            cell_names.append(name)
            tissues.append(ach_to_tissue.get(name, 'Unknown'))
            if name in train_set:
                splits.append('train')
            elif name in val_set:
                splits.append('val_ind')
            elif name in test_set:
                splits.append('test_ind')
            else:
                splits.append('sanger_only')

        print(f"  Running t-SNE on {n_cells} cells...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_cells - 1))
        coords = tsne.fit_transform(cell_embeds)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'{args.model_name} — Cell Embedding t-SNE', fontsize=14, fontweight='bold')

        # By tissue (top 10)
        tissue_counts = pd.Series(tissues).value_counts()
        top_tissues = tissue_counts.head(10).index.tolist()
        cmap = plt.cm.get_cmap('tab10')

        ax = axes[0]
        for i, tissue in enumerate(top_tissues):
            mask = [t == tissue for t in tissues]
            ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.6,
                      color=cmap(i), label=f'{tissue} ({sum(mask)})')
        other_mask = [t not in top_tissues for t in tissues]
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1], s=5, alpha=0.2,
                  color='gray', label=f'Other ({sum(other_mask)})')
        ax.set_title('By Tissue Type')
        ax.legend(fontsize=7, loc='upper right', markerscale=2)

        # By split
        ax = axes[1]
        split_colors = {'train': 'tab:blue', 'val_ind': 'tab:orange',
                       'test_ind': 'tab:red', 'sanger_only': 'tab:green'}
        for split_name, color in split_colors.items():
            mask = [s == split_name for s in splits]
            if sum(mask) > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.6,
                          color=color, label=f'{split_name} ({sum(mask)})')
        ax.set_title('By Data Split')
        ax.legend(fontsize=9, markerscale=2)

        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/05_tsne_cells.png')
        plt.close()
        print(f"  Saved: {args.output_dir}/05_tsne_cells.png")

    # ================================================================
    # PLOT 6: Confusion Matrices
    # ================================================================
    print("\n--- Plot 6: Confusion Matrices ---")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'{args.model_name} — Confusion Matrices', fontsize=14, fontweight='bold')

    for idx, scenario in enumerate(['S1', 'S2', 'S3', 'S4']):
        result = predict_sanger(model, dataset, args.data_dir, scenario, device, eval_tables, generator)
        if result is None:
            continue
        binary = (result['labels'] > 0.5).astype(int)
        pred_binary = (result['preds'] > 0.5).astype(int)
        cm = confusion_matrix(binary, pred_binary)

        ax = axes[idx]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{scenario}\n{descs[scenario]}', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Resistant', 'Sensitive'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Resistant', 'Sensitive'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                       color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/06_confusion_matrices.png')
    plt.close()
    print(f"  Saved: {args.output_dir}/06_confusion_matrices.png")

    # ================================================================
    # Summary text file
    # ================================================================
    print("\n--- Summary ---")
    summary_path = f'{args.output_dir}/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Data: {args.data_dir}\n\n")

        if os.path.exists(log_path):
            log = pd.read_csv(log_path)
            f.write(f"Training: {len(log)} epochs\n")
            f.write(f"Best Val Ind AUC: {log['val_ind_auc'].max():.4f} (epoch {log['val_ind_auc'].idxmax() + 1})\n")
            f.write(f"Final Val Ind AUC: {log['val_ind_auc'].iloc[-1]:.4f}\n\n")

        f.write("Sanger Results:\n")
        for s in ['S1', 'S2', 'S3', 'S4']:
            if s in sanger_aucs:
                f.write(f"  {s}: AUC={sanger_aucs[s]:.4f}, F1={sanger_f1s[s]:.4f}\n")

    print(f"  Saved: {summary_path}")
    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
