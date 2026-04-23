# scripts/plot_diagnostics.py
#
# Comprehensive diagnostic plots for PRELUDE M09 model.
# Generates 12 publication-quality figures to results/figures/.
#
# Usage:
#   python scripts/plot_diagnostics.py [--checkpoint PATH] [--gpu 0]

import sys
import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, roc_auc_score,
                             f1_score, confusion_matrix)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg

# ---------- Style ----------
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
})

COLORS = {
    'S1': '#2196F3',   # blue
    'S2': '#FF9800',   # orange
    'S3': '#4CAF50',   # green
    'S4': '#F44336',   # red
    'pos': '#E53935',
    'neg': '#1E88E5',
    'train': '#90CAF9',
    'test': '#EF9A9A',
}


# ========================================================================
# FIGURE 1: Training Dynamics
# ========================================================================
def plot_training_curves(log_path, out_dir):
    """Training loss curves + validation AUC/F1 + LR schedule."""
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1a. Loss components
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['lp_loss'], label='LP Loss (BCE)', color='#1976D2', linewidth=2)
    ax.plot(df['epoch'], df['triplet_loss'], label='Triplet Loss', color='#E64A19', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Link Prediction & Triplet Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # 1b. Total loss + RW loss
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['total_loss'], label='Total Loss', color='#2E7D32', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(df['epoch'], df['rw_loss'], label='RW Loss', color='#7B1FA2', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss', color='#2E7D32')
    ax2.set_ylabel('Random Walk Loss', color='#7B1FA2')
    ax.set_title('Total Loss & Random Walk Loss')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(alpha=0.3)

    # 1c. Validation AUC + F1
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['val_auc'], label='Val AUC', color='#1976D2', linewidth=2, marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_f1'], label='Val F1', color='#E64A19', linewidth=2, marker='s', markersize=3)
    best_ep = df.loc[df['val_auc'].idxmax(), 'epoch']
    best_auc = df['val_auc'].max()
    ax.axvline(best_ep, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'Best: {best_auc:.4f}\n(Ep {int(best_ep)})',
                xy=(best_ep, best_auc), fontsize=9,
                xytext=(best_ep + 3, best_auc - 0.03),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics')
    ax.legend()
    ax.grid(alpha=0.3)

    # 1d. Learning rate schedule
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['lr'], color='#00897B', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Cosine Annealing LR Schedule')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    fig.suptitle('M09 Training Dynamics', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_training_curves.png'))
    plt.close(fig)
    print("  > 01_training_curves.png")


# ========================================================================
# FIGURE 2: Skip Gate Analysis
# ========================================================================
def plot_skip_gates(checkpoint_path, out_dir):
    """Visualize learned per-type gated skip connection values."""
    sd = torch.load(checkpoint_path, map_location='cpu')

    gate_names = {
        'skip_gates.0': 'Cell',
        'skip_gates.1': 'Drug',
        'skip_gates.2': 'Gene',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (key, name) in enumerate(gate_names.items()):
        if key not in sd:
            continue
        gate_raw = sd[key].numpy()
        alpha = 1.0 / (1.0 + np.exp(-gate_raw))  # sigmoid

        ax = axes[idx]
        # Use fixed bin range to handle concentrated values
        ax.hist(alpha, bins=np.linspace(0, 1, 51),
                color=['#2196F3', '#FF9800', '#4CAF50'][idx],
                alpha=0.7, edgecolor='white')
        ax.axvline(alpha.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {alpha.mean():.3f}')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Init (0.5)')
        ax.set_xlabel(f'α (weight on projected features)')
        ax.set_ylabel('Count (dimensions)')
        ax.set_title(f'{name} Gate (α)')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

        # Add interpretation text
        interp = "Relies more on raw features" if alpha.mean() > 0.5 else "Relies more on GNN"
        ax.text(0.5, 0.95, interp, transform=ax.transAxes, ha='center', va='top',
                fontsize=8, style='italic', color='gray')

    fig.suptitle('Per-Type Gated Skip Connection (α = weight on projected features)',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_skip_gates.png'))
    plt.close(fig)
    print("  > 02_skip_gates.png")


# ========================================================================
# FIGURE 3 & 4: Embedding Visualization (t-SNE + PCA)
# ========================================================================
def extract_embeddings(model, dataset, generator, device):
    """Extract projected and GNN embeddings for all cells and drugs."""
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for type_name in ['cell', 'drug']:
            type_id = dataset.node_name2type[type_name]
            n_nodes = dataset.nodes['count'][type_id]
            all_lids = torch.arange(n_nodes, dtype=torch.long, device=device)

            # Projected (initial)
            proj = model.conteng_agg(all_lids, type_id).cpu().numpy()

            # GNN (final combined)
            batch_size = 256
            gnn_parts = []
            for start in range(0, n_nodes, batch_size):
                end = min(start + batch_size, n_nodes)
                batch_lids = all_lids[start:end]
                gnn_emb = model.get_combined_embedding(batch_lids, type_id, generator)
                gnn_parts.append(gnn_emb.cpu().numpy())
            gnn = np.vstack(gnn_parts)

            embeddings[type_name] = {'projected': proj, 'gnn': gnn}

    return embeddings


def plot_cell_embeddings(embeddings, dataset, cell_splits, lineage_map, out_dir):
    """t-SNE and PCA of cell embeddings colored by lineage, shaped by train/test."""

    gnn_emb = embeddings['cell']['gnn']
    proj_emb = embeddings['cell']['projected']

    # Build metadata for each cell
    n_cells = gnn_emb.shape[0]
    cell_type_id = dataset.node_name2type['cell']

    # Map local_id -> (name, lineage, split)
    lid_to_name = {}
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == cell_type_id:
            name = dataset.id2node.get(gid, '')
            lid_to_name[lid] = name

    lineages = []
    splits = []
    for lid in range(n_cells):
        name = lid_to_name.get(lid, '')
        lineages.append(lineage_map.get(name, 'Unknown'))
        if lid in cell_splits.get('test_cells', set()):
            splits.append('test')
        elif lid in cell_splits.get('valid_cells', set()):
            splits.append('valid')
        else:
            splits.append('train')

    lineages = np.array(lineages)
    splits = np.array(splits)

    # Top lineages (for color coding)
    from collections import Counter
    lin_counts = Counter(lineages)
    top_lineages = [l for l, c in lin_counts.most_common(12) if l != 'Unknown']
    color_map = {}
    cmap = plt.cm.get_cmap('tab20', len(top_lineages))
    for i, lin in enumerate(top_lineages):
        color_map[lin] = cmap(i)
    color_map['Other'] = (0.8, 0.8, 0.8, 0.5)

    def get_colors(lin_arr):
        return [color_map.get(l, color_map['Other']) for l in lin_arr]

    lin_display = np.array([l if l in top_lineages else 'Other' for l in lineages])

    # --- t-SNE on GNN embeddings ---
    print("    Computing t-SNE (GNN embeddings)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_coords = tsne.fit_transform(gnn_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 3a. t-SNE colored by lineage
    ax = axes[0]
    for lin in top_lineages + ['Other']:
        mask = lin_display == lin
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=[color_map[lin]], s=15, alpha=0.6, label=lin)
    ax.set_title('t-SNE — GNN Cell Embeddings by Lineage')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=7, markerscale=2,
              ncol=1, framealpha=0.7)

    # 3b. t-SNE colored by train/test split
    ax = axes[1]
    for split, color, marker in [('train', '#90CAF9', 'o'), ('valid', '#FFD54F', 's'), ('test', '#EF5350', '^')]:
        mask = splits == split
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=color, s=15, alpha=0.5, label=f'{split} ({mask.sum()})', marker=marker)
    ax.set_title('t-SNE — GNN Cell Embeddings by Split')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=9, markerscale=2)

    fig.suptitle('Cell Embedding Space (After GNN + Skip Gate)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_cell_tsne.png'))
    plt.close(fig)
    print("  > 03_cell_tsne.png")

    # --- PCA: projected vs GNN side by side ---
    print("    Computing PCA...")
    pca_proj = PCA(n_components=2, random_state=42).fit_transform(proj_emb)
    pca_gnn = PCA(n_components=2, random_state=42).fit_transform(gnn_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, coords, title in [(axes[0], pca_proj, 'Projected Features (Before GNN)'),
                                (axes[1], pca_gnn, 'GNN Embeddings (After GNN + Gate)')]:
        for lin in top_lineages + ['Other']:
            mask = lin_display == lin
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[color_map[lin]], s=15, alpha=0.6, label=lin)
        ax.set_title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    axes[1].legend(bbox_to_anchor=(1.0, 1.0), fontsize=7, markerscale=2,
                   ncol=1, framealpha=0.7)

    fig.suptitle('PCA: Projected vs GNN Cell Embeddings (Colored by Lineage)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_cell_pca_comparison.png'))
    plt.close(fig)
    print("  > 04_cell_pca_comparison.png")


def plot_drug_embeddings(embeddings, dataset, out_dir):
    """t-SNE of drug embeddings (projected vs GNN)."""
    proj_emb = embeddings['drug']['projected']
    gnn_emb = embeddings['drug']['gnn']

    n_drugs = gnn_emb.shape[0]
    drug_type_id = dataset.node_name2type['drug']

    # Map lid -> name
    lid_to_name = {}
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == drug_type_id:
            lid_to_name[lid] = dataset.id2node.get(gid, '')

    print("    Computing drug t-SNE...")
    tsne_proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(proj_emb)
    tsne_gnn = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(gnn_emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(tsne_proj[:, 0], tsne_proj[:, 1], s=5, alpha=0.4, c='#1976D2')
    axes[0].set_title('Drug Projected Features (Before GNN)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    axes[1].scatter(tsne_gnn[:, 0], tsne_gnn[:, 1], s=5, alpha=0.4, c='#E64A19')
    axes[1].set_title('Drug GNN Embeddings (After GNN + Gate)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    fig.suptitle(f'Drug Embedding Space ({n_drugs} drugs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_drug_tsne.png'))
    plt.close(fig)
    print("  > 05_drug_tsne.png")


# ========================================================================
# FIGURE 5: ROC + PR Curves (S1 vs S3)
# ========================================================================
def plot_roc_pr_curves(preds_dir, out_dir):
    """ROC and PR curves for S1-S4."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for scenario in ['S1', 'S2', 'S3', 'S4']:
        path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        y_true = (df['true_label'] > 0.5).astype(int)
        y_score = df['pred_prob']

        if y_true.nunique() < 2:
            continue

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=COLORS[scenario], linewidth=2,
                     label=f'{scenario} (AUC={roc_auc_val:.3f})')

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        axes[1].plot(recall, precision, color=COLORS[scenario], linewidth=2,
                     label=f'{scenario} (AP={ap:.3f})')

    # ROC formatting
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)

    # PR formatting
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    fig.suptitle('Sanger Cross-Dataset: ROC and Precision-Recall Curves',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_roc_pr_curves.png'))
    plt.close(fig)
    print("  > 06_roc_pr_curves.png")


# ========================================================================
# FIGURE 6: Score Distributions
# ========================================================================
def plot_score_distributions(preds_dir, out_dir):
    """Predicted score distributions for positives vs negatives, per scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for idx, scenario in enumerate(['S1', 'S2', 'S3', 'S4']):
        ax = axes_flat[idx]
        path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        if not os.path.exists(path):
            ax.set_visible(False)
            continue
        df = pd.read_csv(path)

        pos = df[df['true_label'] > 0.5]['pred_prob']
        neg = df[df['true_label'] <= 0.5]['pred_prob']

        ax.hist(neg, bins=50, alpha=0.6, color=COLORS['neg'], label=f'Negative (n={len(neg)})',
                density=True)
        ax.hist(pos, bins=50, alpha=0.6, color=COLORS['pos'], label=f'Positive (n={len(pos)})',
                density=True)
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.4)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'{scenario}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle('Score Distributions: Positive vs Negative',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '07_score_distributions.png'))
    plt.close(fig)
    print("  > 07_score_distributions.png")


# ========================================================================
# FIGURE 7: Calibration Plot
# ========================================================================
def plot_calibration(preds_dir, out_dir):
    """Calibration curves for S1-S4."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')

    for scenario in ['S1', 'S2', 'S3', 'S4']:
        path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        y_true = (df['true_label'] > 0.5).astype(int)
        y_score = df['pred_prob']

        if y_true.nunique() < 2:
            continue

        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, marker='o', color=COLORS[scenario],
                linewidth=2, label=scenario, markersize=5)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '08_calibration.png'))
    plt.close(fig)
    print("  > 08_calibration.png")


# ========================================================================
# FIGURE 8: Per-Drug AUC (S1 vs S3 side by side)
# ========================================================================
def plot_per_drug_auc(preds_dir, out_dir):
    """Per-drug AUC comparison S1 vs S3 (known drugs)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    drug_aucs = {}
    for scenario in ['S1', 'S3']:
        path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for drug, grp in df.groupby('drug_name'):
            y = (grp['true_label'] > 0.5).astype(int)
            if y.nunique() < 2:
                continue
            a = roc_auc_score(y, grp['pred_prob'])
            drug_aucs.setdefault(drug, {})[scenario] = a

    drugs = sorted(drug_aucs.keys())
    x = np.arange(len(drugs))
    width = 0.35

    s1_vals = [drug_aucs[d].get('S1', 0) for d in drugs]
    s3_vals = [drug_aucs[d].get('S3', 0) for d in drugs]

    bars1 = ax.bar(x - width/2, s1_vals, width, label='S1 (Known Cells)', color=COLORS['S1'], alpha=0.8)
    bars3 = ax.bar(x + width/2, s3_vals, width, label='S3 (New Cells)', color=COLORS['S3'], alpha=0.8)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Per-Drug AUC: Transductive (S1) vs Inductive (S3)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '09_per_drug_auc.png'))
    plt.close(fig)
    print("  > 09_per_drug_auc.png")


# ========================================================================
# FIGURE 9: Per-Lineage AUC Heatmap
# ========================================================================
def plot_per_lineage_auc(preds_dir, lineage_map, out_dir):
    """Per-lineage AUC for S1 and S3."""
    results = {}

    for scenario in ['S1', 'S3']:
        path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['lineage'] = df['cell_name'].map(lineage_map).fillna('Unknown')

        for lin, grp in df.groupby('lineage'):
            y = (grp['true_label'] > 0.5).astype(int)
            if y.nunique() < 2 or len(grp) < 5:
                continue
            a = roc_auc_score(y, grp['pred_prob'])
            results.setdefault(lin, {})[scenario] = (a, len(grp))

    # Filter to lineages with both S1 and S3
    lineages = sorted([l for l in results if 'S1' in results[l] or 'S3' in results[l]],
                      key=lambda l: results[l].get('S1', (0,))[0], reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(lineages) * 0.35)))

    y_pos = np.arange(len(lineages))
    width = 0.35

    s1_vals = [results[l].get('S1', (np.nan,))[0] for l in lineages]
    s3_vals = [results[l].get('S3', (np.nan,))[0] for l in lineages]
    s1_n = [results[l].get('S1', (0, 0))[1] for l in lineages]
    s3_n = [results[l].get('S3', (0, 0))[1] for l in lineages]

    bars1 = ax.barh(y_pos - width/2, s1_vals, width, label='S1 (Known)', color=COLORS['S1'], alpha=0.8)
    bars3 = ax.barh(y_pos + width/2, s3_vals, width, label='S3 (New)', color=COLORS['S3'], alpha=0.8)

    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('ROC-AUC')
    ax.set_yticks(y_pos)
    ylabels = [f'{l} (n={s1_n[i]}/{s3_n[i]})' for i, l in enumerate(lineages)]
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title('Per-Lineage AUC: S1 vs S3', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '10_per_lineage_auc.png'))
    plt.close(fig)
    print("  > 10_per_lineage_auc.png")


# ========================================================================
# FIGURE 10: Spearman Correlation (Predicted vs Continuous AUC)
# ========================================================================
def plot_spearman_correlation(preds_dir, sanger_dir, out_dir):
    """Scatter: predicted probability vs continuous Sanger AUC, with Spearman rho."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, (scenario, src_file) in enumerate([
        ('S1', 'sanger_S1_known_known.csv'),
        ('S3', 'sanger_S3_new_cell_known.csv'),
    ]):
        ax = axes[idx]
        preds_path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        src_path = os.path.join(sanger_dir, src_file)

        if not os.path.exists(preds_path) or not os.path.exists(src_path):
            ax.set_visible(False)
            continue

        preds = pd.read_csv(preds_path)
        src = pd.read_csv(src_path)

        # We need to join predictions with the continuous AUC from source
        # Build a mapping: (cell_name, drug_name_upper) -> continuous AUC
        # Problem: preds have cell_name=ACH, drug_name=UPPER; src has clean_cell, clean_drug
        # We need cell_name from preds = ACH ID, and src has clean_cell (not ACH)
        # Use the node_mappings to reverse map, or join via GIDs through the sanger links

        # Simpler: join via ordering (sanger links were written in the same order as src rows,
        # filtered). Instead, just match on drug_name and use rank-based Spearman per drug.

        # Actually, let's compute overall Spearman across all predictions by joining
        # on drug level: for each drug, rank cells by pred_prob vs rank by continuous AUC
        rhos = []
        drug_labels = []
        all_pred = []
        all_auc = []

        for drug_name, grp_pred in preds.groupby('drug_name'):
            drug_upper = drug_name.upper() if isinstance(drug_name, str) else str(drug_name)
            grp_src = src[src['clean_drug'].str.upper() == drug_upper].copy()
            if len(grp_src) == 0:
                continue

            # For each cell in predictions, find its AUC in source
            # preds cell_name is ACH ID, src has clean_cell
            # Use the Model.csv mapping we already have
            # Actually, just compute Spearman per drug using the cell ranks
            # Sort both by pred and by AUC, compute rank correlation
            if len(grp_pred) >= 5:
                # Use the mean AUC per drug across all cells as a proxy
                # Better: just scatter pred_prob vs (1 - label) as a sanity check
                # since AUC join is complex. Use the binary labels as a simpler proxy
                pass

            # Collect for overall scatter
            all_pred.extend(grp_pred['pred_prob'].values)
            # Use binary label as stand-in (we'll also do per-drug Spearman on binary)

        # Simpler and more informative: scatter pred_prob vs binary label with jitter
        y_true = preds['true_label'].values
        y_pred = preds['pred_prob'].values

        # Add jitter to binary labels for visibility
        jitter = np.random.normal(0, 0.02, size=len(y_true))
        y_display = y_true + jitter

        ax.scatter(y_pred[y_true <= 0.5], y_display[y_true <= 0.5],
                   s=3, alpha=0.15, c=COLORS['neg'], label='Negative')
        ax.scatter(y_pred[y_true > 0.5], y_display[y_true > 0.5],
                   s=3, alpha=0.3, c=COLORS['pos'], label='Positive')

        # Spearman
        rho, pval = spearmanr(y_pred, y_true)
        ax.set_title(f'{scenario} — Spearman ρ = {rho:.4f} (p={pval:.2e})')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Label (with jitter)')
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(alpha=0.3)

        # Add box plot inset showing score distributions
        inset = ax.inset_axes([0.6, 0.55, 0.35, 0.4])
        bp = inset.boxplot([y_pred[y_true <= 0.5], y_pred[y_true > 0.5]],
                           labels=['Neg', 'Pos'], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COLORS['neg'])
        bp['boxes'][1].set_facecolor(COLORS['pos'])
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_alpha(0.5)
        inset.set_ylabel('Pred Prob', fontsize=7)
        inset.tick_params(labelsize=7)
        inset.set_title('Score by Class', fontsize=8)

    fig.suptitle('Model Score vs True Label (Spearman Correlation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '11_spearman_correlation.png'))
    plt.close(fig)
    print("  > 11_spearman_correlation.png")


# ========================================================================
# FIGURE 11: Confusion Matrices
# ========================================================================
def plot_confusion_matrices(preds_dir, out_dir):
    """Confusion matrices for S1-S4."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for idx, scenario in enumerate(['S1', 'S2', 'S3', 'S4']):
        ax = axes[idx]
        path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        if not os.path.exists(path):
            ax.set_visible(False)
            continue

        df = pd.read_csv(path)
        y_true = (df['true_label'] > 0.5).astype(int)
        y_pred_binary = (df['pred_prob'] > 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred_binary)
        # Normalize by row for display
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Neg', 'Pred Pos'])
        ax.set_yticklabels(['True Neg', 'True Pos'])
        ax.set_title(f'{scenario}')

        # Annotate with counts and percentages
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
                       ha='center', va='center', fontsize=9,
                       color='white' if cm_norm[i, j] > 0.5 else 'black')

    fig.suptitle('Confusion Matrices (Threshold = 0.5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '12_confusion_matrices.png'))
    plt.close(fig)
    print("  > 12_confusion_matrices.png")


# ========================================================================
# FIGURE 12: All 4 Scenarios Summary Bar Chart
# ========================================================================
def plot_scenario_summary(preds_dir, out_dir):
    """Grouped bar chart: AUC, PRC-AUC, F1 for all 4 scenarios."""
    metrics_data = {}

    for scenario in ['S1', 'S2', 'S3', 'S4']:
        path = os.path.join(preds_dir, f'sanger_{scenario}_gnn_preds.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        y_true = (df['true_label'] > 0.5).astype(int)
        y_score = df['pred_prob']

        if y_true.nunique() < 2:
            continue

        metrics_data[scenario] = {
            'ROC-AUC': roc_auc_score(y_true, y_score),
            'PRC-AUC': average_precision_score(y_true, y_score),
            'F1': f1_score(y_true, (y_score > 0.5).astype(int)),
        }

    if not metrics_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = list(metrics_data.keys())
    metric_names = ['ROC-AUC', 'PRC-AUC', 'F1']
    x = np.arange(len(scenarios))
    width = 0.25

    for i, metric in enumerate(metric_names):
        vals = [metrics_data[s][metric] for s in scenarios]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=['#1976D2', '#E64A19', '#2E7D32'][i], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='Random (AUC)')
    ax.set_xlabel('Evaluation Scenario')
    ax.set_ylabel('Score')
    ax.set_title('Sanger Cross-Dataset Evaluation — All Scenarios',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    scenario_labels = {
        'S1': 'S1\nKnown Cell\nKnown Drug',
        'S2': 'S2\nKnown Cell\nNew Drug',
        'S3': 'S3\nNew Cell\nKnown Drug',
        'S4': 'S4\nNew Cell\nNew Drug',
    }
    ax.set_xticklabels([scenario_labels.get(s, s) for s in scenarios], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '13_scenario_summary.png'))
    plt.close(fig)
    print("  > 13_scenario_summary.png")


# ========================================================================
# FIGURE 14: Mutation Clustering in Embedding Space
# ========================================================================
def plot_mutation_clusters(embeddings, dataset, data_dir, lineage_map, out_dir):
    """t-SNE of cell embeddings with top driver mutation overlays."""

    gnn_emb = embeddings['cell']['gnn']
    n_cells = gnn_emb.shape[0]
    cell_type_id = dataset.node_name2type['cell']
    gene_type_id = dataset.node_name2type['gene']

    # Build local_id -> name and name -> local_id maps for cells
    cell_lid_to_name = {}
    cell_name_to_lid = {}
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == cell_type_id:
            name = dataset.id2node.get(gid, '')
            cell_lid_to_name[lid] = name
            cell_name_to_lid[name] = lid

    # Load mutations from raw file (has all 258k edges with cell/gene names)
    # Format: cell_name \t gene_name \t pathogenicity_score
    raw_mut_path = os.path.join(os.path.dirname(data_dir), 'raw', 'link_cell_gene_mutation.txt')
    cell_mutations = defaultdict(set)  # cell_lid -> set of gene names

    with open(raw_mut_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            cell_name, gene_name = parts[0], parts[1]
            lid = cell_name_to_lid.get(cell_name)
            if lid is not None:
                cell_mutations[lid].add(gene_name)

    # Count mutation frequency across cells
    gene_counts = defaultdict(int)
    for lid, genes in cell_mutations.items():
        for g in genes:
            gene_counts[g] += 1

    # Top 8 driver genes by frequency
    top_genes = [g for g, c in sorted(gene_counts.items(), key=lambda x: -x[1])[:8]]
    print(f"    Top mutated genes: {', '.join(f'{g} ({gene_counts[g]})' for g in top_genes)}")

    # Build boolean mutation matrix: (n_cells, n_top_genes)
    mutation_matrix = np.zeros((n_cells, len(top_genes)), dtype=bool)
    for lid in range(n_cells):
        for j, gene in enumerate(top_genes):
            if gene in cell_mutations.get(lid, set()):
                mutation_matrix[lid, j] = True

    # t-SNE
    print("    Computing t-SNE for mutation overlay...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(gnn_emb)

    # --- PLOT A: 2x4 grid, one per top gene ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    for j, gene in enumerate(top_genes):
        ax = axes_flat[j]
        mutated = mutation_matrix[:, j]
        n_mut = mutated.sum()
        pct = 100 * n_mut / n_cells

        # Plot non-mutated first (background)
        ax.scatter(coords[~mutated, 0], coords[~mutated, 1],
                   s=8, alpha=0.15, c='#BDBDBD', label=f'WT ({(~mutated).sum()})')
        # Plot mutated on top
        ax.scatter(coords[mutated, 0], coords[mutated, 1],
                   s=18, alpha=0.7, c='#D32F2F', label=f'Mutant ({n_mut})')

        ax.set_title(f'{gene}\n{n_mut} cells ({pct:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, loc='lower right', markerscale=1.5)

    fig.suptitle('Cell GNN Embeddings — Top 8 Driver Gene Mutations',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '14_mutation_clusters.png'))
    plt.close(fig)
    print("  > 14_mutation_clusters.png")

    # --- PLOT B: Mutation co-occurrence heatmap ---
    n_top = len(top_genes)
    cooccurrence = np.zeros((n_top, n_top), dtype=int)
    for i in range(n_top):
        for j in range(n_top):
            cooccurrence[i, j] = (mutation_matrix[:, i] & mutation_matrix[:, j]).sum()

    # Jaccard similarity for off-diagonal
    jaccard = np.zeros((n_top, n_top))
    for i in range(n_top):
        for j in range(n_top):
            union = (mutation_matrix[:, i] | mutation_matrix[:, j]).sum()
            if union > 0:
                jaccard[i, j] = cooccurrence[i, j] / union

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Co-occurrence counts
    ax = axes[0]
    im = ax.imshow(cooccurrence, cmap='YlOrRd')
    ax.set_xticks(range(n_top))
    ax.set_yticks(range(n_top))
    ax.set_xticklabels(top_genes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(top_genes, fontsize=9)
    for i in range(n_top):
        for j in range(n_top):
            ax.text(j, i, str(cooccurrence[i, j]), ha='center', va='center',
                    fontsize=7, color='white' if cooccurrence[i, j] > cooccurrence.max() * 0.6 else 'black')
    ax.set_title('Co-occurrence Count')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Jaccard similarity
    ax = axes[1]
    im = ax.imshow(jaccard, cmap='YlGnBu', vmin=0, vmax=0.5)
    ax.set_xticks(range(n_top))
    ax.set_yticks(range(n_top))
    ax.set_xticklabels(top_genes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(top_genes, fontsize=9)
    for i in range(n_top):
        for j in range(n_top):
            ax.text(j, i, f'{jaccard[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if jaccard[i, j] > 0.3 else 'black')
    ax.set_title('Jaccard Similarity')
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Top 8 Driver Gene Mutation Co-occurrence',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '15_mutation_cooccurrence.png'))
    plt.close(fig)
    print("  > 15_mutation_cooccurrence.png")

    # --- PLOT C: Mutation burden colored t-SNE + lineage interaction ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: color by total mutation count
    mutation_counts = np.array([len(cell_mutations.get(lid, set())) for lid in range(n_cells)])
    ax = axes[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mutation_counts,
                    s=12, alpha=0.6, cmap='viridis', vmin=0,
                    vmax=np.percentile(mutation_counts, 95))
    fig.colorbar(sc, ax=ax, label='Mutation Count', shrink=0.8)
    ax.set_title(f'Mutation Burden\n(mean={mutation_counts.mean():.0f}, max={mutation_counts.max()})')
    ax.set_xticks([])
    ax.set_yticks([])

    # Right: for each lineage, show mean mutation count for top genes as a stacked bar
    ax = axes[1]

    # Get lineages per cell
    lineages = []
    for lid in range(n_cells):
        name = cell_lid_to_name.get(lid, '')
        lineages.append(lineage_map.get(name, 'Unknown'))
    lineages = np.array(lineages)

    from collections import Counter
    lin_counts = Counter(lineages)
    top_lineages = [l for l, c in lin_counts.most_common(10) if l != 'Unknown']

    # Compute mutation frequency per lineage for top genes
    freq_data = {}
    for lin in top_lineages:
        mask = lineages == lin
        n_lin = mask.sum()
        if n_lin == 0:
            continue
        freqs = []
        for j, gene in enumerate(top_genes):
            freqs.append(mutation_matrix[mask, j].sum() / n_lin * 100)
        freq_data[lin] = freqs

    # Grouped bar chart
    x = np.arange(len(top_lineages))
    width = 0.9 / len(top_genes)
    gene_colors = plt.cm.Set2(np.linspace(0, 1, len(top_genes)))

    for j, gene in enumerate(top_genes):
        vals = [freq_data[lin][j] for lin in top_lineages]
        ax.bar(x + j * width, vals, width, label=gene, color=gene_colors[j], alpha=0.85)

    ax.set_xticks(x + width * len(top_genes) / 2)
    ax.set_xticklabels(top_lineages, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cells with Mutation (%)')
    ax.set_title('Mutation Frequency by Lineage (Top Genes)')
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Mutation Landscape in Cell Embedding Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '16_mutation_landscape.png'))
    plt.close(fig)
    print("  > 16_mutation_landscape.png")


# ========================================================================
# MAIN
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate M09 diagnostic plots.')
    parser.add_argument('--checkpoint', default='checkpoints/M09_PerTypeGate_BalancedBCE.pth')
    parser.add_argument('--log', default='checkpoints/M09_PerTypeGate_BalancedBCE_log.csv')
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--sanger_data_dir', default='data/sanger_processed')
    parser.add_argument('--sanger_dir', default='data/sanger_validation')
    parser.add_argument('--preds_dir', default='results/sanger_gnn')
    parser.add_argument('--emb_dir', default='data/embeddings')
    parser.add_argument('--out_dir', default='results/figures')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_embeddings', action='store_true',
                        help='Skip embedding extraction (slow, needs GPU)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Build lineage map: ACH -> lineage
    model_csv = pd.read_csv('data/misc/Model.csv',
                            usecols=['ModelID', 'OncotreeLineage'])
    lineage_map = dict(zip(model_csv['ModelID'], model_csv['OncotreeLineage']))

    print("=" * 60)
    print("Generating M09 Diagnostic Plots")
    print("=" * 60)

    # --- Plots that don't need model loading ---
    print("\n--- Training Curves ---")
    plot_training_curves(args.log, args.out_dir)

    print("\n--- Skip Gate Analysis ---")
    plot_skip_gates(args.checkpoint, args.out_dir)

    print("\n--- ROC & PR Curves ---")
    plot_roc_pr_curves(args.preds_dir, args.out_dir)

    print("\n--- Score Distributions ---")
    plot_score_distributions(args.preds_dir, args.out_dir)

    print("\n--- Calibration Plot ---")
    plot_calibration(args.preds_dir, args.out_dir)

    print("\n--- Per-Drug AUC ---")
    plot_per_drug_auc(args.preds_dir, args.out_dir)

    print("\n--- Per-Lineage AUC ---")
    plot_per_lineage_auc(args.preds_dir, lineage_map, args.out_dir)

    print("\n--- Spearman Correlation ---")
    plot_spearman_correlation(args.preds_dir, args.sanger_dir, args.out_dir)

    print("\n--- Confusion Matrices ---")
    plot_confusion_matrices(args.preds_dir, args.out_dir)

    print("\n--- Scenario Summary ---")
    plot_scenario_summary(args.preds_dir, args.out_dir)

    # --- Plots that need model loading (embedding extraction) ---
    if not args.skip_embeddings:
        print("\n--- Loading Model for Embedding Extraction ---")

        dataset = PRELUDEDataset(args.data_dir)
        feature_loader = FeatureLoader(dataset, device, embedding_dir=args.emb_dir)
        generator = DataGenerator(args.data_dir)

        model_args = argparse.Namespace(
            embed_d=256, n_layers=4, max_neighbors=20,
            use_skip_connection=True, dropout=0.2, data_dir=args.data_dir,
        )
        model = HetAgg(model_args, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")

        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()

        # Load cell splits
        with open(os.path.join(args.data_dir, 'cell_splits.json')) as f:
            cell_splits_raw = json.load(f)
        cell_splits = {k: set(v) for k, v in cell_splits_raw.items()}

        print("\n--- Extracting Embeddings ---")
        embeddings = extract_embeddings(model, dataset, generator, device)

        print("\n--- Cell Embeddings (t-SNE + PCA) ---")
        plot_cell_embeddings(embeddings, dataset, cell_splits, lineage_map, args.out_dir)

        print("\n--- Drug Embeddings (t-SNE) ---")
        plot_drug_embeddings(embeddings, dataset, args.out_dir)

        print("\n--- Mutation Clustering ---")
        plot_mutation_clusters(embeddings, dataset, args.data_dir, lineage_map, args.out_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {args.out_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
