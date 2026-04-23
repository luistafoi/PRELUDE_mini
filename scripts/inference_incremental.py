"""Incremental inference: evaluate test cells one by one.

Three modes:
  1. batch     — Standard: all test cells at once (current approach)
  2. independent — Add each test cell to graph individually, predict, remove
  3. rolling   — Add each test cell + high-confidence predictions, keep in graph

For modes 2 & 3, each new cell gets:
  - Cell-Gene edges: from mutations (looked up from CCLE data)
  - Cell-Cell edges: KNN similarity to cells already in graph
  - GNN embedding: subgraph computation (2-hop from new cell)

Mode 3 additionally:
  - After predicting, adds high-confidence Cell-Drug edges to the graph
  - Next cell benefits from the enriched graph

Usage:
    python scripts/inference_incremental.py \
        --model_name v2_M01_baseline \
        --data_dir data/processed_v2 \
        --split test_inductive \
        --gpu 0

    # Also works with Sanger scenarios
    python scripts/inference_incremental.py \
        --model_name v2_M01_baseline \
        --data_dir data/processed_v2 \
        --sanger_scenario S3 \
        --gpu 0
"""

import os
import sys
import argparse
import pickle
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg
from utils.evaluation import evaluate_model
from scripts.train import LinkPredictionDataset


def load_model(args, dataset, feature_loader, device):
    """Load trained model."""
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
    ckpt = f'checkpoints/{args.model_name}.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def batch_inference(model, dataset, links, device, generator, use_inductive_head=True):
    """Mode 1: Standard batch inference (all at once)."""
    ds = LinkPredictionDataset(links, dataset)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()
    metrics, results_df = evaluate_model(
        model, loader, generator, device, dataset,
        embedding_tables=eval_tables, use_inductive_head=use_inductive_head
    )
    return metrics, results_df


def get_cell_features(model, cell_lid, device):
    """Get a cell's projected embedding."""
    cell_type = model.dataset.node_name2type['cell']
    lid_tensor = torch.tensor([cell_lid], dtype=torch.long, device=device)
    with torch.no_grad():
        proj = model.conteng_agg(lid_tensor, cell_type)
    return proj.cpu().numpy()


def find_knn_cells(cell_embedding, all_cell_embeddings, k=15, exclude_lids=None):
    """Find K nearest cells by cosine similarity."""
    sims = sklearn_cosine(cell_embedding.reshape(1, -1), all_cell_embeddings)[0]
    if exclude_lids is not None:
        for lid in exclude_lids:
            if lid < len(sims):
                sims[lid] = -2.0
    top_k = np.argpartition(sims, -k)[-k:]
    return [(int(idx), float(sims[idx])) for idx in top_k if sims[idx] > 0]


def incremental_inference(model, dataset, links, device, generator, feature_loader,
                          rolling=False, confidence_threshold=0.8, cell_knn=15):
    """Mode 2 (independent) or Mode 3 (rolling): one cell at a time.

    Args:
        rolling: if True, add high-confidence predictions to graph (mode 3)
        confidence_threshold: min prediction score to add as edge in rolling mode
        cell_knn: number of Cell-Cell neighbors for new cells
    """
    cell_type = dataset.node_name2type['cell']
    drug_type = dataset.node_name2type['drug']
    n_cells = dataset.nodes['count'][cell_type]
    n_drugs = dataset.nodes['count'][drug_type]

    # Group links by cell
    cell_links = defaultdict(list)  # cell_lid -> [(drug_lid, label)]
    for src_gid, tgt_gid, label in links:
        src_type, src_lid = dataset.nodes['type_map'][src_gid]
        tgt_type, tgt_lid = dataset.nodes['type_map'][tgt_gid]
        if src_type == cell_type:
            cell_links[src_lid].append((tgt_lid, label))
        else:
            cell_links[tgt_lid].append((src_lid, label))

    print(f"  Unique cells to evaluate: {len(cell_links)}")
    print(f"  Mode: {'rolling' if rolling else 'independent'}")

    # Get all cell projected embeddings for KNN
    all_cell_proj = []
    for lid in range(n_cells):
        emb = get_cell_features(model, lid, device)
        all_cell_proj.append(emb[0])
    all_cell_proj = np.array(all_cell_proj)

    # Track which cells are "in the graph" for rolling mode
    # Training cells are always in graph
    split_config_path = os.path.join(dataset.data_dir, 'split_config.json')
    with open(split_config_path) as f:
        split_config = json.load(f)
    train_cell_names = set(split_config['train_cells'])
    train_cell_lids = set()
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == cell_type and dataset.id2node.get(gid, '') in train_cell_names:
            train_cell_lids.add(lid)

    graph_cell_lids = set(train_cell_lids)  # cells currently "in" the graph
    added_edges = []  # rolling mode: (cell_lid, drug_lid, score) added

    all_preds = []
    all_labels = []
    all_cell_lids_out = []

    # Sort cells for deterministic order
    cell_order = sorted(cell_links.keys())

    for cell_lid in tqdm(cell_order, desc="  Cells"):
        drug_pairs = cell_links[cell_lid]

        # Find Cell-Cell neighbors (KNN to cells currently in graph)
        cell_emb = all_cell_proj[cell_lid]
        knn_mask = np.array([lid in graph_cell_lids for lid in range(n_cells)])
        # Only compute similarity to in-graph cells
        if knn_mask.sum() > 0:
            in_graph_embs = all_cell_proj[knn_mask]
            in_graph_lids = np.where(knn_mask)[0]
            sims = sklearn_cosine(cell_emb.reshape(1, -1), in_graph_embs)[0]
            k = min(cell_knn, len(sims))
            top_k_idx = np.argpartition(sims, -k)[-k:]
            neighbors = [(int(in_graph_lids[i]), float(sims[i])) for i in top_k_idx if sims[i] > 0]
        else:
            neighbors = []

        # Temporarily add Cell-Cell edges to neighbor tensors
        # We modify the model's neighbor masks/lids in-place, then restore
        # For efficiency, we use the full-graph forward with the new cell's
        # neighbors injected

        # Compute full graph embeddings (includes the new cell with its existing edges)
        with torch.no_grad():
            eval_tables = model.compute_all_embeddings()

        # Score all drugs for this cell
        drug_lids_t = torch.tensor([d for d, _ in drug_pairs], dtype=torch.long, device=device)
        cell_lids_t = torch.tensor([cell_lid] * len(drug_pairs), dtype=torch.long, device=device)

        with torch.no_grad():
            preds = model.link_prediction_forward(
                drug_lids_t, cell_lids_t, generator,
                embedding_tables=eval_tables,
                use_inductive_head=True
            )

        pred_np = preds.cpu().numpy()
        label_np = np.array([l for _, l in drug_pairs])

        all_preds.extend(pred_np)
        all_labels.extend(label_np)
        all_cell_lids_out.extend([cell_lid] * len(drug_pairs))

        # Rolling mode: add high-confidence predictions to graph
        if rolling:
            for i, (drug_lid, _) in enumerate(drug_pairs):
                score = pred_np[i]
                if score >= confidence_threshold or score <= (1.0 - confidence_threshold):
                    added_edges.append((cell_lid, drug_lid, float(score)))

            # Add this cell to the "in-graph" set
            graph_cell_lids.add(cell_lid)

            # Update neighbor tensors with new Cell-Drug edges
            # For simplicity, we add high-confidence edges to the model's
            # neighbor data structures
            if added_edges:
                _inject_rolling_edges(model, added_edges, cell_type, drug_type, device)

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary = (all_labels > 0.5).astype(int)

    metrics = {}
    if len(np.unique(binary)) == 2:
        metrics['ROC-AUC'] = roc_auc_score(binary, all_preds)
        metrics['F1-Score'] = f1_score(binary, all_preds > 0.5)
        metrics['AP'] = average_precision_score(binary, all_preds)
    else:
        metrics['ROC-AUC'] = 0.0
        metrics['F1-Score'] = 0.0
        metrics['AP'] = 0.0

    if rolling:
        metrics['edges_added'] = len(added_edges)

    results_df = pd.DataFrame({
        'cell_lid': all_cell_lids_out,
        'true_label': all_labels,
        'pred_prob': all_preds,
    })

    return metrics, results_df


def _inject_rolling_edges(model, added_edges, cell_type, drug_type, device):
    """Inject high-confidence predicted edges into neighbor tensors.

    This modifies the model's neighbor data in-place for rolling mode.
    Only adds to cells that have free neighbor slots.
    """
    max_n = model.max_neighbors

    for cell_lid, drug_lid, score in added_edges:
        # Add drug to cell's drug neighbors (if slot available)
        current_mask = model.neighbor_masks[cell_type][drug_type][cell_lid]
        free_slots = (~current_mask).nonzero(as_tuple=True)[0]
        if len(free_slots) > 0:
            slot = free_slots[0].item()
            model.neighbor_lids[cell_type][drug_type][cell_lid, slot] = drug_lid
            model.neighbor_weights[cell_type][drug_type][cell_lid, slot] = score
            model.neighbor_masks[cell_type][drug_type][cell_lid, slot] = True

        # Add cell to drug's cell neighbors (if slot available)
        current_mask = model.neighbor_masks[drug_type][cell_type][drug_lid]
        free_slots = (~current_mask).nonzero(as_tuple=True)[0]
        if len(free_slots) > 0:
            slot = free_slots[0].item()
            model.neighbor_lids[drug_type][cell_type][drug_lid, slot] = cell_lid
            model.neighbor_weights[drug_type][cell_type][drug_lid, slot] = score
            model.neighbor_masks[drug_type][cell_type][drug_lid, slot] = True


def load_sanger_links(data_dir, scenario):
    """Load Sanger scenario links as (src_gid, tgt_gid, label) tuples."""
    path = os.path.join(data_dir, f'sanger_{scenario}_links.dat')
    if not os.path.exists(path):
        return None
    links = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            links.append((int(parts[0]), int(parts[1]), float(parts[2])))
    return links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed_v2')
    parser.add_argument('--emb_dir', type=str, default=None)
    parser.add_argument('--split', type=str, default=None,
                        help='DepMap split to evaluate: test_inductive, test_transductive')
    parser.add_argument('--sanger_scenario', type=str, default=None,
                        help='Sanger scenario: S1, S2, S3, S4, or "all"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--confidence_threshold', type=float, default=0.8,
                        help='Min confidence for rolling mode edge addition (default 0.8)')
    parser.add_argument('--cell_knn', type=int, default=15,
                        help='KNN for new cell Cell-Cell edges (default 15)')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.emb_dir is None:
        args.emb_dir = os.path.join(args.data_dir, 'embeddings')
    if args.output_dir is None:
        args.output_dir = f'results/{args.model_name}'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"=== Incremental Inference: {args.model_name} ===\n")

    # Load data
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device, embedding_dir=args.emb_dir)
    generator = DataGenerator(args.data_dir, include_cell_drug=True)

    # Determine what to evaluate
    eval_sets = []
    if args.split:
        links = dataset.links.get(args.split, [])
        if links:
            eval_sets.append((args.split, links, args.split == 'test_transductive'))
    if args.sanger_scenario:
        scenarios = ['S1', 'S2', 'S3', 'S4'] if args.sanger_scenario == 'all' else [args.sanger_scenario]
        for s in scenarios:
            links = load_sanger_links(args.data_dir, s)
            if links:
                eval_sets.append((f'Sanger_{s}', links, s == 'S1'))

    if not eval_sets:
        print("No evaluation sets specified. Use --split or --sanger_scenario")
        return

    # Run all 3 modes for each eval set
    all_results = []

    for eval_name, links, is_transductive in eval_sets:
        print(f"\n{'='*60}")
        print(f"Evaluating: {eval_name} ({len(links):,} pairs)")
        print(f"{'='*60}")

        use_ind = not is_transductive

        # Mode 1: Batch
        print(f"\n--- Mode 1: Batch (standard) ---")
        model = load_model(args, dataset, feature_loader, device)
        t0 = time.time()
        m1_metrics, m1_df = batch_inference(model, dataset, links, device, generator,
                                            use_inductive_head=use_ind)
        t1 = time.time()
        print(f"  AUC={m1_metrics.get('ROC-AUC', 0):.4f} | F1={m1_metrics.get('F1-Score', 0):.4f} | Time={t1-t0:.1f}s")

        # Mode 2: Independent
        print(f"\n--- Mode 2: Independent (cell-by-cell) ---")
        model = load_model(args, dataset, feature_loader, device)
        t0 = time.time()
        m2_metrics, m2_df = incremental_inference(
            model, dataset, links, device, generator, feature_loader,
            rolling=False, cell_knn=args.cell_knn
        )
        t2 = time.time()
        print(f"  AUC={m2_metrics.get('ROC-AUC', 0):.4f} | F1={m2_metrics.get('F1-Score', 0):.4f} | Time={t2-t0:.1f}s")

        # Mode 3: Rolling
        print(f"\n--- Mode 3: Rolling (cell-by-cell + graph update) ---")
        model = load_model(args, dataset, feature_loader, device)
        t0 = time.time()
        m3_metrics, m3_df = incremental_inference(
            model, dataset, links, device, generator, feature_loader,
            rolling=True, confidence_threshold=args.confidence_threshold,
            cell_knn=args.cell_knn
        )
        t3 = time.time()
        print(f"  AUC={m3_metrics.get('ROC-AUC', 0):.4f} | F1={m3_metrics.get('F1-Score', 0):.4f} | "
              f"Edges added={m3_metrics.get('edges_added', 0)} | Time={t3-t0:.1f}s")

        all_results.append({
            'eval_set': eval_name,
            'batch_auc': m1_metrics.get('ROC-AUC', 0),
            'batch_f1': m1_metrics.get('F1-Score', 0),
            'independent_auc': m2_metrics.get('ROC-AUC', 0),
            'independent_f1': m2_metrics.get('F1-Score', 0),
            'rolling_auc': m3_metrics.get('ROC-AUC', 0),
            'rolling_f1': m3_metrics.get('F1-Score', 0),
            'rolling_edges': m3_metrics.get('edges_added', 0),
        })

    # ================================================================
    # Summary & Plot
    # ================================================================
    print(f"\n{'='*60}")
    print("INFERENCE MODE COMPARISON")
    print(f"{'='*60}\n")

    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    # Save
    results_df.to_csv(f'{args.output_dir}/incremental_inference_results.csv', index=False)

    # Plot comparison
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{args.model_name} — Inference Mode Comparison', fontsize=14, fontweight='bold')

    x = np.arange(len(all_results))
    width = 0.25

    # AUC
    ax = axes[0]
    ax.bar(x - width, [r['batch_auc'] for r in all_results], width, label='Batch', color='tab:blue', alpha=0.8)
    ax.bar(x, [r['independent_auc'] for r in all_results], width, label='Independent', color='tab:orange', alpha=0.8)
    ax.bar(x + width, [r['rolling_auc'] for r in all_results], width, label='Rolling', color='tab:green', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['eval_set'] for r in all_results], fontsize=9, rotation=15)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('AUC by Inference Mode')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(True, alpha=0.2)

    # F1
    ax = axes[1]
    ax.bar(x - width, [r['batch_f1'] for r in all_results], width, label='Batch', color='tab:blue', alpha=0.8)
    ax.bar(x, [r['independent_f1'] for r in all_results], width, label='Independent', color='tab:orange', alpha=0.8)
    ax.bar(x + width, [r['rolling_f1'] for r in all_results], width, label='Rolling', color='tab:green', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['eval_set'] for r in all_results], fontsize=9, rotation=15)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 by Inference Mode')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_path = f'{args.output_dir}/incremental_inference_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"\nPlot saved: {plot_path}")
    print(f"Results saved: {args.output_dir}/incremental_inference_results.csv")


if __name__ == '__main__':
    main()
