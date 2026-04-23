"""TCGA Patient Inference — predict drug sensitivity for real tumor patients.

Takes TCGA patients (expression → VAE embedding, mutations → gene edges,
KNN → similar training cells) and predicts drug sensitivity using the trained model.

Three modes:
  1. batch      — All patients scored with base graph
  2. independent — Each patient added individually, scored, removed
  3. rolling    — Each patient added with edges, high-confidence predictions kept

Usage:
    python scripts/inference_tcga.py --model_name v2_M03_drug_sim --data_dir data/processed_v2 --gpu 0
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg


def load_tcga_eval(data_dir):
    """Load TCGA evaluation data."""
    tcga_dir = os.path.join(data_dir, 'tcga_eval')

    eval_df = pd.read_csv(os.path.join(tcga_dir, 'tcga_eval_all.csv'))
    patient_embeds = np.load(os.path.join(tcga_dir, 'tcga_patient_embeddings.npy'))
    with open(os.path.join(tcga_dir, 'tcga_patient_ids.txt')) as f:
        patient_ids = [line.strip() for line in f]
    with open(os.path.join(tcga_dir, 'tcga_knn_edges.json')) as f:
        knn_edges = json.load(f)
    with open(os.path.join(tcga_dir, 'tcga_patient_mutations.json')) as f:
        patient_mutations = json.load(f)

    return {
        'eval_df': eval_df,
        'patient_embeds': patient_embeds,
        'patient_ids': patient_ids,
        'knn_edges': knn_edges,
        'patient_mutations': patient_mutations,
    }


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
    model.load_state_dict(torch.load(f'checkpoints/{args.model_name}.pth', map_location=device))
    model.eval()
    return model


def batch_inference(model, dataset, tcga_data, device):
    """Mode 1: Score all TCGA patients using base graph embeddings.

    TCGA patients are scored using their VAE embeddings directly through the
    inductive head. No graph modification — pure projection-based scoring.
    """
    eval_df = tcga_data['eval_df']
    patient_embeds = tcga_data['patient_embeds']
    patient_ids = tcga_data['patient_ids']
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    drug_type = dataset.node_name2type['drug']
    cell_type = dataset.node_name2type['cell']

    # Get drug GNN embeddings from the trained graph
    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()
        drug_embeds_all = eval_tables[drug_type]  # (N_drugs, 256)

    # For each patient, score against their drugs
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, row in eval_df.iterrows():
            patient = row['patient']
            drug = row['drug']
            label = row['label']

            p_idx = pid_to_idx.get(patient)
            drug_gid = row['drug_gid']

            if p_idx is None:
                continue

            # Patient embedding → project through cell projection
            p_emb = torch.tensor(patient_embeds[p_idx], dtype=torch.float32, device=device).unsqueeze(0)
            # Project through cell projection layer — use the same path as batch_inference_simple.
            # Temporarily swap a cell slot's features so conteng_agg sees p_emb.
            # (Dead path anyway; fixed to avoid AttributeError if ever called.)
            saved_feat = model.feature_loader.cell_features[0].clone()
            model.feature_loader.cell_features[0] = p_emb.squeeze(0)
            try:
                cell_proj = model.conteng_agg(torch.tensor([0], device=device), cell_type)
            finally:
                model.feature_loader.cell_features[0] = saved_feat

            # Drug embedding from GNN
            drug_lid = dataset.nodes['type_map'][drug_gid][1]
            d_emb = drug_embeds_all[drug_lid].unsqueeze(0)

            # Score with inductive head
            if model.lp_bilinear_ind is not None:
                score = torch.sigmoid(model.lp_bilinear_ind(d_emb, cell_proj)).item()
            else:
                score = torch.sigmoid(model.lp_bilinear(d_emb, cell_proj)).item()

            all_preds.append(score)
            all_labels.append(label)

    return np.array(all_preds), np.array(all_labels)


def batch_inference_simple(model, dataset, tcga_data, device):
    """Mode 1 (simplified): Use projected embeddings and inductive head."""
    eval_df = tcga_data['eval_df']
    patient_embeds = tcga_data['patient_embeds']
    patient_ids = tcga_data['patient_ids']
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    drug_type = dataset.node_name2type['drug']
    cell_type = dataset.node_name2type['cell']

    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()

    # Project all patient embeddings through cell projection
    patient_tensor = torch.tensor(patient_embeds, dtype=torch.float32, device=device)

    # Use the cell feature projection
    feat_loader = model.feature_loader
    cell_proj_layer = None
    for name, module in model.named_modules():
        if 'proj' in name.lower() and 'cell' in name.lower():
            cell_proj_layer = module
            break

    # Simpler: use conteng_agg which handles projection
    # We need to temporarily inject patient embeddings as cell features
    # Instead, compute projection manually
    n_patients = len(patient_embeds)

    # Project through the model's cell projection (same as conteng_agg does internally)
    # The model stores projections in feature_projections dict keyed by type_id
    cell_type_id = cell_type
    with torch.no_grad():
        # Temporarily swap cell features to get projections for TCGA patients
        original_cell_features = model.feature_loader.cell_features.clone()
        model.feature_loader.cell_features = patient_tensor
        # conteng_agg projects features for given local IDs
        all_lids = torch.arange(n_patients, device=device)
        patient_projected = model.conteng_agg(all_lids, cell_type_id)
        # Restore original features
        model.feature_loader.cell_features = original_cell_features

    # Get drug embeddings
    drug_embeds = eval_tables[drug_type]  # (N_drugs, 256)

    # Score each eval pair
    all_preds, all_labels, all_patients, all_drugs = [], [], [], []

    with torch.no_grad():
        for _, row in eval_df.iterrows():
            p_idx = pid_to_idx.get(row['patient'])
            if p_idx is None:
                continue

            drug_gid = row['drug_gid']
            if drug_gid not in dataset.nodes['type_map']:
                continue
            drug_lid = dataset.nodes['type_map'][drug_gid][1]

            p_emb = patient_projected[p_idx].unsqueeze(0)
            d_emb = drug_embeds[drug_lid].unsqueeze(0)

            if model.lp_bilinear_ind is not None:
                score = torch.sigmoid(model.lp_bilinear_ind(d_emb, p_emb)).item()
            else:
                score = torch.sigmoid(model.lp_bilinear(d_emb, p_emb)).item()

            all_preds.append(score)
            all_labels.append(row['label'])
            all_patients.append(row['patient'])
            all_drugs.append(row['drug'])

    return np.array(all_preds), np.array(all_labels), all_patients, all_drugs, patient_projected


def compute_metrics(preds, labels, name=""):
    """Compute AUC, F1, AP."""
    binary = (np.array(labels) > 0.5).astype(int)
    metrics = {}
    if len(np.unique(binary)) == 2 and len(preds) > 0:
        metrics['AUC'] = roc_auc_score(binary, preds)
        metrics['F1'] = f1_score(binary, preds > 0.5)
        metrics['AP'] = average_precision_score(binary, preds)
    else:
        metrics['AUC'] = 0.0
        metrics['F1'] = 0.0
        metrics['AP'] = 0.0
    metrics['N'] = len(preds)
    if name:
        print(f"  {name}: AUC={metrics['AUC']:.4f} | F1={metrics['F1']:.4f} | "
              f"AP={metrics['AP']:.4f} | N={metrics['N']:,}")
    return metrics


def knn_enhanced_inference(model, dataset, tcga_data, device, patient_projected,
                           rolling=False, confidence_threshold=0.8, knn_weight=0.3):
    """Mode 2/3: Enhance patient embeddings with KNN neighbor information.

    For each patient:
    1. Start with their projected VAE embedding
    2. Blend in weighted average of KNN training cells' GNN embeddings
    3. Score drugs with the inductive head
    4. (Rolling) Add high-confidence predictions to a patient response pool

    Args:
        patient_projected: (N_patients, 256) projected TCGA embeddings
        rolling: if True, accumulate patient predictions to inform later patients
        knn_weight: how much to blend KNN neighbor embeddings (0=projection only, 1=KNN only)
    """
    eval_df = tcga_data['eval_df']
    patient_ids = tcga_data['patient_ids']
    knn_edges = tcga_data['knn_edges']
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    drug_type = dataset.node_name2type['drug']
    cell_type = dataset.node_name2type['cell']

    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()
        cell_gnn_embeds = eval_tables[cell_type].cpu().numpy()  # (N_cells, 256)
        drug_embeds = eval_tables[drug_type]  # on device

    patient_proj_np = patient_projected.detach().cpu().numpy()

    # Group eval pairs by patient
    patient_pairs = defaultdict(list)
    for _, row in eval_df.iterrows():
        p_idx = pid_to_idx.get(row['patient'])
        if p_idx is not None and row['drug_gid'] in dataset.nodes['type_map']:
            patient_pairs[row['patient']].append({
                'drug': row['drug'],
                'drug_gid': row['drug_gid'],
                'drug_lid': dataset.nodes['type_map'][row['drug_gid']][1],
                'label': row['label'],
            })

    # Rolling: accumulated high-confidence patient embeddings
    rolling_patient_embeds = {}  # patient_id -> (embedding, [drug_responses])

    all_preds, all_labels, all_patients, all_drugs = [], [], [], []
    patient_order = sorted(patient_pairs.keys())

    for patient in tqdm(patient_order, desc=f"  {'Rolling' if rolling else 'Independent'}"):
        p_idx = pid_to_idx[patient]
        pairs = patient_pairs[patient]

        # Base embedding: projection
        base_emb = patient_proj_np[p_idx]

        # KNN enhancement: blend with similar training cells' GNN embeddings
        knn = knn_edges.get(patient, [])
        if knn:
            neighbor_embeds = []
            neighbor_weights = []
            for cell_lid, sim in knn:
                if cell_lid < len(cell_gnn_embeds):
                    neighbor_embeds.append(cell_gnn_embeds[cell_lid])
                    neighbor_weights.append(sim)

            # Also include rolling patients if available
            if rolling and rolling_patient_embeds:
                # Find similar rolled patients (simple: use all with equal weight)
                for rpid, (remb, _) in rolling_patient_embeds.items():
                    # Small weight for rolled patients
                    neighbor_embeds.append(remb)
                    neighbor_weights.append(0.1)

            if neighbor_embeds:
                neighbor_embeds = np.array(neighbor_embeds)
                neighbor_weights = np.array(neighbor_weights)
                neighbor_weights = neighbor_weights / neighbor_weights.sum()
                knn_emb = (neighbor_embeds * neighbor_weights[:, None]).sum(axis=0)
                enhanced_emb = (1 - knn_weight) * base_emb + knn_weight * knn_emb
            else:
                enhanced_emb = base_emb
        else:
            enhanced_emb = base_emb

        # Score all drugs for this patient
        p_tensor = torch.tensor(enhanced_emb, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            for pair in pairs:
                d_emb = drug_embeds[pair['drug_lid']].unsqueeze(0)
                if model.lp_bilinear_ind is not None:
                    score = torch.sigmoid(model.lp_bilinear_ind(d_emb, p_tensor)).item()
                else:
                    score = torch.sigmoid(model.lp_bilinear(d_emb, p_tensor)).item()

                all_preds.append(score)
                all_labels.append(pair['label'])
                all_patients.append(patient)
                all_drugs.append(pair['drug'])

        # Rolling: add this patient's embedding and high-confidence predictions
        if rolling:
            rolling_patient_embeds[patient] = (enhanced_emb, [
                (p['drug'], s) for p, s in zip(pairs, all_preds[-len(pairs):])
                if s >= confidence_threshold or s <= (1 - confidence_threshold)
            ])

    return np.array(all_preds), np.array(all_labels), all_patients, all_drugs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed_v2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'results/{args.model_name}'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"TCGA INFERENCE: {args.model_name}")
    print("=" * 60)

    # Load everything
    print("\n--- Loading ---")
    emb_dir = os.path.join(args.data_dir, 'embeddings')
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device, embedding_dir=emb_dir)
    generator = DataGenerator(args.data_dir, include_cell_drug=True)
    model = load_model(args, dataset, feature_loader, device)
    tcga_data = load_tcga_eval(args.data_dir)

    eval_df = tcga_data['eval_df']
    print(f"  TCGA patients: {len(tcga_data['patient_ids'])}")
    print(f"  Eval pairs: {len(eval_df):,}")
    print(f"  Sensitive: {(eval_df['label'] == 1).sum():,} | Resistant: {(eval_df['label'] == 0).sum():,}")

    # ================================================================
    # Mode 1: Batch inference
    # ================================================================
    print(f"\n{'='*60}")
    print("Mode 1: Batch Inference")
    print("=" * 60)

    t0 = time.time()
    preds, labels, patients, drugs, patient_projected = batch_inference_simple(model, dataset, tcga_data, device)
    t1 = time.time()

    print(f"\n  Time: {t1-t0:.1f}s")
    overall = compute_metrics(preds, labels, "Overall")

    # Split by known vs new drugs
    results_df = pd.DataFrame({
        'patient': patients, 'drug': drugs,
        'pred_score': preds, 'true_label': labels,
    })
    results_df['drug_known'] = results_df['drug'].isin(
        set(eval_df[eval_df['drug_known']]['drug']) if 'drug_known' in eval_df.columns else set()
    )

    known = results_df[results_df['drug_known']]
    new = results_df[~results_df['drug_known']]
    if len(known) > 0:
        compute_metrics(known['pred_score'].values, known['true_label'].values, "Known Drugs")
    if len(new) > 0 and len(new['true_label'].unique()) > 1:
        compute_metrics(new['pred_score'].values, new['true_label'].values, "New Drugs")

    # Per-drug breakdown
    print(f"\n  Per-drug breakdown (top 15):")
    drug_counts = results_df.groupby('drug').size().sort_values(ascending=False)
    for drug in drug_counts.head(15).index:
        drug_df = results_df[results_df['drug'] == drug]
        binary = (drug_df['true_label'] > 0.5).astype(int)
        if len(binary.unique()) == 2:
            auc = roc_auc_score(binary, drug_df['pred_score'])
            n_s = (binary == 1).sum()
            n_r = (binary == 0).sum()
            print(f"    {drug[:25]:25s} AUC={auc:.3f} (sens={n_s}, res={n_r})")

    # Save results
    results_df.to_csv(os.path.join(args.output_dir, 'tcga_predictions.csv'), index=False)

    # ================================================================
    # Mode 2: Independent (KNN-enhanced)
    # ================================================================
    print(f"\n{'='*60}")
    print("Mode 2: Independent (KNN-enhanced embedding)")
    print("=" * 60)

    t0 = time.time()
    preds2, labels2, patients2, drugs2 = knn_enhanced_inference(
        model, dataset, tcga_data, device, patient_projected, rolling=False)
    t1 = time.time()

    print(f"\n  Time: {t1-t0:.1f}s")
    overall2 = compute_metrics(preds2, labels2, "Overall")

    # ================================================================
    # Mode 3: Rolling (KNN-enhanced + graph enrichment)
    # ================================================================
    print(f"\n{'='*60}")
    print("Mode 3: Rolling (KNN-enhanced + graph enrichment)")
    print("=" * 60)

    t0 = time.time()
    preds3, labels3, patients3, drugs3 = knn_enhanced_inference(
        model, dataset, tcga_data, device, patient_projected, rolling=True)
    t1 = time.time()

    print(f"\n  Time: {t1-t0:.1f}s")
    overall3 = compute_metrics(preds3, labels3, "Overall")

    # ================================================================
    # Comparison
    # ================================================================
    print(f"\n{'='*60}")
    print("TCGA INFERENCE — MODE COMPARISON")
    print(f"{'='*60}")

    comparison = pd.DataFrame([
        {'Mode': 'Batch (projection)', 'AUC': overall['AUC'], 'F1': overall['F1'],
         'AP': overall['AP'], 'N': overall['N']},
        {'Mode': 'Independent (KNN)', 'AUC': overall2['AUC'], 'F1': overall2['F1'],
         'AP': overall2['AP'], 'N': overall2['N']},
        {'Mode': 'Rolling (KNN+enrich)', 'AUC': overall3['AUC'], 'F1': overall3['F1'],
         'AP': overall3['AP'], 'N': overall3['N']},
    ])
    print(f"\n{comparison.to_string(index=False)}")

    comparison.to_csv(os.path.join(args.output_dir, 'tcga_mode_comparison.csv'), index=False)

    # Save all predictions
    all_results = pd.DataFrame({
        'patient': patients, 'drug': drugs,
        'pred_batch': preds, 'pred_knn': preds2, 'pred_rolling': preds3,
        'true_label': labels,
    })
    all_results.to_csv(os.path.join(args.output_dir, 'tcga_predictions_all_modes.csv'), index=False)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("TCGA INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Batch AUC:       {overall['AUC']:.4f}")
    print(f"  Independent AUC: {overall2['AUC']:.4f}")
    print(f"  Rolling AUC:     {overall3['AUC']:.4f}")
    print(f"  Pairs: {overall['N']:,}")
    print(f"\n  Results: {args.output_dir}/")


if __name__ == '__main__':
    main()
