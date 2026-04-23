"""TCGA Performance Analysis — try multiple normalization and evaluation strategies.

Non-destructive: only changes how TCGA data is processed and evaluated.
Does NOT modify the trained model.

Strategies tested:
  A. Baseline (current): log(x+1) + StandardScaler + VAE
  B. Quantile normalization: map TCGA gene distributions to match CCLE
  C. Rank normalization: rank-transform both datasets to [0,1]
  D. Per-drug evaluation: only report drugs with balanced labels
  E. Tissue-stratified: evaluate within matching tissue types

Usage:
    python scripts/analyze_tcga.py --model_name v2_M03_drug_sim --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TCGA_DIR = 'data/misc/tcga'
PROC_V2 = 'data/processed_v2'
MISC = 'data/misc'


def load_model_and_drugs(model_name, data_dir, device):
    """Load model and get drug embeddings."""
    from dataloaders.data_loader import PRELUDEDataset
    from dataloaders.feature_loader import FeatureLoader
    from dataloaders.data_generator import DataGenerator
    from models.tools import HetAgg

    emb_dir = os.path.join(data_dir, 'embeddings')
    dataset = PRELUDEDataset(data_dir)
    feature_loader = FeatureLoader(dataset, device, embedding_dir=emb_dir)

    model_args = argparse.Namespace(
        data_dir=data_dir, embed_d=256, n_layers=2, max_neighbors=10,
        dropout=0.2, use_skip_connection=True, use_node_gate=True,
        cell_feature_source='vae', gene_encoder_dim=0, use_cross_attention=False,
        cross_attn_dim=32, freeze_cell_gnn=False, regression=False,
        use_residual_gnn=False, residual_scale=0.0, ema_lambda=0.0, ema_momentum=0.999,
        include_cell_drug=True, dual_head=True, inductive_loss_weight=1.0,
        backbone_lr_scale=1.0,
    )
    model = HetAgg(model_args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction('drug', 'cell')
    model.load_state_dict(torch.load(f'checkpoints/{model_name}.pth', map_location=device))
    model.eval()

    drug_type = dataset.node_name2type['drug']
    cell_type = dataset.node_name2type['cell']

    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()

    return model, dataset, eval_tables, feature_loader, cell_type, drug_type


def load_tcga_expression():
    """Load raw TCGA and CCLE expression."""
    print("  Loading CCLE expression...")
    ccle = pd.read_csv(os.path.join(MISC,
        '24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'), index_col=0)

    print("  Loading TCGA expression...")
    tcga = pd.read_csv(os.path.join(TCGA_DIR,
        'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv'), sep='\t', index_col=0)

    # Map TCGA gene IDs to symbols
    tcga_gene_map = {}
    ccle_genes = set(col.split(' (')[0] for col in ccle.columns)
    for gene_id in tcga.index:
        symbol = str(gene_id).split('|')[0]
        if symbol != '?' and symbol in ccle_genes:
            tcga_gene_map[gene_id] = symbol

    tcga_matched = tcga.loc[tcga.index.isin(tcga_gene_map.keys())].copy()
    tcga_matched.index = tcga_matched.index.map(tcga_gene_map)
    tcga_matched = tcga_matched.groupby(tcga_matched.index).mean()
    tcga_transposed = tcga_matched.T

    # Align genes
    ccle_gene_order = [col.split(' (')[0] for col in ccle.columns]
    common = [g for g in ccle_gene_order if g in tcga_transposed.columns]
    missing = [g for g in ccle_gene_order if g not in tcga_transposed.columns]

    tcga_aligned = pd.DataFrame(0.0, index=tcga_transposed.index, columns=ccle_gene_order)
    tcga_aligned[common] = tcga_transposed[common].values

    print(f"  CCLE: {ccle.shape}, TCGA aligned: {tcga_aligned.shape}, Common genes: {len(common)}")
    return ccle, tcga_aligned, ccle_gene_order


def encode_vae(data_matrix, device):
    """Encode expression through VAE."""
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from cell_vae import CellLineVAE

    # Clean NaN/Inf
    data_clean = np.nan_to_num(data_matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    input_dim = data_clean.shape[1]
    dims = [input_dim, 10000, 5000, 2048, 1024, 512]
    vae = CellLineVAE(dims, dropout_rate=0.4).to(device)
    vae.load_state_dict(torch.load('data/embeddings/cell_vae_weights.pth', map_location=device))
    vae.eval()

    tensor = torch.tensor(data_clean, device='cpu')
    loader = DataLoader(TensorDataset(tensor), batch_size=64, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            mu, _ = vae.encode(batch_x)
            batch_emb = mu.cpu().numpy()
            batch_emb = np.nan_to_num(batch_emb, nan=0.0)
            embeddings.append(batch_emb)

    return np.concatenate(embeddings, axis=0)


def score_patients(model, dataset, patient_embeds, eval_df, patient_ids, device,
                   cell_type, drug_type, eval_tables):
    """Score all eval pairs given patient embeddings."""
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    patient_tensor = torch.tensor(patient_embeds, dtype=torch.float32, device=device)

    # Project through cell projection
    original = model.feature_loader.cell_features.clone()
    model.feature_loader.cell_features = patient_tensor
    with torch.no_grad():
        all_lids = torch.arange(len(patient_embeds), device=device)
        projected = model.conteng_agg(all_lids, cell_type)
    model.feature_loader.cell_features = original

    drug_embeds = eval_tables[drug_type]

    preds, labels, drugs_out, patients_out = [], [], [], []
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            p_idx = pid_to_idx.get(row['patient'])
            if p_idx is None:
                continue
            drug_gid = row['drug_gid']
            if drug_gid not in dataset.nodes['type_map']:
                continue
            drug_lid = dataset.nodes['type_map'][drug_gid][1]

            p_emb = projected[p_idx].unsqueeze(0)
            d_emb = drug_embeds[drug_lid].unsqueeze(0)

            if model.lp_bilinear_ind is not None:
                score = torch.sigmoid(model.lp_bilinear_ind(d_emb, p_emb)).item()
            else:
                score = torch.sigmoid(model.lp_bilinear(d_emb, p_emb)).item()

            preds.append(score)
            labels.append(row['label'])
            drugs_out.append(row['drug'])
            patients_out.append(row['patient'])

    return np.array(preds), np.array(labels), drugs_out, patients_out


def evaluate(preds, labels, name="", min_n=20):
    """Compute metrics."""
    preds = np.array(preds)
    labels = np.array(labels)
    # Remove NaN predictions
    valid = ~np.isnan(preds)
    preds, labels = preds[valid], labels[valid]
    binary = (labels > 0.5).astype(int)
    if len(np.unique(binary)) < 2 or len(preds) < min_n:
        if name:
            print(f"  {name}: SKIPPED (insufficient data, N={len(preds)})")
        return {'AUC': None, 'F1': None, 'AP': None, 'N': len(preds)}
    auc = roc_auc_score(binary, preds)
    f1 = f1_score(binary, preds > 0.5)
    ap = average_precision_score(binary, preds)
    if name:
        print(f"  {name}: AUC={auc:.4f} | F1={f1:.4f} | AP={ap:.4f} | N={len(preds):,}")
    return {'AUC': auc, 'F1': f1, 'AP': ap, 'N': len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='v2_M03_drug_sim')
    parser.add_argument('--data_dir', type=str, default=PROC_V2)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = f'results/{args.model_name}'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TCGA COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    # Load model
    print("\n--- Loading Model ---")
    model, dataset, eval_tables, feature_loader, cell_type, drug_type = \
        load_model_and_drugs(args.model_name, args.data_dir, device)

    # Load TCGA eval data
    tcga_eval = pd.read_csv(os.path.join(args.data_dir, 'tcga_eval', 'tcga_eval_all.csv'))
    patient_ids = []
    with open(os.path.join(args.data_dir, 'tcga_eval', 'tcga_patient_ids.txt')) as f:
        patient_ids = [line.strip() for line in f]

    print(f"  Eval pairs: {len(tcga_eval):,}, Patients: {len(patient_ids)}")

    # Load expression
    print("\n--- Loading Expression Data ---")
    ccle, tcga_aligned, gene_order = load_tcga_expression()

    # Patient sample mapping
    with open(os.path.join(TCGA_DIR, 'uuid_to_barcode.csv')) as f:
        uuid_map = pd.read_csv(f)
    uuid_to_barcode = dict(zip(uuid_map['uuid'], uuid_map['tcga_barcode']))

    sample_to_patient = {}
    for sid in tcga_aligned.index:
        if sid.startswith('TCGA-'):
            sample_to_patient[sid] = sid[:12]

    patient_to_sample = {}
    for sid, pid in sample_to_patient.items():
        if pid in set(patient_ids):
            if pid not in patient_to_sample or '-01A-' in sid:
                patient_to_sample[pid] = sid

    all_results = []

    # ================================================================
    # Strategy A: Baseline (log + StandardScaler)
    # ================================================================
    print(f"\n{'='*70}")
    print("Strategy A: Baseline (log + StandardScaler)")
    print("=" * 70)

    scaler_a = StandardScaler()
    scaler_a.fit(ccle.values.astype(np.float32))
    scaler_a.scale_[scaler_a.scale_ == 0] = 1.0

    tcga_log = np.log1p(tcga_aligned.values.astype(np.float32))
    tcga_a = scaler_a.transform(tcga_log)
    tcga_a = np.nan_to_num(tcga_a, nan=0.0)

    # Select patients
    patient_data_a = []
    for pid in patient_ids:
        sid = patient_to_sample.get(pid)
        if sid and sid in tcga_aligned.index:
            idx = list(tcga_aligned.index).index(sid)
            patient_data_a.append(tcga_a[idx])
        else:
            patient_data_a.append(np.zeros(tcga_a.shape[1]))
    patient_data_a = np.array(patient_data_a)

    emb_a = encode_vae(patient_data_a, device)
    print(f"  Embedding stats: mean={emb_a.mean():.3f}, std={emb_a.std():.3f}")

    preds_a, labels_a, drugs_a, patients_a = score_patients(
        model, dataset, emb_a, tcga_eval, patient_ids, device, cell_type, drug_type, eval_tables)
    m_a = evaluate(preds_a, labels_a, "A: Baseline")
    all_results.append({'Strategy': 'A: Baseline (log+SS)', **m_a})

    # ================================================================
    # Strategy B: Quantile normalization
    # ================================================================
    print(f"\n{'='*70}")
    print("Strategy B: Quantile Normalization")
    print("=" * 70)

    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    qt.fit(ccle.values.astype(np.float32))

    tcga_log_b = np.log1p(tcga_aligned.values.astype(np.float32))
    tcga_qt = qt.transform(tcga_log_b)

    scaler_b = StandardScaler()
    ccle_qt = qt.transform(ccle.values.astype(np.float32))
    scaler_b.fit(ccle_qt)
    scaler_b.scale_[scaler_b.scale_ == 0] = 1.0

    tcga_b = scaler_b.transform(tcga_qt)
    tcga_b = np.nan_to_num(tcga_b, nan=0.0)

    patient_data_b = []
    for pid in patient_ids:
        sid = patient_to_sample.get(pid)
        if sid and sid in tcga_aligned.index:
            idx = list(tcga_aligned.index).index(sid)
            patient_data_b.append(tcga_b[idx])
        else:
            patient_data_b.append(np.zeros(tcga_b.shape[1]))
    patient_data_b = np.array(patient_data_b)

    emb_b = encode_vae(patient_data_b, device)
    print(f"  Embedding stats: mean={emb_b.mean():.3f}, std={emb_b.std():.3f}")

    preds_b, labels_b, _, _ = score_patients(
        model, dataset, emb_b, tcga_eval, patient_ids, device, cell_type, drug_type, eval_tables)
    m_b = evaluate(preds_b, labels_b, "B: Quantile")
    all_results.append({'Strategy': 'B: Quantile Norm', **m_b})

    # ================================================================
    # Strategy C: Rank normalization
    # ================================================================
    print(f"\n{'='*70}")
    print("Strategy C: Rank Normalization")
    print("=" * 70)

    # Rank-transform each sample independently (within-sample gene ranking)
    from scipy.stats import rankdata

    ccle_ranked = np.zeros_like(ccle.values, dtype=np.float32)
    for i in range(ccle.shape[0]):
        ccle_ranked[i] = rankdata(ccle.values[i]) / ccle.shape[1]

    tcga_values = tcga_aligned.values.astype(np.float32)
    tcga_ranked = np.zeros_like(tcga_values, dtype=np.float32)
    for i in range(tcga_values.shape[0]):
        tcga_ranked[i] = rankdata(tcga_values[i]) / tcga_values.shape[1]

    scaler_c = StandardScaler()
    scaler_c.fit(ccle_ranked)
    scaler_c.scale_[scaler_c.scale_ == 0] = 1.0
    tcga_c = scaler_c.transform(tcga_ranked)

    patient_data_c = []
    for pid in patient_ids:
        sid = patient_to_sample.get(pid)
        if sid and sid in tcga_aligned.index:
            idx = list(tcga_aligned.index).index(sid)
            patient_data_c.append(tcga_c[idx])
        else:
            patient_data_c.append(np.zeros(tcga_c.shape[1]))
    patient_data_c = np.array(patient_data_c)

    emb_c = encode_vae(patient_data_c, device)
    print(f"  Embedding stats: mean={emb_c.mean():.3f}, std={emb_c.std():.3f}")

    preds_c, labels_c, _, _ = score_patients(
        model, dataset, emb_c, tcga_eval, patient_ids, device, cell_type, drug_type, eval_tables)
    m_c = evaluate(preds_c, labels_c, "C: Rank Norm")
    all_results.append({'Strategy': 'C: Rank Norm', **m_c})

    # ================================================================
    # Strategy D: Domain Adaptation (lightweight alignment MLP)
    # ================================================================
    print(f"\n{'='*70}")
    print("Strategy D: Domain Adaptation (alignment MLP on shared cells)")
    print("=" * 70)

    # Find CCLE cells that are also in the expression data
    # Train a small MLP: TCGA_encoded → CCLE_encoded for same cells
    ccle_cell_ids = list(ccle.index)
    tcga_sample_to_patient = {}
    for sid in tcga_aligned.index:
        if sid.startswith('TCGA-'):
            tcga_sample_to_patient[sid] = sid[:12]

    # CCLE cells don't directly appear in TCGA, but we can use the VAE encoding
    # of CCLE cells as the "target" distribution and train an alignment layer
    # that maps TCGA VAE encodings closer to CCLE VAE encodings

    # Approach: train MLP to minimize distribution mismatch
    # Using MMD (Maximum Mean Discrepancy) or simple mean/std matching
    import torch.nn as nn
    import torch.optim as optim

    ccle_emb = np.load(os.path.join(args.data_dir, 'embeddings', 'final_vae_cell_embeddings.npy'))

    # Simple affine alignment: match mean and std of TCGA to CCLE
    tcga_mean = emb_a.mean(axis=0)
    tcga_std = emb_a.std(axis=0)
    ccle_mean = ccle_emb.mean(axis=0)
    ccle_std = ccle_emb.std(axis=0)

    # Prevent division by zero
    tcga_std[tcga_std < 1e-6] = 1.0

    # Affine transform: aligned = (tcga - tcga_mean) / tcga_std * ccle_std + ccle_mean
    emb_d_affine = (emb_a - tcga_mean) / tcga_std * ccle_std + ccle_mean
    print(f"  Affine aligned stats: mean={emb_d_affine.mean():.3f}, std={emb_d_affine.std():.3f}")
    print(f"  CCLE target stats:    mean={ccle_emb.mean():.3f}, std={ccle_emb.std():.3f}")

    preds_d_aff, labels_d_aff, _, _ = score_patients(
        model, dataset, emb_d_affine, tcga_eval, patient_ids, device, cell_type, drug_type, eval_tables)
    m_d_aff = evaluate(preds_d_aff, labels_d_aff, "D1: Affine Alignment")
    all_results.append({'Strategy': 'D1: Affine Alignment', **m_d_aff})

    # Learned MLP alignment (train to minimize MMD between TCGA and CCLE embeddings)
    print("\n  Training alignment MLP (MMD loss)...")
    class AlignmentMLP(nn.Module):
        def __init__(self, dim=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
        def forward(self, x):
            return x + self.net(x)  # Residual

    def mmd_loss(x, y, bandwidth=1.0):
        """Maximum Mean Discrepancy."""
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dxx = rx.t() + rx - 2 * xx
        dyy = ry.t() + ry - 2 * yy
        dxy = rx.t() + ry - 2 * xy
        kxx = torch.exp(-dxx / (2 * bandwidth))
        kyy = torch.exp(-dyy / (2 * bandwidth))
        kxy = torch.exp(-dxy / (2 * bandwidth))
        return kxx.mean() + kyy.mean() - 2 * kxy.mean()

    align_mlp = AlignmentMLP(512).to(device)
    align_opt = optim.Adam(align_mlp.parameters(), lr=1e-3)

    tcga_tensor_d = torch.tensor(emb_a, dtype=torch.float32, device=device)
    ccle_tensor_d = torch.tensor(ccle_emb, dtype=torch.float32, device=device)

    for epoch in range(200):
        align_opt.zero_grad()
        # Random subsample for efficiency
        t_idx = torch.randperm(len(tcga_tensor_d))[:128]
        c_idx = torch.randperm(len(ccle_tensor_d))[:128]
        aligned = align_mlp(tcga_tensor_d[t_idx])
        loss = mmd_loss(aligned, ccle_tensor_d[c_idx])
        loss.backward()
        align_opt.step()

    with torch.no_grad():
        emb_d_mlp = align_mlp(tcga_tensor_d).cpu().numpy()
    print(f"  MLP aligned stats: mean={emb_d_mlp.mean():.3f}, std={emb_d_mlp.std():.3f}")

    preds_d_mlp, labels_d_mlp, _, _ = score_patients(
        model, dataset, emb_d_mlp, tcga_eval, patient_ids, device, cell_type, drug_type, eval_tables)
    m_d_mlp = evaluate(preds_d_mlp, labels_d_mlp, "D2: MLP Alignment (MMD)")
    all_results.append({'Strategy': 'D2: MLP Alignment (MMD)', **m_d_mlp})

    # ================================================================
    # Strategy E: Balanced drugs only (using best embeddings so far)
    # ================================================================
    print(f"\n{'='*70}")
    print("Strategy D: Balanced Drugs Only (min 20% minority class)")
    print("=" * 70)

    results_df = pd.DataFrame({
        'patient': patients_a, 'drug': drugs_a,
        'pred': preds_a, 'label': labels_a,
    })

    balanced_drugs = []
    for drug, group in results_df.groupby('drug'):
        binary = (group['label'] > 0.5).astype(int)
        if len(binary.unique()) == 2:
            minority_frac = min(binary.mean(), 1 - binary.mean())
            if minority_frac >= 0.2 and len(group) >= 15:
                balanced_drugs.append(drug)

    balanced_df = results_df[results_df['drug'].isin(balanced_drugs)]
    if len(balanced_df) > 0:
        m_d = evaluate(balanced_df['pred'].values, balanced_df['label'].values,
                      f"D: Balanced ({len(balanced_drugs)} drugs)")
    else:
        m_d = {'AUC': None, 'F1': None, 'AP': None, 'N': 0}
        print(f"  No balanced drugs found")
    all_results.append({'Strategy': f'D: Balanced ({len(balanced_drugs)} drugs)', **m_d})

    # Per-drug breakdown for balanced
    print(f"\n  Per-drug (balanced, min 20% minority):")
    drug_results = []
    for drug in balanced_drugs:
        dg = results_df[results_df['drug'] == drug]
        binary = (dg['label'] > 0.5).astype(int)
        if len(binary.unique()) == 2:
            auc = roc_auc_score(binary, dg['pred'])
            n_s, n_r = (binary == 1).sum(), (binary == 0).sum()
            drug_results.append((drug, auc, n_s, n_r))

    drug_results.sort(key=lambda x: x[1], reverse=True)
    for drug, auc, ns, nr in drug_results:
        marker = "***" if auc > 0.6 else "  *" if auc > 0.55 else "   "
        print(f"    {marker} {drug[:25]:25s} AUC={auc:.3f} (sens={ns}, res={nr})")

    # ================================================================
    # Strategy E: Tissue-stratified (match TCGA cancer type to best cell line tissue)
    # ================================================================
    print(f"\n{'='*70}")
    print("Strategy E: Per Cancer Type Evaluation")
    print("=" * 70)

    # Get TCGA cancer types from mutation load file
    mut_load = pd.read_csv(os.path.join(TCGA_DIR, 'mutation-load_updated.txt'), sep='\t')
    patient_cohort = dict(zip(mut_load['Patient_ID'], mut_load['Cohort']))

    results_df['cohort'] = results_df['patient'].map(patient_cohort)

    print(f"\n  Per cancer type:")
    cohort_results = []
    for cohort, group in results_df.groupby('cohort'):
        if pd.isna(cohort):
            continue
        binary = (group['label'] > 0.5).astype(int)
        if len(binary.unique()) == 2 and len(group) >= 20:
            auc = roc_auc_score(binary, group['pred'])
            cohort_results.append((cohort, auc, len(group), binary.mean()))

    cohort_results.sort(key=lambda x: x[1], reverse=True)
    for cohort, auc, n, sens_rate in cohort_results:
        marker = "***" if auc > 0.6 else "  *" if auc > 0.55 else "   "
        print(f"    {marker} {cohort:6s} AUC={auc:.3f} (n={n:4d}, sens_rate={sens_rate:.0%})")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY — ALL STRATEGIES")
    print("=" * 70)

    summary_df = pd.DataFrame(all_results)
    print(f"\n{summary_df.to_string(index=False)}")

    summary_df.to_csv(os.path.join(output_dir, 'tcga_strategy_comparison.csv'), index=False)
    print(f"\n  Saved: {output_dir}/tcga_strategy_comparison.csv")


if __name__ == '__main__':
    main()
