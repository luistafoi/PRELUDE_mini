"""Step 6: Build TCGA evaluation dataset for cross-domain inference.

Takes TCGA patient data (expression, mutations, drug outcomes) and creates
evaluation files compatible with our inference pipeline.

Steps:
  1. Map TCGA drugs to our graph drugs
  2. Map clinical outcomes to binary labels
  3. Normalize TCGA expression to CCLE space and encode through VAE
  4. Build patient-gene edges from mutations (with pathogenicity)
  5. Compute patient-cell similarity (KNN to training cells)
  6. Write evaluation files

Usage:
    python scripts/pipeline_v2/step6_tcga_eval.py
    python scripts/pipeline_v2/step6_tcga_eval.py --data_dir data/processed_v2 --tcga_dir data/misc/tcga
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TCGA_DIR = 'data/misc/tcga'
PROC_V2 = 'data/processed_v2'
MISC = 'data/misc'


def step1_match_drugs(tcga_dir, proc_v2):
    """Match TCGA drug names to our graph drugs."""
    print("=== Step 1: Drug Matching ===")

    drug_outcomes = pd.read_csv(os.path.join(tcga_dir, 'patient_drug_outcome_2.csv'))

    # Our drug names
    our_drugs = set()
    with open(os.path.join(proc_v2, 'node.dat')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if int(parts[2]) == 1:
                our_drugs.add(parts[1])

    # Direct + fuzzy matching
    tcga_to_ours = {}
    our_clean = {d.replace('-', '').replace(' ', '').upper(): d for d in our_drugs}

    for tcga_drug in drug_outcomes['drug_id'].unique():
        upper = str(tcga_drug).upper().strip()
        clean = upper.replace('-', '').replace(' ', '')

        if upper in our_drugs:
            tcga_to_ours[tcga_drug] = upper
        elif clean in our_clean:
            tcga_to_ours[tcga_drug] = our_clean[clean]

    print(f"  TCGA drugs: {drug_outcomes['drug_id'].nunique()}")
    print(f"  Matched to graph: {len(tcga_to_ours)}")
    print(f"  Sample matches: {list(tcga_to_ours.items())[:5]}")

    return tcga_to_ours


def step2_map_outcomes(tcga_dir, drug_map):
    """Map clinical outcomes to binary labels and filter to matched drugs."""
    print("\n=== Step 2: Outcome Mapping ===")

    drug_outcomes = pd.read_csv(os.path.join(tcga_dir, 'patient_drug_outcome_2.csv'))

    # Binary mapping
    sensitive_outcomes = {'Complete Response', 'Partial Response'}
    resistant_outcomes = {'Progressive Disease', 'No Response'}
    exclude_outcomes = {'Treatment Ongoing', 'Unknown', 'Stable Disease',
                       'Not Reported', 'Treatment Stopped Due to Toxicity'}

    drug_outcomes['label'] = drug_outcomes['Treatment outcome'].map(
        lambda x: 1 if x in sensitive_outcomes else (0 if x in resistant_outcomes else -1)
    )

    # Filter to matched drugs and valid labels
    drug_outcomes['graph_drug'] = drug_outcomes['drug_id'].map(drug_map)
    valid = drug_outcomes[(drug_outcomes['graph_drug'].notna()) & (drug_outcomes['label'] >= 0)].copy()

    # Extract TCGA patient ID (format: UUID_suffix or TCGA-XX-XXXX)
    valid['patient_short'] = valid['patient_id'].apply(
        lambda x: x.split('_')[0][:36] if '_' in str(x) else str(x)[:12]
    )

    print(f"  Total records: {len(drug_outcomes):,}")
    print(f"  After drug matching: {len(drug_outcomes[drug_outcomes['graph_drug'].notna()]):,}")
    print(f"  After label filtering: {len(valid):,}")
    print(f"  Sensitive: {(valid['label'] == 1).sum():,}")
    print(f"  Resistant: {(valid['label'] == 0).sum():,}")
    print(f"  Unique patients: {valid['patient_short'].nunique()}")
    print(f"  Unique drugs: {valid['graph_drug'].nunique()}")

    return valid


def step3_encode_expression(tcga_dir, misc_dir, proc_v2, device):
    """Normalize TCGA expression to CCLE space and encode through VAE."""
    print("\n=== Step 3: Expression Encoding ===")

    # Load CCLE expression for scaler fitting
    print("  Loading CCLE expression for scaler...")
    ccle_expr = pd.read_csv(os.path.join(misc_dir,
        '24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'), index_col=0)
    ccle_genes = set()
    gene_col_map = {}
    for col in ccle_expr.columns:
        symbol = col.split(' (')[0] if ' (' in col else col
        ccle_genes.add(symbol)
        gene_col_map[symbol] = col

    print(f"  CCLE: {ccle_expr.shape[0]} cells x {ccle_expr.shape[1]} genes")

    # Load TCGA expression
    print("  Loading TCGA expression (this may take a minute)...")
    tcga_expr = pd.read_csv(os.path.join(tcga_dir,
        'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv'), sep='\t', index_col=0)
    print(f"  TCGA raw: {tcga_expr.shape[0]} genes x {tcga_expr.shape[1]} samples")

    # TCGA gene IDs: format is "GENE|ENTREZ" — extract gene symbol
    tcga_gene_map = {}
    for gene_id in tcga_expr.index:
        parts = str(gene_id).split('|')
        symbol = parts[0] if parts[0] != '?' else None
        if symbol and symbol in ccle_genes:
            tcga_gene_map[gene_id] = symbol

    print(f"  TCGA genes matching CCLE: {len(tcga_gene_map)}")

    # Subset and transpose TCGA to (samples x genes)
    tcga_matched = tcga_expr.loc[tcga_expr.index.isin(tcga_gene_map.keys())].copy()
    tcga_matched.index = tcga_matched.index.map(tcga_gene_map)
    # Handle duplicate gene symbols — take mean
    tcga_matched = tcga_matched.groupby(tcga_matched.index).mean()
    tcga_transposed = tcga_matched.T  # (samples x genes)
    print(f"  TCGA transposed: {tcga_transposed.shape}")

    # Align genes to CCLE order
    ccle_gene_order = [col.split(' (')[0] for col in ccle_expr.columns]
    common_genes = [g for g in ccle_gene_order if g in tcga_transposed.columns]
    missing_genes = [g for g in ccle_gene_order if g not in tcga_transposed.columns]
    print(f"  Common genes: {len(common_genes)}, Missing: {len(missing_genes)}")

    # Build aligned matrix (fill missing with 0)
    tcga_aligned = pd.DataFrame(0.0, index=tcga_transposed.index, columns=ccle_gene_order)
    tcga_aligned[common_genes] = tcga_transposed[common_genes].values

    # Log transform TCGA if not already (CCLE is log(TPM+1))
    # TCGA RSEM values can be large — apply log(x+1) to match CCLE
    if tcga_aligned.max().max() > 100:
        print("  Applying log(x+1) transform to TCGA...")
        tcga_aligned = np.log1p(tcga_aligned)

    # Fit scaler on CCLE, transform TCGA
    print("  Fitting StandardScaler on CCLE...")
    scaler = StandardScaler()
    scaler.fit(ccle_expr.values.astype(np.float32))
    scaler.scale_[scaler.scale_ == 0] = 1.0

    tcga_scaled = scaler.transform(tcga_aligned.values.astype(np.float32))
    tcga_scaled = np.nan_to_num(tcga_scaled, nan=0.0)

    print(f"  TCGA scaled stats: mean={tcga_scaled.mean():.3f}, std={tcga_scaled.std():.3f}")

    # Load VAE and encode
    print("  Loading VAE...")
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from cell_vae import CellLineVAE

    input_dim = ccle_expr.shape[1]
    dims = [input_dim, 10000, 5000, 2048, 1024, 512]
    vae = CellLineVAE(dims, dropout_rate=0.4).to(device)
    vae.load_state_dict(torch.load('data/embeddings/cell_vae_weights.pth', map_location=device))
    vae.eval()

    print("  Encoding TCGA through VAE...")
    tcga_tensor = torch.tensor(tcga_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tcga_tensor), batch_size=64, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            mu, _ = vae.encode(batch_x)
            embeddings.append(mu.cpu().numpy())

    tcga_embeddings = np.concatenate(embeddings, axis=0)
    tcga_sample_ids = list(tcga_aligned.index)

    print(f"  TCGA embeddings: {tcga_embeddings.shape}")
    print(f"  Embedding stats: mean={tcga_embeddings.mean():.4f}, std={tcga_embeddings.std():.4f}")

    # Compare to CCLE embedding stats
    ccle_emb = np.load(os.path.join(proc_v2, 'embeddings', 'final_vae_cell_embeddings.npy'))
    print(f"  CCLE embedding stats: mean={ccle_emb.mean():.4f}, std={ccle_emb.std():.4f}")

    return tcga_embeddings, tcga_sample_ids, ccle_emb


def step4_build_patient_edges(tcga_dir, proc_v2):
    """Build patient-gene edges from TCGA mutations."""
    print("\n=== Step 4: Patient-Gene Edges ===")

    # Use pre-processed patient-gene pairs
    pg = pd.read_csv(os.path.join(tcga_dir, 'multi_type_patient_gene_pairs.tsv'), sep='\t')
    print(f"  Raw patient-gene pairs: {len(pg):,}")

    # Our graph genes
    graph_genes = set()
    with open(os.path.join(proc_v2, 'node.dat')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if int(parts[2]) == 2:
                graph_genes.add(parts[1])

    # Filter to graph genes
    pg_filtered = pg[pg['Hugo_Symbol'].isin(graph_genes)]
    print(f"  After filtering to graph genes: {len(pg_filtered):,}")
    print(f"  Patients: {pg_filtered['patient_id'].nunique()}")
    print(f"  Genes: {pg_filtered['Hugo_Symbol'].nunique()}")

    # Build patient -> [genes] mapping
    patient_genes = defaultdict(list)
    for _, row in pg_filtered.iterrows():
        patient_genes[row['patient_id']].append(row['Hugo_Symbol'])

    return patient_genes


def step5_match_patients_to_samples(valid_outcomes, tcga_sample_ids, tcga_dir):
    """Match drug outcome patient IDs to expression sample IDs."""
    print("\n=== Step 5: Patient-Sample Matching ===")

    # Load UUID -> barcode mapping (from GDC API)
    uuid_map_path = os.path.join(tcga_dir, 'uuid_to_barcode.csv')
    if os.path.exists(uuid_map_path):
        uuid_map = pd.read_csv(uuid_map_path)
        uuid_to_barcode = dict(zip(uuid_map['uuid'], uuid_map['tcga_barcode']))
        print(f"  Loaded UUID mapping: {len(uuid_to_barcode)} entries")
    else:
        uuid_to_barcode = {}
        print("  WARNING: uuid_to_barcode.csv not found. Run GDC API resolution first.")

    # Expression samples -> patient barcode (first 12 chars)
    sample_patient_map = {}
    for sid in tcga_sample_ids:
        if sid.startswith('TCGA-'):
            patient = sid[:12]
            sample_patient_map[sid] = patient

    expression_patients = set(sample_patient_map.values())

    # Drug outcome UUIDs -> TCGA barcodes
    patient_uuid_to_tcga = {}
    outcome_patients = set()

    for pid in valid_outcomes['patient_id'].unique():
        uuid_part = str(pid).split('_')[0]
        barcode = uuid_to_barcode.get(uuid_part)
        if barcode:
            patient_uuid_to_tcga[pid] = barcode
            outcome_patients.add(barcode)

    matched_patients = outcome_patients & expression_patients

    # Find best sample per patient (prefer -01A tumor samples)
    patient_to_sample = {}
    for sid, pid in sample_patient_map.items():
        if pid in matched_patients:
            if pid not in patient_to_sample or '-01A-' in sid:
                patient_to_sample[pid] = sid

    print(f"  Drug outcome patients (resolved): {len(outcome_patients)}")
    print(f"  Expression patients: {len(expression_patients)}")
    print(f"  Matched: {len(matched_patients)}")
    print(f"  With expression sample: {len(patient_to_sample)}")

    return patient_to_sample, patient_uuid_to_tcga


def step6_write_eval_files(valid_outcomes, tcga_embeddings, tcga_sample_ids,
                           ccle_emb, patient_to_sample, patient_uuid_to_tcga,
                           patient_genes, proc_v2, output_dir):
    """Write TCGA evaluation files."""
    print("\n=== Step 6: Write Evaluation Files ===")

    os.makedirs(output_dir, exist_ok=True)

    # Load graph mappings
    with open(os.path.join(proc_v2, 'id_mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
    drug_to_gid = mappings['drug_to_gid']
    gene_to_gid = mappings['gene_to_gid']
    cell_to_gid = mappings['cell_to_gid']

    # Sample ID -> embedding index
    sample_to_idx = {sid: i for i, sid in enumerate(tcga_sample_ids)}

    # Build TCGA patient embeddings (aligned to matched patients)
    tcga_patients = []
    tcga_patient_embeds = []
    tcga_patient_mutations = {}

    for tcga_id, sample_id in patient_to_sample.items():
        idx = sample_to_idx.get(sample_id)
        if idx is not None:
            tcga_patients.append(tcga_id)
            tcga_patient_embeds.append(tcga_embeddings[idx])
            # Mutations for this patient
            tcga_patient_mutations[tcga_id] = patient_genes.get(tcga_id, [])

    tcga_patient_embeds = np.array(tcga_patient_embeds)
    print(f"  TCGA patients with embeddings: {len(tcga_patients)}")

    # KNN: find most similar training cells for each TCGA patient
    print("  Computing KNN to training cells...")
    sims = sklearn_cosine(tcga_patient_embeds, ccle_emb)
    tcga_knn = {}
    for i, tcga_id in enumerate(tcga_patients):
        top_k = np.argsort(sims[i])[-15:]
        tcga_knn[tcga_id] = [(int(j), float(sims[i, j])) for j in top_k]

    # Save patient embeddings
    np.save(os.path.join(output_dir, 'tcga_patient_embeddings.npy'), tcga_patient_embeds)
    with open(os.path.join(output_dir, 'tcga_patient_ids.txt'), 'w') as f:
        for pid in tcga_patients:
            f.write(f"{pid}\n")

    # Save KNN edges
    with open(os.path.join(output_dir, 'tcga_knn_edges.json'), 'w') as f:
        json.dump(tcga_knn, f)

    # Save patient mutations
    with open(os.path.join(output_dir, 'tcga_patient_mutations.json'), 'w') as f:
        json.dump(tcga_patient_mutations, f)

    # Build evaluation pairs: (patient, drug, label)
    tcga_patient_set = set(tcga_patients)
    eval_pairs = []
    skipped_no_patient = 0
    skipped_no_drug = 0

    for _, row in valid_outcomes.iterrows():
        pid = row['patient_id']
        tcga_id = patient_uuid_to_tcga.get(pid)
        if not tcga_id or tcga_id not in tcga_patient_set:
            skipped_no_patient += 1
            continue

        drug = row['graph_drug']
        if drug not in drug_to_gid:
            skipped_no_drug += 1
            continue

        eval_pairs.append({
            'patient': tcga_id,
            'drug': drug,
            'drug_gid': drug_to_gid[drug],
            'label': int(row['label']),
            'outcome': row['Treatment outcome'],
        })

    eval_df = pd.DataFrame(eval_pairs)
    print(f"  Evaluation pairs: {len(eval_df):,}")
    print(f"  Skipped (no patient match): {skipped_no_patient}")
    print(f"  Skipped (no drug match): {skipped_no_drug}")
    print(f"  Patients: {eval_df['patient'].nunique()}")
    print(f"  Drugs: {eval_df['drug'].nunique()}")
    print(f"  Sensitive: {(eval_df['label'] == 1).sum():,}")
    print(f"  Resistant: {(eval_df['label'] == 0).sum():,}")

    # Separate into known-drug vs new-drug scenarios
    # Load training drug set
    with open(os.path.join(proc_v2, 'split_config.json')) as f:
        sc = json.load(f)
    # Training drugs = drugs that appear in training LP links
    train_lp = []
    with open(os.path.join(proc_v2, 'train_lp_links.dat')) as f:
        for line in f:
            parts = line.strip().split('\t')
            train_lp.append((int(parts[0]), int(parts[1])))

    gid_to_name = {}
    with open(os.path.join(proc_v2, 'node.dat')) as f:
        for line in f:
            parts = line.strip().split('\t')
            gid_to_name[int(parts[0])] = parts[1]

    train_drugs = set()
    for src, tgt in train_lp:
        for gid in [src, tgt]:
            name = gid_to_name.get(gid, '')
            if name in drug_to_gid:
                train_drugs.add(name)

    eval_df['drug_known'] = eval_df['drug'].isin(train_drugs)
    known_drug = eval_df[eval_df['drug_known']]
    new_drug = eval_df[~eval_df['drug_known']]

    print(f"\n  TCGA-Known (drugs in training): {len(known_drug):,} pairs, "
          f"{known_drug['patient'].nunique()} patients, {known_drug['drug'].nunique()} drugs")
    print(f"  TCGA-New (drugs NOT in training): {len(new_drug):,} pairs, "
          f"{new_drug['patient'].nunique()} patients, {new_drug['drug'].nunique()} drugs")

    # Save evaluation files
    eval_df.to_csv(os.path.join(output_dir, 'tcga_eval_all.csv'), index=False)
    known_drug.to_csv(os.path.join(output_dir, 'tcga_eval_known_drugs.csv'), index=False)
    new_drug.to_csv(os.path.join(output_dir, 'tcga_eval_new_drugs.csv'), index=False)

    # Summary
    summary = {
        'n_patients': len(tcga_patients),
        'n_eval_pairs': len(eval_df),
        'n_sensitive': int((eval_df['label'] == 1).sum()),
        'n_resistant': int((eval_df['label'] == 0).sum()),
        'n_known_drug_pairs': len(known_drug),
        'n_new_drug_pairs': len(new_drug),
        'n_drugs_matched': eval_df['drug'].nunique(),
        'n_drugs_known': known_drug['drug'].nunique(),
        'n_drugs_new': new_drug['drug'].nunique(),
        'embedding_shape': list(tcga_patient_embeds.shape),
    }
    with open(os.path.join(output_dir, 'tcga_eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to: {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, fname)) / 1024
        print(f"    {fname} ({size:.0f} KB)")

    return eval_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=PROC_V2)
    parser.add_argument('--tcga_dir', type=str, default=TCGA_DIR)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'tcga_eval')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("TCGA EVALUATION PIPELINE")
    print("=" * 60)

    # Step 1: Match drugs
    drug_map = step1_match_drugs(args.tcga_dir, args.data_dir)

    # Step 2: Map outcomes
    valid_outcomes = step2_map_outcomes(args.tcga_dir, drug_map)

    # Step 3: Encode expression
    tcga_embeddings, tcga_sample_ids, ccle_emb = step3_encode_expression(
        args.tcga_dir, MISC, args.data_dir, device)

    # Step 4: Patient-gene edges
    patient_genes = step4_build_patient_edges(args.tcga_dir, args.data_dir)

    # Step 5: Match patients to samples
    patient_to_sample, patient_uuid_to_tcga = step5_match_patients_to_samples(
        valid_outcomes, tcga_sample_ids, args.tcga_dir)

    # Step 6: Write eval files
    eval_df = step6_write_eval_files(
        valid_outcomes, tcga_embeddings, tcga_sample_ids, ccle_emb,
        patient_to_sample, patient_uuid_to_tcga, patient_genes,
        args.data_dir, args.output_dir)

    print(f"\n{'='*60}")
    print("TCGA EVALUATION DATA READY")
    print(f"{'='*60}")
    print(f"\n  Next: Run inference with scripts/inference_tcga.py")


if __name__ == '__main__':
    main()
