"""Step 3b: Compute Drug-Drug similarity edges using Morgan2 Tanimoto.

Creates Drug-Drug edges based on molecular fingerprint similarity.
KNN approach: top K most similar drugs per drug (same as Cell-Cell).

Appends new edge type to existing link.dat and updates info.dat.
Non-destructive: creates a new link.dat with Drug-Drug edges added.

Usage:
    python scripts/pipeline_v2/step3b_drug_similarity.py
    python scripts/pipeline_v2/step3b_drug_similarity.py --drug_knn 10 --radius 2 --nbits 2048
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except ImportError:
    print("ERROR: RDKit is required. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

PROC_V2 = 'data/processed_v2'


def compute_morgan_fingerprints(smiles_dict, radius=2, nbits=2048):
    """Compute Morgan fingerprints for all drugs.

    Args:
        smiles_dict: {drug_name: SMILES}
        radius: Morgan fingerprint radius (2 = ECFP4)
        nbits: number of bits in fingerprint

    Returns:
        fps: {drug_name: RDKit fingerprint object}
        failed: list of drug names that failed
    """
    fps = {}
    failed = []

    for name, smiles in tqdm(smiles_dict.items(), desc="Computing fingerprints"):
        if not smiles or pd.isna(smiles):
            failed.append(name)
            continue

        # Handle comma-separated SMILES (salt forms) — take the largest fragment
        if ',' in str(smiles):
            parts = str(smiles).split(',')
            smiles = max(parts, key=len).strip()

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed.append(name)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps[name] = fp
        except Exception:
            failed.append(name)

    return fps, failed


def compute_tanimoto_knn(fps, drug_names, k):
    """Compute KNN Drug-Drug edges using Tanimoto similarity.

    Args:
        fps: {drug_name: RDKit fingerprint}
        drug_names: ordered list of drug names (matching GID order)
        k: number of nearest neighbors per drug

    Returns:
        edges: list of (drug_name_a, drug_name_b, similarity)
    """
    # Filter to drugs that have fingerprints
    valid_drugs = [d for d in drug_names if d in fps]
    valid_fps = [fps[d] for d in valid_drugs]
    n = len(valid_drugs)

    print(f"  Computing Tanimoto similarity for {n} drugs (K={k})...")

    # Compute full similarity matrix using RDKit bulk
    # For large N, compute row by row to save memory
    edge_set = set()
    edges = []

    for i in tqdm(range(n), desc="  KNN search"):
        # Compute similarities from drug i to all others
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps)
        sims[i] = -1.0  # exclude self

        # Get top-K
        top_k_idx = np.argpartition(sims, -k)[-k:]

        for j in top_k_idx:
            if sims[j] <= 0:
                continue
            a, b = valid_drugs[i], valid_drugs[j]
            pair = (min(a, b), max(a, b))
            if pair not in edge_set:
                edge_set.add(pair)
                edges.append((a, b, float(sims[j])))

    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_knn', type=int, default=10,
                        help='Number of most similar drugs per drug (default 10)')
    parser.add_argument('--min_similarity', type=float, default=0.3,
                        help='Minimum Tanimoto similarity to create an edge (default 0.3)')
    parser.add_argument('--radius', type=int, default=2,
                        help='Morgan fingerprint radius (2=ECFP4, 3=ECFP6, default 2)')
    parser.add_argument('--nbits', type=int, default=2048,
                        help='Number of bits in Morgan fingerprint (default 2048)')
    args = parser.parse_args()

    # Load master drugs
    master_drugs = pd.read_csv(f'{PROC_V2}/master_drugs.csv')
    graph_drugs = master_drugs[master_drugs['has_smiles']].copy()

    # Load ID mappings
    with open(f'{PROC_V2}/id_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    drug_to_gid = mappings['drug_to_gid']

    print(f"=== Drug-Drug Similarity (Morgan{args.radius} Tanimoto, K={args.drug_knn}) ===\n")

    # Build SMILES dict
    smiles_dict = {}
    for _, row in graph_drugs.iterrows():
        name = row['canonical_name']
        smiles = row['smiles']
        if name in drug_to_gid and pd.notna(smiles):
            smiles_dict[name] = str(smiles)

    print(f"  Drugs with SMILES: {len(smiles_dict)}")

    # Compute fingerprints
    fps, failed = compute_morgan_fingerprints(smiles_dict, radius=args.radius, nbits=args.nbits)
    print(f"  Fingerprints computed: {len(fps)}")
    print(f"  Failed (invalid SMILES): {len(failed)}")
    if failed and len(failed) <= 10:
        print(f"    Failed: {failed}")

    # Compute KNN edges
    drug_names_ordered = sorted(drug_to_gid.keys(), key=lambda x: drug_to_gid[x])
    edges = compute_tanimoto_knn(fps, drug_names_ordered, args.drug_knn)

    # Filter by minimum similarity
    before = len(edges)
    edges = [(a, b, s) for a, b, s in edges if s >= args.min_similarity]
    print(f"  After min_similarity >= {args.min_similarity}: {len(edges):,} edges (removed {before - len(edges):,} noisy)")

    print(f"\n  Drug-Drug edges: {len(edges):,} unique pairs")

    # Stats
    sims = [s for _, _, s in edges]
    if sims:
        print(f"  Similarity range: [{min(sims):.4f}, {max(sims):.4f}]")
        print(f"  Mean: {np.mean(sims):.4f}, Median: {np.median(sims):.4f}")

    # Check coverage of Sanger-only drugs
    sanger_only = set(graph_drugs[~graph_drugs['in_prism']]['canonical_name'])
    sanger_with_dd = set()
    for a, b, _ in edges:
        if a in sanger_only:
            sanger_with_dd.add(a)
        if b in sanger_only:
            sanger_with_dd.add(b)
    print(f"\n  Sanger-only drugs with Drug-Drug edges: {len(sanger_with_dd)}/{len(sanger_only)}")

    # Write Drug-Drug edge file (separate, to be merged)
    dd_path = f'{PROC_V2}/drug_drug_similarity_edges.txt'
    with open(dd_path, 'w') as f:
        for a, b, sim in edges:
            gid_a = drug_to_gid[a]
            gid_b = drug_to_gid[b]
            # Bidirectional
            f.write(f"{gid_a}\t{gid_b}\t5\t{sim:.4f}\n")
            f.write(f"{gid_b}\t{gid_a}\t5\t{sim:.4f}\n")

    n_lines = len(edges) * 2
    print(f"\n  Saved: {dd_path} ({n_lines:,} edges, bidirectional)")

    # Now append to link.dat
    print(f"\n--- Updating link.dat ---")

    # Read existing link.dat
    with open(f'{PROC_V2}/link.dat') as f:
        existing_edges = f.readlines()

    # Count existing edge type 5 (if any from previous run)
    existing_no_dd = [e for e in existing_edges if not e.strip().endswith('\t5\t') and '\t5\t' not in e]
    # More robust: parse and filter
    existing_no_dd = []
    for line in existing_edges:
        parts = line.strip().split('\t')
        if len(parts) >= 3 and parts[2] != '5':
            existing_no_dd.append(line)

    # Append Drug-Drug edges
    with open(dd_path) as f:
        dd_lines = f.readlines()

    with open(f'{PROC_V2}/link.dat', 'w') as f:
        for line in existing_no_dd:
            f.write(line if line.endswith('\n') else line + '\n')
        for line in dd_lines:
            f.write(line if line.endswith('\n') else line + '\n')

    total = len(existing_no_dd) + len(dd_lines)
    print(f"  Updated link.dat: {total:,} edges ({len(existing_no_dd):,} existing + {len(dd_lines):,} Drug-Drug)")

    # Update info.dat
    with open(f'{PROC_V2}/info.dat') as f:
        info = json.load(f)

    info['link.dat']['5'] = ['drug', 'drug', 'drug-drug_similarity', n_lines]
    with open(f'{PROC_V2}/info.dat', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Updated info.dat with edge type 5")

    print(f"\n{'='*60}")
    print(f"DRUG-DRUG SIMILARITY COMPLETE")
    print(f"{'='*60}")
    print(f"  Edge type: 5 (drug-drug_similarity)")
    print(f"  Fingerprint: Morgan{args.radius} ({args.nbits} bits)")
    print(f"  KNN: {args.drug_knn} per drug")
    print(f"  Edges: {n_lines:,} (bidirectional)")
    print(f"\n  Next: re-run step4 and step5 to rebuild splits and neighbors")


if __name__ == '__main__':
    main()
