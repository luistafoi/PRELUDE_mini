"""Step 5: Build feature files and generate neighbor pickle.

Creates:
  - Cell VAE embeddings aligned to graph node order
  - Drug MoleculeSTM embeddings aligned to graph node order
  - Gene ESM embeddings aligned to graph node order
  - Neighbor pickle (with --include_cell_drug option)

Usage:
    python scripts/pipeline_v2/step5_build_features.py
    python scripts/pipeline_v2/step5_build_features.py --include_cell_drug
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MISC = 'data/misc'
PROC_V2 = 'data/processed_v2'
EMB_DIR = 'data/embeddings'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_cell_drug', action='store_true',
                        help='Include Cell-Drug edges in neighbor pickle for dynamic isolation')
    parser.add_argument('--max_neighbors', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(f'{PROC_V2}/embeddings', exist_ok=True)

    # Load ID mappings
    with open(f'{PROC_V2}/id_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    cell_to_gid = mappings['cell_to_gid']
    drug_to_gid = mappings['drug_to_gid']
    gene_to_gid = mappings['gene_to_gid']

    # Reverse maps
    gid_to_cell = {v: k for k, v in cell_to_gid.items()}
    gid_to_drug = {v: k for k, v in drug_to_gid.items()}
    gid_to_gene = {v: k for k, v in gene_to_gid.items()}

    n_cells = len(cell_to_gid)
    n_drugs = len(drug_to_gid)
    n_genes = len(gene_to_gid)

    # ==========================================
    # CELL FEATURES (VAE)
    # ==========================================
    print("=== Cell Features (VAE) ===")

    # Load existing VAE embeddings
    vae_emb = np.load(f'{EMB_DIR}/final_vae_cell_embeddings.npy')
    with open(f'{EMB_DIR}/final_vae_cell_names.txt') as f:
        vae_names = [line.strip() for line in f if line.strip()]

    vae_map = {name.upper(): vae_emb[i] for i, name in enumerate(vae_names)}
    feat_dim = vae_emb.shape[1]

    # Align to graph order
    cell_features = np.zeros((n_cells, feat_dim), dtype=np.float32)
    cell_names_ordered = []
    matched = 0
    for ach_id in sorted(cell_to_gid.keys(), key=lambda x: cell_to_gid[x]):
        lid = cell_to_gid[ach_id] - 0  # cells start at GID 0
        vec = vae_map.get(ach_id.upper())
        if vec is not None:
            cell_features[lid] = vec
            matched += 1
        cell_names_ordered.append(ach_id)

    np.save(f'{PROC_V2}/embeddings/final_vae_cell_embeddings.npy', cell_features)
    with open(f'{PROC_V2}/embeddings/final_vae_cell_names.txt', 'w') as f:
        for name in cell_names_ordered:
            f.write(f"{name}\n")

    print(f"  Shape: {cell_features.shape}")
    print(f"  Matched: {matched}/{n_cells}")
    print(f"  Saved: {PROC_V2}/embeddings/final_vae_cell_embeddings.npy")

    # ==========================================
    # DRUG FEATURES (MoleculeSTM)
    # ==========================================
    print("\n=== Drug Features (MoleculeSTM) ===")

    drugs_df = pd.read_csv(f'{EMB_DIR}/drugs_with_embeddings.csv')
    drug_emb_map = {}
    emb_cols = [c for c in drugs_df.columns if c.startswith('embedding_')]
    for _, row in drugs_df.iterrows():
        name = str(row['Drug']).upper()
        vec = row[emb_cols].values.astype(np.float32)
        drug_emb_map[name] = vec

    drug_feat_dim = len(emb_cols)
    drug_features = np.zeros((n_drugs, drug_feat_dim), dtype=np.float32)
    drug_names_ordered = []
    matched = 0
    for drug_name in sorted(drug_to_gid.keys(), key=lambda x: drug_to_gid[x]):
        lid = drug_to_gid[drug_name] - n_cells  # drugs start after cells
        vec = drug_emb_map.get(drug_name)
        if vec is not None:
            drug_features[lid] = vec
            matched += 1
        drug_names_ordered.append(drug_name)

    # Save as CSV matching existing format
    drug_df_out = pd.DataFrame(drug_features, columns=emb_cols)
    drug_df_out.insert(0, 'Drug', drug_names_ordered)
    drug_df_out.to_csv(f'{PROC_V2}/embeddings/drugs_with_embeddings.csv', index=False)

    print(f"  Shape: ({n_drugs}, {drug_feat_dim})")
    print(f"  Matched: {matched}/{n_drugs}")
    print(f"  Saved: {PROC_V2}/embeddings/drugs_with_embeddings.csv")

    # ==========================================
    # GENE FEATURES (ESM)
    # ==========================================
    print("\n=== Gene Features (ESM) ===")

    with open(f'{EMB_DIR}/gene_embeddings_esm_by_symbol.pkl', 'rb') as f:
        esm_map = pickle.load(f)

    # Determine ESM dim from first entry
    sample_vec = next(iter(esm_map.values()))
    if isinstance(sample_vec, np.ndarray):
        gene_feat_dim = sample_vec.shape[0]
    else:
        gene_feat_dim = len(sample_vec)

    gene_features = {}
    matched = 0
    for gene_name in sorted(gene_to_gid.keys(), key=lambda x: gene_to_gid[x]):
        vec = esm_map.get(gene_name)
        if vec is not None:
            if isinstance(vec, np.ndarray):
                gene_features[gene_name] = vec.astype(np.float32)
            else:
                gene_features[gene_name] = np.array(vec, dtype=np.float32)
            matched += 1
        else:
            gene_features[gene_name] = np.zeros(gene_feat_dim, dtype=np.float32)

    with open(f'{PROC_V2}/embeddings/gene_embeddings_esm_by_symbol.pkl', 'wb') as f:
        pickle.dump(gene_features, f)

    print(f"  Dim: {gene_feat_dim}")
    print(f"  Matched: {matched}/{n_genes}")
    print(f"  Saved: {PROC_V2}/embeddings/gene_embeddings_esm_by_symbol.pkl")

    # ==========================================
    # NEIGHBOR PICKLE
    # ==========================================
    print(f"\n=== Neighbor Pickle (include_cell_drug={args.include_cell_drug}) ===")

    # Load structural graph
    neighbors = defaultdict(lambda: defaultdict(list))
    neighbor_weights = defaultdict(lambda: defaultdict(list))

    # Build node type map: gid -> type
    node_type = {}
    with open(f'{PROC_V2}/node.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_type[int(parts[0])] = int(parts[2])

    # Type IDs
    CELL_TYPE, DRUG_TYPE, GENE_TYPE = 0, 1, 2

    with open(f'{PROC_V2}/train.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            src, tgt, etype = int(parts[0]), int(parts[1]), int(parts[2])
            weight = float(parts[3])

            st = node_type.get(src)
            tt = node_type.get(tgt)

            # Skip Cell-Drug edges if not including them
            is_cell_drug = (st == CELL_TYPE and tt == DRUG_TYPE) or (st == DRUG_TYPE and tt == CELL_TYPE)
            if is_cell_drug and not args.include_cell_drug:
                continue

            # Bidirectional
            neighbors[src][etype].append(tgt)
            neighbor_weights[src][etype].append(weight)
            neighbors[tgt][etype].append(src)
            neighbor_weights[tgt][etype].append(weight)

    # Build local ID lookups
    # type_map: gid -> (type_id, local_id)
    type_counts = {CELL_TYPE: n_cells, DRUG_TYPE: n_drugs, GENE_TYPE: n_genes}
    type_map = {}
    cell_lid = 0
    drug_lid = 0
    gene_lid = 0
    with open(f'{PROC_V2}/node.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            gid, ntype = int(parts[0]), int(parts[2])
            if ntype == CELL_TYPE:
                type_map[gid] = (CELL_TYPE, cell_lid)
                cell_lid += 1
            elif ntype == DRUG_TYPE:
                type_map[gid] = (DRUG_TYPE, drug_lid)
                drug_lid += 1
            elif ntype == GENE_TYPE:
                type_map[gid] = (GENE_TYPE, gene_lid)
                gene_lid += 1

    all_node_types = [CELL_TYPE, DRUG_TYPE, GENE_TYPE]
    PAD_VALUE = -1
    MAX_NEIGHBORS = args.max_neighbors

    precomputed = {}
    empty_by_type = {
        nt: {'ids': [PAD_VALUE] * MAX_NEIGHBORS, 'weights': [0.0] * MAX_NEIGHBORS}
        for nt in all_node_types
    }

    for gid in tqdm(sorted(type_map.keys()), desc="Building neighbors"):
        center_type, center_lid = type_map[gid]

        # Parse neighbors by type
        parsed = defaultdict(list)
        for rtype, neigh_gids in neighbors[gid].items():
            weights = neighbor_weights[gid][rtype]
            for i, ng in enumerate(neigh_gids):
                neigh_info = type_map.get(ng)
                if neigh_info:
                    neigh_type, neigh_lid = neigh_info
                    w = weights[i] if i < len(weights) else 1.0
                    parsed[neigh_type].append((neigh_lid, w))

        # Sample and pad
        padded = {}
        for nt in all_node_types:
            neigh_list = parsed.get(nt, [])
            if len(neigh_list) > MAX_NEIGHBORS:
                import random
                sampled = random.sample(neigh_list, MAX_NEIGHBORS)
                ids = [x[0] for x in sampled]
                weights = [x[1] for x in sampled]
            elif len(neigh_list) > 0:
                ids = [x[0] for x in neigh_list] + [PAD_VALUE] * (MAX_NEIGHBORS - len(neigh_list))
                weights = [x[1] for x in neigh_list] + [0.0] * (MAX_NEIGHBORS - len(neigh_list))
            else:
                ids = [PAD_VALUE] * MAX_NEIGHBORS
                weights = [0.0] * MAX_NEIGHBORS
            padded[nt] = {'ids': ids, 'weights': weights}

        precomputed[(center_type, center_lid)] = padded

    pkl_path = f'{PROC_V2}/train_neighbors_preprocessed.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(precomputed, f)

    # Count Cell-Drug neighbors
    cd_count = 0
    dc_count = 0
    for (ct, lid), neighs in precomputed.items():
        if ct == CELL_TYPE:
            cd_count += sum(1 for x in neighs[DRUG_TYPE]['ids'] if x != PAD_VALUE)
        elif ct == DRUG_TYPE:
            dc_count += sum(1 for x in neighs[CELL_TYPE]['ids'] if x != PAD_VALUE)

    print(f"  Saved: {pkl_path}")
    print(f"  Total nodes: {len(precomputed)}")
    print(f"  Cell->Drug entries: {cd_count}")
    print(f"  Drug->Cell entries: {dc_count}")

    print(f"\n{'='*60}")
    print("FEATURES AND NEIGHBORS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
