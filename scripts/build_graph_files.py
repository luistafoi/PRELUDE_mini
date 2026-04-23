# scripts/build_graph_files.py
"""
Constructs the final graph files (node.dat, link.dat, info.dat) and node
mapping files from curated raw link files.

Feature-validated: only includes nodes that have pre-computed embeddings
and meet the inclusion rules defined in the change plan.

Edge types:
  0: Cell-Drug  (prediction target, GMM binary label)
  1: Gene-Gene  (binary weight=1)
  2: Cell-Gene  (pathogenicity score [0,1])
  3: Drug-Gene  (normalized interaction score [0,1])
  4: Cell-Cell  (cosine similarity [0,1])
"""
import pandas as pd
import json
import pickle
from collections import defaultdict
import os
import argparse
import sys


def load_feature_whitelists(args):
    """Load embedding files to create sets of valid node names (UPPERCASE)."""
    print("--- Loading Feature Whitelists ---")

    # Cells
    with open(args.cell_names_file) as f:
        valid_cells = {line.strip().upper() for line in f if line.strip()}
    print(f"  > Cells with embeddings: {len(valid_cells)}")

    # Drugs
    df_drugs = pd.read_csv(args.drug_embed_file, usecols=[0])
    valid_drugs = set(df_drugs.iloc[:, 0].str.strip().str.upper())
    print(f"  > Drugs with embeddings: {len(valid_drugs)}")

    # Genes
    with open(args.gene_embed_file, 'rb') as f:
        gene_dict = pickle.load(f)
    valid_genes = {str(k).upper() for k in gene_dict.keys()}
    print(f"  > Genes with embeddings: {len(valid_genes)}")

    return valid_cells, valid_drugs, valid_genes


def main(args):
    valid_cells, valid_drugs, valid_genes = load_feature_whitelists(args)

    # Whitelist lookup by node type ID
    whitelist = {0: valid_cells, 1: valid_drugs, 2: valid_genes}
    type_names = {0: "cell", 1: "drug", 2: "gene"}

    # Edge type definitions: (file_path, src_type, tgt_type, edge_type_id, relation_name, n_columns)
    edge_defs = [
        {
            'name': 'Cell-Drug',
            'path': os.path.join(args.raw_dir, "link_cell_drug_labeled.txt"),
            'src_type': 0, 'tgt_type': 1,
            'edge_type': 0,
            'relation': 'cell-drug_interaction',
            'expected_cols': 4,  # cell, drug, type_from_file, label
            'weight_col': 3,     # 4th column = GMM label
        },
        {
            'name': 'Gene-Gene',
            'path': os.path.join(args.raw_dir, "link_gene_gene.txt"),
            'src_type': 2, 'tgt_type': 2,
            'edge_type': 1,
            'relation': 'gene-interacts_gene',
            'expected_cols': 3,
            'weight_col': 2,
        },
        {
            'name': 'Cell-Gene',
            'path': os.path.join(args.raw_dir, "link_cell_gene_mutation.txt"),
            'src_type': 0, 'tgt_type': 2,
            'edge_type': 2,
            'relation': 'cell-gene_mutation',
            'expected_cols': 3,
            'weight_col': 2,
        },
        {
            'name': 'Drug-Gene',
            'path': os.path.join(args.raw_dir, "link_gene_drug_relation.txt"),
            'src_type': 2, 'tgt_type': 1,  # file format: gene, drug, weight
            'edge_type': 3,
            'relation': 'drug-gene_inhibition',
            'expected_cols': 3,
            'weight_col': 2,
        },
        {
            'name': 'Cell-Cell',
            'path': os.path.join(args.raw_dir, "link_cell_cell_similarity.txt"),
            'src_type': 0, 'tgt_type': 0,
            'edge_type': 4,
            'relation': 'cell-cell_similarity',
            'expected_cols': 3,
            'weight_col': 2,
        },
    ]

    # --- Build graph ---
    node2id = {}        # (name, type) -> global_id
    node_names = {}     # global_id -> name (for node.dat output)
    node_type_map = {}  # global_id -> type
    next_id = [0]  # mutable for closure
    all_edges = []
    link_info = {}

    def get_or_create_node(name, ntype):
        """Get existing node ID or create new one. Keyed by (name, type) to avoid collisions."""
        key = (name, ntype)
        if key not in node2id:
            node2id[key] = next_id[0]
            node_names[next_id[0]] = name
            node_type_map[next_id[0]] = ntype
            next_id[0] += 1
        return node2id[key]

    # Controls to exclude from prediction edges (vehicle/positive controls)
    EXCLUDED_DRUGS = {'DMSO'}
    print(f"\n  Excluding control compounds from Cell-Drug edges: {EXCLUDED_DRUGS}")

    print("\n--- Building graph from raw link files ---")

    for edef in edge_defs:
        path = edef['path']
        if not os.path.exists(path):
            print(f"  > Warning: {edef['name']} file not found at {path}. Skipping.")
            link_info[str(edef['edge_type'])] = [
                type_names[edef['src_type']],
                type_names[edef['tgt_type']],
                edef['relation'], 0]
            continue

        print(f"  > Processing {edef['name']} from {path}...")

        # Read file efficiently
        df = pd.read_csv(path, sep='\t', header=None, dtype=str)
        n_cols = df.shape[1]

        # Parse columns based on file format
        src_col = df.iloc[:, 0].str.strip().str.upper()
        tgt_col = df.iloc[:, 1].str.strip().str.upper()

        # Weight column
        if edef['weight_col'] < n_cols:
            weight_col = pd.to_numeric(df.iloc[:, edef['weight_col']], errors='coerce')
        elif n_cols == 4:
            # Cell-drug format: cell, drug, type, label
            weight_col = pd.to_numeric(df.iloc[:, 3], errors='coerce')
        else:
            weight_col = pd.Series([1.0] * len(df))

        src_type = edef['src_type']
        tgt_type = edef['tgt_type']
        src_whitelist = whitelist[src_type]
        tgt_whitelist = whitelist[tgt_type]

        # Vectorized validation
        valid_mask = src_col.isin(src_whitelist) & tgt_col.isin(tgt_whitelist)

        # For Cell-Drug: also exclude control compounds
        is_cell_drug = (edef['edge_type'] == 0)
        if is_cell_drug:
            drug_col = tgt_col if src_type == 0 else src_col
            valid_mask = valid_mask & ~drug_col.isin(EXCLUDED_DRUGS)

        valid_src = src_col[valid_mask].values
        valid_tgt = tgt_col[valid_mask].values
        valid_weight = weight_col[valid_mask].values

        # --- Deduplication for Cell-Drug edges ---
        # Store the fraction of positive replicates as a soft label (confidence weight).
        # e.g., 3 replicates [1,1,0] → weight 0.667 instead of hard binary 1
        if is_cell_drug:
            from collections import defaultdict
            pair_labels = defaultdict(list)
            n_raw = len(valid_src)
            for i in range(n_raw):
                pair_labels[(valid_src[i], valid_tgt[i])].append(valid_weight[i])

            dedup_src, dedup_tgt, dedup_weight = [], [], []
            n_mixed = 0
            for (s, t), labels in pair_labels.items():
                pos_frac = sum(1 for l in labels if l == 1.0) / len(labels)
                if 0 < pos_frac < 1:
                    n_mixed += 1
                dedup_src.append(s)
                dedup_tgt.append(t)
                dedup_weight.append(round(pos_frac, 4))

            print(f"    Dedup: {n_raw:,} measurements → {len(dedup_src):,} unique pairs "
                  f"({n_mixed:,} had conflicting labels, stored as soft labels)")
            valid_src = dedup_src
            valid_tgt = dedup_tgt
            valid_weight = dedup_weight

        # Create nodes and edges
        edge_type_id = edef['edge_type']
        count = 0
        for i in range(len(valid_src)):
            src_id = get_or_create_node(valid_src[i], src_type)
            tgt_id = get_or_create_node(valid_tgt[i], tgt_type)
            w = valid_weight[i]
            all_edges.append(f"{src_id}\t{tgt_id}\t{edge_type_id}\t{w}")
            count += 1

        link_info[str(edge_type_id)] = [
            type_names[src_type], type_names[tgt_type],
            edef['relation'], count]
        print(f"    {count:,} valid edges")

    # --- Write outputs ---
    os.makedirs(args.output_dir, exist_ok=True)

    # node.dat
    node_dat_path = os.path.join(args.output_dir, "node.dat")
    with open(node_dat_path, 'w') as f:
        for nid in range(next_id[0]):
            name = node_names[nid]
            ntype = node_type_map[nid]
            f.write(f"{nid}\t{name}\t{ntype}\n")

    # link.dat
    link_dat_path = os.path.join(args.output_dir, "link.dat")
    with open(link_dat_path, 'w') as f:
        f.write("\n".join(all_edges))
        if all_edges:
            f.write("\n")

    # info.dat
    type_counts = defaultdict(int)
    for nid in range(next_id[0]):
        type_counts[node_type_map[nid]] += 1

    info = {
        "dataset": "PRELUDE_HetGNN_v2",
        "node.dat": {
            str(k): [type_names[k], type_counts[k]]
            for k in sorted(type_counts.keys())
        },
        "link.dat": link_info,
    }
    info_dat_path = os.path.join(args.output_dir, "info.dat")
    with open(info_dat_path, 'w') as f:
        json.dump(info, f, indent=4)

    # node_mappings.json (name -> global_id)
    # For names shared across types (e.g., "SAG" is both a drug and gene),
    # we namespace them as "NAME__drug", "NAME__gene" etc.
    seen_names = {}  # name -> list of (gid, ntype)
    for (name, ntype), gid in node2id.items():
        seen_names.setdefault(name, []).append((gid, ntype))

    name_to_id = {}
    collisions = 0
    for name, entries in seen_names.items():
        if len(entries) == 1:
            name_to_id[name] = entries[0][0]
        else:
            collisions += 1
            for gid, ntype in entries:
                name_to_id[f"{name}__{type_names[ntype]}"] = gid
    if collisions > 0:
        print(f"  > Warning: {collisions} name collision(s) across types (namespaced in mappings)")
    map_path = os.path.join(args.output_dir, "node_mappings.json")
    with open(map_path, 'w') as f:
        json.dump(name_to_id, f, indent=4)

    # --- Summary ---
    print(f"\n--- Graph Summary ---")
    for k in sorted(type_counts.keys()):
        print(f"  > {type_names[k]}: {type_counts[k]:,} nodes")
    print(f"  > Total nodes: {next_id[0]:,}")
    print(f"  > Total edges: {len(all_edges):,}")
    for etype, info_entry in sorted(link_info.items()):
        print(f"  > Edge type {etype} ({info_entry[2]}): {info_entry[3]:,} edges")
    print(f"\n  Output: {args.output_dir}/")
    print("Graph construction complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build feature-validated graph .dat files.")
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory with curated raw link files.')
    parser.add_argument('--output-dir', default='data/processed',
                        help='Directory to save output .dat files.')
    parser.add_argument('--cell-names-file',
                        default='data/embeddings/final_vae_cell_names.txt')
    parser.add_argument('--drug-embed-file',
                        default='data/embeddings/drugs_with_embeddings.csv')
    parser.add_argument('--gene-embed-file',
                        default='data/embeddings/gene_embeddings_esm_by_symbol.pkl')

    args = parser.parse_args()
    main(args)
