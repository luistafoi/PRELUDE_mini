"""Step 3: Build graph files (node.dat, link.dat, info.dat) from resolved data.

Edge quality approach:
  - Cell-Drug (type 0): GMM ratio-matched labels from step2
  - Gene-Gene (type 1): Binary PPI edges (no filtering needed)
  - Cell-Gene (type 2): 3-component GMM on pathogenicity, >80% posterior for pathogenic
  - Drug-Gene (type 3): Typed inhibitory only + score threshold (excludes untyped/activating)
  - Cell-Cell (type 4): KNN cosine similarity (top K per cell, ensures connectivity)

Usage:
    python scripts/pipeline_v2/step3_build_graph.py
    python scripts/pipeline_v2/step3_build_graph.py --cell_knn 15 --dgi_min_score 0.1
"""

import os
import sys
import json
import argparse
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

MISC = 'data/misc'
PROC_V2 = 'data/processed_v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_knn', type=int, default=15,
                        help='Number of most-similar cells per cell for Cell-Cell edges (default 15)')
    parser.add_argument('--dgi_min_score', type=float, default=0.1,
                        help='Min DGIdb interaction score for Drug-Gene edges (default 0.1)')
    parser.add_argument('--pathogenicity_gmm_posterior', type=float, default=0.8,
                        help='Min posterior probability for GMM pathogenic class (default 0.8)')
    args = parser.parse_args()

    os.makedirs(PROC_V2, exist_ok=True)

    # Load master tables
    master_cells = pd.read_csv(f'{PROC_V2}/master_cells.csv')
    master_drugs = pd.read_csv(f'{PROC_V2}/master_drugs.csv')
    master_genes = pd.read_csv(f'{PROC_V2}/master_genes.csv')

    graph_cells = master_cells[master_cells['in_graph']].sort_values('ach_id')
    graph_drugs = master_drugs[master_drugs['has_smiles']].sort_values('canonical_name')
    graph_genes = master_genes[master_genes['in_graph']].sort_values('hugo_symbol')

    print(f"Graph nodes: {len(graph_cells)} cells, {len(graph_drugs)} drugs, {len(graph_genes)} genes")

    # ==========================================
    # ASSIGN GLOBAL IDs
    # ==========================================
    print("\n--- Assigning global IDs ---")

    cell_to_gid = {}
    drug_to_gid = {}
    gene_to_gid = {}

    gid = 0
    for _, row in graph_cells.iterrows():
        cell_to_gid[row['ach_id']] = gid
        gid += 1
    for _, row in graph_drugs.iterrows():
        drug_to_gid[row['canonical_name']] = gid
        gid += 1
    for _, row in graph_genes.iterrows():
        gene_to_gid[row['hugo_symbol']] = gid
        gid += 1

    total_nodes = gid
    print(f"  Total nodes: {total_nodes}")

    # ==========================================
    # WRITE node.dat
    # ==========================================
    print("\n--- Writing node.dat ---")
    node_path = f'{PROC_V2}/node.dat'
    with open(node_path, 'w') as f:
        for name, g in sorted(cell_to_gid.items(), key=lambda x: x[1]):
            f.write(f"{g}\t{name}\t0\n")
        for name, g in sorted(drug_to_gid.items(), key=lambda x: x[1]):
            f.write(f"{g}\t{name}\t1\n")
        for name, g in sorted(gene_to_gid.items(), key=lambda x: x[1]):
            f.write(f"{g}\t{name}\t2\n")
    print(f"  Saved: {node_path} ({total_nodes} nodes)")

    # ==========================================
    # BUILD EDGES
    # ==========================================
    all_edges = []
    edge_counts = {}

    # --- Edge Type 0: Cell-Drug (PRISM labeled) ---
    print("\n--- Edge Type 0: Cell-Drug (PRISM) ---")
    prism_labeled = pd.read_csv(f'{PROC_V2}/prism_labeled.csv')
    cd_count = 0
    for _, row in prism_labeled.iterrows():
        cell_gid = cell_to_gid.get(row['cell_ach'])
        drug_gid = drug_to_gid.get(row['drug_name'])
        if cell_gid is not None and drug_gid is not None:
            if row['excluded']:
                weight = 0.5
            else:
                weight = float(row['label'])
            all_edges.append(f"{cell_gid}\t{drug_gid}\t0\t{weight}")
            cd_count += 1
    edge_counts['Cell-Drug'] = cd_count
    print(f"  Cell-Drug edges: {cd_count:,}")

    # --- Edge Type 1: Gene-Gene (PPI, bidirectional, deduplicated) ---
    print("\n--- Edge Type 1: Gene-Gene (PPI) ---")
    gg = pd.read_csv(f'{MISC}/GeneGene_Interactions_EntrezMapped_Filtered.csv')
    gg_edge_set = set()
    for _, row in gg.iterrows():
        g1 = gene_to_gid.get(row['Gene1'])
        g2 = gene_to_gid.get(row['Gene2'])
        if g1 is not None and g2 is not None:
            pair = (min(g1, g2), max(g1, g2))
            gg_edge_set.add(pair)
    gg_count = 0
    for g1, g2 in gg_edge_set:
        all_edges.append(f"{g1}\t{g2}\t1\t1.0")
        all_edges.append(f"{g2}\t{g1}\t1\t1.0")
        gg_count += 2
    edge_counts['Gene-Gene'] = gg_count
    print(f"  Gene-Gene edges: {gg_count:,} (bidirectional, {len(gg_edge_set):,} unique pairs)")

    # --- Edge Type 2: Cell-Gene (Mutations, 3-component GMM) ---
    print("\n--- Edge Type 2: Cell-Gene (Mutations, 3-GMM) ---")
    mut = pd.read_csv(f'{MISC}/OmicsSomaticMutations_24Q2.csv', low_memory=False)

    model_col = 'ModelID'
    hugo_col = 'HugoSymbol'
    path_cols = [c for c in mut.columns if any(k in c.lower() for k in ['revel', 'polyphen'])]

    graph_cell_set = set(cell_to_gid.keys())
    graph_gene_set = set(gene_to_gid.keys())

    mut_filtered = mut[
        mut[model_col].isin(graph_cell_set) &
        mut[hugo_col].isin(graph_gene_set)
    ].copy()

    for col in path_cols:
        mut_filtered[col] = pd.to_numeric(mut_filtered[col], errors='coerce')
    mut_filtered['max_path'] = mut_filtered[path_cols].max(axis=1)

    # Aggregate per (cell, gene) — take max pathogenicity
    cg_pairs = mut_filtered.groupby([model_col, hugo_col])['max_path'].max().reset_index()

    # Fit 3-component GMM on scores
    scored = cg_pairs['max_path'].dropna().values
    print(f"  Scored cell-gene pairs: {len(scored):,}")

    gmm3 = GaussianMixture(n_components=3, random_state=42, n_init=5).fit(scored.reshape(-1, 1))
    means = gmm3.means_.flatten()
    order = np.argsort(means)  # benign, uncertain, pathogenic

    print(f"  3-GMM components:")
    for i, idx in enumerate(order):
        label = ['benign', 'uncertain', 'pathogenic'][i]
        std = np.sqrt(gmm3.covariances_.flatten()[idx])
        print(f"    {label}: mean={means[idx]:.3f}, std={std:.3f}, weight={gmm3.weights_[idx]:.3f}")

    # Get posterior probability of pathogenic class
    posteriors = gmm3.predict_proba(scored.reshape(-1, 1))
    pathogenic_idx = order[-1]
    is_pathogenic = posteriors[:, pathogenic_idx] >= args.pathogenicity_gmm_posterior

    min_score = scored[is_pathogenic].min() if is_pathogenic.any() else 0.9
    print(f"  Pathogenic (>={args.pathogenicity_gmm_posterior*100:.0f}% posterior): score >= {min_score:.3f}, n={is_pathogenic.sum():,}")

    # Build index for scored pairs
    scored_pairs = cg_pairs.dropna(subset=['max_path']).copy()
    scored_pairs['is_pathogenic'] = is_pathogenic

    # Also include pairs with NO score but in graph (treat as unknown, exclude)
    # Only keep pathogenic
    pathogenic_pairs = scored_pairs[scored_pairs['is_pathogenic']]

    cg_count = 0
    for _, row in pathogenic_pairs.iterrows():
        cell_gid = cell_to_gid.get(row[model_col])
        gene_gid = gene_to_gid.get(row[hugo_col])
        if cell_gid is not None and gene_gid is not None:
            weight = min(1.0, row['max_path'])
            all_edges.append(f"{cell_gid}\t{gene_gid}\t2\t{weight:.4f}")
            cg_count += 1
    edge_counts['Cell-Gene'] = cg_count
    print(f"  Cell-Gene edges: {cg_count:,} (GMM pathogenic only)")

    # --- Edge Type 3: Drug-Gene (DGIdb, typed inhibitory + score threshold) ---
    print(f"\n--- Edge Type 3: Drug-Gene (DGIdb, inhibitory + score>={args.dgi_min_score}) ---")
    dgi = pd.read_csv(f'{MISC}/DGIdb_Interactions_Enriched_v2.csv')

    # Only typed inhibitory interactions
    inhibitory_types = {'inhibitor', 'antagonist', 'blocker', 'negative modulator', 'inverse agonist'}

    dgi_filtered = dgi[
        dgi['gene_name'].isin(graph_gene_set) &
        dgi['drug_name'].str.upper().isin(set(drug_to_gid.keys())) &
        dgi['interaction_type'].str.lower().isin(inhibitory_types) &
        (dgi['interaction_score'] >= args.dgi_min_score)
    ].copy()

    n_typed_inhibitory = len(dgi[dgi['interaction_type'].str.lower().isin(inhibitory_types)])
    n_after_score = len(dgi_filtered)
    print(f"  Total DGIdb: {len(dgi):,}")
    print(f"  Typed inhibitory: {n_typed_inhibitory:,}")
    print(f"  After score >= {args.dgi_min_score} + graph filter: {n_after_score:,}")

    # Deduplicate per (drug, gene) — take max score
    dgi_filtered['drug_upper'] = dgi_filtered['drug_name'].str.upper()
    dg_pairs = dgi_filtered.groupby(['drug_upper', 'gene_name']).agg(
        score=('interaction_score', 'max')
    ).reset_index()

    # Normalize scores to [0, 1]
    if len(dg_pairs) > 0 and dg_pairs['score'].max() > 0:
        log_scores = np.log1p(dg_pairs['score'].values)
        min_s, max_s = log_scores.min(), log_scores.max()
        if max_s > min_s:
            dg_pairs['weight'] = (log_scores - min_s) / (max_s - min_s)
        else:
            dg_pairs['weight'] = 1.0
    else:
        dg_pairs['weight'] = 1.0

    dg_count = 0
    for _, row in dg_pairs.iterrows():
        drug_gid = drug_to_gid.get(row['drug_upper'])
        gene_gid = gene_to_gid.get(row['gene_name'])
        if drug_gid is not None and gene_gid is not None:
            all_edges.append(f"{drug_gid}\t{gene_gid}\t3\t{row['weight']:.4f}")
            dg_count += 1
    edge_counts['Drug-Gene'] = dg_count
    print(f"  Drug-Gene edges: {dg_count:,} (typed inhibitory, score >= {args.dgi_min_score})")

    # --- Edge Type 4: Cell-Cell (KNN Cosine Similarity) ---
    print(f"\n--- Edge Type 4: Cell-Cell (KNN, K={args.cell_knn}) ---")

    expr = pd.read_csv(f'{MISC}/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv',
                       index_col=0)
    graph_cell_ids = sorted(cell_to_gid.keys())
    expr_graph = expr.loc[expr.index.isin(graph_cell_ids)]
    expr_ordered = expr_graph.reindex([c for c in graph_cell_ids if c in expr_graph.index])

    print(f"  Computing cosine similarity for {len(expr_ordered)} cells...")
    cos_sim = cosine_similarity(expr_ordered.values)
    np.fill_diagonal(cos_sim, -2.0)  # exclude self-loops

    cell_ids_ordered = list(expr_ordered.index)
    cc_count = 0
    edge_set = set()

    K = args.cell_knn
    for i in range(len(cell_ids_ordered)):
        scores = cos_sim[i]
        top_indices = np.argpartition(scores, -K)[-K:]
        for j in top_indices:
            if i == j or scores[j] <= 0:
                continue
            gid_i = cell_to_gid[cell_ids_ordered[i]]
            gid_j = cell_to_gid[cell_ids_ordered[j]]
            sim = float(scores[j])
            # Bidirectional, dedup
            pair = (min(gid_i, gid_j), max(gid_i, gid_j))
            if pair not in edge_set:
                edge_set.add(pair)
                all_edges.append(f"{gid_i}\t{gid_j}\t4\t{sim:.4f}")
                all_edges.append(f"{gid_j}\t{gid_i}\t4\t{sim:.4f}")
                cc_count += 2

    edge_counts['Cell-Cell'] = cc_count
    avg_neighbors = cc_count / len(cell_ids_ordered) if cell_ids_ordered else 0
    print(f"  Cell-Cell edges: {cc_count:,} (bidirectional)")
    print(f"  Avg neighbors per cell: {avg_neighbors:.1f}")

    # Also build triplet map for triplet loss
    print(f"  Building triplet map...")
    np.fill_diagonal(cos_sim, -2.0)
    triplet_map = {}
    NUM_POS, NUM_NEG = 3, 5

    for i in range(len(cell_ids_ordered)):
        scores = cos_sim[i]
        pos_idx = np.argpartition(scores, -NUM_POS)[-NUM_POS:]
        scores_neg = scores.copy()
        scores_neg[scores_neg < -1.5] = 2.0
        neg_idx = np.argpartition(scores_neg, NUM_NEG)[:NUM_NEG]

        pos_names = [cell_ids_ordered[j] for j in pos_idx if j != i]
        neg_names = [cell_ids_ordered[j] for j in neg_idx if j != i]

        if pos_names and neg_names:
            triplet_map[cell_ids_ordered[i]] = {'pos': pos_names, 'neg': neg_names}

    triplet_path = f'{PROC_V2}/cell_drug_triplet_map.pkl'
    with open(triplet_path, 'wb') as f:
        pickle.dump({'triplet_map': triplet_map}, f)
    print(f"  Triplet map: {len(triplet_map)} cells -> {triplet_path}")

    # ==========================================
    # WRITE link.dat
    # ==========================================
    print(f"\n--- Writing link.dat ---")
    link_path = f'{PROC_V2}/link.dat'
    with open(link_path, 'w') as f:
        for edge in all_edges:
            f.write(edge + '\n')
    total_edges = len(all_edges)
    print(f"  Saved: {link_path} ({total_edges:,} edges)")

    # ==========================================
    # WRITE info.dat
    # ==========================================
    print("\n--- Writing info.dat ---")
    info = {
        'dataset': 'PRELUDE_HetGNN_v2',
        'node.dat': {
            '0': ['cell', len(cell_to_gid)],
            '1': ['drug', len(drug_to_gid)],
            '2': ['gene', len(gene_to_gid)],
        },
        'link.dat': {
            '0': ['cell', 'drug', 'cell-drug_interaction', edge_counts['Cell-Drug']],
            '1': ['gene', 'gene', 'gene-interacts_gene', edge_counts['Gene-Gene']],
            '2': ['cell', 'gene', 'cell-gene_mutation', edge_counts['Cell-Gene']],
            '3': ['gene', 'drug', 'drug-gene_inhibition', edge_counts['Drug-Gene']],
            '4': ['cell', 'cell', 'cell-cell_similarity', edge_counts['Cell-Cell']],
        },
    }

    info_path = f'{PROC_V2}/info.dat'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {info_path}")

    # Save ID mappings
    mappings = {
        'cell_to_gid': cell_to_gid,
        'drug_to_gid': drug_to_gid,
        'gene_to_gid': gene_to_gid,
    }
    with open(f'{PROC_V2}/id_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print(f"GRAPH BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Nodes: {total_nodes:,}")
    print(f"    Cells: {len(cell_to_gid):,}")
    print(f"    Drugs: {len(drug_to_gid):,}")
    print(f"    Genes: {len(gene_to_gid):,}")
    print(f"\n  Edges: {total_edges:,}")
    for etype, count in edge_counts.items():
        print(f"    {etype}: {count:,}")
    print(f"\n  Edge Quality:")
    print(f"    Cell-Drug: GMM ratio-matched labels (77% concordance with Sanger)")
    print(f"    Cell-Gene: 3-GMM pathogenic posterior >= {args.pathogenicity_gmm_posterior}")
    print(f"    Drug-Gene: Typed inhibitory only, score >= {args.dgi_min_score}")
    print(f"    Cell-Cell: KNN (K={args.cell_knn})")
    print(f"    Gene-Gene: Binary PPI (unfiltered)")


if __name__ == '__main__':
    main()
