# scripts/precompute_cell_similarity.py
#
# Computes cell-cell cosine similarity from gene expression and outputs:
# 1. Cell-Cell edge file (top K% most similar, bidirectional) for graph structure
# 2. Triplet map (pos/neg per cell) for triplet loss
#
# This script is INDEPENDENT of the graph — it only needs the raw expression
# file and a list of cell names with embeddings. It does NOT depend on
# PRELUDEDataset or any processed graph files.

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Precompute cell-cell similarity edges and triplet map.")
    parser.add_argument('--misc-dir', default='data/misc',
                        help='Directory with source data (expression CSV).')
    parser.add_argument('--emb-dir', default='data/embeddings',
                        help='Directory with cell embedding names.')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Output directory for Cell-Cell edge file.')
    parser.add_argument('--processed-dir', default='data/processed',
                        help='Output directory for triplet map pkl.')
    parser.add_argument('--top-pct', type=float, default=0.01,
                        help='Top percentage of most similar cells to keep as edges (default 0.01 = 1%%).')
    parser.add_argument('--num-pos', type=int, default=1,
                        help='Number of positive (most similar) cells per anchor for triplet loss.')
    parser.add_argument('--num-neg', type=int, default=5,
                        help='Number of negative (least similar) cells per anchor for triplet loss.')
    args = parser.parse_args()

    expr_file = os.path.join(
        args.misc_dir,
        "24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
    cell_names_file = os.path.join(args.emb_dir, "final_vae_cell_names.txt")
    edge_output = os.path.join(args.raw_dir, "link_cell_cell_similarity.txt")
    triplet_output = os.path.join(args.processed_dir, "cell_triplet_map.pkl")

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load cells that have embeddings (= cells in our graph)
    # ------------------------------------------------------------------
    print("--- Precomputing Cell-Cell Similarity ---")

    if not os.path.exists(cell_names_file):
        print(f"FATAL: Cell names file not found: {cell_names_file}")
        return
    with open(cell_names_file) as f:
        emb_cells = set(line.strip().upper() for line in f if line.strip())
    print(f"  > Cells with embeddings: {len(emb_cells)}")

    # ------------------------------------------------------------------
    # 2. Load gene expression for those cells
    # ------------------------------------------------------------------
    if not os.path.exists(expr_file):
        print(f"FATAL: Expression file not found: {expr_file}")
        return

    print(f"  > Loading expression data from: {expr_file}")
    df_expr = pd.read_csv(expr_file, index_col=0)
    df_expr.index = df_expr.index.str.strip().str.upper()

    # Filter to cells that have embeddings
    common_cells = sorted(emb_cells & set(df_expr.index))
    df_expr = df_expr.loc[common_cells]
    print(f"  > Cells with expression + embeddings: {len(df_expr)}")

    cell_names = df_expr.index.tolist()
    n_cells = len(cell_names)

    # ------------------------------------------------------------------
    # 3. Compute pairwise cosine similarity
    # ------------------------------------------------------------------
    print(f"  > Computing {n_cells}x{n_cells} cosine similarity matrix...")
    sim_matrix = cosine_similarity(df_expr.values)
    np.fill_diagonal(sim_matrix, -2.0)  # exclude self-loops

    # ------------------------------------------------------------------
    # 4. Build Cell-Cell edges (top K% per cell, bidirectional)
    # ------------------------------------------------------------------
    top_k = max(1, int(n_cells * args.top_pct))
    print(f"  > Top {args.top_pct*100:.1f}% = top {top_k} neighbors per cell")

    edge_set = set()  # (cell_a, cell_b, sim_score) as strings for dedup

    for i in tqdm(range(n_cells), desc="  - Building edges"):
        scores = sim_matrix[i]
        # Get indices of top-k most similar cells
        top_indices = np.argpartition(scores, -top_k)[-top_k:]

        for j in top_indices:
            if i == j:
                continue
            sim_score = float(scores[j])
            # Bidirectional: add both directions
            a, b = cell_names[i], cell_names[j]
            edge_set.add((a, b, round(sim_score, 6)))
            edge_set.add((b, a, round(sim_score, 6)))

    # Write edge file
    with open(edge_output, 'w') as f:
        for a, b, w in sorted(edge_set):
            f.write(f"{a}\t{b}\t{w}\n")

    # Count unique cells in edges
    edge_cells = set()
    for a, b, _ in edge_set:
        edge_cells.add(a)
        edge_cells.add(b)

    print(f"  > Cell-Cell edges: {len(edge_set)} (bidirectional)")
    print(f"  > Unique cells in edges: {len(edge_cells)}")
    print(f"  > Avg neighbors per cell: {len(edge_set) / len(edge_cells):.1f}")
    print(f"  > Saved to: {edge_output}")

    # ------------------------------------------------------------------
    # 5. Build triplet map (for triplet loss)
    # ------------------------------------------------------------------
    print(f"  > Building triplet map (pos={args.num_pos}, neg={args.num_neg})...")

    # Reset diagonal for triplet selection
    np.fill_diagonal(sim_matrix, -2.0)

    # Triplet map: cell_name -> {pos: [names], neg: [names]}
    triplet_map = {}

    for i in tqdm(range(n_cells), desc="  - Building triplet map"):
        scores = sim_matrix[i]

        # Most similar = positives
        pos_indices = np.argpartition(scores, -args.num_pos)[-args.num_pos:]
        # Least similar = negatives (excluding the diagonal sentinel)
        # Reset sentinel temporarily for negative selection
        scores_for_neg = scores.copy()
        scores_for_neg[scores_for_neg < -1.5] = 2.0  # push sentinels high
        neg_indices = np.argpartition(scores_for_neg, args.num_neg)[:args.num_neg]

        pos_names = [cell_names[j] for j in pos_indices if j != i]
        neg_names = [cell_names[j] for j in neg_indices if j != i]

        if pos_names and neg_names:
            triplet_map[cell_names[i]] = {
                'pos': pos_names,
                'neg': neg_names,
            }

    with open(triplet_output, 'wb') as f:
        pickle.dump(triplet_map, f)

    print(f"  > Triplet map: {len(triplet_map)} cells")
    print(f"  > Saved to: {triplet_output}")
    print("Done.")


if __name__ == "__main__":
    main()
