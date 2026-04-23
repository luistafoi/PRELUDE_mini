"""Path-based interpretability for drug-cell predictions.

Enumerates 2-hop and 3-hop paths through the heterogeneous graph that support
a predicted (cell, drug) interaction, scores them by product of edge weights,
and exports the top-k paths per pair for literature review.

Node types:       0=cell, 1=drug, 2=gene
Edge types:
    0: cell-drug  (interaction label 0/0.5/1)
    1: gene-gene  (interaction score)
    2: cell-gene  (mutation pathogenicity)
    3: gene-drug  (inhibition strength)
    4: cell-cell  (expression similarity)
    5: drug-drug  (Tanimoto similarity)

Supporting path templates (for a target pair C→D):
    2-hop:
        C -[mut]-  G -[inhib]-  D
        C -[sim]-  C' -[sens]-  D
        C -[sens]- D' -[sim]-   D
    3-hop:
        C -[mut]- G -[gg]-   G' -[inhib]- D
        C -[mut]- G -[inhib]- D' -[sim]-  D
        C -[sim]- C' -[mut]- G  -[inhib]- D
        C -[sim]- C' -[sens]- D' -[sim]-  D

Usage:
    python scripts/analyze_paths.py \\
        --model_name v2_M03_drug_sim \\
        --data_dir data/processed_v2 \\
        --predictions_dir results/v2_M03_drug_sim/predictions \\
        --top_k_paths 15 \\
        --cells_per_drug 5
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


# Default curated drug list for path analysis.
# Top performers (high predicted discrimination in TCGA/Sanger):
#   Tamoxifen, Gemcitabine, Oxaliplatin, Dacarbazine, Etoposide
# Marginal / mechanistically interesting (for literature review):
#   Irinotecan, Temozolomide, Paclitaxel, Cisplatin, Docetaxel
DEFAULT_DRUGS = [
    "TAMOXIFEN", "GEMCITABINE", "OXALIPLATIN", "DACARBAZINE", "ETOPOSIDE",
    "IRINOTECAN", "TEMOZOLOMIDE", "PACLITAXEL", "CISPLATIN", "DOCETAXEL",
]


def load_graph(data_dir):
    """Load node and link data into adjacency structures."""
    id2name = {}
    name2id = {}
    id2type = {}
    nodes_by_type = defaultdict(list)

    with open(os.path.join(data_dir, "node.dat")) as f:
        for line in f:
            nid, name, ntype = line.rstrip("\n").split("\t")
            nid = int(nid); ntype = int(ntype)
            id2name[nid] = name
            name2id[name.upper()] = nid
            id2type[nid] = ntype
            nodes_by_type[ntype].append(nid)

    # Adjacency: edges[etype][u] = list of (v, weight)
    edges = {e: defaultdict(list) for e in range(6)}

    with open(os.path.join(data_dir, "link.dat")) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            u, v, et, w = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            edges[et][u].append((v, w))

    return id2name, name2id, id2type, nodes_by_type, edges


def find_drug_ids(name2id, id2type, drug_names):
    """Resolve drug names to global IDs with fuzzy matching."""
    resolved = {}
    all_drug_names = {n: i for n, i in name2id.items() if id2type[i] == 1}

    for wanted in drug_names:
        w = wanted.upper()
        if w in all_drug_names:
            resolved[wanted] = all_drug_names[w]
            continue
        # substring match
        candidates = [n for n in all_drug_names if w in n]
        if candidates:
            best = min(candidates, key=len)
            resolved[wanted] = all_drug_names[best]
            print(f"  [fuzzy] {wanted} -> {best}")
        else:
            print(f"  [miss]  {wanted} not found")
    return resolved


def enumerate_paths(cell_id, drug_id, edges, id2name, id2type, max_branch=50):
    """Enumerate supporting 2-hop and 3-hop paths from cell -> drug.

    Returns list of dicts with pattern, nodes, edge_types, weights, score.
    """
    paths = []

    def edge_name(et):
        return {0: "sens", 1: "gg", 2: "mut", 3: "inhib", 4: "csim", 5: "dsim"}[et]

    # ------ 2-hop ------
    # C -mut- G -inhib- D
    for g, w1 in edges[2].get(cell_id, []):
        for d2, w2 in edges[3].get(g, []):
            if d2 == drug_id:
                paths.append({
                    "pattern": "C-mut-G-inhib-D",
                    "nodes": [cell_id, g, drug_id],
                    "node_names": [id2name[cell_id], id2name[g], id2name[drug_id]],
                    "edge_types": ["mut", "inhib"],
                    "weights": [w1, w2],
                    "score": w1 * w2,
                })

    # C -csim- C' -sens- D  (only count sensitive labels > 0.5)
    for c2, w1 in edges[4].get(cell_id, [])[:max_branch]:
        for d2, w2 in edges[0].get(c2, []):
            if d2 == drug_id and w2 > 0.5:
                paths.append({
                    "pattern": "C-sim-C'-sens-D",
                    "nodes": [cell_id, c2, drug_id],
                    "node_names": [id2name[cell_id], id2name[c2], id2name[drug_id]],
                    "edge_types": ["csim", "sens"],
                    "weights": [w1, w2],
                    "score": w1 * w2,
                })

    # C -sens- D' -dsim- D (only from known sensitive drug)
    for d_prime, w1 in edges[0].get(cell_id, []):
        if w1 <= 0.5 or d_prime == drug_id or id2type.get(d_prime) != 1:
            continue
        for d2, w2 in edges[5].get(d_prime, []):
            if d2 == drug_id:
                paths.append({
                    "pattern": "C-sens-D'-sim-D",
                    "nodes": [cell_id, d_prime, drug_id],
                    "node_names": [id2name[cell_id], id2name[d_prime], id2name[drug_id]],
                    "edge_types": ["sens", "dsim"],
                    "weights": [w1, w2],
                    "score": w1 * w2,
                })

    # ------ 3-hop ------
    # C -mut- G -gg- G' -inhib- D
    seen_gg = set()
    for g, w1 in edges[2].get(cell_id, []):
        for g2, w2 in edges[1].get(g, [])[:max_branch]:
            if (g, g2) in seen_gg:
                continue
            seen_gg.add((g, g2))
            for d2, w3 in edges[3].get(g2, []):
                if d2 == drug_id:
                    paths.append({
                        "pattern": "C-mut-G-gg-G'-inhib-D",
                        "nodes": [cell_id, g, g2, drug_id],
                        "node_names": [id2name[cell_id], id2name[g], id2name[g2], id2name[drug_id]],
                        "edge_types": ["mut", "gg", "inhib"],
                        "weights": [w1, w2, w3],
                        "score": w1 * w2 * w3,
                    })

    # C -mut- G -inhib- D' -dsim- D
    for g, w1 in edges[2].get(cell_id, []):
        for d_prime, w2 in edges[3].get(g, []):
            if d_prime == drug_id:
                continue
            for d2, w3 in edges[5].get(d_prime, []):
                if d2 == drug_id:
                    paths.append({
                        "pattern": "C-mut-G-inhib-D'-sim-D",
                        "nodes": [cell_id, g, d_prime, drug_id],
                        "node_names": [id2name[cell_id], id2name[g], id2name[d_prime], id2name[drug_id]],
                        "edge_types": ["mut", "inhib", "dsim"],
                        "weights": [w1, w2, w3],
                        "score": w1 * w2 * w3,
                    })

    # C -csim- C' -mut- G -inhib- D
    for c2, w1 in edges[4].get(cell_id, [])[:max_branch]:
        for g, w2 in edges[2].get(c2, [])[:max_branch]:
            for d2, w3 in edges[3].get(g, []):
                if d2 == drug_id:
                    paths.append({
                        "pattern": "C-sim-C'-mut-G-inhib-D",
                        "nodes": [cell_id, c2, g, drug_id],
                        "node_names": [id2name[cell_id], id2name[c2], id2name[g], id2name[drug_id]],
                        "edge_types": ["csim", "mut", "inhib"],
                        "weights": [w1, w2, w3],
                        "score": w1 * w2 * w3,
                    })

    return paths


def summarize_mechanism(paths, top_k=5):
    """Collapse top paths into a short human-readable mechanism summary."""
    if not paths:
        return "No supporting paths found."
    # Collect genes that appear in top-k supporting paths
    genes = []
    similar_drugs = []
    similar_cells = []
    for p in paths[:top_k]:
        for nid, nname, ntype_hint in zip(p["nodes"], p["node_names"], p["edge_types"] + [None]):
            pass
        # simpler: walk node_names with known positions
        if "G" in p["pattern"]:
            # gene nodes are middle entries (index 1 or 2)
            for i, et in enumerate(p["edge_types"]):
                if et in ("mut", "inhib", "gg"):
                    if i + 1 < len(p["node_names"]):
                        name = p["node_names"][i + 1]
                        # Heuristic: gene names are uppercase short
                        if name.isupper() and len(name) < 12 and not name.startswith("ACH-"):
                            genes.append(name)
        if "D'" in p["pattern"]:
            # similar drug is index 1 or 2
            similar_drugs.append(p["node_names"][-2])
        if "C'" in p["pattern"]:
            similar_cells.append(p["node_names"][1])

    parts = []
    if genes:
        top_genes = list(dict.fromkeys(genes))[:5]
        parts.append(f"targets genes: {', '.join(top_genes)}")
    if similar_drugs:
        top_sd = list(dict.fromkeys(similar_drugs))[:3]
        parts.append(f"similar to: {', '.join(top_sd)}")
    if similar_cells:
        top_sc = list(dict.fromkeys(similar_cells))[:3]
        parts.append(f"analog cells: {', '.join(top_sc)}")
    return "; ".join(parts) if parts else "Weak indirect support only."


def load_predictions(predictions_dir):
    """Load test_inductive + Sanger prediction CSVs if present."""
    dfs = {}
    for name in ["test_inductive_predictions.csv", "sanger_S3_predictions.csv",
                 "sanger_S4_predictions.csv"]:
        path = os.path.join(predictions_dir, name)
        if os.path.exists(path):
            dfs[name.replace("_predictions.csv", "")] = pd.read_csv(path)
    return dfs


def pick_cells_for_drug(pred_df, drug_name, n_high=5, n_marginal=3):
    """Pick top-N most-confident sensitive cells + a few marginal (score ~0.5)."""
    sub = pred_df[pred_df["drug"].str.upper() == drug_name.upper()]
    if len(sub) == 0:
        return []
    sub = sub.sort_values("pred_score", ascending=False)
    high = sub.head(n_high)
    # marginal = scores closest to 0.5
    marg = sub.iloc[(sub["pred_score"] - 0.5).abs().argsort()].head(n_marginal)
    picks = []
    for _, r in high.iterrows():
        picks.append((int(r["cell_gid"]), r["cell"], float(r["pred_score"]),
                      int(r["true_label"]), "high_confidence"))
    for _, r in marg.iterrows():
        picks.append((int(r["cell_gid"]), r["cell"], float(r["pred_score"]),
                      int(r["true_label"]), "marginal"))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="v2_M03_drug_sim")
    ap.add_argument("--data_dir", type=str, default="data/processed_v2")
    ap.add_argument("--predictions_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--drugs", type=str, nargs="+", default=None,
                    help="Override drug list (case-insensitive names)")
    ap.add_argument("--top_k_paths", type=int, default=15)
    ap.add_argument("--cells_per_drug", type=int, default=5)
    ap.add_argument("--marginal_cells", type=int, default=3)
    args = ap.parse_args()

    if args.predictions_dir is None:
        args.predictions_dir = f"results/{args.model_name}/predictions"
    if args.output_dir is None:
        args.output_dir = f"results/{args.model_name}/interpretability"
    os.makedirs(args.output_dir, exist_ok=True)

    drug_names = args.drugs if args.drugs else DEFAULT_DRUGS

    print("=" * 60)
    print(f"PATH ANALYSIS: {args.model_name}")
    print("=" * 60)

    print("\n--- Loading graph ---")
    id2name, name2id, id2type, nodes_by_type, edges = load_graph(args.data_dir)
    print(f"  Cells: {len(nodes_by_type[0])}, Drugs: {len(nodes_by_type[1])}, "
          f"Genes: {len(nodes_by_type[2])}")

    print("\n--- Resolving drug names ---")
    drug_ids = find_drug_ids(name2id, id2type, drug_names)
    print(f"  Resolved {len(drug_ids)}/{len(drug_names)}")

    print("\n--- Loading predictions ---")
    pred_dfs = load_predictions(args.predictions_dir)
    if not pred_dfs:
        print(f"  [warn] No predictions found in {args.predictions_dir}")
        print("         Run scripts/export_predictions.py first to generate them.")
        return
    for k, v in pred_dfs.items():
        print(f"  {k}: {len(v)} pairs")

    # Use test_inductive as the primary source; fall back to S3/S4 or the first available.
    if "test_inductive" in pred_dfs:
        primary = pred_dfs["test_inductive"]
    elif "sanger_S3" in pred_dfs:
        primary = pred_dfs["sanger_S3"]
    else:
        primary = next(iter(pred_dfs.values()))

    all_records = []
    mechanism_rows = []

    for drug_name, drug_gid in drug_ids.items():
        print(f"\n--- {drug_name} (gid={drug_gid}) ---")
        picks = pick_cells_for_drug(primary, drug_name,
                                     n_high=args.cells_per_drug,
                                     n_marginal=args.marginal_cells)
        if not picks:
            print(f"  No predictions found for {drug_name}")
            continue
        print(f"  {len(picks)} cells selected")

        for cell_gid, cell_name, score, label, category in picks:
            paths = enumerate_paths(cell_gid, drug_gid, edges, id2name, id2type)
            paths.sort(key=lambda p: p["score"], reverse=True)
            top = paths[:args.top_k_paths]

            for rank, p in enumerate(top):
                all_records.append({
                    "drug": drug_name,
                    "drug_gid": drug_gid,
                    "cell": cell_name,
                    "cell_gid": cell_gid,
                    "pred_score": score,
                    "true_label": label,
                    "category": category,
                    "path_rank": rank + 1,
                    "pattern": p["pattern"],
                    "path": " -> ".join(p["node_names"]),
                    "edge_types": "|".join(p["edge_types"]),
                    "weights": "|".join(f"{w:.4f}" for w in p["weights"]),
                    "path_score": p["score"],
                    "n_hops": len(p["edge_types"]),
                })

            mechanism_rows.append({
                "drug": drug_name,
                "cell": cell_name,
                "pred_score": score,
                "true_label": label,
                "category": category,
                "n_paths_found": len(paths),
                "top_path_score": top[0]["score"] if top else 0.0,
                "mechanism_summary": summarize_mechanism(top, top_k=5),
            })
            tag = "HIGH" if category == "high_confidence" else "MARG"
            print(f"  [{tag}] {cell_name}  score={score:.3f}  paths={len(paths)}")

    # Save outputs
    paths_df = pd.DataFrame(all_records)
    mech_df = pd.DataFrame(mechanism_rows)

    paths_path = os.path.join(args.output_dir, "top_paths.csv")
    mech_path = os.path.join(args.output_dir, "mechanism_summaries.csv")
    json_path = os.path.join(args.output_dir, "top_paths.json")

    paths_df.to_csv(paths_path, index=False)
    mech_df.to_csv(mech_path, index=False)

    # Grouped JSON for literature review
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        grouped[rec["drug"]][rec["cell"]].append({
            "rank": rec["path_rank"],
            "pattern": rec["pattern"],
            "path": rec["path"],
            "score": rec["path_score"],
            "weights": rec["weights"],
        })
    with open(json_path, "w") as f:
        json.dump({drug: dict(cells) for drug, cells in grouped.items()},
                  f, indent=2)

    print("\n--- Complete ---")
    print(f"  {paths_path}  ({len(paths_df)} paths)")
    print(f"  {mech_path}   ({len(mech_df)} summaries)")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
