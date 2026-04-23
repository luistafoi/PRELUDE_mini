# scripts/prep_sanger_graph.py
#
# Builds data/sanger_processed/ — a copy of the training graph augmented with
# Sanger S3/S4 (inductive) cells AND S2/S4 (new) drugs.
# S1 (transductive) cells are already in the graph.
#
# Outputs:
#   data/sanger_processed/
#     node.dat, link.dat, train.dat, info.dat, node_mappings.json
#     cell_triplet_map.pkl (copy)
#     train_lp_links.dat, valid_*_links.dat, test_*_links.dat (copies)
#     sanger_S1_links.dat, sanger_S2_links.dat, sanger_S3_links.dat, sanger_S4_links.dat
#     embeddings/  (augmented cell .npy/.txt, augmented drug .csv, symlink for gene)

import sys
import os
import argparse
import json
import shutil
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from scipy.optimize import root_scalar
from tqdm import tqdm

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# GMM Relabeling (mirrors label_links_with_gmm.py exactly)
# ---------------------------------------------------------------------------

def _find_intersection(gmm, comp1_idx, comp2_idx, bounds):
    """Finds the point where two Gaussian components have equal posterior probability."""
    def diff_func(x):
        probs = gmm.predict_proba(np.array([[x]])).ravel()
        if comp1_idx >= len(probs) or comp2_idx >= len(probs):
            return 0
        return probs[comp1_idx] - probs[comp2_idx]

    try:
        f_lower = diff_func(bounds[0])
        f_upper = diff_func(bounds[1])
        if np.sign(f_lower) == np.sign(f_upper):
            return np.mean(bounds)
        result = root_scalar(diff_func, bracket=bounds, method='brentq')
        return result.root
    except (ValueError, RuntimeError):
        return np.mean(bounds)


def _fit_gmm_and_find_thresholds(scores: np.ndarray):
    """Fits GMMs with 1-3 components, selects best by BIC, finds thresholds."""
    X = scores.reshape(-1, 1)
    if X.shape[0] < 3:
        return [], GaussianMixture(n_components=1).fit(X)

    best_gmm = None
    min_bic = np.inf

    for n in range(1, 4):
        if X.shape[0] < n:
            continue
        try:
            gmm = GaussianMixture(n_components=n, random_state=0, n_init=5).fit(X)
            bic = gmm.bic(X)
            if bic < min_bic:
                min_bic = bic
                best_gmm = gmm
        except ValueError:
            continue

    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=1).fit(X)

    n_components = best_gmm.n_components
    if n_components == 1:
        return [], best_gmm

    order = np.argsort(best_gmm.means_.ravel())
    sorted_means = best_gmm.means_[order]

    thresholds = []
    for i in range(n_components - 1):
        comp1_original_idx = order[i]
        comp2_original_idx = order[i + 1]
        lower_bound = sorted_means[i][0]
        upper_bound = sorted_means[i + 1][0]
        if np.isclose(lower_bound, upper_bound):
            threshold = lower_bound
        else:
            padding = (upper_bound - lower_bound) * 0.1
            threshold = _find_intersection(
                best_gmm, comp1_original_idx, comp2_original_idx,
                (lower_bound - padding, upper_bound + padding))
        thresholds.append(threshold)

    return sorted(thresholds), best_gmm


def _classify_score(score, thresholds, n_components):
    """Classifies a score: lower = Positive (sensitive), higher = Negative."""
    if n_components == 1:
        return "Uncertain"
    elif n_components == 2:
        return "Positive" if score <= thresholds[0] else "Negative"
    else:  # 3 components
        if score <= thresholds[0]:
            return "Positive"
        elif score <= thresholds[1]:
            return "Uncertain"
        else:
            return "Negative"


def relabel_sanger_gmm(df, value_col='IC50', drug_col='clean_drug', min_points=10):
    """Relabel Sanger data using per-drug GMM on IC50 (same method as DepMap LMFI).

    Lower IC50 = more sensitive = Positive (label=1), matching the DepMap convention
    where lower LMFI = more sensitive = Positive (label=1).

    Returns DataFrame with updated 'label' column and rows with 'Uncertain' dropped.
    """
    print("\n--- Relabeling Sanger data with per-drug GMM ---")
    all_labeled = []

    for drug_name, group in df.groupby(drug_col):
        labeled_group = group.copy()
        scores = group[value_col].dropna().values

        if len(scores) < min_points:
            labeled_group['gmm_class'] = "Uncertain"
            print(f"  {drug_name:25s}  n={len(scores):4d}  -> Uncertain (too few points)")
        else:
            thresholds, gmm = _fit_gmm_and_find_thresholds(scores)
            labeled_group['gmm_class'] = labeled_group[value_col].apply(
                lambda x: _classify_score(x, thresholds, gmm.n_components))

            n_pos = (labeled_group['gmm_class'] == 'Positive').sum()
            n_neg = (labeled_group['gmm_class'] == 'Negative').sum()
            n_unc = (labeled_group['gmm_class'] == 'Uncertain').sum()
            old_pos = labeled_group['label'].sum()
            thr_str = ', '.join(f'{t:.2f}' for t in thresholds) if thresholds else 'none'
            print(f"  {drug_name:25s}  n={len(scores):4d}  "
                  f"GMM: pos={n_pos} neg={n_neg} unc={n_unc}  "
                  f"(old pos={old_pos})  thresh=[{thr_str}]  k={gmm.n_components}")

        all_labeled.append(labeled_group)

    df_labeled = pd.concat(all_labeled)

    # Drop uncertain, assign binary label
    df_certain = df_labeled[df_labeled['gmm_class'] != 'Uncertain'].copy()
    df_certain['label'] = (df_certain['gmm_class'] == 'Positive').astype(int)

    n_before = len(df)
    n_after = len(df_certain)
    n_pos = df_certain['label'].sum()
    print(f"\n  Summary: {n_before} -> {n_after} pairs ({n_before - n_after} uncertain dropped)")
    print(f"  New balance: {n_pos}/{n_after} positive ({n_pos/n_after:.1%})")

    return df_certain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_cell_name(name: str) -> str:
    """Strip all non-alphanumeric, uppercase — same logic as prep_sanger_subsets."""
    return re.sub(r'[^A-Z0-9]', '', str(name).upper().strip())


def build_stripped_to_ach(model_csv: str):
    """Build StrippedCellLineName (uppercase) -> ACH ModelID from Model.csv."""
    df = pd.read_csv(model_csv, usecols=['ModelID', 'StrippedCellLineName'])
    df['key'] = df['StrippedCellLineName'].str.strip().str.upper()
    # Keep first ACH per stripped name (shouldn't have collisions)
    return dict(zip(df['key'], df['ModelID']))


def build_entrez_to_hugo(misc_dir: str) -> dict:
    """Entrez GeneID -> HUGO symbol (same logic as curate_raw_links.py)."""
    mapping = {}

    expr_file = os.path.join(misc_dir,
                             "24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
    if os.path.exists(expr_file):
        cols = pd.read_csv(expr_file, nrows=0).columns
        for col in cols[1:]:
            m = re.match(r'^(.*)\s\((\d+)\)$', col)
            if m:
                mapping[int(m.group(2))] = m.group(1).upper()

    mut_file = os.path.join(misc_dir, "OmicsSomaticMutations_24Q2.csv")
    if os.path.exists(mut_file):
        df = pd.read_csv(mut_file, usecols=['HugoSymbol', 'EntrezGeneID'],
                         low_memory=False)
        df = df.dropna(subset=['HugoSymbol', 'EntrezGeneID']).drop_duplicates(
            subset=['EntrezGeneID'])
        for _, row in df.iterrows():
            eid = int(row['EntrezGeneID'])
            hugo = str(row['HugoSymbol']).upper().strip()
            if eid not in mapping:
                mapping[eid] = hugo

    return mapping


def score_mutations(df, entrez_to_hugo, graph_genes):
    """Apply identical pathogenicity scoring as curate_raw_links.py.

    Args:
        df: mutation DataFrame with standard DepMap columns
        entrez_to_hugo: Entrez -> HUGO mapping
        graph_genes: set of UPPERCASE gene symbols present in graph

    Returns:
        DataFrame with columns [ModelID, gene_name, weight]
    """
    df = df.copy()
    df.dropna(subset=['ModelID', 'EntrezGeneID'], inplace=True)
    df['EntrezGeneID'] = df['EntrezGeneID'].astype(int)

    # Map Entrez -> HUGO
    df['gene_name'] = df['EntrezGeneID'].map(entrez_to_hugo)
    df.dropna(subset=['gene_name'], inplace=True)
    df['gene_name'] = df['gene_name'].str.upper()

    # Filter to genes in graph
    df = df[df['gene_name'].isin(graph_genes)].copy()

    # --- Pathogenicity scoring (identical to curate_raw_links.py) ---
    df['weight'] = np.nan

    flag_weights = {
        'LikelyLoF': 1.0,
        'Hotspot': 0.9,
        'HessDriver': 0.9,
        'OncogeneHighImpact': 0.9,
        'TumorSuppressorHighImpact': 0.9,
    }
    for col, w in flag_weights.items():
        if col not in df.columns:
            continue
        mask = df[col] == True
        df.loc[mask, 'weight'] = df.loc[mask, 'weight'].fillna(0).clip(lower=0)
        df.loc[mask, 'weight'] = np.maximum(df.loc[mask, 'weight'], w)

    # REVEL score >= 0.5
    if 'RevelScore' in df.columns:
        revel_mask = df['RevelScore'].notna() & (df['RevelScore'] >= 0.5)
        df.loc[revel_mask, 'weight'] = np.fmax(
            df.loc[revel_mask, 'weight'].fillna(0), df.loc[revel_mask, 'RevelScore'])

    # AlphaMissense likely_pathogenic
    if 'AMClass' in df.columns and 'AMPathogenicity' in df.columns:
        am_mask = (df['AMClass'] == 'likely_pathogenic') & df['AMPathogenicity'].notna()
        df.loc[am_mask, 'weight'] = np.fmax(
            df.loc[am_mask, 'weight'].fillna(0), df.loc[am_mask, 'AMPathogenicity'])

    # Drop non-qualifying mutations
    df = df.dropna(subset=['weight'])

    # Aggregate per (cell, gene): max weight
    agg = df.groupby(['ModelID', 'gene_name'])['weight'].max().reset_index()
    agg['weight'] = agg['weight'].round(4)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build augmented graph for Sanger cross-dataset evaluation.")
    parser.add_argument('--output-dir', default='data/sanger_processed',
                        help='Output directory for augmented graph.')
    parser.add_argument('--source-dir', default='data/processed',
                        help='Source graph directory to augment.')
    parser.add_argument('--sanger-dir', default='data/sanger_validation',
                        help='Directory with Sanger CSV files.')
    parser.add_argument('--misc-dir', default='data/misc',
                        help='Directory with DepMap source files.')
    parser.add_argument('--emb-dir', default='data/embeddings',
                        help='Directory with original embedding files.')
    parser.add_argument('--top-pct', type=float, default=0.01,
                        help='Top %% of cells for Cell-Cell similarity edges (default 1%%).')
    args = parser.parse_args()

    OUT = args.output_dir
    SRC = args.source_dir
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, 'embeddings'), exist_ok=True)

    print("=" * 60)
    print("Sanger Graph Augmentation — prep_sanger_graph.py")
    print("=" * 60)

    # ==================================================================
    # STEP 1: Load existing graph structures
    # ==================================================================
    print("\n--- Step 1: Loading existing graph ---")

    # Load node.dat
    existing_nodes = {}  # name -> (gid, type)
    existing_gids = {}   # gid -> (name, type)
    with open(os.path.join(SRC, 'node.dat')) as f:
        for line in f:
            gid_s, name, ntype_s = line.strip().split('\t')
            gid, ntype = int(gid_s), int(ntype_s)
            existing_nodes[name] = (gid, ntype)
            existing_gids[gid] = (name, ntype)

    max_gid = max(existing_gids.keys())
    print(f"  > Existing nodes: {len(existing_nodes)} (max GID: {max_gid})")

    # Load node_mappings.json
    with open(os.path.join(SRC, 'node_mappings.json')) as f:
        node_mappings = json.load(f)

    # Identify graph cells/drugs/genes
    graph_cells = {name for name, (_, t) in existing_nodes.items() if t == 0}
    graph_drugs = {name.upper() for name, (_, t) in existing_nodes.items() if t == 1}
    graph_genes = {name.upper() for name, (_, t) in existing_nodes.items() if t == 2}

    print(f"  > Graph cells: {len(graph_cells)}, drugs: {len(graph_drugs)}, genes: {len(graph_genes)}")

    # ==================================================================
    # STEP 2: Map Sanger cells -> ACH IDs, filter drugs
    # ==================================================================
    print("\n--- Step 2: Mapping Sanger cells to ACH IDs ---")

    stripped_to_ach = build_stripped_to_ach(os.path.join(args.misc_dir, 'Model.csv'))
    print(f"  > Model.csv mapping: {len(stripped_to_ach)} entries")

    # Load Sanger VAE embeddings (indexed by clean_cell name)
    sanger_vae = pd.read_csv(
        os.path.join(args.sanger_dir, 'sanger_vae_embeddings.csv'), index_col=0)
    sanger_vae.index = sanger_vae.index.astype(str)
    sanger_vae_names = set(sanger_vae.index)
    print(f"  > Sanger VAE embeddings: {len(sanger_vae_names)} cells")

    # --- Map S3 (new/inductive) cells ---
    s3_df = pd.read_csv(os.path.join(args.sanger_dir, 'sanger_S3_new_cell_known.csv'))
    s3_clean_cells = s3_df['clean_cell'].unique()

    s3_cell_map = {}  # clean_cell -> ACH
    s3_dropped = {'no_ach': [], 'no_vae': [], 'in_graph': []}

    for clean_cell in s3_clean_cells:
        norm = normalize_cell_name(clean_cell)
        ach = stripped_to_ach.get(norm)
        if ach is None:
            s3_dropped['no_ach'].append(clean_cell)
            continue
        if clean_cell not in sanger_vae_names:
            s3_dropped['no_vae'].append(clean_cell)
            continue
        if ach in graph_cells:
            # Already in graph — shouldn't happen for S3 but safety check
            s3_dropped['in_graph'].append(clean_cell)
            continue
        s3_cell_map[clean_cell] = ach

    print(f"  > S3 mapped: {len(s3_cell_map)} cells")
    for reason, cells in s3_dropped.items():
        if cells:
            print(f"    Dropped ({reason}): {len(cells)} — {cells[:3]}...")

    # --- Map S1 (known/transductive) cells ---
    s1_df = pd.read_csv(os.path.join(args.sanger_dir, 'sanger_S1_known_known.csv'))
    s1_clean_cells = s1_df['clean_cell'].unique()

    s1_cell_map = {}  # clean_cell -> ACH
    s1_dropped = {'no_ach': [], 'not_in_graph': []}

    for clean_cell in s1_clean_cells:
        norm = normalize_cell_name(clean_cell)
        ach = stripped_to_ach.get(norm)
        if ach is None:
            s1_dropped['no_ach'].append(clean_cell)
            continue
        if ach not in graph_cells:
            s1_dropped['not_in_graph'].append(clean_cell)
            continue
        s1_cell_map[clean_cell] = ach

    print(f"  > S1 mapped: {len(s1_cell_map)} cells")
    for reason, cells in s1_dropped.items():
        if cells:
            print(f"    Dropped ({reason}): {len(cells)} — {cells[:3]}...")

    # --- Load S2 (known cells, new drugs) ---
    s2_path = os.path.join(args.sanger_dir, 'sanger_S2_known_new_drug.csv')
    if os.path.exists(s2_path):
        s2_df = pd.read_csv(s2_path)
        print(f"\n  > S2 data: {len(s2_df)} rows, {s2_df['clean_drug'].nunique()} drugs, "
              f"{s2_df['clean_cell'].nunique()} cells")
    else:
        s2_df = pd.DataFrame(columns=['Cell Line Name', 'Drug Name', 'label',
                                       'clean_cell', 'clean_drug', 'IC50', 'AUC'])
        print("  > S2 data: not found, skipping")

    # --- Load S4 (new cells, new drugs) ---
    s4_path = os.path.join(args.sanger_dir, 'sanger_S4_new_new.csv')
    if os.path.exists(s4_path):
        s4_df = pd.read_csv(s4_path)
        print(f"  > S4 data: {len(s4_df)} rows, {s4_df['clean_drug'].nunique()} drugs, "
              f"{s4_df['clean_cell'].nunique()} cells")
    else:
        s4_df = pd.DataFrame(columns=['Cell Line Name', 'Drug Name', 'label',
                                       'clean_cell', 'clean_drug', 'IC50', 'AUC'])
        print("  > S4 data: not found, skipping")

    # --- Load Sanger drug embeddings (for new drugs in S2/S4) ---
    sanger_drug_emb_path = os.path.join(args.sanger_dir, 'sanger_drug_embeddings_final.csv')
    sanger_drug_emb = pd.read_csv(sanger_drug_emb_path)
    sanger_drug_emb_names = set(sanger_drug_emb.iloc[:, 0].str.upper())
    print(f"  > Sanger drug embeddings: {len(sanger_drug_emb_names)} drugs (256-dim)")

    # --- Map S2 cells (known, should be in graph — same logic as S1) ---
    s2_cell_map = {}
    if len(s2_df) > 0:
        for clean_cell in s2_df['clean_cell'].unique():
            norm = normalize_cell_name(clean_cell)
            ach = stripped_to_ach.get(norm)
            if ach is not None and ach in graph_cells:
                s2_cell_map[clean_cell] = ach
        print(f"  > S2 cells mapped (in graph): {len(s2_cell_map)} / {s2_df['clean_cell'].nunique()}")

    # --- Map S4 cells (new — reuse S3 cell mapping, same cells) ---
    s4_cell_map = {}
    if len(s4_df) > 0:
        for clean_cell in s4_df['clean_cell'].unique():
            norm = normalize_cell_name(clean_cell)
            ach = stripped_to_ach.get(norm)
            if ach is None:
                continue
            if clean_cell not in sanger_vae_names:
                continue
            if ach in graph_cells:
                continue  # Filter out cells already in graph
            s4_cell_map[clean_cell] = ach
        print(f"  > S4 cells mapped (new, with VAE): {len(s4_cell_map)} / {s4_df['clean_cell'].nunique()}")

    # --- Identify known vs new drugs across all scenarios ---
    all_sanger_drugs = (set(s3_df['clean_drug'].unique()) | set(s1_df['clean_drug'].unique())
                       | set(s2_df['clean_drug'].unique()) | set(s4_df['clean_drug'].unique()))
    known_drugs = {d for d in all_sanger_drugs if d.upper() in graph_drugs}
    new_drugs = {d for d in all_sanger_drugs if d.upper() not in graph_drugs
                 and d.upper() in sanger_drug_emb_names}
    dropped_drugs = all_sanger_drugs - known_drugs - new_drugs

    # Drop Bortezomib — it's a positive control in PRISM (352K rows vs ~3K for
    # normal drugs), dosed on every plate as a viability reference.  Its PRISM
    # GMM labels are not comparable to Sanger IC50/AUC.
    CONTROL_DRUGS = {'BORTEZOMIB'}
    for d in list(known_drugs):
        if d.upper() in CONTROL_DRUGS:
            print(f"  > Excluding PRISM control drug from evaluation: {d}")
            known_drugs.discard(d)
            dropped_drugs.add(d)

    print(f"  > Known drugs (in graph): {len(known_drugs)}")
    print(f"  > New drugs (with embeddings): {len(new_drugs)}")
    print(f"  > Dropped drugs (no graph node, no embedding, or control): {len(dropped_drugs)}")
    if dropped_drugs:
        print(f"    {sorted(dropped_drugs)[:10]}...")

    # Valid drugs for each scenario
    s1_valid_drugs = known_drugs   # S1: known cells, known drugs
    s3_valid_drugs = known_drugs   # S3: new cells, known drugs
    s2_valid_drugs = new_drugs     # S2: known cells, new drugs
    s4_valid_drugs = new_drugs     # S4: new cells, new drugs

    # --- Relabel ALL scenarios using per-drug GMM on AUC ---
    # AUC (dose-response area) aligns better with PRISM LMFI than IC50 does,
    # because both measure total viability rather than inflection-point potency.
    # Combine all scenarios so each drug's GMM sees all available cells.
    s1_df['_source'] = 'S1'
    s2_df['_source'] = 'S2'
    s3_df['_source'] = 'S3'
    s4_df['_source'] = 'S4'
    combined = pd.concat([s1_df, s2_df, s3_df, s4_df], ignore_index=True)
    combined = relabel_sanger_gmm(combined, value_col='AUC', drug_col='clean_drug')
    s1_df = combined[combined['_source'] == 'S1'].drop(columns=['_source', 'gmm_class'])
    s2_df = combined[combined['_source'] == 'S2'].drop(columns=['_source', 'gmm_class'])
    s3_df = combined[combined['_source'] == 'S3'].drop(columns=['_source', 'gmm_class'])
    s4_df = combined[combined['_source'] == 'S4'].drop(columns=['_source', 'gmm_class'])

    # ==================================================================
    # STEP 3: Extract Cell-Gene mutations for S3 cells
    # ==================================================================
    print("\n--- Step 3: Extracting Cell-Gene mutations for S3 cells ---")

    entrez_to_hugo = build_entrez_to_hugo(args.misc_dir)
    print(f"  > Entrez->HUGO mapping: {len(entrez_to_hugo)} genes")

    s3_ach_ids = set(s3_cell_map.values())

    mut_file = os.path.join(args.misc_dir, "OmicsSomaticMutations_24Q2.csv")
    use_cols = ['ModelID', 'EntrezGeneID', 'LikelyLoF', 'Hotspot',
                'HessDriver', 'OncogeneHighImpact',
                'TumorSuppressorHighImpact', 'RevelScore', 'AMClass',
                'AMPathogenicity']

    print(f"  > Reading mutations from: {mut_file}")
    mut_df = pd.read_csv(mut_file, usecols=use_cols, low_memory=False)

    # Filter to S3 ACH IDs only
    mut_df = mut_df[mut_df['ModelID'].isin(s3_ach_ids)]
    print(f"  > S3 mutations (raw): {len(mut_df):,}")

    cg_edges = score_mutations(mut_df, entrez_to_hugo, graph_genes)
    print(f"  > S3 Cell-Gene edges: {len(cg_edges):,}")
    print(f"    Unique S3 cells with mutations: {cg_edges['ModelID'].nunique()}")
    print(f"    Unique genes: {cg_edges['gene_name'].nunique()}")

    # Cells with mutations vs without
    s3_with_muts = set(cg_edges['ModelID'].unique())
    s3_without_muts = s3_ach_ids - s3_with_muts
    print(f"  > S3 cells without qualifying mutations: {len(s3_without_muts)}")

    # ==================================================================
    # STEP 4: Compute Cell-Cell similarity for S3 cells
    # ==================================================================
    print("\n--- Step 4: Computing Cell-Cell similarity for S3 cells ---")

    # Load DepMap expression (ACH-indexed)
    depmap_expr_file = os.path.join(
        args.misc_dir, "24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
    print(f"  > Loading DepMap expression...")
    depmap_expr = pd.read_csv(depmap_expr_file, index_col=0)
    depmap_expr.index = depmap_expr.index.str.strip().str.upper()

    # Filter DepMap to cells in graph
    depmap_in_graph = sorted(graph_cells & set(depmap_expr.index))
    depmap_expr_graph = depmap_expr.loc[depmap_in_graph]
    print(f"  > DepMap cells in graph with expression: {len(depmap_expr_graph)}")

    # Load Sanger expression (clean_cell-indexed)
    sanger_expr_file = os.path.join(args.sanger_dir, 'sanger_aligned_tpm.csv')
    print(f"  > Loading Sanger expression...")
    sanger_expr = pd.read_csv(sanger_expr_file, index_col=0)
    sanger_expr.index = sanger_expr.index.astype(str)

    # Filter Sanger to S3 cells that we mapped
    s3_clean_in_expr = sorted(set(s3_cell_map.keys()) & set(sanger_expr.index))
    sanger_expr_s3 = sanger_expr.loc[s3_clean_in_expr]
    print(f"  > S3 cells with expression: {len(sanger_expr_s3)}")

    # Align columns (intersection of genes)
    common_genes = sorted(set(depmap_expr_graph.columns) & set(sanger_expr_s3.columns))
    print(f"  > Common expression genes: {len(common_genes)}")

    depmap_mat = depmap_expr_graph[common_genes].fillna(0).values
    sanger_mat = sanger_expr_s3[common_genes].fillna(0).values

    # Cross cosine similarity: S3 × DepMap
    print(f"  > Computing cross similarity ({len(sanger_expr_s3)} × {len(depmap_expr_graph)})...")
    cross_sim = cosine_similarity(sanger_mat, depmap_mat)

    # For each S3 cell, take top-K most similar DepMap cells
    n_depmap = len(depmap_expr_graph)
    top_k = max(1, int(n_depmap * args.top_pct))
    print(f"  > Top {args.top_pct*100:.1f}% = top {top_k} DepMap neighbors per S3 cell")

    cc_edges = []  # (s3_ach, depmap_ach, similarity)
    s3_clean_list = sanger_expr_s3.index.tolist()
    depmap_ach_list = depmap_expr_graph.index.tolist()

    for i in tqdm(range(len(s3_clean_list)), desc="  - Building CC edges"):
        scores = cross_sim[i]
        top_indices = np.argpartition(scores, -top_k)[-top_k:]

        s3_ach = s3_cell_map[s3_clean_list[i]]
        for j in top_indices:
            depmap_ach = depmap_ach_list[j]
            sim = float(scores[j])
            # Bidirectional
            cc_edges.append((s3_ach, depmap_ach, round(sim, 6)))
            cc_edges.append((depmap_ach, s3_ach, round(sim, 6)))

    print(f"  > Cell-Cell edges (S3↔DepMap): {len(cc_edges)}")

    # ==================================================================
    # STEP 5: Build augmented graph files
    # ==================================================================
    print("\n--- Step 5: Building augmented graph files ---")

    # --- 5a. node.dat ---
    # Copy existing, append S3 cells with new GIDs, then new drugs
    s3_ach_to_gid = {}
    next_gid = max_gid + 1

    # Sort S3 ACH IDs for deterministic ordering
    # Include S4 cells too (same new cells as S3)
    all_new_cell_achs = set(s3_cell_map.values()) | set(s4_cell_map.values())
    for ach in sorted(all_new_cell_achs):
        if ach not in s3_ach_to_gid:
            s3_ach_to_gid[ach] = next_gid
            next_gid += 1

    print(f"  > New cells (S3/S4): {len(s3_ach_to_gid)} (GIDs {max_gid+1} - {next_gid-1})")

    # Assign GIDs for new drugs (type 1)
    new_drug_to_gid = {}
    drug_gid_start = next_gid
    for drug_name in sorted(new_drugs):
        new_drug_to_gid[drug_name.upper()] = next_gid
        next_gid += 1

    print(f"  > New drugs (S2/S4): {len(new_drug_to_gid)} (GIDs {drug_gid_start} - {next_gid-1})")

    # Write node.dat
    with open(os.path.join(SRC, 'node.dat')) as f_in, \
         open(os.path.join(OUT, 'node.dat'), 'w') as f_out:
        for line in f_in:
            f_out.write(line)
        # Append new cells (type 0)
        for ach, gid in sorted(s3_ach_to_gid.items(), key=lambda x: x[1]):
            f_out.write(f"{gid}\t{ach}\t0\n")
        # Append new drugs (type 1)
        for drug_name, gid in sorted(new_drug_to_gid.items(), key=lambda x: x[1]):
            f_out.write(f"{gid}\t{drug_name}\t1\n")

    # Update node_mappings
    aug_mappings = dict(node_mappings)
    for ach, gid in s3_ach_to_gid.items():
        aug_mappings[ach] = gid
    for drug_name, gid in new_drug_to_gid.items():
        aug_mappings[drug_name] = gid
    with open(os.path.join(OUT, 'node_mappings.json'), 'w') as f:
        json.dump(aug_mappings, f, indent=4)

    # --- 5b. Build new structural edges ---
    new_edges = []  # (src_gid, tgt_gid, ltype, weight)

    # Cell-Gene mutation edges (type 2) for S3 cells
    cg_count = 0
    for _, row in cg_edges.iterrows():
        ach = row['ModelID']
        gene = row['gene_name']
        weight = row['weight']
        cell_gid = s3_ach_to_gid.get(ach)
        gene_info = existing_nodes.get(gene)
        if cell_gid is not None and gene_info is not None:
            gene_gid = gene_info[0]
            new_edges.append((cell_gid, gene_gid, 2, weight))
            cg_count += 1

    print(f"  > New Cell-Gene edges: {cg_count}")

    # Cell-Cell similarity edges (type 4) for S3 cells
    cc_count = 0
    for s3_ach, depmap_ach, sim in cc_edges:
        gid_a = s3_ach_to_gid.get(s3_ach) or existing_nodes.get(s3_ach, (None,))[0]
        gid_b = s3_ach_to_gid.get(depmap_ach) or existing_nodes.get(depmap_ach, (None,))[0]
        if gid_a is not None and gid_b is not None:
            new_edges.append((gid_a, gid_b, 4, sim))
            cc_count += 1

    print(f"  > New Cell-Cell edges: {cc_count}")

    # --- 5c. link.dat: copy existing + append new structural ---
    with open(os.path.join(SRC, 'link.dat')) as f_in, \
         open(os.path.join(OUT, 'link.dat'), 'w') as f_out:
        for line in f_in:
            f_out.write(line)
        for src, tgt, ltype, w in new_edges:
            f_out.write(f"{src}\t{tgt}\t{ltype}\t{w}\n")

    # --- 5d. train.dat: copy existing + append new structural ---
    with open(os.path.join(SRC, 'train.dat')) as f_in, \
         open(os.path.join(OUT, 'train.dat'), 'w') as f_out:
        for line in f_in:
            f_out.write(line)
        for src, tgt, ltype, w in new_edges:
            f_out.write(f"{src}\t{tgt}\t{ltype}\t{w}\n")

    # --- 5e. info.dat: update counts ---
    with open(os.path.join(SRC, 'info.dat')) as f:
        info = json.load(f)

    # Update cell count
    old_cell_count = info['node.dat']['0'][1]
    info['node.dat']['0'][1] = old_cell_count + len(s3_ach_to_gid)

    # Update drug count
    old_drug_count = info['node.dat']['1'][1]
    info['node.dat']['1'][1] = old_drug_count + len(new_drug_to_gid)

    # Update edge counts
    info['link.dat']['2'][3] += cg_count   # cell-gene
    info['link.dat']['4'][3] += cc_count   # cell-cell

    with open(os.path.join(OUT, 'info.dat'), 'w') as f:
        json.dump(info, f, indent=4)

    print(f"  > info.dat updated: cells {old_cell_count} -> {info['node.dat']['0'][1]}, "
          f"drugs {old_drug_count} -> {info['node.dat']['1'][1]}")

    # --- 5f. Copy existing split files and triplet map ---
    copy_files = [
        'train_lp_links.dat',
        'valid_transductive_links.dat',
        'valid_inductive_links.dat',
        'test_transductive_links.dat',
        'test_inductive_links.dat',
        'cell_triplet_map.pkl',
        'cell_splits.json',
    ]
    for fname in copy_files:
        src_path = os.path.join(SRC, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(OUT, fname))

    # --- 5g. Write Sanger evaluation link files ---

    def _resolve_drug_gid(drug_name):
        """Resolve drug name to GID (existing or new)."""
        upper = drug_name.upper()
        info = existing_nodes.get(upper)
        if info is not None:
            return info[0]
        return new_drug_to_gid.get(upper)

    # S1 links: known cells, known drugs
    s1_link_count = 0
    with open(os.path.join(OUT, 'sanger_S1_links.dat'), 'w') as f:
        for _, row in s1_df.iterrows():
            if row['clean_drug'] not in s1_valid_drugs:
                continue
            ach = s1_cell_map.get(row['clean_cell'])
            if ach is None:
                continue
            cell_info = existing_nodes.get(ach)
            drug_gid = _resolve_drug_gid(row['clean_drug'])
            if cell_info is not None and drug_gid is not None:
                f.write(f"{cell_info[0]}\t{drug_gid}\t{float(row['label'])}\n")
                s1_link_count += 1
    print(f"  > sanger_S1_links.dat: {s1_link_count} pairs")

    # S2 links: known cells, NEW drugs
    s2_link_count = 0
    with open(os.path.join(OUT, 'sanger_S2_links.dat'), 'w') as f:
        for _, row in s2_df.iterrows():
            if row['clean_drug'] not in s2_valid_drugs:
                continue
            ach = s2_cell_map.get(row['clean_cell'])
            if ach is None:
                continue
            cell_info = existing_nodes.get(ach)
            drug_gid = _resolve_drug_gid(row['clean_drug'])
            if cell_info is not None and drug_gid is not None:
                f.write(f"{cell_info[0]}\t{drug_gid}\t{float(row['label'])}\n")
                s2_link_count += 1
    print(f"  > sanger_S2_links.dat: {s2_link_count} pairs")

    # S3 links: NEW cells, known drugs
    s3_link_count = 0
    with open(os.path.join(OUT, 'sanger_S3_links.dat'), 'w') as f:
        for _, row in s3_df.iterrows():
            if row['clean_drug'] not in s3_valid_drugs:
                continue
            ach = s3_cell_map.get(row['clean_cell'])
            if ach is None:
                continue
            cell_gid = s3_ach_to_gid.get(ach)
            drug_gid = _resolve_drug_gid(row['clean_drug'])
            if cell_gid is not None and drug_gid is not None:
                f.write(f"{cell_gid}\t{drug_gid}\t{float(row['label'])}\n")
                s3_link_count += 1
    print(f"  > sanger_S3_links.dat: {s3_link_count} pairs")

    # S4 links: NEW cells, NEW drugs
    s4_link_count = 0
    with open(os.path.join(OUT, 'sanger_S4_links.dat'), 'w') as f:
        for _, row in s4_df.iterrows():
            if row['clean_drug'] not in s4_valid_drugs:
                continue
            ach = s4_cell_map.get(row['clean_cell'])
            if ach is None:
                continue
            cell_gid = s3_ach_to_gid.get(ach)
            drug_gid = _resolve_drug_gid(row['clean_drug'])
            if cell_gid is not None and drug_gid is not None:
                f.write(f"{cell_gid}\t{drug_gid}\t{float(row['label'])}\n")
                s4_link_count += 1
    print(f"  > sanger_S4_links.dat: {s4_link_count} pairs")

    # ==================================================================
    # STEP 6: Augment embedding files
    # ==================================================================
    print("\n--- Step 6: Augmenting embedding files ---")

    # Load existing cell embeddings
    orig_embeds = np.load(os.path.join(args.emb_dir, 'final_vae_cell_embeddings.npy'))
    with open(os.path.join(args.emb_dir, 'final_vae_cell_names.txt')) as f:
        orig_names = [line.strip() for line in f if line.strip()]

    print(f"  > Original cell embeddings: {orig_embeds.shape}")

    # Load Sanger VAE embeddings
    sanger_vae_full = pd.read_csv(
        os.path.join(args.sanger_dir, 'sanger_vae_embeddings.csv'), index_col=0)
    sanger_vae_full.index = sanger_vae_full.index.astype(str)

    # Build new embeddings: append S3 cells in sorted GID order
    new_embeds_list = []
    new_names_list = []

    for ach, gid in sorted(s3_ach_to_gid.items(), key=lambda x: x[1]):
        # Find clean_cell name for this ACH
        clean_cell = None
        for cc, a in s3_cell_map.items():
            if a == ach:
                clean_cell = cc
                break
        if clean_cell is not None and clean_cell in sanger_vae_full.index:
            vec = sanger_vae_full.loc[clean_cell].values.astype(np.float32)
            new_embeds_list.append(vec)
            new_names_list.append(ach)  # Store ACH ID to match node.dat
        else:
            # Should not happen (filtered earlier), but use zeros as fallback
            print(f"  > WARNING: No VAE embedding for {ach} ({clean_cell}), using zeros")
            new_embeds_list.append(np.zeros(orig_embeds.shape[1], dtype=np.float32))
            new_names_list.append(ach)

    if new_embeds_list:
        new_embeds = np.stack(new_embeds_list)
        aug_embeds = np.vstack([orig_embeds, new_embeds])
        aug_names = orig_names + new_names_list
    else:
        aug_embeds = orig_embeds
        aug_names = orig_names

    print(f"  > Augmented cell embeddings: {aug_embeds.shape}")

    np.save(os.path.join(OUT, 'embeddings', 'final_vae_cell_embeddings.npy'), aug_embeds)
    with open(os.path.join(OUT, 'embeddings', 'final_vae_cell_names.txt'), 'w') as f:
        for name in aug_names:
            f.write(f"{name}\n")

    # Augment drug embeddings: merge existing PRISM + new Sanger drugs
    emb_out = os.path.join(OUT, 'embeddings')
    if new_drug_to_gid:
        orig_drug_emb = pd.read_csv(os.path.join(args.emb_dir, 'drugs_with_embeddings.csv'))
        sanger_drug_emb_full = pd.read_csv(sanger_drug_emb_path)
        # Normalize drug names for matching
        sanger_drug_emb_full.iloc[:, 0] = sanger_drug_emb_full.iloc[:, 0].str.upper()
        # Filter to only new drugs
        new_drug_rows = sanger_drug_emb_full[
            sanger_drug_emb_full.iloc[:, 0].isin(new_drug_to_gid.keys())
        ]
        aug_drug_emb = pd.concat([orig_drug_emb, new_drug_rows], ignore_index=True)
        drug_emb_path = os.path.join(emb_out, 'drugs_with_embeddings.csv')
        aug_drug_emb.to_csv(drug_emb_path, index=False)
        print(f"  > Augmented drug embeddings: {len(orig_drug_emb)} + {len(new_drug_rows)} "
              f"= {len(aug_drug_emb)} drugs")
    else:
        # No new drugs — symlink original
        dst = os.path.join(emb_out, 'drugs_with_embeddings.csv')
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(os.path.join(args.emb_dir, 'drugs_with_embeddings.csv')), dst)
        print(f"  > Symlinked drug embeddings (no new drugs)")

    # Symlink gene embeddings (always unchanged)
    gene_dst = os.path.join(emb_out, 'gene_embeddings_esm_by_symbol.pkl')
    if os.path.exists(gene_dst) or os.path.islink(gene_dst):
        os.remove(gene_dst)
    os.symlink(os.path.abspath(os.path.join(args.emb_dir, 'gene_embeddings_esm_by_symbol.pkl')),
               gene_dst)
    print(f"  > Symlinked gene embeddings")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("Augmented graph built successfully!")
    print(f"  Output directory: {OUT}")
    print(f"  Total nodes: {next_gid}")
    print(f"    Cells: {info['node.dat']['0'][1]} ({len(s3_ach_to_gid)} new)")
    print(f"    Drugs: {info['node.dat']['1'][1]} ({len(new_drug_to_gid)} new)")
    print(f"    Genes: {info['node.dat']['2'][1]}")
    print(f"  New structural edges: {len(new_edges)}")
    print(f"    Cell-Gene (S3/S4 mutations): {cg_count}")
    print(f"    Cell-Cell (S3/S4↔DepMap sim): {cc_count}")
    print(f"    Drug edges for new drugs: 0 (cold-start, skip gate handles)")
    print(f"  Evaluation pairs:")
    print(f"    S1 (known cells, known drugs): {s1_link_count}")
    print(f"    S2 (known cells, new drugs):   {s2_link_count}")
    print(f"    S3 (new cells, known drugs):   {s3_link_count}")
    print(f"    S4 (new cells, new drugs):     {s4_link_count}")

    for name, df_s, cmap, vdrugs in [
        ('S1', s1_df, s1_cell_map, s1_valid_drugs),
        ('S2', s2_df, s2_cell_map, s2_valid_drugs),
        ('S3', s3_df, s3_cell_map, s3_valid_drugs),
        ('S4', s4_df, s4_cell_map, s4_valid_drugs),
    ]:
        usable = df_s[df_s['clean_cell'].isin(cmap) & df_s['clean_drug'].isin(vdrugs)]
        if len(usable) > 0:
            print(f"  {name} label balance: {usable['label'].mean():.1%} sensitive (GMM)")

    print("=" * 60)
    print("\nNext: run generate_neighbors.py --data_dir", OUT)


if __name__ == '__main__':
    main()
