"""Step 2: Label Cell-Drug interactions with GMM + confidence margin.

Fits GMM on PRISM and Sanger AUC distributions separately.
Applies a confidence margin around the intersection to exclude ambiguous pairs.
Outputs labeled interaction files for graph building.

Creates:
  - data/processed_v2/prism_labeled.csv: cell, drug, auc, label, excluded
  - data/processed_v2/sanger_labeled.csv: cell, drug, auc, ic50, label, excluded

Usage:
    python scripts/pipeline_v2/step2_label_gmm.py
    python scripts/pipeline_v2/step2_label_gmm.py --prism_margin 0.05 --sanger_margin 0.025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq

MISC = 'data/misc'
OUT = 'data/processed_v2'


def fit_gmm_and_label(values, name, margin):
    """Fit 2-component GMM, find intersection, apply margin.

    Args:
        values: 1D array of AUC values
        name: dataset name for printing
        margin: half-width of exclusion zone around intersection

    Returns:
        intersection, labels array (1=sensitive, 0=resistant, -1=excluded)
    """
    X = values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    sens_idx = np.argmin(means)  # lower AUC = more sensitive
    res_idx = np.argmax(means)

    print(f"\n  {name} GMM Results:")
    print(f"    Sensitive component: mean={means[sens_idx]:.4f}, std={stds[sens_idx]:.4f}, weight={weights[sens_idx]:.3f}")
    print(f"    Resistant component: mean={means[res_idx]:.4f}, std={stds[res_idx]:.4f}, weight={weights[res_idx]:.3f}")

    # Find intersection
    try:
        def gmm_diff(x):
            probs = gmm.predict_proba(np.array([[x]]))[0]
            return probs[sens_idx] - probs[res_idx]
        intersection = brentq(gmm_diff, means[sens_idx], means[res_idx])
    except (ValueError, RuntimeError):
        intersection = np.mean(means)
        print(f"    WARNING: brentq failed, using midpoint")

    print(f"    Intersection: {intersection:.4f}")
    print(f"    Margin: ±{margin:.4f}")

    lower = intersection - margin
    upper = intersection + margin

    # Label
    labels = np.full(len(values), -1, dtype=int)  # -1 = excluded
    labels[values < lower] = 1   # sensitive
    labels[values > upper] = 0   # resistant

    n_sens = (labels == 1).sum()
    n_res = (labels == 0).sum()
    n_excl = (labels == -1).sum()
    print(f"    Sensitive (label=1): {n_sens:,} ({100*n_sens/len(values):.1f}%)")
    print(f"    Resistant (label=0): {n_res:,} ({100*n_res/len(values):.1f}%)")
    print(f"    Excluded (margin):   {n_excl:,} ({100*n_excl/len(values):.1f}%)")
    print(f"    Threshold: AUC < {lower:.4f} = sensitive, AUC > {upper:.4f} = resistant")

    return intersection, labels, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prism_margin', type=float, default=0.05,
                        help='Half-width of exclusion zone for PRISM (default 0.05)')
    parser.add_argument('--sanger_margin', type=float, default=0.025,
                        help='Half-width of exclusion zone for Sanger (default 0.025)')
    parser.add_argument('--ratio_match', action='store_true',
                        help='Match PRISM sens/res ratio to Sanger instead of using GMM intersection')
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)

    # Load master mapping tables
    master_cells = pd.read_csv(f'{OUT}/master_cells.csv')
    master_drugs = pd.read_csv(f'{OUT}/master_drugs.csv')
    sanger_name_map = pd.read_csv(f'{OUT}/sanger_cell_name_to_ach.csv')
    sanger_name_to_ach = dict(zip(sanger_name_map['sanger_name'], sanger_name_map['ach_id']))

    graph_cells = set(master_cells[master_cells['in_graph']]['ach_id'])
    graph_drugs = set(master_drugs[master_drugs['has_smiles']]['canonical_name'])

    # ==========================================
    # PRISM LABELING
    # ==========================================
    print("=" * 60)
    print("STEP 2a: PRISM AUC Labeling")
    print("=" * 60)

    prism = pd.read_csv(f'{MISC}/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv',
                        usecols=['depmap_id', 'name', 'auc', 'ic50'], low_memory=False)

    # Filter to graph cells and drugs
    prism['drug_upper'] = prism['name'].str.upper().str.strip()
    prism_valid = prism[
        prism['depmap_id'].isin(graph_cells) &
        prism['drug_upper'].isin(graph_drugs) &
        prism['auc'].notna()
    ].copy()

    print(f"  Valid PRISM measurements: {len(prism_valid):,}")
    print(f"  Cells: {prism_valid['depmap_id'].nunique()}, Drugs: {prism_valid['name'].nunique()}")

    # Average replicates per (cell, drug) pair
    prism_pairs = prism_valid.groupby(['depmap_id', 'drug_upper']).agg(
        auc_mean=('auc', 'mean'),
        auc_std=('auc', 'std'),
        n_replicates=('auc', 'count'),
        ic50_mean=('ic50', 'mean'),
    ).reset_index()
    print(f"  Unique pairs after averaging: {len(prism_pairs):,}")

    # Fit GMM and label
    auc_values = prism_pairs['auc_mean'].values
    prism_intersection, prism_labels, prism_lower, prism_upper = fit_gmm_and_label(
        auc_values, "PRISM", args.prism_margin
    )

    if args.ratio_match:
        # Override PRISM thresholds to match Sanger's sens/res ratio (~32% / ~54%)
        # This dramatically improves cross-dataset concordance (49% → 77%)
        target_sens = 0.32
        target_res = 0.54
        prism_lower = np.percentile(auc_values, target_sens * 100)
        prism_upper = np.percentile(auc_values, (1 - target_res) * 100)
        prism_labels = np.full(len(auc_values), -1, dtype=int)
        prism_labels[auc_values < prism_lower] = 1
        prism_labels[auc_values > prism_upper] = 0

        n_sens = (prism_labels == 1).sum()
        n_res = (prism_labels == 0).sum()
        n_excl = (prism_labels == -1).sum()
        print(f"\n  RATIO-MATCHED override (matching Sanger distribution):")
        print(f"    Sensitive: AUC < {prism_lower:.4f} -> {n_sens:,} ({100*n_sens/len(auc_values):.1f}%)")
        print(f"    Resistant: AUC > {prism_upper:.4f} -> {n_res:,} ({100*n_res/len(auc_values):.1f}%)")
        print(f"    Excluded: {n_excl:,} ({100*n_excl/len(auc_values):.1f}%)")

    prism_pairs['label'] = prism_labels
    prism_pairs['excluded'] = prism_labels == -1
    prism_pairs.rename(columns={'depmap_id': 'cell_ach', 'drug_upper': 'drug_name'}, inplace=True)

    prism_pairs.to_csv(f'{OUT}/prism_labeled.csv', index=False)
    print(f"\n  Saved: {OUT}/prism_labeled.csv")

    # ==========================================
    # SANGER LABELING
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 2b: Sanger AUC Labeling")
    print("=" * 60)

    sanger = pd.read_csv(f'{MISC}/PANCANCER_IC_Fri Jan 16 16_28_57 2026.csv')
    sanger['ach'] = sanger['Cell Line Name'].map(sanger_name_to_ach)
    sanger['drug_upper'] = sanger['Drug Name'].str.upper().str.strip()

    sanger_valid = sanger[
        sanger['ach'].isin(graph_cells) &
        sanger['drug_upper'].isin(graph_drugs) &
        sanger['AUC'].notna()
    ].copy()

    print(f"  Valid Sanger measurements: {len(sanger_valid):,}")
    print(f"  Cells: {sanger_valid['ach'].nunique()}, Drugs: {sanger_valid['Drug Name'].nunique()}")

    # Average replicates
    sanger_pairs = sanger_valid.groupby(['ach', 'drug_upper']).agg(
        auc_mean=('AUC', 'mean'),
        ic50_mean=('IC50', 'mean'),
        n_replicates=('AUC', 'count'),
    ).reset_index()
    print(f"  Unique pairs after averaging: {len(sanger_pairs):,}")

    # Fit GMM and label
    sanger_auc_values = sanger_pairs['auc_mean'].values
    sanger_intersection, sanger_labels, sanger_lower, sanger_upper = fit_gmm_and_label(
        sanger_auc_values, "Sanger", args.sanger_margin
    )

    sanger_pairs['label'] = sanger_labels
    sanger_pairs['excluded'] = sanger_labels == -1
    sanger_pairs.rename(columns={'ach': 'cell_ach', 'drug_upper': 'drug_name'}, inplace=True)

    sanger_pairs.to_csv(f'{OUT}/sanger_labeled.csv', index=False)
    print(f"\n  Saved: {OUT}/sanger_labeled.csv")

    # ==========================================
    # CONCORDANCE
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 2c: Cross-Dataset Concordance")
    print("=" * 60)

    # Find pairs that exist in both
    prism_set = set(zip(prism_pairs['cell_ach'], prism_pairs['drug_name']))
    sanger_set = set(zip(sanger_pairs['cell_ach'], sanger_pairs['drug_name']))
    overlap_pairs = prism_set & sanger_set

    print(f"  Pairs in PRISM:   {len(prism_set):,}")
    print(f"  Pairs in Sanger:  {len(sanger_set):,}")
    print(f"  Pairs in both:    {len(overlap_pairs):,}")

    if overlap_pairs:
        # Build concordance table
        prism_idx = prism_pairs.set_index(['cell_ach', 'drug_name'])
        sanger_idx = sanger_pairs.set_index(['cell_ach', 'drug_name'])

        concordance_rows = []
        for cell, drug in overlap_pairs:
            p = prism_idx.loc[(cell, drug)]
            s = sanger_idx.loc[(cell, drug)]
            concordance_rows.append({
                'cell_ach': cell,
                'drug_name': drug,
                'prism_auc': p['auc_mean'],
                'prism_label': p['label'],
                'sanger_auc': s['auc_mean'],
                'sanger_label': s['label'],
            })

        conc = pd.DataFrame(concordance_rows)

        # Only compare pairs where both have non-excluded labels
        both_labeled = conc[(conc['prism_label'] >= 0) & (conc['sanger_label'] >= 0)]
        if len(both_labeled) > 0:
            agree = (both_labeled['prism_label'] == both_labeled['sanger_label']).sum()
            total = len(both_labeled)
            print(f"\n  Concordance (both labeled):")
            print(f"    Pairs: {total:,}")
            print(f"    Agree: {agree:,} ({100*agree/total:.1f}%)")
            print(f"    Disagree: {total-agree:,} ({100*(total-agree)/total:.1f}%)")

            # Breakdown
            tp = ((both_labeled['prism_label'] == 1) & (both_labeled['sanger_label'] == 1)).sum()
            tn = ((both_labeled['prism_label'] == 0) & (both_labeled['sanger_label'] == 0)).sum()
            fp = ((both_labeled['prism_label'] == 1) & (both_labeled['sanger_label'] == 0)).sum()
            fn = ((both_labeled['prism_label'] == 0) & (both_labeled['sanger_label'] == 1)).sum()
            print(f"    Both sensitive:  {tp:,}")
            print(f"    Both resistant:  {tn:,}")
            print(f"    PRISM sens / Sanger res: {fp:,}")
            print(f"    PRISM res / Sanger sens: {fn:,}")

        # Spearman correlation on AUC values
        from scipy.stats import spearmanr, pearsonr
        sp = spearmanr(conc['prism_auc'], conc['sanger_auc'])
        pr = pearsonr(conc['prism_auc'], conc['sanger_auc'])
        print(f"\n  AUC Correlation ({len(conc):,} pairs):")
        print(f"    Spearman: {sp.statistic:.4f} (p={sp.pvalue:.2e})")
        print(f"    Pearson:  {pr.statistic:.4f} (p={pr.pvalue:.2e})")

        conc.to_csv(f'{OUT}/concordance.csv', index=False)
        print(f"\n  Saved: {OUT}/concordance.csv")

    # ==========================================
    # SAVE CONFIG
    # ==========================================
    config = {
        'prism_margin': args.prism_margin,
        'sanger_margin': args.sanger_margin,
        'prism_intersection': float(prism_intersection),
        'sanger_intersection': float(sanger_intersection),
        'prism_lower_threshold': float(prism_lower),
        'prism_upper_threshold': float(prism_upper),
        'sanger_lower_threshold': float(sanger_lower),
        'sanger_upper_threshold': float(sanger_upper),
    }

    import json
    with open(f'{OUT}/labeling_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved: {OUT}/labeling_config.json")


if __name__ == '__main__':
    main()
