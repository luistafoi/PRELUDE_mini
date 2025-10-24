# scripts/label_links_with_gmm.py

"""
Applies a Gaussian Mixture Model (GMM) to classify drug-cell interactions.

This script reads raw drug response data, fits a GMM to the response scores
for each drug individually, and classifies each interaction as Positive,
Negative, or Uncertain. It then outputs a new link file containing only the
high-confidence Positive (1) and Negative (0) links, ready for graph
construction.
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import root_scalar
from tqdm import tqdm
import argparse

# --- Helper Functions for GMM Fitting and Classification ---

def find_intersection(gmm, comp1_idx, comp2_idx, bounds):
    """Finds the point where two Gaussian components have equal posterior probability."""
    def diff_func(x):
        probs = gmm.predict_proba(np.array([[x]])).ravel()
        return probs[comp1_idx] - probs[comp2_idx]
    
    try:
        result = root_scalar(diff_func, bracket=bounds, method='brentq')
        return result.root
    except (ValueError, RuntimeError):
        return np.mean(bounds)

def fit_gmm_and_find_thresholds(scores: np.ndarray):
    """Fits GMMs with 1-3 components, selects the best one via BIC, and finds thresholds."""
    X = scores.reshape(-1, 1)
    
    gmms = [GaussianMixture(n_components=n, random_state=0, n_init=10).fit(X) for n in range(1, 4)]
    bics = [gmm.bic(X) for gmm in gmms]
    best_gmm = gmms[np.argmin(bics)]
    
    n_components = best_gmm.n_components
    if n_components == 1:
        return [], best_gmm

    order = np.argsort(best_gmm.means_.ravel())
    sorted_means = best_gmm.means_[order]

    thresholds = []
    for i in range(n_components - 1):
        comp1_original_idx = order[i]
        comp2_original_idx = order[i+1]
        lower_bound, upper_bound = sorted_means[i][0], sorted_means[i+1][0]
        threshold = find_intersection(best_gmm, comp1_original_idx, comp2_original_idx, (lower_bound, upper_bound))
        thresholds.append(threshold)
        
    return sorted(thresholds), best_gmm

def classify_score(score, thresholds, n_components):
    """Classifies a score based on the GMM-derived thresholds."""
    if n_components == 1:
        return "Uncertain"
    elif n_components == 2:
        # Lower LMFI score is better (more effective)
        return "Positive" if score < thresholds[0] else "Negative"
    else: # n_components == 3
        if score < thresholds[0]:
            return "Positive"
        elif score < thresholds[1]:
            return "Uncertain"
        else:
            return "Negative"

# --- Main Execution ---

def main(args):
    print(f"Reading raw drug response data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    df_cleaned = df.dropna(subset=[args.value_col, args.drug_col, args.cell_col])

    # Extract DepMap ID for cells
    df_cleaned['depmap_id'] = df_cleaned[args.cell_col].str.split('::').str[0]

    all_labeled_dfs = []
    drug_groups = df_cleaned.groupby(args.drug_col)

    for drug_name, group in tqdm(drug_groups, desc="Fitting GMMs per drug"):
        labeled_group = group.copy()
        if len(group) < args.min_data_points:
            labeled_group['label'] = "Uncertain"
        else:
            thresholds, gmm = fit_gmm_and_find_thresholds(group[args.value_col].values)
            labeled_group['label'] = labeled_group[args.value_col].apply(
                lambda x: classify_score(x, thresholds, gmm.n_components)
            )
        all_labeled_dfs.append(labeled_group)

    df_labeled = pd.concat(all_labeled_dfs)
    
    # Filter out uncertain links and create the final output DataFrame
    df_final = df_labeled[df_labeled['label'] != "Uncertain"].copy()
    df_final['binary_label'] = df_final['label'].apply(lambda x: 1 if x == 'Positive' else 0)
    df_final['type'] = 0 # Set standard link type for cell-drug

    output_df = df_final[['depmap_id', args.drug_col, 'type', 'binary_label']]
    
    # Save the new, labeled link file
    output_df.to_csv(args.output_file, sep="\t", index=False, header=False)
    
    print("\nProcessing complete.")
    print(f"  > Total Positive links identified: {len(df_final[df_final['label'] == 'Positive'])}")
    print(f"  > Total Negative links identified: {len(df_final[df_final['label'] == 'Negative'])}")
    print(f" Labeled link file saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify drug-cell links using per-drug GMMs.")
    parser.add_argument('--input-file', default='/data/luis/PRELUDE/data/misc/Repurposing_Public_24Q2_LMFI_NORMALIZED_with_DrugNames.csv',
                        help='Path to the raw drug response CSV file.')
    parser.add_argument('--output-file', default='data/raw/link_cell_drug_labeled.txt',
                        help='Path to save the new labeled link file.')
    parser.add_argument('--value-col', default='LMFI.normalized', help='Column with drug effectiveness scores.')
    parser.add_argument('--drug-col', default='name', help='Column with drug names.')
    parser.add_argument('--cell-col', default='row_id', help='Column with cell line IDs.')
    parser.add_argument('--min-data-points', type=int, default=100, help='Minimum interactions for a drug to be processed by GMM.')
    
    args = parser.parse_args()
    main(args)