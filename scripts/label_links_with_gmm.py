# scripts/label_links_with_gmm.py

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import root_scalar
from tqdm import tqdm
import argparse
import os
import sys # Added for path checks

# --- Helper Functions for GMM Fitting and Classification ---

def find_intersection(gmm, comp1_idx, comp2_idx, bounds):
    """Finds the point where two Gaussian components have equal posterior probability."""
    def diff_func(x):
        # Reshape x for predict_proba which expects 2D array [n_samples, n_features]
        probs = gmm.predict_proba(np.array([[x]])).ravel()
        # Handle cases where predict_proba might return fewer probabilities than expected
        # though unlikely with fitted GMM. Safety check:
        if comp1_idx >= len(probs) or comp2_idx >= len(probs):
            print(f"Warning: Index out of bounds in predict_proba. Comp idx: {comp1_idx}, {comp2_idx}, Probs: {probs}")
            return 0 # Or handle appropriately
        return probs[comp1_idx] - probs[comp2_idx]

    try:
        # Brentq requires the function values at the endpoints to have opposite signs.
        # Check signs before calling root_scalar to avoid ValueError.
        f_lower = diff_func(bounds[0])
        f_upper = diff_func(bounds[1])
        if np.sign(f_lower) == np.sign(f_upper):
            # If signs are the same, intersection might be outside bounds or components overlap heavily.
            # Return midpoint as a reasonable fallback.
            # print(f"Warning: Signs at bracket endpoints are the same ({f_lower:.2f}, {f_upper:.2f}). Returning midpoint for intersection.")
            return np.mean(bounds)

        result = root_scalar(diff_func, bracket=bounds, method='brentq')
        return result.root
    except (ValueError, RuntimeError) as e:
        # Catch potential errors during root finding (e.g., bracket issues not caught by sign check)
        # print(f"Warning: root_scalar failed ({e}). Returning midpoint.")
        return np.mean(bounds)

def fit_gmm_and_find_thresholds(scores: np.ndarray):
    """Fits GMMs with 1-3 components, selects the best one via BIC, and finds thresholds."""
    X = scores.reshape(-1, 1)
    if X.shape[0] < 3: # Need at least n_components samples
        print("Warning: Too few data points (<3) to fit GMM. Returning single component.")
        return [], GaussianMixture(n_components=1).fit(X) # Fit with 1 component

    best_gmm = None
    min_bic = np.inf

    # Try fitting GMM with 1 to 3 components
    for n in range(1, 4):
        if X.shape[0] < n: # Cannot fit if fewer samples than components
            continue
        try:
            gmm = GaussianMixture(n_components=n, random_state=0, n_init=5).fit(X) # Reduced n_init
            bic = gmm.bic(X)
            if bic < min_bic:
                min_bic = bic
                best_gmm = gmm
        except ValueError as e:
            # Catch errors during GMM fitting (e.g., covariance issues)
            print(f"Warning: GMM fitting failed for n={n} components: {e}")
            continue # Try next number of components

    if best_gmm is None: # If all fits failed, default to 1 component
        print("Warning: All GMM fits failed. Defaulting to 1 component.")
        best_gmm = GaussianMixture(n_components=1).fit(X)


    n_components = best_gmm.n_components
    if n_components == 1:
        return [], best_gmm # No thresholds for a single component

    # Sort components by mean
    order = np.argsort(best_gmm.means_.ravel())
    sorted_means = best_gmm.means_[order]

    thresholds = []
    # Find intersection between adjacent sorted components
    for i in range(n_components - 1):
        comp1_original_idx = order[i]     # Original index of the lower mean component
        comp2_original_idx = order[i+1] # Original index of the higher mean component
        # Define bounds slightly wider than means to ensure root is bracketed
        lower_bound = sorted_means[i][0]
        upper_bound = sorted_means[i+1][0]
        # Ensure bounds are distinct
        if np.isclose(lower_bound, upper_bound):
            # If means are too close, place threshold at the mean
             threshold = lower_bound
        else:
            # Widen bounds slightly to help root finding
            padding = (upper_bound - lower_bound) * 0.1
            threshold = find_intersection(best_gmm, comp1_original_idx, comp2_original_idx,
                                          (lower_bound - padding, upper_bound + padding))

        thresholds.append(threshold)

    return sorted(thresholds), best_gmm

def classify_score(score, thresholds, n_components):
    """Classifies a score based on the GMM-derived thresholds."""
    if n_components == 1:
        return "Uncertain" # Cannot classify with only one component
    elif n_components == 2:
        # Assuming lower score is better (Positive)
        return "Positive" if score <= thresholds[0] else "Negative"
    else: # n_components == 3
        if score <= thresholds[0]:
            return "Positive"
        elif score <= thresholds[1]:
            return "Uncertain" # Middle component is uncertain
        else:
            return "Negative"

# --- Main Execution ---

def main(args):
    print(f"Reading raw drug response data from: {args.input_file}")
    if not os.path.exists(args.input_file):
        print(f"FATAL ERROR: Input file not found at {args.input_file}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"FATAL ERROR reading input CSV file: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_cols = [args.value_col, args.drug_col, args.cell_col]
    if not all(col in df.columns for col in required_cols):
        print(f"FATAL ERROR: Input CSV missing one or more required columns: {required_cols}")
        sys.exit(1)
        
    # Prepare data
    df_cleaned = df.dropna(subset=required_cols)
    if df_cleaned.empty:
         print("FATAL ERROR: No valid data after dropping NaNs in required columns.")
         sys.exit(1)

    # Extract DepMap ID (assuming format 'DEPMAP_ID::CellName')
    # Use str.split only if '::' is expected, otherwise use the column directly
    if '::' in df_cleaned[args.cell_col].iloc[0]:
         df_cleaned['cell_id_clean'] = df_cleaned[args.cell_col].str.split('::').str[0].str.strip()
    else:
         df_cleaned['cell_id_clean'] = df_cleaned[args.cell_col].str.strip() # Assume column is already the ID


    all_labeled_dfs = []
    # Ensure drug names are standardized (e.g., uppercase) before grouping
    df_cleaned[args.drug_col] = df_cleaned[args.drug_col].astype(str).str.strip().str.upper()
    drug_groups = df_cleaned.groupby(args.drug_col)

    print(f"Processing {len(drug_groups)} drugs...")
    for drug_name, group in tqdm(drug_groups, desc="Fitting GMMs per drug"):
        labeled_group = group.copy()
        scores = group[args.value_col].values
        
        if len(scores) < args.min_data_points:
            labeled_group['label'] = "Uncertain"
        else:
            thresholds, gmm = fit_gmm_and_find_thresholds(scores)
            labeled_group['label'] = labeled_group[args.value_col].apply(
                lambda x: classify_score(x, thresholds, gmm.n_components)
            )
        all_labeled_dfs.append(labeled_group)

    if not all_labeled_dfs:
         print("FATAL ERROR: No drugs processed. Check input data and grouping.")
         sys.exit(1)
         
    df_labeled = pd.concat(all_labeled_dfs)

    # Filter out uncertain links and create final output
    df_final = df_labeled[df_labeled['label'] != "Uncertain"].copy()
    
    if df_final.empty:
        print("Warning: No certain (Positive/Negative) links found after GMM classification.")
        # Create an empty file or handle as needed
        open(args.output_file, 'w').close()
    else:
        # Assign binary label (1=Positive, 0=Negative)
        df_final['binary_label'] = df_final['label'].apply(lambda x: 1 if x == 'Positive' else 0)
        # Assign standard link type ID (e.g., 0 for cell-drug)
        # Make sure this matches the type expected by build_graph_files.py
        df_final['type'] = 0

        # Select and order columns for the output file
        output_df = df_final[['cell_id_clean', args.drug_col, 'type', 'binary_label']]

        # Save the new, labeled link file
        try:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            output_df.to_csv(args.output_file, sep="\t", index=False, header=False)
        except Exception as e:
             print(f"FATAL ERROR saving output file {args.output_file}: {e}")
             sys.exit(1)


    print("\nProcessing complete.")
    print(f"  > Total Positive links identified: {len(df_final[df_final['label'] == 'Positive'])}")
    print(f"  > Total Negative links identified: {len(df_final[df_final['label'] == 'Negative'])}")
    print(f"  Labeled link file saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify drug-cell links using per-drug GMMs.")
    # Use relative paths assuming script is run from project root
    parser.add_argument('--input-file', default='data/misc/Repurposing_Public_24Q2_LMFI_NORMALIZED_with_DrugNames.csv',
                        help='Path to the raw drug response CSV file.')
    parser.add_argument('--output-file', default='data/raw/link_cell_drug_labeled.txt',
                        help='Path to save the new labeled link file.')
    # Column names based on the provided default file
    parser.add_argument('--value-col', default='LMFI.normalized', help='Column with drug effectiveness scores.')
    parser.add_argument('--drug-col', default='name', help='Column with drug names.')
    parser.add_argument('--cell-col', default='row_id', help='Column with cell line IDs (potentially DepMapID::CellName).')
    parser.add_argument('--min-data-points', type=int, default=100,
                        help='Minimum interactions per drug to fit GMM (otherwise marked Uncertain).')

    args = parser.parse_args()
    main(args)