import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import argparse
import os
import re

# --- CONFIG ---
LOWER_IS_SENSITIVE = True 

def normalize_string(s):
    if pd.isna(s): return ""
    s = str(s).upper()
    return re.sub(r'[^A-Z0-9]', '', s)

def extract_ccle_name(s):
    if pd.isna(s): return ""
    return str(s).split('_')[0]

def fit_gmm_probability(values, n_components=2):
    """
    Fits a GMM to 1D data and returns the probability that each point
    belongs to the 'Sensitive' population.
    """
    # 1. Handle Binary Data
    unique_vals = np.unique(values)
    if len(unique_vals) <= 2:
        return values 

    # 2. Handle Continuous Data
    X = values.reshape(-1, 1)
    
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        means = gmm.means_.flatten()
        if LOWER_IS_SENSITIVE:
            sens_idx = np.argmin(means) 
        else:
            sens_idx = np.argmax(means) 
            
        probs = gmm.predict_proba(X)[:, sens_idx]
        return probs
        
    except Exception as e:
        # Don't print full stack trace, just the error
        # print(f"    [GMM Fit Error] {e}")
        return np.zeros_like(values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanger_file', type=str, required=True, help='Sanger CSV with IC50/AUC')
    parser.add_argument('--depmap_file', type=str, required=True, help='DepMap file')
    parser.add_argument('--metadata_file', type=str, required=True, help='Metadata CSV')
    parser.add_argument('--output_dir', type=str, default='results/concordance_analysis')
    parser.add_argument('--sanger_metric', type=str, default='IC50', help='IC50 or AUC')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Sanger
    print("1. Loading Sanger Data...")
    df_sanger = pd.read_csv(args.sanger_file)
    if 'clean_cell' in df_sanger.columns:
        df_sanger = df_sanger.rename(columns={'clean_cell': 'cell_name', 'clean_drug': 'drug'})
    
    # 2. Map IDs
    print("2. Mapping IDs...")
    df_meta = pd.read_csv(args.metadata_file)
    name_to_id = {}
    
    name_col = next((c for c in ['ccle_name', 'CellLineName', 'StrippedCellLineName'] if c in df_meta.columns), None)
    id_col = next((c for c in ['depmap_id', 'ModelID', 'Broad_ID'] if c in df_meta.columns), None)
    
    for idx, row in df_meta.iterrows():
        short_name = extract_ccle_name(row[name_col])
        norm_name = normalize_string(short_name)
        name_to_id[norm_name] = row[id_col]
        
    df_sanger['norm_cell'] = df_sanger['cell_name'].apply(normalize_string)
    df_sanger['ACH_ID'] = df_sanger['norm_cell'].map(name_to_id)
    df_sanger = df_sanger.dropna(subset=['ACH_ID'])

    # 3. Load DepMap
    print("3. Loading DepMap...")
    try:
        df_depmap = pd.read_csv(args.depmap_file)
        if 'ACH-000' in str(df_depmap.iloc[0,0]): raise ValueError
    except:
        df_depmap = pd.read_csv(args.depmap_file, sep='\t', header=None, 
                                names=['ACH_ID', 'drug', 'unused', 'depmap_val'],
                                usecols=['ACH_ID', 'drug', 'depmap_val'])

    # 4. Merge
    print("4. Merging...")
    df_sanger['join_drug'] = df_sanger['drug'].apply(normalize_string)
    df_depmap['join_drug'] = df_depmap['drug'].apply(normalize_string)
    
    merged = pd.merge(df_sanger, df_depmap, on=['ACH_ID', 'join_drug'], how='inner')
    print(f"   Found {len(merged)} overlapping points.")

    # 5. Analyze
    print("\n5. Analyzing Consistency per Drug...")
    drug_stats = []
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for drug in merged['join_drug'].unique():
        subset = merged[merged['join_drug'] == drug]
        
        # --- FIX: DROP NaNs BEFORE ANALYSIS ---
        subset = subset.dropna(subset=[args.sanger_metric, 'depmap_val'])
        
        if len(subset) < 20: continue 
        
        # Get Values
        sanger_vals = subset[args.sanger_metric].values.astype(float)
        depmap_vals = subset['depmap_val'].values.astype(float)
        
        # Log transform IC50 for fitting (handle negatives if any)
        if args.sanger_metric == 'IC50':
            # Clip negative or zero values to a small epsilon
            sanger_vals_fit = np.log(np.maximum(sanger_vals, 1e-6))
        else:
            sanger_vals_fit = sanger_vals

        sanger_probs = fit_gmm_probability(sanger_vals_fit)
        
        # Correlation
        if np.std(sanger_probs) == 0 or np.std(depmap_vals) == 0:
            correlation = 0 # Cannot correlate constant arrays
        else:
            correlation = np.corrcoef(sanger_probs, depmap_vals)[0,1]
        
        drug_stats.append({
            'Drug': subset['drug_x'].iloc[0],
            'N': len(subset),
            'Correlation': correlation
        })

    # 6. Save
    df_res = pd.DataFrame(drug_stats).sort_values('Correlation', ascending=False)
    out_csv = os.path.join(args.output_dir, "drug_consistency_gmm.csv")
    df_res.to_csv(out_csv, index=False)
    
    good_drugs = df_res[df_res['Correlation'] > 0.15]
    
    print(f"\n--- ANALYSIS COMPLETE ---")
    print(f"Total Drugs: {len(df_res)}")
    print(f"Consistent Drugs (Corr > 0.15): {len(good_drugs)}")
    print(f"\nTop 5 Consistent:\n{df_res.head(5)}")
    print(f"Stats saved to: {out_csv}")

if __name__ == "__main__":
    main()