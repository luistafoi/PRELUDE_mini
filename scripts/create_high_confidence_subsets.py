import pandas as pd
import os
import argparse

def normalize_string(s):
    if pd.isna(s): return ""
    import re
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concordance_file', type=str, required=True, 
                        help='Path to drug_consistency_gmm.csv')
    parser.add_argument('--input_dir', type=str, default='data/sanger_validation',
                        help='Directory containing S1, S2, S3, S4 csv files')
    parser.add_argument('--threshold', type=float, default=0.15, 
                        help='Minimum correlation to keep a drug (default: 0.15)')
    args = parser.parse_args()

    # 1. Load the Good Drugs
    print(f"Loading concordance stats from {args.concordance_file}...")
    df_conc = pd.read_csv(args.concordance_file)
    
    # Filter
    good_drugs = df_conc[df_conc['Correlation'] > args.threshold]['Drug'].tolist()
    
    # Normalize for matching
    good_drugs_norm = set([normalize_string(d) for d in good_drugs])
    
    print(f"Selected {len(good_drugs_norm)} High-Confidence Drugs (Corr > {args.threshold})")
    print(f"Drugs: {sorted(list(good_drugs_norm))}")

    # 2. Process each Subset File
    subsets = ['sanger_S1_known_known.csv', 'sanger_S2_known_new_drug.csv', 
               'sanger_S3_new_cell_known.csv', 'sanger_S4_new_new.csv']

    for filename in subsets:
        input_path = os.path.join(args.input_dir, filename)
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (Not found)")
            continue
            
        print(f"\nProcessing {filename}...")
        df = pd.read_csv(input_path)
        
        # Ensure we have a clean drug column to match against
        # Your S3 files usually have 'clean_drug' or 'drug_name'
        col_to_check = 'clean_drug'
        if col_to_check not in df.columns:
            # Fallback
            col_to_check = 'drug' if 'drug' in df.columns else None
            
        if not col_to_check:
            print(f"   Error: Could not find drug column in {filename}")
            continue

        # Normalize the drug column in the test set
        df['norm_drug_temp'] = df[col_to_check].apply(normalize_string)
        
        # Filter
        df_filtered = df[df['norm_drug_temp'].isin(good_drugs_norm)].copy()
        
        # Drop temp column
        df_filtered = df_filtered.drop(columns=['norm_drug_temp'])
        
        # Save
        output_filename = filename.replace('.csv', '_HighConf.csv')
        output_path = os.path.join(args.input_dir, output_filename)
        
        df_filtered.to_csv(output_path, index=False)
        print(f"   Original: {len(df)} pairs -> Filtered: {len(df_filtered)} pairs")
        print(f"   Saved to {output_path}")

if __name__ == "__main__":
    main()