import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import re

def normalize_string(s):
    """
    Standard normalization: Upper case + keep ONLY A-Z and 0-9.
    Example: "NCI-H1437" -> "NCIH1437"
    """
    if pd.isna(s): return ""
    s = str(s).upper()
    return re.sub(r'[^A-Z0-9]', '', s)

def extract_ccle_name(s):
    """
    Parses 'KYSE510_OESOPHAGUS' -> 'KYSE510'
    """
    if pd.isna(s): return ""
    # Split by the first underscore and take the first part
    return str(s).split('_')[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depmap_file', type=str, required=True, help='Path to link_cell_drug_labeled.txt')
    parser.add_argument('--sanger_file', type=str, default='data/sanger_validation/sanger_S1_known_known.csv')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to Repurposing...Meta_Data.csv')
    parser.add_argument('--output_dir', type=str, default='results/correlation_analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Sanger
    print("1. Loading Sanger Data...")
    df_sanger = pd.read_csv(args.sanger_file)
    if 'clean_cell' in df_sanger.columns:
        df_sanger = df_sanger.rename(columns={'clean_cell': 'cell_name', 'clean_drug': 'drug', 'label': 'sanger_label'})

    # 2. Build Mapping Dictionary from Metadata
    print(f"2. Building ID Map from {args.metadata_file}...")
    df_meta = pd.read_csv(args.metadata_file)
    
    # We want to map: Normalized(Short Name) -> DepMap_ID
    # e.g. "KYSE510" -> "ACH-000824"
    
    name_to_id = {}
    for idx, row in df_meta.iterrows():
        full_ccle = row['ccle_name']
        depmap_id = row['depmap_id']
        
        # Logic: Extract "KYSE510" from "KYSE510_OESOPHAGUS"
        short_name = extract_ccle_name(full_ccle)
        norm_name = normalize_string(short_name)
        
        name_to_id[norm_name] = depmap_id

    # 3. Apply Map to Sanger
    print("3. Mapping Sanger Cells to ACH IDs...")
    df_sanger['norm_cell_name'] = df_sanger['cell_name'].apply(normalize_string)
    df_sanger['ACH_ID'] = df_sanger['norm_cell_name'].map(name_to_id)
    
    # Check drop rate
    missing_mask = df_sanger['ACH_ID'].isna()
    if missing_mask.sum() > 0:
        print(f"   [Warn] Could not map {missing_mask.sum()} Sanger cells (e.g., {df_sanger[missing_mask]['cell_name'].iloc[0]}). Dropping.")
        df_sanger = df_sanger.dropna(subset=['ACH_ID'])
    
    print(f"   Mapped Sanger Rows: {len(df_sanger)}")

    # 4. Load DepMap (TXT)
    print("4. Loading DepMap Data...")
    try:
        # Columns: Cell(ACH-ID), Drug, Unused, Label
        df_depmap = pd.read_csv(args.depmap_file, sep='\t', header=None, 
                                names=['ACH_ID', 'drug', 'unused', 'depmap_label'],
                                usecols=['ACH_ID', 'drug', 'depmap_label'])
    except Exception as e:
        print(f"Error loading DepMap file: {e}")
        return

    # 5. Normalize Drugs & Merge
    print("5. Normalizing Drugs and Merging...")
    df_sanger['join_drug'] = df_sanger['drug'].apply(normalize_string)
    df_depmap['join_drug'] = df_depmap['drug'].apply(normalize_string)

    merged = pd.merge(df_sanger, df_depmap, on=['ACH_ID', 'join_drug'], how='inner')
    
    if len(merged) == 0:
        print("CRITICAL ERROR: No overlapping pairs found.")
        return

    print(f"   Found {len(merged)} overlapping pairs.")

    # 6. Statistics
    concordance = (merged['sanger_label'] == merged['depmap_label']).mean()
    print(f"\n--- GLOBAL STATISTICS ---")
    print(f"Overall Label Concordance: {concordance:.2%}")

    # Drug Stats
    print("\n6. Analyzing Per-Drug Correlation...")
    drug_stats = []
    for drug_norm, group in merged.groupby('join_drug'):
        if len(group) < 10: continue 
        conc = (group['sanger_label'] == group['depmap_label']).mean()
        drug_stats.append({
            'Drug': group['drug_x'].iloc[0],
            'Concordance': conc,
            'N_Pairs': len(group)
        })
    
    df_drugs = pd.DataFrame(drug_stats).sort_values('Concordance', ascending=False)
    out_file = os.path.join(args.output_dir, "drug_concordance.csv")
    df_drugs.to_csv(out_file, index=False)
    
    good_drugs = df_drugs[df_drugs['Concordance'] > 0.6]
    print(f"\n--- RECOMMENDATION ---")
    print(f"Reliable Drugs (>60% Agreement): {len(good_drugs)} / {len(df_drugs)}")
    print(f"Stats saved to {out_file}")

if __name__ == "__main__":
    main()