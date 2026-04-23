import pandas as pd
import os
import re

# --- CONFIG ---
# Paths (Update these to your actual locations)
DEPMAP_FILE = "data/misc/Repurposing_Public_24Q2_LMFI_NORMALIZED_with_DrugNames.csv"
ID_MAP_FILE = "data/misc/model_ids.csv"
SANGER_FILE = "data/misc/PANCANCER_IC_Fri Jan 16 16_28_57 2026.csv"
OUTPUT_DIR = "data/sanger_validation"

def normalize_string(s):
    """
    Removes punctuation and converts to uppercase for safer matching.
    Example: 'KYSE-510' -> 'KYSE510', 'Camptothecin (Topotecan)' -> 'CAMPTOTHECIN'
    """
    if pd.isna(s): return ""
    s = str(s).upper()
    # Remove special chars but keep alphanumeric
    return re.sub(r'[^A-Z0-9]', '', s)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Loading Data...")
    df_depmap = pd.read_csv(DEPMAP_FILE)
    df_map = pd.read_csv(ID_MAP_FILE)
    df_sanger = pd.read_csv(SANGER_FILE)
    
    print(f"   DepMap Rows: {len(df_depmap)}")
    print(f"   Sanger Rows: {len(df_sanger)}")

    # --- STEP 2: DEFINE "KNOWN" UNIVERSE (TRAINING DATA) ---
    print("\n2. Identifying Known Drugs & Cells from Training Data...")
    
    # Clean Drug Names
    train_drugs = set(df_depmap['name'].apply(normalize_string).unique())
    
    # Identify Training Cells via Row ID -> CCLE Name
    # Filter ID map to only include rows present in DepMap data
    valid_row_ids = set(df_depmap['row_id'].unique())
    df_map_filtered = df_map[df_map['row_id'].isin(valid_row_ids)].copy()
    
    # Extract Cell Model Name (e.g., 'KYSE510' from 'KYSE510_OESOPHAGUS')
    # Logic: Split by '_' and take the first part
    df_map_filtered['cell_model_clean'] = df_map_filtered['ccle_name'].apply(
        lambda x: normalize_string(str(x).split('_')[0])
    )
    
    train_cells = set(df_map_filtered['cell_model_clean'].unique())
    
    print(f"   Known Training Drugs: {len(train_drugs)}")
    print(f"   Known Training Cells: {len(train_cells)}")

    # --- STEP 3: PROCESS SANGER DATA ---
    print("\n3. Processing Sanger Data...")
    
    # Clean Names for Matching
    df_sanger['clean_drug'] = df_sanger['Drug Name'].apply(normalize_string)
    df_sanger['clean_cell'] = df_sanger['Cell Line Name'].apply(normalize_string)
    
    # Binarize Response (Create the Label)
    # Logic: Sensitive (1) if IC50 < Max Conc. Resistant (0) otherwise.
    # Note: ln(IC50) is often provided. Check if raw or log. 
    # GDSC usually provides ln(IC50). Max Conc is usually uM.
    # If IC50 is negative (log scale), and Max Conc is positive (uM), we need to be careful.
    # *Looking at your snippet:* Camptothecin IC50 is -1.46 (likely ln(uM)), Max Conc is 0.1 (uM).
    # We need to convert Max Conc to ln(Max Conc) to compare!
    
    import numpy as np
    # Assuming 'Max Conc' is in uM and 'IC50' is ln(IC50) in uM
    df_sanger['max_conc_ln'] = np.log(df_sanger['Max Conc'])
    
    # Label: 1 (Sensitive) if IC50 < Max Conc
    df_sanger['label'] = (df_sanger['IC50'] < df_sanger['max_conc_ln']).astype(int)
    
    print(f"   Sanger Class Balance: {df_sanger['label'].mean():.2%} Sensitive")

    # --- STEP 4: SPLIT INTO SUBSETS ---
    print("\n4. Splitting into 4 Subsets...")
    
    # Flags for overlap
    df_sanger['is_known_drug'] = df_sanger['clean_drug'].isin(train_drugs)
    df_sanger['is_known_cell'] = df_sanger['clean_cell'].isin(train_cells)
    
    # S1: Known Cell, Known Drug
    s1 = df_sanger[df_sanger['is_known_cell'] & df_sanger['is_known_drug']]
    
    # S2: Known Cell, New Drug
    s2 = df_sanger[df_sanger['is_known_cell'] & ~df_sanger['is_known_drug']]
    
    # S3: New Cell, Known Drug (CRITICAL FOR INDUCTIVE TEST)
    s3 = df_sanger[~df_sanger['is_known_cell'] & df_sanger['is_known_drug']]
    
    # S4: New Cell, New Drug
    s4 = df_sanger[~df_sanger['is_known_cell'] & ~df_sanger['is_known_drug']]
    
    # --- STEP 5: SAVE ---
    # Save standard columns: cell_name, drug_name, label, clean_cell, clean_drug
    cols = ['Cell Line Name', 'Drug Name', 'label', 'clean_cell', 'clean_drug', 'IC50', 'AUC']
    
    s1[cols].to_csv(f"{OUTPUT_DIR}/sanger_S1_known_known.csv", index=False)
    s2[cols].to_csv(f"{OUTPUT_DIR}/sanger_S2_known_new_drug.csv", index=False)
    s3[cols].to_csv(f"{OUTPUT_DIR}/sanger_S3_new_cell_known.csv", index=False)
    s4[cols].to_csv(f"{OUTPUT_DIR}/sanger_S4_new_new.csv", index=False)
    
    print("\n--- Summary of Subsets ---")
    print(f"S1 (Known Cell, Known Drug): {len(s1)} pairs")
    print(f"S2 (Known Cell, New Drug):   {len(s2)} pairs")
    print(f"S3 (New Cell, Known Drug):   {len(s3)} pairs  <-- Primary Inductive Test")
    print(f"S4 (New Cell, New Drug):     {len(s4)} pairs")
    print(f"Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()