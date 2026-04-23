import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import argparse
import time
import os

def get_smiles(query_name):
    """Try to get isomeric SMILES, fall back to canonical."""
    try:
        # Search by name
        c = pcp.get_compounds(query_name, 'name')
        if c:
            return getattr(c[0], 'isomeric_smiles', None) or c[0].canonical_smiles
    except:
        pass
    return None

def main():
    parser = argparse.ArgumentParser()
    # CHANGED: Now points to the actual response data
    parser.add_argument('--input_csv', type=str, required=True, 
                        help='Path to PANCANCER_IC...csv')
    parser.add_argument('--output_csv', type=str, default='data/sanger_validation/unique_drugs_combined.csv')
    args = parser.parse_args()

    print(f"Loading drugs from {args.input_csv}...")
    # Load entire file
    df = pd.read_csv(args.input_csv, low_memory=False)
    
    # Check for correct columns (User provided: 'Drug Name', 'Drug ID')
    if 'Drug Name' not in df.columns or 'Drug ID' not in df.columns:
        print(f"Error: File must contain 'Drug Name' and 'Drug ID'. Found: {df.columns.tolist()}")
        return

    # Extract unique pairs
    print("Extracting unique drugs...")
    unique_drugs = df[['Drug Name', 'Drug ID']].drop_duplicates()
    
    # Rename for consistency with downstream scripts
    unique_drugs = unique_drugs.rename(columns={'Drug Name': 'DRUG_NAME', 'Drug ID': 'DRUG_ID'})
    
    # Sort for tidiness
    unique_drugs = unique_drugs.sort_values('DRUG_ID')
    
    print(f"Processing {len(unique_drugs)} unique drugs found in response data...")

    smiles_list = []
    
    # Loop with progress bar
    for idx, row in tqdm(unique_drugs.iterrows(), total=len(unique_drugs)):
        did = row['DRUG_ID']
        name = str(row['DRUG_NAME']).strip()
        
        # 1. Try Main Name
        smiles = get_smiles(name)
        
        # 2. Append Result
        smiles_list.append({
            'DRUG_ID': did,
            'DRUG_NAME': name,
            'SMILES': smiles
        })
        
        # Polite delay for API
        time.sleep(0.1)

    # Save
    df_out = pd.DataFrame(smiles_list)
    found = df_out['SMILES'].notna().sum()
    print(f"Found SMILES for {found} / {len(df_out)} drugs.")
    
    # Save only valid ones for the next step
    df_final = df_out.dropna(subset=['SMILES'])
    
    # Create output dir if needed
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_final.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")

if __name__ == "__main__":
    main()