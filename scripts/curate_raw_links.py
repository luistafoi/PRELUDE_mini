# scripts/curate_raw_links.py

import pandas as pd
import re
from tqdm import tqdm
import argparse
import os
import sys # Added for path checks

# --- Curation Functions ---

def curate_cell_gene_links(args):
    """Generates cell-gene links from mutations, converting Entrez IDs to gene names."""
    print("\n--- Curating Cell-Gene Links (from Mutations) ---")
    
    mutation_file = os.path.join(args.misc_dir, "OmicsSomaticMutations_24Q2.csv")
    expression_file = os.path.join(args.misc_dir, "24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
    output_file = os.path.join(args.raw_dir, "link_cell_gene_mutation.txt")

    # Check existence of input files
    if not os.path.exists(mutation_file):
        print(f"  > Error: Mutation file not found at {mutation_file}. Skipping cell-gene link creation.")
        return
    if not os.path.exists(expression_file):
        print(f"  > Error: Expression file not found at {expression_file}. Skipping cell-gene link creation.")
        return

    try:
        print(f"  > Building gene map from: {expression_file}")
        # Use pandas to read just the header quickly
        df_expr_cols = pd.read_csv(expression_file, nrows=0).columns
        entrez_to_name_map = {}
        # Iterate from the second column (index 1) onwards
        for col in df_expr_cols[1:]:
            match = re.search(r'^(.*) \((\d+)\)$', col)
            if match:
                # Store Entrez ID (int) -> Gene Name (str)
                entrez_to_name_map[int(match.group(2))] = match.group(1)
        
        if not entrez_to_name_map:
             print("  > Warning: Could not extract gene names/Entrez IDs from expression file header.")
             # Decide how to handle this - maybe proceed without mapping? For now, we'll continue.

        print(f"  > Reading mutation data from: {mutation_file}")
        # Read only necessary columns to save memory
        df_mutation = pd.read_csv(mutation_file, usecols=['ModelID', 'EntrezGeneID', 'LikelyLoF'], low_memory=False)
        
        # Filter for LikelyLoF mutations and drop rows with missing values in key columns
        df_filtered = df_mutation[df_mutation['LikelyLoF'] == True].copy()
        df_filtered.dropna(subset=['ModelID', 'EntrezGeneID'], inplace=True)
        
        # Ensure EntrezGeneID is integer type for mapping
        df_filtered['EntrezGeneID'] = df_filtered['EntrezGeneID'].astype(int)

        # Map Entrez IDs to gene names
        df_filtered['gene_name'] = df_filtered['EntrezGeneID'].map(entrez_to_name_map)
        
        # Drop rows where mapping failed (gene not in expression file header)
        df_filtered.dropna(subset=['gene_name'], inplace=True)
        
        # Keep only unique cell-gene pairs (ignore multiple mutations in same gene)
        df_filtered.drop_duplicates(subset=['ModelID', 'gene_name'], inplace=True)
        
        # Assign weight (could be based on LikelyLoF if needed, here just 1)
        df_filtered['weight'] = 1
        output_df = df_filtered[['ModelID', 'gene_name', 'weight']]

        # Save the curated links
        output_df.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"  > Saved {len(output_df)} curated cell-gene links to: {output_file}")

    except Exception as e:
        print(f"  > An unexpected error occurred during cell-gene link creation: {e}")


def curate_gene_drug_links(args):
    """Curates gene-drug links from DGIdb, filtering for inhibitory types."""
    print("\n--- Curating Gene-Drug Links (from DGIdb) ---")
    
    dgidb_file = os.path.join(args.misc_dir, "DGIdb_Interactions_Enriched_v2.csv")
    output_file = os.path.join(args.raw_dir, "link_gene_drug_relation.txt") # Renamed for clarity
    
    # Define inhibitory types (case-insensitive matching is good practice)
    INHIBITORY_TYPES = {'inhibitor', 'blocker', 'negative modulator', 'inverse agonist'}

    if not os.path.exists(dgidb_file):
        print(f"  > Error: DGIdb file not found at {dgidb_file}. Skipping gene-drug link creation.")
        return

    try:
        print(f"  > Reading DGIdb data from: {dgidb_file}")
        # Read only necessary columns
        df = pd.read_csv(dgidb_file, usecols=['gene_name', 'drug_name', 'interaction_type'])        
        # Pre-process interaction types: lowercase and split if multiple types exist per row
        df.dropna(subset=['interaction_types'], inplace=True) # Drop rows missing interaction types
        df['interaction_list'] = df['interaction_type'].str.lower().str.split(',')        
        # Filter rows where at least one interaction type is inhibitory
        print(f"  > Filtering for inhibitory interaction types...")
        mask = df['interaction_list'].apply(lambda types: any(t.strip() in INHIBITORY_TYPES for t in types))
        df_filtered = df[mask].copy()
        
        # Clean up gene and drug names and remove duplicates
        df_filtered.dropna(subset=['gene_name', 'drug_name'], inplace=True)
        df_filtered['gene_name'] = df_filtered['gene_name'].str.strip()
        df_filtered['drug_name'] = df_filtered['drug_name'].str.strip().str.upper() # Standardize drug names to upper
        df_filtered.drop_duplicates(subset=['gene_name', 'drug_name'], inplace=True)

        # Assign weight
        df_filtered['weight'] = 1
        output_df = df_filtered[['gene_name', 'drug_name', 'weight']]

        # Save curated links
        output_df.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"  > Saved {len(output_df)} curated gene-drug links to: {output_file}")

    except Exception as e:
        print(f"  > An unexpected error occurred during gene-drug link creation: {e}")


def process_gene_gene_links(args):
    """Processes gene-gene links, creating bidirectional edges with weight 1."""
    print("\n--- Processing Gene-Gene Links ---")
    
    gene_gene_file = os.path.join(args.misc_dir, "GeneGene_Interactions_EntrezMapped_Filtered.csv")
    output_file = os.path.join(args.raw_dir, "link_gene_gene.txt")

    if not os.path.exists(gene_gene_file):
        print(f"  > Error: Gene-gene file not found at {gene_gene_file}. Skipping gene-gene link creation.")
        return
        
    try:
        print(f"  > Reading gene-gene data from: {gene_gene_file}")
        # Assuming header is Gene1, Gene2, Weight or similar
        df = pd.read_csv(gene_gene_file) 
        # Ensure correct column names if different
        df.rename(columns={'Gene1': 'gene1', 'Gene2': 'gene2'}, inplace=True, errors='ignore') 
        df.dropna(subset=["gene1", "gene2"], inplace=True)
        
        # Clean names (strip whitespace)
        df['gene1'] = df['gene1'].astype(str).str.strip()
        df['gene2'] = df['gene2'].astype(str).str.strip()

        link_lines = set() # Use a set to automatically handle duplicates and self-loops if present
        print(f"  > Creating bidirectional links...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  - Processing"):
            g1, g2 = row['gene1'], row['gene2']
            if g1 != g2: # Avoid self-loops if not desired
                # Add forward and backward link with weight 1
                link_lines.add(f"{g1}\t{g2}\t1")
                link_lines.add(f"{g2}\t{g1}\t1")

        with open(output_file, "w") as f:
            # Sort for consistent output (optional)
            f.write("\n".join(sorted(list(link_lines))))
        
        print(f"  > Saved {len(link_lines)} unique, bidirectional gene-gene links to: {output_file}")

    except Exception as e:
        print(f"  > An unexpected error occurred during gene-gene link creation: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Curate all raw link files for the PRELUDE graph.")
    # Use consistent naming for directories
    parser.add_argument('--misc-dir', default='data/misc',
                        help='Directory containing the original source data files (e.g., CSVs).')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory to save the curated, raw link files (tab-separated text).')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.raw_dir, exist_ok=True)
    
    print("Starting raw link file curation process...")
    curate_cell_gene_links(args)
    curate_gene_drug_links(args)
    process_gene_gene_links(args)
    
    print("\n All raw link files have been curated.")