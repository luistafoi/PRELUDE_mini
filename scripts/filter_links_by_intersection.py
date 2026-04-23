# scripts/filter_links_by_intersection.py

"""
Filters the curated raw link files to include only the intersection of drugs
that are present in both the cell-drug and gene-drug datasets.

This script performs the following steps:
1.  Builds a canonical map to resolve all drug name aliases to a single
    consistent name based on a shared SMILES string.
2.  Identifies the set of canonical drug names that have links in BOTH the
    cell-drug and gene-drug datasets.
3.  Filters the raw link files to keep only interactions involving these
    well-characterized, overlapping drugs.
4.  Saves the final, filtered raw link files, which are then ready for
    the `build_graph_files.py` script.
"""

import pandas as pd
from collections import defaultdict
import argparse
import os

def filter_links(args):
    """Main function to filter links based on the drug intersection."""
    
    # --- Step 1: Build the Canonical Drug Map ---
    print("--- Step 1: Building canonical drug map from all sources ---")
    try:
        df_smiles = pd.read_csv(args.smiles_map_file)
        df_smiles.dropna(subset=['drug_name', 'SMILES'], inplace=True)
        
        smiles_to_names_map = defaultdict(set)
        for _, row in df_smiles.iterrows():
            smiles_to_names_map[row['SMILES']].add(row['drug_name'].upper())

        alias_to_canonical_map = {}
        for smiles, names in smiles_to_names_map.items():
            if not names: continue
            # Rule: Choose the shortest, non-numeric name as the canonical one
            canonical_name = min([name for name in names if not name.isnumeric()], key=len, default=min(names, key=len))
            for alias in names:
                alias_to_canonical_map[alias] = canonical_name
        
        print(f"  > Created a map to resolve {len(alias_to_canonical_map)} unique drug aliases.")

    except FileNotFoundError:
        print(f"  > Error: SMILES map file not found at {args.smiles_map_file}. Aborting.")
        return

    # --- Step 2: Identify the Overlapping Drug Set ---
    print("\n--- Step 2: Identifying the set of overlapping drugs ---")
    try:
        # Load cell-drug links and find their canonical names
        df_cell_drug = pd.read_csv(args.cell_drug_file, sep='\t', header=None, names=['cell', 'drug', 'type', 'label'])
        cell_drug_canonical_names = set(df_cell_drug['drug'].str.upper().map(alias_to_canonical_map).dropna())
        print(f"  > Found {len(cell_drug_canonical_names)} unique canonical drugs in the cell-drug file.")

        # Load gene-drug links and find their canonical names
        df_gene_drug = pd.read_csv(args.gene_drug_file, sep='\t', header=None, names=['gene', 'drug', 'weight'])
        gene_drug_canonical_names = set(df_gene_drug['drug'].str.upper().map(alias_to_canonical_map).dropna())
        print(f"  > Found {len(gene_drug_canonical_names)} unique canonical drugs in the gene-drug file.")

        # Find the intersection
        overlapping_drugs = cell_drug_canonical_names.intersection(gene_drug_canonical_names)
        print(f"  > Found {len(overlapping_drugs)} drugs in the intersection. These will be kept.")

    except FileNotFoundError as e:
        print(f"  > Error: A required link file was not found. Aborting. {e}")
        return

    # --- Step 3: Filter and Save the Link Files ---
    print("\n--- Step 3: Filtering and saving the final link files ---")
    
    # Filter cell-drug links
    df_cell_drug['canonical_drug'] = df_cell_drug['drug'].str.upper().map(alias_to_canonical_map)
    df_cell_drug_filtered = df_cell_drug[df_cell_drug['canonical_drug'].isin(overlapping_drugs)]
    # Use the canonical name in the final output
    output_cd = df_cell_drug_filtered[['cell', 'canonical_drug', 'type', 'label']]
    output_cd.to_csv(args.output_cell_drug, sep='\t', index=False, header=False)
    print(f"  > Saved {len(output_cd)} filtered cell-drug links to {args.output_cell_drug}")

    # Filter gene-drug links
    df_gene_drug['canonical_drug'] = df_gene_drug['drug'].str.upper().map(alias_to_canonical_map)
    df_gene_drug_filtered = df_gene_drug[df_gene_drug['canonical_drug'].isin(overlapping_drugs)]
    # Use the canonical name in the final output
    output_gd = df_gene_drug_filtered[['gene', 'canonical_drug', 'weight']]
    output_gd.to_csv(args.output_gene_drug, sep='\t', index=False, header=False)
    print(f"  > Saved {len(output_gd)} filtered gene-drug links to {args.output_gene_drug}")

    print("\n Filtering complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter raw link files to the intersection of well-characterized drugs.")
    parser.add_argument('--smiles-map-file', default='data/misc/unique_drugs_combined.csv',
                        help='Path to the file mapping drug names to SMILES strings.')
    parser.add_argument('--cell-drug-file', default='data/raw/link_cell_drug_labeled.txt',
                        help='Path to the curated (GMM-labeled) cell-drug link file.')
    parser.add_argument('--gene-drug-file', default='data/raw/link_gene_drug_relation.txt',
                        help='Path to the curated (inhibitory) gene-drug link file.')
    
    # Define output paths for the new, filtered files
    parser.add_argument('--output-cell-drug', default='data/raw/link_cell_drug_filtered.txt',
                        help='Path to save the final, filtered cell-drug links.')
    parser.add_argument('--output-gene-drug', default='data/raw/link_gene_drug_filtered.txt',
                        help='Path to save the final, filtered gene-drug links.')
    
    args = parser.parse_args()
    filter_links(args)
