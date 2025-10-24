# scripts/curate_raw_links.py

"""
Consolidates all raw data curation steps into a single script.

This script performs the following steps:
1.  Curates cell-gene links from somatic mutation data based on 'LikelyLoF'.
2.  Curates gene-drug links from DGIdb based on inhibitory interaction types.
3.  Processes gene-gene interaction links.
4.  Saves the final, curated raw link files to the `data/raw` directory,
    ready for the `build_graph_files.py` script.
"""

import pandas as pd
import re
from tqdm import tqdm
import argparse
import os

# --- Curation Functions ---

def curate_cell_gene_links(args):
    """Generates cell-gene links from mutations, converting Entrez IDs to gene names."""
    print("\n--- Curating Cell-Gene Links (from Mutations) ---")
    
    mutation_file = os.path.join(args.misc_dir, "OmicsSomaticMutations_24Q2.csv")
    expression_file = os.path.join(args.misc_dir, "24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
    output_file = os.path.join(args.raw_dir, "link_cell_gene_mutation.txt")

    try:
        print(f"  > Building gene map from: {expression_file}")
        df_expr_cols = pd.read_csv(expression_file, nrows=0).columns
        entrez_to_name_map = {}
        for col in df_expr_cols[1:]:
            match = re.search(r'^(.*) \((\d+)\)$', col)
            if match:
                entrez_to_name_map[int(match.group(2))] = match.group(1)
        
        print(f"  > Reading mutation data from: {mutation_file}")
        df_mutation = pd.read_csv(mutation_file, low_memory=False)
        df_filtered = df_mutation[['ModelID', 'EntrezGeneID', 'LikelyLoF']].copy()
        df_filtered.dropna(inplace=True)
        df_filtered['EntrezGeneID'] = df_filtered['EntrezGeneID'].astype(int)

        df_filtered['gene_name'] = df_filtered['EntrezGeneID'].map(entrez_to_name_map)
        df_filtered.dropna(subset=['gene_name'], inplace=True)
        df_filtered.drop_duplicates(subset=['ModelID', 'gene_name'], inplace=True)
        
        df_filtered['weight'] = df_filtered['LikelyLoF'].astype(int)
        output_df = df_filtered[['ModelID', 'gene_name', 'weight']]

        output_df.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"Saved {len(output_df)} curated cell-gene links to: {output_file}")

    except FileNotFoundError as e:
        print(f"  > Error: File not found, skipping cell-gene link creation. {e}")

def curate_gene_drug_links(args):
    """Curates gene-drug links from DGIdb, filtering for inhibitory types."""
    print("\n--- Curating Gene-Drug Links (from DGIdb) ---")
    
    dgidb_file = os.path.join(args.misc_dir, "DGIdb_Interactions_Enriched_v2.csv")
    output_file = os.path.join(args.raw_dir, "link_gene_drug_relation.txt")
    
    INHIBITORY_TYPES = ['inhibitor', 'blocker', 'negative modulator', 'inverse agonist']

    try:
        print(f"  > Reading DGIdb data from: {dgidb_file}")
        df = pd.read_csv(dgidb_file)
        
        print(f"  > Filtering for inhibitory interaction types...")
        df_filtered = df[df['interaction_type'].isin(INHIBITORY_TYPES)].copy()
        df_filtered.dropna(subset=['gene_name', 'drug_name'], inplace=True)
        
        df_filtered['weight'] = 1
        output_df = df_filtered[['gene_name', 'drug_name', 'weight']]

        output_df.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"Saved {len(output_df)} curated gene-drug links to: {output_file}")

    except FileNotFoundError as e:
        print(f"  > Error: File not found, skipping gene-drug link creation. {e}")

def process_gene_gene_links(args):
    """Processes gene-gene links, creating bidirectional edges."""
    print("\n--- Processing Gene-Gene Links ---")
    
    gene_gene_file = os.path.join(args.misc_dir, "GeneGene_Interactions_EntrezMapped_Filtered.csv")
    output_file = os.path.join(args.raw_dir, "link_gene_gene.txt")

    try:
        print(f"  > Reading gene-gene data from: {gene_gene_file}")
        df = pd.read_csv(gene_gene_file)
        df.dropna(subset=["Gene1", "Gene2"], inplace=True)

        link_lines = []
        for _, row in df.iterrows():
            # The weight is 1, and the type will be assigned by the build script
            link_lines.append(f"{row['Gene1']}\t{row['Gene2']}\t1")
            link_lines.append(f"{row['Gene2']}\t{row['Gene1']}\t1")

        with open(output_file, "w") as f:
            f.write("\n".join(link_lines))
        
        print(f"Saved {len(link_lines)} bidirectional gene-gene links to: {output_file}")

    except FileNotFoundError as e:
        print(f"  > Error: File not found, skipping gene-gene link creation. {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Curate all raw link files for the PRELUDE graph.")
    parser.add_argument('--misc-dir', default='data/misc',
                        help='Directory containing the original source data files.')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory to save the curated, raw link files.')
    
    args = parser.parse_args()
    
    os.makedirs(args.raw_dir, exist_ok=True)
    
    curate_cell_gene_links(args)
    curate_gene_drug_links(args)
    process_gene_gene_links(args)
    
    print("\nAll raw link files have been curated.")
