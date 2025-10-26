# scripts/build_graph_files.py

"""
Constructs the final graph files (node.dat, link.dat, info.dat) and a node
mapping file.

This version performs a critical feature-validation step:
1.  It first loads whitelists of all cells, drugs, and genes that have
    pre-computed embeddings.
2.  It then builds the graph, ADDING ONLY nodes that are present in these
    whitelists and links that connect two valid nodes.
This fixes errors from nodes existing in the graph but missing from feature files.
"""
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import os
import argparse
import pickle
import numpy as np
import sys

def _load_feature_whitelists(args):
    """
    Loads all embedding files to create sets of valid node names.
    This assumes cell_vae.py has been run to create its outputs.
    """
    print("--- Loading Feature Whitelists (to build clean graph) ---")
    
    # 1. Cell Whitelist
    valid_cells = set()
    cell_names_path = "data/embeddings/final_vae_cell_names.txt" # From cell_vae.py
    if not os.path.exists(cell_names_path):
        print(f"FATAL ERROR: Cell names file not found at {cell_names_path}")
        print("Please run scripts/cell_vae.py first.")
        sys.exit(1)
    with open(cell_names_path, 'r') as f:
        valid_cells = {line.strip().upper() for line in f if line.strip()}
    print(f"  > Loaded {len(valid_cells)} valid cell names.")

    # 2. Drug Whitelist
    valid_drugs = set()
    drug_file_path = "data/embeddings/drugs_with_embeddings.csv"
    if not os.path.exists(drug_file_path):
        print(f"FATAL ERROR: Drug embedding file not found at {drug_file_path}")
        sys.exit(1)
    df_drugs = pd.read_csv(drug_file_path)
    # Assumes first column is the drug name
    drug_col_name = df_drugs.columns[0]
    valid_drugs = set(df_drugs[drug_col_name].str.strip().str.upper())
    print(f"  > Loaded {len(valid_drugs)} valid drug names.")

    # 3. Gene Whitelist
    valid_genes = set()
    gene_file_path = "data/embeddings/gene_embeddings_esm_by_symbol.pkl"
    if not os.path.exists(gene_file_path):
        print(f"FATAL ERROR: Gene embedding file not found at {gene_file_path}")
        sys.exit(1)
    with open(gene_file_path, 'rb') as f:
        gene_dict = pickle.load(f)
        valid_genes = {str(k).upper() for k in gene_dict.keys()}
    print(f"  > Loaded {len(valid_genes)} valid gene names.")
    
    return valid_cells, valid_drugs, valid_genes

def main(args):
    # --- Load Feature Whitelists FIRST ---
    valid_cells, valid_drugs, valid_genes = _load_feature_whitelists(args)

    # Helper function to check if a node is valid
    def is_valid_node(node_name, n_type):
        if n_type == 0: # Cell
            return node_name.upper() in valid_cells
        elif n_type == 1: # Drug
            return node_name.upper() in valid_drugs
        elif n_type == 2: # Gene
            return node_name.upper() in valid_genes
        return False

    # --- Configuration ---
    link_paths = {
        "cell-drug": args.cell_drug_file,
        "gene-drug": args.gene_drug_file,
        "gene-cell": os.path.join(args.raw_dir, "link_cell_gene_mutation.txt"),
        "gene-gene": os.path.join(args.raw_dir, "link_gene_gene.txt")
    }
    type_map = {
        "cell-drug": (0, 1, "cell-drug_interaction"),
        "gene-cell": (0, 2, "cell-gene_mutation"), # Swapped from (2,0) to match file? check file format
        "gene-drug": (2, 1, "gene-drug_inhibition"),
        "gene-gene": (2, 2, "gene-interacts_gene")
    }
    
    # --- Initialization ---
    node2id = {}         # map {node_name_standardized: id}
    node_types = {}      # map {node_name_standardized: type_id}
    node_name_final = {} # map {id: node_name_standardized}
    next_id = 0
    all_edges = []
    link_info = {}
    link_type_counter = 0

    print("\n--- Building graph files from raw sources ---")
    for name, path in link_paths.items():
        if not os.path.exists(path):
            print(f"  > Warning: File not found for {name} at {path}. Skipping.")
            continue

        print(f"  > Processing {name} links from {path}...")
        
        df = pd.read_csv(path, sep="\t", header=None)
        if df.shape[1] == 3:
            df.columns = ["src", "tgt", "weight"]
        elif df.shape[1] == 4:
            df.columns = ["src", "tgt", "type_from_file", "weight"]
        else:
            print(f"  > Warning: Skipping file {path} (unexpected columns: {df.shape[1]}).")
            continue

        src_type, tgt_type, relation_name = type_map[name]

        df["src"] = df["src"].astype(str).str.strip()
        df["tgt"] = df["tgt"].astype(str).str.strip()
        # Standardize all names for checking
        df["src_std"] = df["src"].str.upper()
        df["tgt_std"] = df["tgt"].str.upper()
        
        valid_links_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  - {name}"):
            src, tgt = row["src_std"], row["tgt_std"]
            
            # --- FEATURE VALIDATION CHECK ---
            if not (is_valid_node(src, src_type) and is_valid_node(tgt, tgt_type)):
                continue # Skip this link, nodes are missing features

            # --- Node Creation (if new) ---
            for node_name_std, ntype in [(src, src_type), (tgt, tgt_type)]:
                if node_name_std not in node2id:
                    node2id[node_name_std] = next_id
                    node_types[node_name_std] = ntype
                    node_name_final[next_id] = node_name_std # Store final name
                    next_id += 1

            src_id = node2id[src]
            tgt_id = node2id[tgt]
            weight = row["weight"]

            all_edges.append(f"{src_id}\t{tgt_id}\t{link_type_counter}\t{weight}")
            valid_links_count += 1
        
        type_names = {0: "cell", 1: "drug", 2: "gene"}
        link_info[str(link_type_counter)] = [ 
            type_names[src_type],
            type_names[tgt_type],
            relation_name,
            valid_links_count # Store count of *valid* links
        ]
        link_type_counter += 1

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Save node.dat and mappings ---
    node_dat_path = os.path.join(args.output_dir, "node.dat")
    map_path = os.path.join(args.output_dir, "node_mappings.json")
    
    with open(node_dat_path, "w") as f:
        # Use the id->name map to write in correct ID order
        for nid in range(next_id):
            node_name = node_name_final[nid]
            ntype = node_types[node_name]
            f.write(f"{nid}\t{node_name}\t{ntype}\n")
            
    with open(map_path, 'w') as f:
        json.dump(node2id, f, indent=4)

    # --- Save link.dat ---
    link_dat_path = os.path.join(args.output_dir, "link.dat")
    with open(link_dat_path, "w") as f:
        f.write("\n".join(all_edges))

    # --- Save info.dat ---
    type_counts = defaultdict(int)
    for ntype in node_types.values():
        type_counts[ntype] += 1
    info = {
        "dataset": "PRELUDE_CellDrugGene_Network_Filtered", # Updated name
        "node.dat": { "0": ["cell", type_counts[0]], "1": ["drug", type_counts[1]], "2": ["gene", type_counts[2]] },
        "link.dat": link_info
    }
    info_dat_path = os.path.join(args.output_dir, "info.dat")
    with open(info_dat_path, "w") as f:
        json.dump(info, f, indent=4)

    # --- Summary ---
    print("\n--- Summary (Filtered Graph) ---")
    print(f"  > Wrote {len(node2id)} nodes to {node_dat_path}")
    print(f"  > Wrote mapping for {len(node2id)} nodes to {map_path}")
    print(f"  > Wrote {len(all_edges)} links to {link_dat_path}")
    print(f"  > Wrote metadata to {info_dat_path}")
    print("\nGraph construction complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build feature-validated graph .dat files.")
    parser.add_argument('--cell-drug-file', default='data/raw/link_cell_drug_labeled.txt', # Use the GMM output
                        help='Path to the labeled cell-drug link file.')
    parser.add_argument('--gene-drug-file', default='data/raw/link_gene_drug_relation.txt',
                        help='Path to the curated gene-drug link file.')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory containing other curated link files (gene-gene, etc.).')
    parser.add_argument('--output-dir', default='data/processed',
                        help='Directory to save the output .dat files.')
    
    # --- Feature Path Arguments (optional, but good to add) ---
    # Although hardcoded in _load_feature_whitelists, adding them here makes it explicit
    parser.add_argument('--cell-names-file', default='data/embeddings/final_vae_cell_names.txt')
    parser.add_argument('--drug-embed-file', default='data/embeddings/drugs_with_embeddings.csv')
    parser.add_argument('--gene-embed-file', default='data/embeddings/gene_embeddings_esm_by_symbol.pkl')

    args = parser.parse_args()
    main(args)