# scripts/create_splits.py

import pandas as pd
import numpy as np
import os
import random
import sys
from collections import defaultdict

# --- Configuration ---
PROCESSED_DIR = "data/processed"
LINK_DAT_FILE = os.path.join(PROCESSED_DIR, "link.dat")
NODE_DAT_FILE = os.path.join(PROCESSED_DIR, "node.dat")

# Output files
TRAIN_GRAPH_FILE = os.path.join(PROCESSED_DIR, "train.dat") # Structural graph
TRAIN_LP_FILE = os.path.join(PROCESSED_DIR, "train_lp_links.dat") # Supervised LP targets
VALID_INDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "valid_inductive_links.dat")
TEST_INDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "test_inductive_links.dat")

# Split Ratios
VALID_RATIO = 0.10  # 10% of applicable nodes for inductive validation
TEST_RATIO = 0.10   # 10% of applicable nodes for inductive test

# Constants from info.dat (ensure these match your data)
CELL_TYPE_ID = 0
DRUG_TYPE_ID = 1
GENE_TYPE_ID = 2
CELL_DRUG_LINK_TYPE = 0 # From your log: '0': ['cell', 'drug', ...]

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Inductive Data Splitting for PRELUDE_mini (Corrected) ---")

    # --- Step 0: Check Input Files ---
    if not os.path.exists(LINK_DAT_FILE):
        print(f"FATAL ERROR: link.dat not found at {LINK_DAT_FILE}. Run build_graph_files.py first.")
        sys.exit(1)
    if not os.path.exists(NODE_DAT_FILE):
        print(f"FATAL ERROR: node.dat not found at {NODE_DAT_FILE}. Run build_graph_files.py first.")
        sys.exit(1)

    # --- Step 1: Load Data and Identify Candidate Nodes ---
    print(" > Loading graph data...")
    all_links_list = []
    with open(LINK_DAT_FILE, 'r') as f:
        for line in f:
            try:
                src, tgt, rtype, weight = line.strip().split('\t')
                all_links_list.append({'src': int(src), 'tgt': int(tgt), 'type': int(rtype), 'weight': float(weight)})
            except ValueError:
                print(f"Warning: Skipping malformed line in link.dat: {line.strip()}")
    
    df_links = pd.DataFrame(all_links_list)

    # Load node types
    node_types = {}
    all_gene_nodes = set()
    with open(NODE_DAT_FILE, 'r') as f:
        for line in f:
            nid, _, ntype = line.strip().split('\t')
            nid, ntype = int(nid), int(ntype)
            node_types[nid] = ntype
            if ntype == GENE_TYPE_ID:
                all_gene_nodes.add(nid)

    # Identify all cells and drugs involved in cell-drug links
    df_cd_links = df_links[df_links['type'] == CELL_DRUG_LINK_TYPE].copy()
    
    candidate_cells = sorted(list(pd.unique(df_cd_links['src']))) # Assuming src is always cell
    candidate_drugs = sorted(list(pd.unique(df_cd_links['tgt']))) # Assuming tgt is always drug
    
    random.shuffle(candidate_cells)
    random.shuffle(candidate_drugs)

    print(f" > Found {len(candidate_cells)} cells involved in cell-drug links.")
    print(f" > Found {len(candidate_drugs)} drugs involved in cell-drug links.")
    print(f" > Found {len(all_gene_nodes)} gene nodes (will be shared context).")


    # --- Step 2: Create Inductive Node Splits for BOTH Cells and Drugs ---
    print("\n--- Step 2: Creating inductive splits for BOTH cells and drugs ---")
    
    # Split Cells
    num_valid_cells = int(len(candidate_cells) * VALID_RATIO)
    num_test_cells = int(len(candidate_cells) * TEST_RATIO)
    valid_cell_set = set(candidate_cells[:num_valid_cells])
    test_cell_set = set(candidate_cells[num_valid_cells : num_valid_cells + num_test_cells])
    train_cell_set = set(candidate_cells[num_valid_cells + num_test_cells:])

    # Split Drugs
    num_valid_drugs = int(len(candidate_drugs) * VALID_RATIO)
    num_test_drugs = int(len(candidate_drugs) * TEST_RATIO)
    valid_drug_set = set(candidate_drugs[:num_valid_drugs])
    test_drug_set = set(candidate_drugs[num_valid_drugs : num_valid_drugs + num_test_drugs])
    train_drug_set = set(candidate_drugs[num_valid_drugs + num_test_drugs:])

    print(f"  > Train Cells: {len(train_cell_set)} | Valid Cells: {len(valid_cell_set)} | Test Cells: {len(test_cell_set)}")
    print(f"  > Train Drugs: {len(train_drug_set)} | Valid Drugs: {len(valid_drug_set)} | Test Drugs: {len(test_drug_set)}")

    # --- Step 3: Create Final Link Files Based on Node Splits ---
    print("\n--- Step 3: Creating final link files based on node splits ---")

    # 3a. Create train_lp_links.dat (Train-Train)
    train_lp_links = df_cd_links[
        df_cd_links['src'].isin(train_cell_set) & 
        df_cd_links['tgt'].isin(train_drug_set)
    ]
    train_lp_links[['src', 'tgt', 'weight']].to_csv(TRAIN_LP_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(train_lp_links)} links for (Train Cell, Train Drug) to {TRAIN_LP_FILE}")

    # 3b. Create valid_inductive_links.dat (Valid-Valid)
    valid_inductive_links = df_cd_links[
        df_cd_links['src'].isin(valid_cell_set) & 
        df_cd_links['tgt'].isin(valid_drug_set)
    ]
    valid_inductive_links[['src', 'tgt', 'weight']].to_csv(VALID_INDUCTIVE_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(valid_inductive_links)} links for (Valid Cell, Valid Drug) to {VALID_INDUCTIVE_FILE}")

    # 3c. Create test_inductive_links.dat (Test-Test)
    test_inductive_links = df_cd_links[
        df_cd_links['src'].isin(test_cell_set) & 
        df_cd_links['tgt'].isin(test_drug_set)
    ]
    test_inductive_links[['src', 'tgt', 'weight']].to_csv(TEST_INDUCTIVE_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(test_inductive_links)} links for (Test Cell, Test Drug) to {TEST_INDUCTIVE_FILE}")

    # 3d. Create train.dat (Structural Graph for Message Passing)
    # This includes ALL links (C-G, D-G, G-G, C-D) where BOTH
    # nodes are in the training sets (train_cells, train_drugs, all_genes).
    
    training_nodes_set = train_cell_set.union(train_drug_set).union(all_gene_nodes)
    print(f"  > Total nodes in training graph (train_cells + train_drugs + all_genes): {len(training_nodes_set)}")

    # Filter all links to keep only those that connect two training nodes
    final_train_graph_df = df_links[
        df_links['src'].isin(training_nodes_set) & 
        df_links['tgt'].isin(training_nodes_set)
    ]

    # Save the structural graph file
    final_train_graph_df.to_csv(TRAIN_GRAPH_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(final_train_graph_df)} links for the GNN training graph structure to {TRAIN_GRAPH_FILE}")


    print("\nInductive data splitting complete.")
