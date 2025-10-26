# scripts/create_splits.py

import pandas as pd
import numpy as np
import os
import random
import sys # Added for path checks
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
# TEST_TRANSDUCTIVE_FILE removed

# Split Ratios
VALID_CELL_RATIO = 0.10  # 10% of applicable cells for inductive validation
TEST_CELL_RATIO = 0.10   # 10% of applicable cells for inductive test
# TEST_1_LINK_RATIO removed

# Constants from info.dat (ensure these match your data)
CELL_TYPE_ID = 0
DRUG_TYPE_ID = 1
GENE_TYPE_ID = 2
CELL_DRUG_LINK_TYPE = 0 # Example: cell -> drug
# Add other link types if needed for structural graph
CELL_GENE_LINK_TYPE = 2 # Example: cell -> gene
DRUG_GENE_LINK_TYPE = 4 # Example: drug -> gene
GENE_GENE_LINK_TYPE = 6 # Example: gene -> gene

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Inductive Data Splitting for PRELUDE_mini ---")

    # --- Step 0: Check Input Files ---
    if not os.path.exists(LINK_DAT_FILE):
        print(f"FATAL ERROR: link.dat not found at {LINK_DAT_FILE}. Run build_graph_files.py first.")
        sys.exit(1)
    if not os.path.exists(NODE_DAT_FILE):
        print(f"FATAL ERROR: node.dat not found at {NODE_DAT_FILE}. Run build_graph_files.py first.")
        sys.exit(1)

    # --- Step 1: Load Data and Identify Candidate Cells ---
    print(" > Loading graph data...")
    # Load all links first
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
    with open(NODE_DAT_FILE, 'r') as f:
        for line in f:
            nid, _, ntype = line.strip().split('\t')
            node_types[int(nid)] = int(ntype)

    # Identify cells involved in cell-drug links
    df_cd_links = df_links[df_links['type'] == CELL_DRUG_LINK_TYPE].copy()
    candidate_cells = list(pd.unique(df_cd_links['src'])) # Assuming src is always cell for type 0
    print(f" > Found {len(candidate_cells)} cells involved in cell-drug links.")

    # --- Step 2: Create Inductive Splits (Validation and Test) ---
    print("\n--- Step 2: Creating inductive splits from candidate cells ---")
    random.shuffle(candidate_cells)

    num_valid_cells = int(len(candidate_cells) * VALID_CELL_RATIO)
    num_test_cells = int(len(candidate_cells) * TEST_CELL_RATIO)

    valid_cell_set = set(candidate_cells[:num_valid_cells])
    test_cell_set = set(candidate_cells[num_valid_cells : num_valid_cells + num_test_cells])
    train_cell_set = set(candidate_cells[num_valid_cells + num_test_cells:]) # Remaining cells

    # Get the corresponding links
    valid_inductive_links = df_cd_links[df_cd_links['src'].isin(valid_cell_set)]
    test_inductive_links = df_cd_links[df_cd_links['src'].isin(test_cell_set)]

    # Save inductive splits (only src, tgt needed usually)
    valid_inductive_links[['src', 'tgt']].to_csv(VALID_INDUCTIVE_FILE, sep='\t', index=False, header=False)
    test_inductive_links[['src', 'tgt']].to_csv(TEST_INDUCTIVE_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(valid_inductive_links)} links for {len(valid_cell_set)} validation cells to {VALID_INDUCTIVE_FILE}")
    print(f"  > Saved {len(test_inductive_links)} links for {len(test_cell_set)} test cells to {TEST_INDUCTIVE_FILE}")

    # --- Step 3: Create Final Training Files ---
    print("\n--- Step 3: Creating final training files ---")

    # 3a. Create train_lp_links.dat (Supervised LP Training Targets)
    # These are the cell-drug links involving ONLY the training cells
    train_lp_links = df_cd_links[df_cd_links['src'].isin(train_cell_set)]
    train_lp_links[['src', 'tgt']].to_csv(TRAIN_LP_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(train_lp_links)} links for the link prediction training task to {TRAIN_LP_FILE}")

    # 3b. Create train.dat (Structural Graph for Message Passing)
    # This includes:
    #   - All non-cell-drug links (gene-gene, cell-gene, drug-gene)
    #   - The cell-drug links used for LP training (train_lp_links)
    # Why? The GNN needs the full context around the training nodes.
    
    # Get all links EXCEPT the inductive validation and test cell-drug links
    links_to_keep_mask = ~df_links['src'].isin(valid_cell_set | test_cell_set) | (df_links['type'] != CELL_DRUG_LINK_TYPE)
    final_train_graph_df = df_links[links_to_keep_mask]

    # Save the structural graph file
    final_train_graph_df.to_csv(TRAIN_GRAPH_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(final_train_graph_df)} links for the GNN training graph structure to {TRAIN_GRAPH_FILE}")

    # --- Step 4: Transductive Test File (Removed) ---
    # print("\n--- Step 4: Transductive Split (Test 1 - Removed) ---")
    # The logic for creating TEST_TRANSDUCTIVE_FILE is removed.

    print("\nInductive data splitting complete for PRELUDE_mini.")