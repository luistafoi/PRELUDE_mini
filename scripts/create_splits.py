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
NODE_DAT_FILE = os.path.join(PROCESSED_DIR, "node.dat") # Still needed for node types

# --- Output files (Using original names where possible) ---
TRAIN_GRAPH_FILE = os.path.join(PROCESSED_DIR, "train.dat") # Structural graph (contains all nodes)
TRAIN_LP_FILE = os.path.join(PROCESSED_DIR, "train_lp_links.dat") # Supervised LP targets
# Explicitly name validation/test files as transductive
VALID_TRANSDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "valid_transductive_links.dat")
TEST_TRANSDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "test_transductive_links.dat")
# Ensure old inductive files are removed if rerunning pipeline
VALID_INDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "valid_inductive_links.dat")
TEST_INDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "test_inductive_links.dat")


# Split Ratios for LINKS
VALID_LINK_RATIO = 0.10  # 10% of cell-drug links for transductive validation
TEST_LINK_RATIO = 0.10   # 10% of cell-drug links for transductive test
# TRAIN_LINK_RATIO is implicitly 1.0 - VALID_LINK_RATIO - TEST_LINK_RATIO

# Constants from info.dat (ensure these match your data)
CELL_TYPE_ID = 0
DRUG_TYPE_ID = 1
GENE_TYPE_ID = 2
CELL_DRUG_LINK_TYPE = 0 # Example: cell -> drug

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Transductive Data Splitting for PRELUDE_mini ---")
    print("--- Mode: Splitting LINKS, keeping ALL NODES in training graph ---")


    # --- Step 0: Check Input Files ---
    if not os.path.exists(LINK_DAT_FILE):
        print(f"FATAL ERROR: link.dat not found at {LINK_DAT_FILE}. Run build_graph_files.py first.")
        sys.exit(1)
    if not os.path.exists(NODE_DAT_FILE):
        print(f"FATAL ERROR: node.dat not found at {NODE_DAT_FILE}. Run build_graph_files.py first.")
        sys.exit(1)

    # --- Cleanup potentially conflicting old inductive files ---
    if os.path.exists(VALID_INDUCTIVE_FILE):
        print(f" > Removing old inductive file: {VALID_INDUCTIVE_FILE}")
        os.remove(VALID_INDUCTIVE_FILE)
    if os.path.exists(TEST_INDUCTIVE_FILE):
        print(f" > Removing old inductive file: {TEST_INDUCTIVE_FILE}")
        os.remove(TEST_INDUCTIVE_FILE)


    # --- Step 1: Load ALL Links ---
    print(" > Loading ALL graph links from link.dat...")
    all_links_list = []
    cell_drug_links_list = [] # Store cell-drug links separately for splitting
    other_links_list = []     # Store non-cell-drug links

    with open(LINK_DAT_FILE, 'r') as f:
        for line in f:
            try:
                src, tgt, rtype, weight = line.strip().split('\t')
                link_data = {'src': int(src), 'tgt': int(tgt), 'type': int(rtype), 'weight': float(weight)}
                all_links_list.append(link_data) # Keep a full list for reference if needed

                if link_data['type'] == CELL_DRUG_LINK_TYPE:
                    # Store only src, tgt, and maybe weight for the LP task files
                    cell_drug_links_list.append((link_data['src'], link_data['tgt'], link_data['weight']))
                else:
                    other_links_list.append(link_data) # Keep full info for structural graph

            except ValueError:
                print(f"Warning: Skipping malformed line in link.dat: {line.strip()}")

    print(f" > Found {len(cell_drug_links_list)} cell-drug links to split.")
    print(f" > Found {len(other_links_list)} other structural links (gene-gene, etc.).")

    # --- Step 2: Split Cell-Drug LINKS Transductively ---
    print("\n--- Step 2: Splitting cell-drug LINKS into train/validation/test ---")
    random.seed(42) # Ensure reproducible splits
    random.shuffle(cell_drug_links_list)

    num_cd_links = len(cell_drug_links_list)
    num_valid_links = int(num_cd_links * VALID_LINK_RATIO)
    num_test_links = int(num_cd_links * TEST_LINK_RATIO)
    num_train_links = num_cd_links - num_valid_links - num_test_links

    # Assign links to sets
    valid_cd_links = cell_drug_links_list[:num_valid_links]
    test_cd_links = cell_drug_links_list[num_valid_links : num_valid_links + num_test_links]
    train_cd_links_for_lp = cell_drug_links_list[num_valid_links + num_test_links:]

    # Save the split link files (src, tgt only for LP task)
    pd.DataFrame(valid_cd_links, columns=['src', 'tgt', 'weight'])[['src', 'tgt']].to_csv(
        VALID_TRANSDUCTIVE_FILE, sep='\t', index=False, header=False
    )
    pd.DataFrame(test_cd_links, columns=['src', 'tgt', 'weight'])[['src', 'tgt']].to_csv(
        TEST_TRANSDUCTIVE_FILE, sep='\t', index=False, header=False
    )
    pd.DataFrame(train_cd_links_for_lp, columns=['src', 'tgt', 'weight'])[['src', 'tgt']].to_csv(
        TRAIN_LP_FILE, sep='\t', index=False, header=False
    )

    print(f"  > Saved {len(train_cd_links_for_lp)} links for LP training to {TRAIN_LP_FILE}")
    print(f"  > Saved {len(valid_cd_links)} links for transductive validation to {VALID_TRANSDUCTIVE_FILE}")
    print(f"  > Saved {len(test_cd_links)} links for transductive testing to {TEST_TRANSDUCTIVE_FILE}")

    # --- Step 3: Create the Training Graph File (`train.dat`) ---
    # This graph includes ALL nodes and all links EXCEPT the held-out cell-drug links.
    print("\n--- Step 3: Creating the transductive training graph file ---")

    # We need the cell-drug links assigned to training, but with their full data (type, weight)
    # Create a set of (src, tgt, weight) tuples for efficient lookup
    train_lp_link_set = set(train_cd_links_for_lp)

    train_cd_links_for_graph = [
        link for link in all_links_list
        if link['type'] == CELL_DRUG_LINK_TYPE and (link['src'], link['tgt'], link['weight']) in train_lp_link_set
    ]

    # Combine non-cell-drug links with the training cell-drug links
    final_train_graph_links = other_links_list + train_cd_links_for_graph

    # Convert back to DataFrame for easy saving
    df_train_graph = pd.DataFrame(final_train_graph_links)

    # Save the structural graph file (ensure correct columns/order if needed by downstream tools)
    # Saving as src, tgt, type, weight
    df_train_graph[['src', 'tgt', 'type', 'weight']].to_csv(
        TRAIN_GRAPH_FILE, sep='\t', index=False, header=False, float_format='%.6f' # Adjust format if needed
    )

    # We need to know how many nodes are actually in this graph file
    # Get unique nodes involved in the training graph links
    training_graph_nodes = pd.unique(df_train_graph[['src', 'tgt']].values.ravel('K'))
    print(f"  > Saved {len(df_train_graph)} links involving {len(training_graph_nodes)} nodes to {TRAIN_GRAPH_FILE}")
    print(f"  > NOTE: This training graph contains ALL nodes present in the original link.dat,")
    print(f"          but only includes cell-drug links assigned to the training set.")

    print("\nTransductive data splitting complete for PRELUDE_mini.")
