# scripts/create_splits.py

import pandas as pd
import numpy as np
import os
import random

# --- Configuration ---
PROCESSED_DIR = "data/processed"
LINK_DAT_FILE = os.path.join(PROCESSED_DIR, "link.dat")
NODE_DAT_FILE = os.path.join(PROCESSED_DIR, "node.dat")

# Output files
TRAIN_GRAPH_FILE = os.path.join(PROCESSED_DIR, "train.dat")
VALID_INDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "valid_inductive_links.dat")
TEST_INDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "test_inductive_links.dat")
TEST_TRANSDUCTIVE_FILE = os.path.join(PROCESSED_DIR, "test_transductive_links.dat")
TRAIN_LP_FILE = os.path.join(PROCESSED_DIR, "train_lp_links.dat")

# Split Ratios
VALID_CELL_RATIO = 0.10  # 10% of well-connected cells for inductive validation
TEST_2_CELL_RATIO = 0.10   # 10% of well-connected cells for inductive test (Test 2)
TEST_1_LINK_RATIO = 0.05   # 5% of remaining links for transductive test (Test 1)

CELL_TYPE_ID = 0
CELL_DRUG_LINK_TYPE = 0
CELL_GENE_LINK_TYPE = 2 # As per our last correction

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Advanced Data Splitting ---")

    # --- Step 1: Load Data and Identify Well-Connected Cells ---
    print(" > Loading graph data to identify well-connected cells...")
    df_links = pd.read_csv(LINK_DAT_FILE, sep='\t', header=None, names=['src', 'tgt', 'type', 'label'])
    
    cells_with_drug_links = set(df_links[df_links['type'] == CELL_DRUG_LINK_TYPE]['src'].unique())
    cells_with_gene_links = set(df_links[df_links['type'] == CELL_GENE_LINK_TYPE]['src'].unique())
    well_connected_cells = list(cells_with_drug_links.intersection(cells_with_gene_links))
    
    print(f" > Found {len(well_connected_cells)} cells with both drug and gene links. These will be used for inductive splits.")

    # --- Step 2: Create Inductive Splits (Validation and Test 2) ---
    print("\n--- Step 2: Creating inductive splits from well-connected cells ---")
    random.shuffle(well_connected_cells)

    num_valid_cells = int(len(well_connected_cells) * VALID_CELL_RATIO)
    num_test_2_cells = int(len(well_connected_cells) * TEST_2_CELL_RATIO)

    valid_cell_set = set(well_connected_cells[:num_valid_cells])
    test_2_cell_set = set(well_connected_cells[num_valid_cells : num_valid_cells + num_test_2_cells])
    
    df_cd_links = df_links[df_links['type'] == CELL_DRUG_LINK_TYPE]
    valid_inductive_links = df_cd_links[df_cd_links['src'].isin(valid_cell_set)]
    test_2_inductive_links = df_cd_links[df_cd_links['src'].isin(test_2_cell_set)]

    valid_inductive_links.to_csv(VALID_INDUCTIVE_FILE, sep='\t', index=False, header=False)
    test_2_inductive_links.to_csv(TEST_INDUCTIVE_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(valid_inductive_links)} links for {len(valid_cell_set)} validation cells.")
    print(f"  > Saved {len(test_2_inductive_links)} links for {len(test_2_cell_set)} test 2 cells.")

    # --- Step 3: Create Transductive Split (Test 1) ---
    print("\n--- Step 3: Creating transductive split (Test 1) ---")
    
    # The remaining cells are all cells that were not used in the inductive splits
    all_cd_cells = set(pd.unique(df_cd_links['src']))
    remaining_cell_set = all_cd_cells - valid_cell_set - test_2_cell_set
    remaining_cd_links = df_cd_links[df_cd_links['src'].isin(remaining_cell_set)]
    
    test_1_transductive_links = remaining_cd_links.sample(frac=TEST_1_LINK_RATIO, random_state=42)
    
    test_1_transductive_links.to_csv(TEST_TRANSDUCTIVE_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(test_1_transductive_links)} links for Test 1.")

    # --- Step 4: Create the Final Training Files ---
    print("\n--- Step 4: Creating final training files ---")
    
    train_lp_links = remaining_cd_links.drop(test_1_transductive_links.index)
    train_lp_links.to_csv(TRAIN_LP_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(train_lp_links)} links for the main link prediction training task.")

    other_links = df_links[df_links['type'] != CELL_DRUG_LINK_TYPE]
    # The final training graph contains the LP links AND the links for the inductive sets
    # This is because the nodes must exist in the graph, we just sever their drug links
    train_graph_df = pd.concat([train_lp_links, other_links, valid_inductive_links, test_2_inductive_links])
    
    # Now, remove the drug links for the inductive sets to create the final structural graph
    final_train_graph_df = train_graph_df[~train_graph_df['src'].isin(valid_cell_set | test_2_cell_set) | (train_graph_df['type'] != CELL_DRUG_LINK_TYPE)]

    final_train_graph_df.to_csv(TRAIN_GRAPH_FILE, sep='\t', index=False, header=False)
    print(f"  > Saved {len(final_train_graph_df)} total links for the final training graph structure.")

    print("\n✅ Advanced data splitting complete.")
