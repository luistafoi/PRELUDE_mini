# scripts/test_splits.py

import pandas as pd
import numpy as np
import os
import random
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
PROCESSED_DIR = "data/processed"
LINK_DAT_FILE = os.path.join(PROCESSED_DIR, "link.dat")
NODE_DAT_FILE = os.path.join(PROCESSED_DIR, "node.dat")

# Split Ratios from the original script
VALID_CELL_RATIO = 0.10
TEST_2_CELL_RATIO = 0.10
TEST_1_LINK_RATIO = 0.05

CELL_TYPE_ID = 0
CELL_DRUG_LINK_TYPE = 0
# As per the info.dat file, the cell-gene link type is 2.
CELL_GENE_LINK_TYPE = 2 

def run_split_logic_for_testing():
    """
    Executes the core logic of create_splits.py but returns the dataframes
    in memory instead of writing them to files.
    """
    try:
        df_links = pd.read_csv(LINK_DAT_FILE, sep='\t', header=None, names=['src', 'tgt', 'type', 'label'])
        df_nodes = pd.read_csv(NODE_DAT_FILE, sep='\t', header=None, names=['id', 'name', 'type'])
    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. Please run the full preprocessing pipeline first.")
        print(f"File not found: {e.filename}")
        return None

    # --- Pre-Check: Analyzing Cell Node Connectivity ---
    print("\n--- Pre-Check: Analyzing Cell Node Connectivity ---")
    cells_with_drug_links = set(df_links[df_links['type'] == CELL_DRUG_LINK_TYPE]['src'].unique())
    print(f"  > Found {len(cells_with_drug_links)} unique cells with at least one drug link.")
    
    cells_with_gene_links = set(df_links[df_links['type'] == CELL_GENE_LINK_TYPE]['src'].unique())
    print(f"  > Found {len(cells_with_gene_links)} unique cells with at least one gene link.")
    
    cells_in_both = cells_with_drug_links.intersection(cells_with_gene_links)
    print(f"  > Overlap: Found {len(cells_in_both)} cells with BOTH drug and gene links.")
    
    # The inductive splits should be sampled from this well-connected pool
    cell_nodes_for_inductive_split = list(cells_in_both)
    
    random.seed(42)
    random.shuffle(cell_nodes_for_inductive_split)

    # Split cell nodes for inductive sets
    num_valid_cells = int(len(cell_nodes_for_inductive_split) * VALID_CELL_RATIO)
    num_test_2_cells = int(len(cell_nodes_for_inductive_split) * TEST_2_CELL_RATIO)

    valid_cell_set = set(cell_nodes_for_inductive_split[:num_valid_cells])
    test_2_cell_set = set(cell_nodes_for_inductive_split[num_valid_cells : num_valid_cells + num_test_2_cells])
    
    all_cd_cells = set(pd.unique(df_links[df_links['type'] == CELL_DRUG_LINK_TYPE]['src']))
    remaining_cell_set = all_cd_cells - valid_cell_set - test_2_cell_set

    # Extract links for each set
    df_cd_links = df_links[df_links['type'] == CELL_DRUG_LINK_TYPE]
    valid_inductive_links = df_cd_links[df_cd_links['src'].isin(valid_cell_set)]
    test_2_inductive_links = df_cd_links[df_cd_links['src'].isin(test_2_cell_set)]
    
    remaining_cd_links = df_cd_links[df_cd_links['src'].isin(remaining_cell_set)]
    test_1_transductive_links = remaining_cd_links.sample(frac=TEST_1_LINK_RATIO, random_state=42)
    
    train_lp_links = remaining_cd_links.drop(test_1_transductive_links.index)
    other_links = df_links[df_links['type'] != CELL_DRUG_LINK_TYPE]
    train_graph_df = pd.concat([train_lp_links, other_links])

    return {
        "valid_cell_set": valid_cell_set,
        "test_2_cell_set": test_2_cell_set,
        "valid_links": valid_inductive_links,
        "test_2_links": test_2_inductive_links,
        "test_1_links": test_1_transductive_links,
        "original_graph": df_links,
        "final_train_graph": train_graph_df,
        "cells_in_both": cells_in_both,
        "nodes_df": df_nodes
    }

# --- Main Verification Script ---
if __name__ == "__main__":
    print("--- Verifying Advanced Data Splitting Logic ---")
    
    results = run_split_logic_for_testing()
    
    if results:
        # --- Pre-Check Results Display ---
        id_to_name_map = dict(zip(results["nodes_df"]['id'], results["nodes_df"]['name']))
        if results["cells_in_both"]:
            print("    - Example overlapping cell names:")
            sample_overlap_ids = random.sample(list(results["cells_in_both"]), min(5, len(results["cells_in_both"])))
            for cell_id in sample_overlap_ids:
                print(f"      - {id_to_name_map.get(cell_id, 'Unknown Name')}")

        # --- START OF NEW SECTION: Summary of Split Sizes ---
        print("\n--- Summary of Generated Splits ---")
        print(f"  > Validation Set (Inductive): {len(results['valid_links'])} links from {len(results['valid_cell_set'])} unique cells.")
        print(f"  > Test Set 2 (Inductive): {len(results['test_2_links'])} links from {len(results['test_2_cell_set'])} unique cells.")
        print(f"  > Test Set 1 (Transductive): {len(results['test_1_links'])} individual links.")
        # --- END OF NEW SECTION ---

        # --- Verification Check 1: Disjoint Sets ---
        print("\n--- Verification Check 1: Disjoint Sets ---")
        valid_cells = results["valid_cell_set"]
        test_2_cells = results["test_2_cell_set"]
        test_1_links = results["test_1_links"]
        
        val_test2_overlap = valid_cells.intersection(test_2_cells)
        assert len(val_test2_overlap) == 0
        print("  ✅ PASSED: Validation and Test 2 cell sets are mutually exclusive.")
        
        test_1_cells = set(pd.unique(test_1_links['src']))
        test1_val_overlap = test_1_cells.intersection(valid_cells)
        assert len(test1_val_overlap) == 0
        print("  ✅ PASSED: Test 1 links do not contain any cells from the Validation set.")
        
        test1_test2_overlap = test_1_cells.intersection(test_2_cells)
        assert len(test1_test2_overlap) == 0
        print("  ✅ PASSED: Test 1 links do not contain any cells from the Test 2 set.")

        # --- Verification Check 2: Link Severing ---
        print("\n--- Verification Check 2: Link Severing for Inductive Sets ---")
        original_graph = results["original_graph"]
        final_train_graph = results["final_train_graph"]

        for name, cell_set in [("Validation", valid_cells), ("Test 2", test_2_cells)]:
            print(f"\n  Checking a sample of cells from the {name} set...")
            if not cell_set:
                print("    - No cells in this set to check.")
                continue
                
            sample_cells = random.sample(list(cell_set), min(3, len(cell_set)))
            
            for cell_id in sample_cells:
                before_drug_links = original_graph[(original_graph['src'] == cell_id) & (original_graph['type'] == CELL_DRUG_LINK_TYPE)].shape[0]
                before_gene_links = original_graph[(original_graph['src'] == cell_id) & (original_graph['type'] == CELL_GENE_LINK_TYPE)].shape[0]
                
                after_drug_links = final_train_graph[(final_train_graph['src'] == cell_id) & (final_train_graph['type'] == CELL_DRUG_LINK_TYPE)].shape[0]
                after_gene_links = final_train_graph[(final_train_graph['src'] == cell_id) & (final_train_graph['type'] == CELL_GENE_LINK_TYPE)].shape[0]
                
                print(f"    - Cell ID {cell_id}:")
                print(f"      - Drug Links: {before_drug_links} (Before) -> {after_drug_links} (After)")
                print(f"      - Gene Links: {before_gene_links} (Before) -> {after_gene_links} (After)")
                
                assert after_drug_links == 0
                assert before_gene_links == after_gene_links

        print("\n  ✅ PASSED: Drug links for inductive sets were correctly severed from the training graph.")
        
        print("\n🎉 All verification checks passed. The splitting logic is correct.")
