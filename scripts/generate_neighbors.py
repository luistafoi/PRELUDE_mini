# scripts/generate_neighbors.py

import sys
import os
import pickle
import random
from collections import defaultdict
from tqdm import tqdm # For progress bar

# Add project root to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator

# --- Configuration ---
DATA_DIR = "data/processed"
# We'll still write the .txt file for compatibility, but it won't be used by the model
OUTPUT_TXT_FILE = os.path.join(DATA_DIR, "train_neighbors.txt")
# THIS IS THE NEW, FAST FILE
OUTPUT_PKL_FILE = os.path.join(DATA_DIR, "train_neighbors_preprocessed.pkl")
# This MUST match the 'max_neighbors_per_type' in models/tools.py
MAX_NEIGHBORS = 10 
# Use -1 as a padding value (LIDs are always >= 0)
PAD_VALUE = -1 

# --- Main Script ---
if __name__ == "__main__":
    print("This script generates TWO neighbor files:")
    print(f"  1. {OUTPUT_TXT_FILE} (Slow string-based, for legacy/debugging)")
    print(f"  2. {OUTPUT_PKL_FILE} (FAST pre-processed .pkl file for training)")

    # --- Load Necessary Data Structures ---
    print("\nLoading dataset...")
    try:
        dataset = PRELUDEDataset(DATA_DIR)
    except Exception as e:
        print(f"FATAL ERROR loading PRELUDEDataset: {e}")
        sys.exit(1)

    print("Initializing DataGenerator (to access neighbor dictionary)...")
    try:
        generator = DataGenerator(DATA_DIR)
    except Exception as e:
        print(f"FATAL ERROR initializing DataGenerator: {e}")
        sys.exit(1)

    # --- Pre-computation ---
    if not (hasattr(generator, 'neighbors') and generator.neighbors):
         print("FATAL ERROR: generator.neighbors dictionary not found or is empty.")
         sys.exit(1)
    if not (hasattr(dataset, 'nodes') and 'type_map' in dataset.nodes):
         print("FATAL ERROR: dataset.nodes['type_map'] not found.")
         sys.exit(1)

    print(f"\nPre-processing neighbors (padding/sampling to {MAX_NEIGHBORS})...")
    
    # This will be our final dictionary
    # Format: {(center_type_id, center_local_id): {neighbor_type_id: [padded_list_of_lids]}}
    precomputed_neighbors_dict = {}
    
    # We also still write the text file for the old loader
    txt_file_lines = []

    all_node_type_ids = sorted(dataset.nodes['count'].keys())
    
    # A fallback empty structure for nodes with no neighbors
    empty_padded_neighbors_by_type = {
        nt_id: [PAD_VALUE] * MAX_NEIGHBORS for nt_id in all_node_type_ids
    }

    # Iterate through all nodes defined in the dataset's node.dat
    for center_global_id in tqdm(sorted(dataset.id2node.keys()), desc="Pre-processing nodes"):
        
        center_info = dataset.nodes['type_map'].get(center_global_id)
        if center_info is None: continue
        
        center_type_id, center_local_id = center_info
        center_type_name = dataset.node_type2name.get(center_type_id)
        if center_type_name is None: continue
        
        center_node_str_key = f"{center_type_name}{center_local_id}"

        # 1. Parse all neighbors from strings to (type_id, local_id)
        # This dict will hold {neighbor_type_id: [list_of_neighbor_lids]}
        parsed_neighbors_by_type = defaultdict(list)
        
        neighbor_strings_for_txt_file = [] # For the old file

        if center_global_id in generator.neighbors:
            for rtype, target_gids in generator.neighbors[center_global_id].items():
                for neighbor_global_id in target_gids:
                    neighbor_info = dataset.nodes['type_map'].get(neighbor_global_id)
                    if neighbor_info:
                        neighbor_type_id, neighbor_local_id = neighbor_info
                        # Add the LID to our new dict
                        parsed_neighbors_by_type[neighbor_type_id].append(neighbor_local_id)
                        
                        # Add the string to the old list
                        neighbor_type_name = dataset.node_type2name.get(neighbor_type_id)
                        if neighbor_type_name:
                             neighbor_strings_for_txt_file.append(f"{neighbor_type_name}{neighbor_local_id}")

        # 2. Apply Sampling & Padding to the parsed LIDs
        padded_neighbors_for_node = {}

        for nt_id in all_node_type_ids:
            neigh_list = parsed_neighbors_by_type.get(nt_id, [])
            
            if len(neigh_list) > MAX_NEIGHBORS:
                # Sample
                padded_list = random.sample(neigh_list, MAX_NEIGHBORS)
            elif 0 < len(neigh_list) <= MAX_NEIGHBORS:
                # Pad
                padded_list = neigh_list + [PAD_VALUE] * (MAX_NEIGHBORS - len(neigh_list))
            else:
                # All padding
                padded_list = [PAD_VALUE] * MAX_NEIGHBORS
            
            padded_neighbors_for_node[nt_id] = padded_list
            
        # 3. Save to our new dictionary
        precomputed_neighbors_dict[(center_type_id, center_local_id)] = padded_neighbors_for_node
        
        # 4. Save to the old .txt file list
        if neighbor_strings_for_txt_file:
            unique_sorted_neighbors = sorted(list(set(neighbor_strings_for_txt_file)))
            txt_file_lines.append(f"{center_node_str_key}:{','.join(unique_sorted_neighbors)}\n")

    # --- Save PKL File ---
    print(f"\nSaving pre-processed neighbors to {OUTPUT_PKL_FILE}...")
    with open(OUTPUT_PKL_FILE, "wb") as f:
        pickle.dump(precomputed_neighbors_dict, f)
    print("  > .pkl file saved.")

    # --- Save TXT File ---
    print(f"Saving legacy string neighbors to {OUTPUT_TXT_FILE}...")
    with open(OUTPUT_TXT_FILE, "w") as f:
        f.writelines(txt_file_lines)
    print(f"  > .txt file saved ({len(txt_file_lines)} nodes).")

    print("\nGeneration complete.")
