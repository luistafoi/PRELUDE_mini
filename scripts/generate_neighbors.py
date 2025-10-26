# scripts/generate_neighbors.py

import sys
import os
from collections import defaultdict
from tqdm import tqdm # For progress bar

# Add project root to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator # Still needed for its loaded data

# --- Configuration ---
# Ensure these point to the correct processed data directory
DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_DIR, "train_neighbors.txt")

# --- Main Script ---
if __name__ == "__main__":
    print("This script generates the training neighbor list using DIRECT connections")
    print("(excluding cell-drug links, as filtered during DataGenerator init).")
    print(f"Output will be saved to: {OUTPUT_FILE}\n")

    # --- Load Necessary Data Structures ---
    print("Loading dataset...")
    try:
        # PRELUDEDataset loads node info and mappings
        dataset = PRELUDEDataset(DATA_DIR)
    except Exception as e:
        print(f"FATAL ERROR loading PRELUDEDataset: {e}")
        sys.exit(1)

    print("Initializing DataGenerator (to access neighbor dictionary)...")
    try:
        # DataGenerator loads links and builds the filtered self.neighbors
        generator = DataGenerator(DATA_DIR)
    except Exception as e:
        print(f"FATAL ERROR initializing DataGenerator: {e}")
        sys.exit(1)

    # --- Generate Neighbors from generator.neighbors ---
    print(f"Generating neighbor strings and writing to {OUTPUT_FILE}...")
    
    # Check if neighbor dict was built
    if not hasattr(generator, 'neighbors') or not generator.neighbors:
         print("FATAL ERROR: generator.neighbors dictionary not found or is empty.")
         print("Ensure DataGenerator._build_neighbor_dicts() ran successfully.")
         sys.exit(1)
         
    # Check if node mappings exist
    if not hasattr(dataset, 'nodes') or 'type_map' not in dataset.nodes:
         print("FATAL ERROR: dataset.nodes['type_map'] not found.")
         sys.exit(1)
    if not hasattr(dataset, 'node_type2name'):
         print("FATAL ERROR: dataset.node_type2name mapping not found.")
         sys.exit(1)


    lines_written = 0
    with open(OUTPUT_FILE, "w") as f:
        # Iterate through all nodes defined in the dataset's node.dat
        for center_global_id in tqdm(sorted(dataset.id2node.keys()), desc="Processing nodes"):
            
            # Get center node's type and local ID
            center_info = dataset.nodes['type_map'].get(center_global_id)
            if center_info is None:
                print(f"Warning: Skipping node {center_global_id}, not found in type_map.")
                continue
            center_type, center_local_id = center_info
            center_type_name = dataset.node_type2name.get(center_type)
            if center_type_name is None:
                 print(f"Warning: Skipping node {center_global_id}, unknown type {center_type}.")
                 continue
            center_node_str = f"{center_type_name}{center_local_id}"

            # Aggregate all direct neighbors from the filtered generator.neighbors
            all_neighbor_global_ids = []
            if center_global_id in generator.neighbors:
                for rtype, target_gids in generator.neighbors[center_global_id].items():
                    all_neighbor_global_ids.extend(target_gids)

            # Convert neighbor global IDs to strings ("type_name" + "local_id")
            neighbor_strings = []
            for neighbor_global_id in all_neighbor_global_ids:
                neighbor_info = dataset.nodes['type_map'].get(neighbor_global_id)
                if neighbor_info:
                    neighbor_type, neighbor_local_id = neighbor_info
                    neighbor_type_name = dataset.node_type2name.get(neighbor_type)
                    if neighbor_type_name:
                        neighbor_strings.append(f"{neighbor_type_name}{neighbor_local_id}")
                    # else: # Should not happen if data is consistent
                    #     print(f"Warning: Unknown type {neighbor_type} for neighbor {neighbor_global_id}")
                # else: # Should not happen if data is consistent
                #     print(f"Warning: Neighbor {neighbor_global_id} not in type_map.")


            if neighbor_strings:
                # Make unique and sort (optional, but good for consistency)
                unique_sorted_neighbors = sorted(list(set(neighbor_strings)))
                f.write(f"{center_node_str}:{','.join(unique_sorted_neighbors)}\n")
                lines_written += 1

    print(f"\nGeneration complete. Wrote neighbor lists for {lines_written} nodes to {OUTPUT_FILE}.")