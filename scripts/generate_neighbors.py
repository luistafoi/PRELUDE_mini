# scripts/generate_neighbors.py

import sys
import os
import pickle
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

# Add project root to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from config.args import read_args

# --- Main Script ---
if __name__ == "__main__":
    # --- STEP 1: Parse Arguments ---
    parser = argparse.ArgumentParser()

    default_dir = "data/processed"
    try:
        default_dir = read_args().data_dir
    except (Exception, SystemExit):
        pass

    parser.add_argument('--data_dir', type=str, default=default_dir,
                        help='Directory containing the graph data.')
    parser.add_argument('--max_neighbors', type=int, default=10,
                        help='Max neighbors per type to sample. Must match --max_neighbors in training.')
    parser.add_argument('--include_cell_drug', action='store_true',
                        help='Include Cell-Drug edges in GNN neighbors (for dynamic isolation masking).')
    parser.add_argument('--dedup_symmetric', action='store_true',
                        help='Skip duplicate edges across directions (for link.dat files that store symmetric edges as both (A,B) and (B,A)).')

    args, _ = parser.parse_known_args()
    DATA_DIR = args.data_dir
    INCLUDE_CD = args.include_cell_drug
    DEDUP_SYMMETRIC = args.dedup_symmetric

    OUTPUT_TXT_FILE = os.path.join(DATA_DIR, "train_neighbors.txt")
    OUTPUT_PKL_FILE = os.path.join(DATA_DIR, "train_neighbors_preprocessed.pkl")

    MAX_NEIGHBORS = args.max_neighbors
    PAD_VALUE = -1

    print(f"Generating neighbor files for: {DATA_DIR}")
    print(f"  1. {OUTPUT_TXT_FILE}")
    print(f"  2. {OUTPUT_PKL_FILE}")

    # --- Load Necessary Data Structures ---
    print("\nLoading dataset...")
    try:
        dataset = PRELUDEDataset(DATA_DIR)
    except Exception as e:
        print(f"FATAL ERROR loading PRELUDEDataset: {e}")
        sys.exit(1)

    print(f"Initializing DataGenerator (include_cell_drug={INCLUDE_CD}, dedup_symmetric={DEDUP_SYMMETRIC})...")
    try:
        generator = DataGenerator(DATA_DIR, include_cell_drug=INCLUDE_CD, dedup_symmetric=DEDUP_SYMMETRIC)
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

    # Final dictionary: (center_type, center_lid) -> {nt_id: {'ids': [...], 'weights': [...]}}
    precomputed_neighbors_dict = {}
    txt_file_lines = []

    all_node_type_ids = sorted(dataset.nodes['count'].keys())

    for center_global_id in tqdm(sorted(dataset.id2node.keys()), desc="Pre-processing nodes"):

        center_info = dataset.nodes['type_map'].get(center_global_id)
        if center_info is None:
            continue

        center_type_id, center_local_id = center_info
        center_type_name = dataset.node_type2name.get(center_type_id)
        if center_type_name is None:
            continue

        center_node_str_key = f"{center_type_name}{center_local_id}"

        # 1. Parse neighbors with weights
        parsed_neighbors_by_type = defaultdict(list)  # nt_id -> [(local_id, weight), ...]
        neighbor_strings_for_txt_file = []

        if center_global_id in generator.neighbors:
            for rtype, target_gids in generator.neighbors[center_global_id].items():
                # Get corresponding weights
                weights = generator.neighbor_weights.get(center_global_id, {}).get(rtype, [])
                for idx, neighbor_global_id in enumerate(target_gids):
                    neighbor_info = dataset.nodes['type_map'].get(neighbor_global_id)
                    if neighbor_info:
                        neighbor_type_id, neighbor_local_id = neighbor_info
                        w = weights[idx] if idx < len(weights) else 1.0
                        parsed_neighbors_by_type[neighbor_type_id].append((neighbor_local_id, w))

                        neighbor_type_name = dataset.node_type2name.get(neighbor_type_id)
                        if neighbor_type_name:
                            neighbor_strings_for_txt_file.append(f"{neighbor_type_name}{neighbor_local_id}")

        # 2. Apply Sampling & Padding (preserving weight alignment)
        padded_neighbors_for_node = {}

        for nt_id in all_node_type_ids:
            neigh_list = parsed_neighbors_by_type.get(nt_id, [])

            if len(neigh_list) > MAX_NEIGHBORS:
                sampled = random.sample(neigh_list, MAX_NEIGHBORS)
                padded_ids = [x[0] for x in sampled]
                padded_weights = [x[1] for x in sampled]
            elif 0 < len(neigh_list) <= MAX_NEIGHBORS:
                padded_ids = [x[0] for x in neigh_list] + [PAD_VALUE] * (MAX_NEIGHBORS - len(neigh_list))
                padded_weights = [x[1] for x in neigh_list] + [0.0] * (MAX_NEIGHBORS - len(neigh_list))
            else:
                padded_ids = [PAD_VALUE] * MAX_NEIGHBORS
                padded_weights = [0.0] * MAX_NEIGHBORS

            padded_neighbors_for_node[nt_id] = {
                'ids': padded_ids,
                'weights': padded_weights,
            }

        # 3. Save
        precomputed_neighbors_dict[(center_type_id, center_local_id)] = padded_neighbors_for_node

        # 4. Save to old txt format (no weights)
        if neighbor_strings_for_txt_file:
            unique_sorted_neighbors = sorted(list(set(neighbor_strings_for_txt_file)))
            txt_file_lines.append(f"{center_node_str_key}:{','.join(unique_sorted_neighbors)}\n")

    # --- Save Files ---
    print(f"\nSaving pre-processed neighbors to {OUTPUT_PKL_FILE}...")
    with open(OUTPUT_PKL_FILE, "wb") as f:
        pickle.dump(precomputed_neighbors_dict, f)
    print("  > .pkl file saved.")

    print(f"Saving legacy string neighbors to {OUTPUT_TXT_FILE}...")
    with open(OUTPUT_TXT_FILE, "w") as f:
        f.writelines(txt_file_lines)
    print(f"  > .txt file saved ({len(txt_file_lines)} nodes).")

    print("\nGeneration complete.")
