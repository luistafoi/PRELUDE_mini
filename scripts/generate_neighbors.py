# scripts/generate_neighbors.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator

# --- Configuration ---
DATA_DIR = "data/processed"
OUTPUT_FILE = "data/processed/train_neighbors.txt"

# Define how many neighbors to sample for each node type via random walks
# This should be tuned based on the graph's density for each type
# Format: {node_type_id: num_samples}
NUM_SAMPLES_PER_TYPE = {
    0: 20,  # cell
    1: 20,  # drug
    2: 20   # gene
}

# --- Main Script ---
if __name__ == "__main__":
    print("This script will generate the training neighbor list using random walks with restart.")
    print(f"Output will be saved to: {OUTPUT_FILE}\n")

    # The DataGenerator needs the PRELUDEDataset object for mappings
    print("Loading dataset...")
    dataset = PRELUDEDataset(DATA_DIR)
    
    print("Initializing DataGenerator...")
    generator = DataGenerator(DATA_DIR)
    # Inject the dataset object into the generator
    generator.dataset = dataset 
    
    # Run the generation process
    generator.gen_train_neighbors_with_restart(OUTPUT_FILE, NUM_SAMPLES_PER_TYPE)
    
    print(f"\nGeneration complete. Neighbor list saved to {OUTPUT_FILE}.")