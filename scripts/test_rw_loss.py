# scripts/test_rw_loss.py

import sys
import os
import torch
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg

print("--- Fast Test for Random Walk (RW) Loss ---")

# --- 1. Load Components ---
print(" > Loading components...")
args = read_args()
args.use_vae_encoder = True
args.use_skip_connection = True
args.embed_d = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PRELUDEDataset("data/processed")
feature_loader = FeatureLoader(dataset, device)
generator = DataGenerator("data/processed").load_train_neighbors("data/processed/train_neighbors.txt")
model = HetAgg(args, dataset, feature_loader, device).to(device)

# --- 2. Generate a Small Batch of RW Triples ---
print(" > Generating a small RW batch...")
rw_pairs = generator.generate_rw_triples(walk_length=5, window_size=2, num_walks=1)
all_node_ids = list(dataset.id2node.keys())
rw_batch = []
for center, pos in rw_pairs[:20]: # Use a small batch of 20
    neg = random.choice(all_node_ids)
    while neg == pos:
        neg = random.choice(all_node_ids)
    rw_batch.append((center, pos, neg))

# --- 3. Run the RW Loss Function ---
print(" > Calculating RW loss...")
model.train() # Set model to train mode to test batchnorm fix
try:
    loss_rw = model.self_supervised_rw_loss(rw_batch, generator)
    print(f"RW Loss calculated successfully: {loss_rw.item():.4f}")
    print("\nThe fix is working correctly!")
except Exception as e:
    print(f"\nTest Failed: {e}")