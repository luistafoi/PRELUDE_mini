# scripts/test_tools.py

import sys
import os
import torch
import random
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg

# --- Prerequisites ---
NEIGHBOR_FILE = "data/processed/train_neighbors.txt"
if not os.path.exists(NEIGHBOR_FILE):
    print(f"Error: Neighbor file not found at '{NEIGHBOR_FILE}'")
    print("Please run 'python scripts/generate_neighbors.py' first.")
    sys.exit(1)
# ---------------------


# 1. --- Load all components ---
print("--- Loading all data and model components ---")
args = read_args()
args.use_vae_encoder = True
args.use_skip_connection = True
args.use_node_isolation = True
args.embed_d = 256 # Match VAE latent dim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PRELUDEDataset("data/processed")
feature_loader = FeatureLoader(dataset, device)
generator = DataGenerator("data/processed").load_train_neighbors(NEIGHBOR_FILE)

model = HetAgg(args, dataset, feature_loader, device).to(device)
model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
print("All components loaded and initialized successfully.\n")


# 2. --- Test Link Prediction ---
print("--- Testing Link Prediction (Relation Type 0: cell -> drug) ---")
# Use the generator to get sample pairs
generator.build_edge_set()
pos_pairs = generator.get_positive_pairs(0)
neg_pairs = generator.sample_negative_pairs(0, num_samples=2)

# Prepare batch
cell_gids = [pos_pairs[0][0], neg_pairs[0][0]]
drug_gids = [pos_pairs[0][1], neg_pairs[0][1]]
labels = torch.tensor([1.0, 0.0], device=device)

# Convert global IDs to local IDs for the model
cell_lids = [dataset.nodes['type_map'][gid][1] for gid in cell_gids]
drug_lids = [dataset.nodes['type_map'][gid][1] for gid in drug_gids]

# --- Forward Pass ---
print("  > Running link_prediction_forward...")
model.eval()
with torch.no_grad():
    scores = model.link_prediction_forward(drug_lids, cell_lids, generator)
print(f"    Scores: {scores.cpu().numpy()}")
print("Forward pass complete.")

# --- Loss Calculation ---
print("  > Running link_prediction_loss...")
model.train()
loss = model.link_prediction_loss(drug_lids, cell_lids, labels, generator, isolation_ratio=0.5)
print(f"    Loss: {loss.item():.4f}")
print("Loss calculation complete.\n")


# 3. --- Test Self-Supervised RW Loss ---
print("--- Testing Self-Supervised RW Loss ---")
print("  > Generating RW triples...")
rw_pairs = generator.generate_rw_triples(walk_length=5, window_size=2, num_walks=1)

if rw_pairs:
    # Get all node IDs to sample from for true negatives
    all_node_ids = list(dataset.id2node.keys())
    
    # Create a batch of (center, positive, negative) triples with global IDs
    rw_batch = []
    for center, pos in rw_pairs[:10]: # Test with 10 triples
        neg = random.choice(all_node_ids)
        # Ensure negative sample is not the same as positive
        while neg == pos:
            neg = random.choice(all_node_ids)
        rw_batch.append((center, pos, neg))
    
    print(f"  > Testing with a batch of {len(rw_batch)} triples...")
    rw_loss = model.self_supervised_rw_loss(rw_batch, generator)
    print(f"    RW Loss: {rw_loss.item():.4f}")
    print("Self-supervised loss calculation complete.")
else:
    print("No random walk triples were generated, skipping test.")

print("\nAll tests passed!")