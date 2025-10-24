import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset

dataset = PRELUDEDataset("data/processed")

print("PRELUDEDataset loaded successfully!")
print(f"Total nodes: {len(dataset.node2id)}")

# Total number of edges
total_edges = sum(len(v) for v in dataset.links['data'].values())
print(f"Total edges: {total_edges}")

# Breakdown per relation type
print("\nEdges by relation type:")
for rel_type, edges in dataset.links['data'].items():
    print(f"  • Type {rel_type}: {len(edges)} edges")

# Sample 10 random edges
print("\nSample links (up to 10 total):")
all_edges = []
for rel_type, edges in dataset.links['data'].items():
    for edge in edges:
        all_edges.append((rel_type, *edge))

sample_edges = random.sample(all_edges, min(10, len(all_edges)))
for rel_type, src, tgt, weight in sample_edges:
    print(f"  • Type {rel_type}: ({src} → {tgt}, weight={weight:.4f})")

# --- Step 4 testing: Validate Cell Feature Integration ---
if hasattr(dataset, 'cell_features_raw') and dataset.cell_features_raw is not None:
    print(f"\nCell features loaded! Shape: {dataset.cell_features_raw.shape}")
    print(f"Valid cell name count: {len(dataset.cell_list_ordered)}")
    
    # Pick 3 random local IDs from the mapping and show feature vector shape
    print("Verifying cell feature mapping:")
    sample_local_ids = random.sample(list(dataset.cell_local_id_to_feature_idx.keys()), k=3)
    for lid in sample_local_ids:
        feature_idx = dataset.cell_local_id_to_feature_idx[lid]
        feat_vec = dataset.cell_features_raw[feature_idx]
        print(f"  - Local ID {lid} → feature index {feature_idx} → shape: {feat_vec.shape}")
else:
    print("No cell features loaded. Double-check VAE alignment.")