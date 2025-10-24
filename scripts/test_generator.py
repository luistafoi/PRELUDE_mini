# scripts/test_generator.py

import sys
import os
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_generator import DataGenerator

print("--- Testing Refactored DataGenerator ---")
dg = DataGenerator("data/processed")

print("\nDataGenerator initialized successfully.")
print(f"  - Total nodes: {len(dg.node2id)}")
print(f"  - Total relation types: {len(dg.links)}")

# --- Test 1: Edge Counts ---
print("\n--- Test 1: Verifying Edge Counts ---")
for rel_type, edges in sorted(dg.links.items()):
    print(f"  - Relation type {rel_type}: {len(edges)} edges")

# --- Test 2: Neighbor Sampling ---
print("\n--- Test 2: Sampling Neighbors ---")
# Sample from a random node to ensure it works generally
random_node_id = random.choice(list(dg.node2id.values()))
# Assuming relation type 0 (cell-drug) exists and has nodes with neighbors
sampled = dg.sample_neighbors(random_node_id, 0, n_samples=5)
print(f"  - Sampled neighbors for a random node (type 0): {sampled}")
print("Neighbor sampling is functional.")

# --- Test 3: Positive/Negative Pair Sampling ---
print("\n--- Test 3: Sampling Positive/Negative Pairs ---")
# Use relation type 2 (gene-drug) as an example
RELATION_TO_TEST = 2
dg.build_edge_set()

pos_pairs = dg.get_positive_pairs(RELATION_TO_TEST)
# Sample a smaller number for a quick test
num_neg_samples = min(len(pos_pairs) * 2, 200) 
neg_pairs = dg.sample_negative_pairs(RELATION_TO_TEST, num_samples=num_neg_samples)

print(f"  - Sampled for relation type {RELATION_TO_TEST}:")
print(f"    - Positive samples found: {len(pos_pairs)}")
print(f"    - Negative samples generated: {len(neg_pairs)}")
if neg_pairs:
    print(f"    - Example negative pair: {neg_pairs[0]}")
print("Positive/negative pair sampling is functional.")

# --- Test 4: Random Walk Triple Generation ---
print("\n--- Test 4: Generating Random Walk Triples ---")
# Generate a small number of walks for a quick test
rw_triples = dg.generate_rw_triples(walk_length=5, window_size=2, num_walks=1)
print(f"  - Generated {len(rw_triples)} skip-gram pairs from a small walk set.")
if rw_triples:
    print(f"  - Sample pairs: {rw_triples[:5]}")
print(" Random walk generation is functional.")

print("\nAll DataGenerator tests passed!")