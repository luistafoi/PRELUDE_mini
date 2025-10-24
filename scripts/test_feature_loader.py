# scripts/test_feature_loader.py

import sys
import os
import torch
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader

# --- 1. Load the main dataset object ---
print("Loading PRELUDEDataset...")
dataset = PRELUDEDataset("data/processed")
print("Dataset loaded successfully.")

# --- 2. Initialize the FeatureLoader ---
print("\nInitializing FeatureLoader...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_loader = FeatureLoader(dataset, device)

# --- 3. Perform Sanity Checks ---
print("\n--- Running Sanity Checks ---")

# Check Drug Features
drug_type_id = dataset.node_name2type["drug"]
num_drugs_in_graph = dataset.nodes['count'][drug_type_id]
drug_feat_shape = feature_loader.drug_features.shape

print(f"\nDrug Feature Validation:")
print(f"  - Number of drugs in graph: {num_drugs_in_graph}")
print(f"  - Drug feature tensor shape: {drug_feat_shape}")
assert num_drugs_in_graph == drug_feat_shape[0], "Mismatch in number of drugs!"
print("Drug count matches tensor dimension.")

# Check Gene Features
gene_type_id = dataset.node_name2type["gene"]
num_genes_in_graph = dataset.nodes['count'][gene_type_id]
gene_feat_shape = feature_loader.gene_features.shape

print(f"\nGene Feature Validation:")
print(f"  - Number of genes in graph: {num_genes_in_graph}")
print(f"  - Gene feature tensor shape: {gene_feat_shape}")
assert num_genes_in_graph == gene_feat_shape[0], "Mismatch in number of genes!"
print("Gene count matches tensor dimension.")

# --- 4. Verify Content Alignment ---
print("\n--- Verifying Content Alignment (Random Samples) ---")

# Find the mapping from local ID back to global ID for easy name lookup
local_to_global = {ntype: {} for ntype in dataset.nodes['count']}
for global_id, (ntype, local_id) in dataset.nodes['type_map'].items():
    local_to_global[ntype][local_id] = global_id

# Verify 3 random drugs
print("\nVerifying 3 random drug samples...")
for _ in range(3):
    random_local_id = random.randint(0, num_drugs_in_graph - 1)
    global_id = local_to_global[drug_type_id][random_local_id]
    node_name = dataset.id2node[global_id]
    
    feature_vec = feature_loader.drug_features[random_local_id]
    
    print(f"  - Drug: '{node_name}' (Local ID: {random_local_id})")
    print(f"    - Feature Vector Shape: {feature_vec.shape}")
    print(f"    - Feature Vector Sum: {feature_vec.sum():.4f}") # A non-zero sum indicates features were loaded

# Verify 3 random genes
print("\n🔎 Verifying 3 random gene samples...")
for _ in range(3):
    random_local_id = random.randint(0, num_genes_in_graph - 1)
    global_id = local_to_global[gene_type_id][random_local_id]
    node_name = dataset.id2node[global_id]
    
    feature_vec = feature_loader.gene_features[random_local_id]
    
    print(f"  - Gene: '{node_name}' (Local ID: {random_local_id})")
    print(f"    - Feature Vector Shape: {feature_vec.shape}")
    print(f"    - Feature Vector Sum: {feature_vec.sum():.4f}")

print("\nAll tests passed!")