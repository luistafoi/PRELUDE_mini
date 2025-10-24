# scripts/test_graph_integrity.py

import sys
import os
import torch
import random
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg

# --- 1. Load Components ---
print("--- Loading all data and model components ---")
args = read_args()
args.use_vae_encoder = True
args.embed_d = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PRELUDEDataset("data/processed")
feature_loader = FeatureLoader(dataset, device)
generator = DataGenerator("data/processed")
model = HetAgg(args, dataset, feature_loader, device).to(device)
print("All components loaded.\n")

# --- 2. Test Node Initialization ---
print("--- Part 1: Verifying Initial Node Features ---")
node_types_to_test = {
    "cell": dataset.node_name2type["cell"],
    "drug": dataset.node_name2type["drug"],
    "gene": dataset.node_name2type["gene"],
}

for name, type_id in node_types_to_test.items():
    sample_lids = []
    if name == "cell":
        if len(dataset.valid_cell_local_ids) >= 3:
            sample_lids = random.sample(dataset.valid_cell_local_ids, 3)
        else:
            sample_lids = dataset.valid_cell_local_ids
    else:
        num_nodes_of_type = dataset.nodes['count'][type_id]
        sample_lids = list(range(min(3, num_nodes_of_type)))
    
    with torch.no_grad():
        features = model.conteng_agg(sample_lids, type_id)
    
    print(f"Testing '{name}' nodes...")
    print(f"  - Requested {len(sample_lids)} feature vectors.")
    print(f"  - Received tensor of shape: {features.shape}")
    assert features.shape == (len(sample_lids), args.embed_d)
    assert features.sum() != 0
    print(f"'{name}' nodes are being initialized correctly.\n")
    
# --- 3. Test Random Walk Composition ---
print("--- Part 2: Analyzing Random Walk Composition ---")

gene_gene_link_type = 3
# --- THIS LINE IS CORRECTED ---
if gene_gene_link_type in dataset.links['train'] and dataset.links['train'][gene_gene_link_type]:
    print(f"  - Found {len(dataset.links['train'][gene_gene_link_type])} gene-gene links in the training dataset.")
else:
    print("  - Warning: No gene-gene links found in the training dataset.")

print("  - Generating a sample of 100 random walks to analyze...")
walks = []
all_node_gids = list(dataset.id2node.keys())
for _ in range(100):
    walk = [random.choice(all_node_gids)]
    for _ in range(10): # Walk length of 10
        cur = walk[-1]
        
        all_neighbors = []
        if cur in generator.neighbors:
            for rtype in generator.neighbors[cur]:
                all_neighbors.extend(generator.neighbors[cur][rtype])
        
        if not all_neighbors:
            break
        walk.append(random.choice(all_neighbors))
    walks.append(walk)
    
gene_type_id = dataset.node_name2type["gene"]
gene_nodes_in_walks = 0
gene_gene_steps = 0

for walk in walks:
    for i, node_gid in enumerate(walk):
        if dataset.nodes['type_map'][node_gid][0] == gene_type_id:
            gene_nodes_in_walks += 1
            
        if i > 0:
            prev_node_gid = walk[i-1]
            if dataset.nodes['type_map'][prev_node_gid][0] == gene_type_id and \
               dataset.nodes['type_map'][node_gid][0] == gene_type_id:
                gene_gene_steps += 1
                
print("\nAnalysis Results:")
print(f"  - Total gene nodes found across all walks: {gene_nodes_in_walks}")
print(f"  - Total gene-gene steps taken in walks: {gene_gene_steps}")

assert gene_nodes_in_walks > 0, "No gene nodes were found in the random walks!"
assert gene_gene_steps > 0, "No gene-gene links were traversed in the random walks!"
print(" Random walks correctly include gene nodes and traverse gene-gene links.\n")

print("All integrity tests passed!")