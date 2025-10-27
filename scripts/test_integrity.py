# scripts/test_integrity.py

"""
Performs a full integrity check on the processed data files.

This version adds detailed logging for leaking nodes.

This script checks for:
1.  Data Leakage (Edges): Ensures no links from the validation or test
    sets appear in the training sets.
2.  Data Leakage (Nodes): Ensures that the inductive validation/test sets
    use cell/drug nodes that are NOT in the training set.
    *** UPDATED: Now lists 10% of leaking nodes. ***
3.  Feature Integrity: Samples random nodes of each type to ensure their
    loaded embeddings are not zero-vectors.
4.  Data Sampling: Prints random examples from each link set for
    human-readable verification.
"""

import os
import sys
import random
import torch
import numpy as np

# --- Add project root to path to import dataloaders ---
# This assumes the script is run from the project root (PRELUDE_mini)
sys.path.append(os.path.abspath(os.path.dirname(__name__)))
# -----------------------------------------------------

try:
    from dataloaders.data_loader import PRELUDEDataset
    from dataloaders.feature_loader import FeatureLoader
except ImportError:
    print("FATAL ERROR: Could not import dataloaders.")
    print("Please make sure you are running this script from the project root directory (PRELUDE_mini).")
    sys.exit(1)


def load_links_as_set(filepath):
    """Helper function to load a link file into a set of (src, tgt) tuples."""
    if not os.path.exists(filepath):
        print(f"  > Warning: Link file not found at {filepath}, returning empty set.")
        return set()
    
    edges = set()
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                edges.add((int(parts[0]), int(parts[1])))
    return edges

def get_inductive_nodes(edge_set, dataset):
    """Helper to get all cell/drug nodes from a set of edges."""
    nodes_cd = set()
    cell_id = dataset.node_name2type.get('cell', -1)
    drug_id = dataset.node_name2type.get('drug', -1)
    
    for src, tgt in edge_set:
        if src in dataset.nodes['type_map']:
            if dataset.nodes['type_map'][src][0] == cell_id or dataset.nodes['type_map'][src][0] == drug_id:
                nodes_cd.add(src)
        if tgt in dataset.nodes['type_map']:
            if dataset.nodes['type_map'][tgt][0] == cell_id or dataset.nodes['type_map'][tgt][0] == drug_id:
                nodes_cd.add(tgt)
    return nodes_cd

def check_embeddings(node_type_name, feature_tensor, dataset, n_samples=10):
    """Samples embeddings for a node type and checks for zero-vectors."""
    print(f"\n--- Checking '{node_type_name}' Embeddings ---")
    
    node_type_id = dataset.node_name2type.get(node_type_name, -1)
    if node_type_id == -1:
        print("  > ERROR: Node type not found in dataset.")
        return

    # Get all global IDs for this type
    global_ids = [gid for gid, (ntype, _) in dataset.nodes['type_map'].items() if ntype == node_type_id]
    
    if not global_ids:
        print(f"  > No nodes found for type '{node_type_name}'.")
        return

    # Sample N global IDs
    sampled_global_ids = random.sample(global_ids, min(n_samples, len(global_ids)))
    
    print(f"Sampling {len(sampled_global_ids)} random '{node_type_name}' nodes:")
    zero_vectors_found = 0
    
    for gid in sampled_global_ids:
        node_name = dataset.id2node.get(gid, "N/A")
        # Get the LOCAL id
        _, local_id = dataset.nodes['type_map'][gid]
        
        # Get the embedding vector
        embedding = feature_tensor[local_id]
        
        # Check if it's a zero vector
        is_zero = torch.all(embedding == 0).item()
        
        print(f"  - GID: {gid:6} | LocalID: {local_id:5} | Name: {node_name:20} | Is Zero: {is_zero}")
        
        if is_zero:
            # This is a critical check. A 'True' here means the feature failed to load.
            # Your 'Warning: No feature found...' messages in the loader would also show this.
            zero_vectors_found += 1
            
    if zero_vectors_found > 0:
        print(f"  > !!! WARNING: Found {zero_vectors_found} zero-vector embeddings! Check logs for 'No feature found' warnings.")
    else:
        print(f"  > SUCCESS: All sampled embeddings are non-zero.")

def print_edge_samples(name, edge_set, dataset, n_samples=20):
    """Prints random human-readable edge examples from a set."""
    print(f"\n--- Sampling {n_samples} Edges from '{name}' (Total: {len(edge_set)}) ---")
    
    if not edge_set:
        print("  > Set is empty.")
        return
        
    # --- THIS IS THE FIX ---
    # Manually create the reverse mapping from the dataset object
    if not hasattr(dataset, 'node_name2type'):
         print("  > ERROR: Dataset object missing 'node_name2type' attribute.")
         return
    type2node = {v: k for k, v in dataset.node_name2type.items()}
    # --- END FIX ---

    sampled_edges = random.sample(list(edge_set), min(n_samples, len(edge_set)))
    
    for src_gid, tgt_gid in sampled_edges:
        try:
            src_name = dataset.id2node[src_gid]
            tgt_name = dataset.id2node[tgt_gid]
            
            src_type_id, _ = dataset.nodes['type_map'][src_gid]
            tgt_type_id, _ = dataset.nodes['type_map'][tgt_gid]
            
            # Use the new 'type2node' dictionary
            src_type_name = type2node[src_type_id]
            tgt_type_name = type2node[tgt_type_id]
            
            print(f"  - ({src_type_name:4}) {src_name:20} <--> ({tgt_type_name:4}) {tgt_name:20}")
        except KeyError as e:
            print(f"  - ERROR: Could not find node info for edge ({src_gid}, {tgt_gid}). Missing key: {e}")


def main():
    DATA_DIR = 'data/processed'
    print(f"--- 1. Initializing Dataset and FeatureLoader from {DATA_DIR} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Dataset (loads node.dat, train.dat, etc.)
    print("Loading PRELUDEDataset (Note: Its own summary will print here)...")
    dataset = PRELUDEDataset(DATA_DIR)
    print("PRELUDEDataset loading complete.")

    # 2. Initialize FeatureLoader (loads embeddings)
    print("\nLoading FeatureLoader (Note: Its own summary will print here)...")
    loader = FeatureLoader(dataset, device)
    print("FeatureLoader loading complete.")

    
    print("\n--- 2. Checking for Data Leakage ---")
    
    # --- Load all edge sets ---
    print("Loading all link files into sets...")
    train_lp_edges = load_links_as_set(os.path.join(DATA_DIR, 'train_lp_links.dat'))
    valid_edges = load_links_as_set(os.path.join(DATA_DIR, 'valid_inductive_links.dat'))
    test_edges = load_links_as_set(os.path.join(DATA_DIR, 'test_inductive_links.dat'))
    train_graph_edges = load_links_as_set(os.path.join(DATA_DIR, 'train.dat'))
    print("...loading complete.")

    # --- Check for EDGE leakage ---
    print("\nChecking for EDGE overlap (should be 0):")
    print(f"  - Train LP <-> Valid Overlap:   {len(train_lp_edges.intersection(valid_edges))}")
    print(f"  - Train LP <-> Test Overlap:    {len(train_lp_edges.intersection(test_edges))}")
    print(f"  - Train Graph <-> Valid Overlap: {len(train_graph_edges.intersection(valid_edges))}")
    print(f"  - Train Graph <-> Test Overlap:  {len(train_graph_edges.intersection(test_edges))}")

    # --- Check for inductive NODE leakage ---
    print("\nChecking for inductive NODE (cell/drug) overlap (should be 0):")
    train_nodes_cd = get_inductive_nodes(train_lp_edges, dataset)
    valid_nodes_cd = get_inductive_nodes(valid_edges, dataset)
    test_nodes_cd = get_inductive_nodes(test_edges, dataset)
    
    print(f"  - Train (Cell/Drug) Nodes: {len(train_nodes_cd)}")
    print(f"  - Valid (Cell/Drug) Nodes: {len(valid_nodes_cd)}")
    print(f"  - Test (Cell/Drug) Nodes:  {len(test_nodes_cd)}")

    # --- *** NEW DETAILED LEAKAGE REPORT *** ---
    type2node = {v: k for k, v in dataset.node_name2type.items()}

    valid_overlap = train_nodes_cd.intersection(valid_nodes_cd)
    print(f"\n  - !!! Train <-> Valid Node Overlap: {len(valid_overlap)} !!!")
    if valid_overlap:
        print(f"  --- Listing first 10% (up to 150) of {len(valid_overlap)} LEAKING Valid Nodes ---")
        limit = min(max(1, int(len(valid_overlap) * 0.1)), 150)
        for i, gid in enumerate(list(valid_overlap)):
            if i >= limit:
                break
            try:
                node_name = dataset.id2node[gid]
                ntype, _ = dataset.nodes['type_map'][gid]
                ntype_name = type2node[ntype]
                print(f"    - GID: {gid:6} | Type: {ntype_name:4} | Name: {node_name}")
            except Exception as e:
                print(f"    - GID: {gid:6} | Error retrieving node info: {e}")

    test_overlap = train_nodes_cd.intersection(test_nodes_cd)
    print(f"\n  - !!! Train <-> Test Node Overlap: {len(test_overlap)} !!!")
    if test_overlap:
        print(f"  --- Listing first 10% (up to 150) of {len(test_overlap)} LEAKING Test Nodes ---")
        limit = min(max(1, int(len(test_overlap) * 0.1)), 150)
        for i, gid in enumerate(list(test_overlap)):
            if i >= limit:
                break
            try:
                node_name = dataset.id2node[gid]
                ntype, _ = dataset.nodes['type_map'][gid]
                ntype_name = type2node[ntype]
                print(f"    - GID: {gid:6} | Type: {ntype_name:4} | Name: {node_name}")
            except Exception as e:
                print(f"    - GID: {gid:6} | Error retrieving node info: {e}")
    # --- *** END NEW REPORT *** ---

    
    # --- Check 3: Feature Integrity ---
    check_embeddings('cell', loader.cell_features, dataset)
    check_embeddings('drug', loader.drug_features, dataset)
    check_embeddings('gene', loader.gene_features, dataset)

    # --- Check 4: Edge Examples ---
    print_edge_samples('Train LP Links', train_lp_edges, dataset)
    print_edge_samples('Valid Inductive Links', valid_edges, dataset)
    print_edge_samples('Test Inductive Links', test_edges, dataset)
    
    print("\n--- Integrity Check Complete ---")

if __name__ == '__main__':
    main()

