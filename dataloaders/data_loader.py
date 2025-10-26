# dataloaders/data_loader.py

import os
import json
import numpy as np
import torch
from collections import defaultdict, Counter
import sys

class PRELUDEDataset:
    def __init__(self, data_dir):
        """Initializes the dataset by loading graph structure and pre-split links."""
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            print(f"FATAL ERROR: Dataset directory not found at {data_dir}")
            sys.exit(1)
            
        self.info = self._load_info()
        
        # Dictionary to hold graph structure and pre-split links
        self.links = {
            'graph': defaultdict(list), # Structural links for GNN message passing
            'train_lp': [],             # Positive links for LP training
            'train_lp_set': set(),      # Set version for faster negative sampling checks
            'valid_inductive': [],      # Validation links (inductive)
            'test_transductive': [],    # Test links (transductive, if file exists)
            'test_inductive': [],       # Test links (inductive)
        }

        self._load_nodes() # Loads self.node2id, self.id2node, self.node_types

        # Define standard type mappings
        self.node_name2type = {"cell": 0, "drug": 1, "gene": 2}
        self.node_type2name = {v: k for k, v in self.node_name2type.items()}

        # Build node structure dictionary required by GNN
        self._build_node_structure_dict() # Creates self.nodes dictionary

        # Load structural links used for message passing
        self._load_graph_links()

        # Load pre-split link prediction sets
        self._load_lp_splits()

        # Build map for local_id -> global_id (might be useful)
        self._build_local_to_global_map()

        # Build link type lookup based on info.dat
        self._build_link_type_lookup()

        print(f"INFO: PRELUDEDataset initialization complete for {data_dir}.")
        self.summary() # Print summary

    def _load_info(self):
        """Loads metadata from info.dat."""
        path = os.path.join(self.data_dir, 'info.dat')
        if not os.path.exists(path):
            print(f"FATAL ERROR: info.dat not found at {path}")
            sys.exit(1)
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"FATAL ERROR loading or parsing info.dat: {e}")
            sys.exit(1)

    def _load_nodes(self):
        """Loads node IDs, names, and types from node.dat."""
        path = os.path.join(self.data_dir, 'node.dat')
        if not os.path.exists(path):
            print(f"FATAL ERROR: node.dat not found at {path}")
            sys.exit(1)
            
        self.node2id = {}
        self.id2node = {}
        self.node_types = {} # Maps global_id -> type_id
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    nid, name, ntype = line.strip().split('\t')
                    nid = int(nid)
                    ntype = int(ntype)
                    self.node2id[name] = nid
                    self.id2node[nid] = name
                    self.node_types[nid] = ntype
            print(f"  > Loaded info for {len(self.id2node)} nodes from node.dat.")
        except Exception as e:
             print(f"FATAL ERROR reading node.dat: {e}")
             sys.exit(1)

    def _build_node_structure_dict(self):
        """Builds the self.nodes dictionary (count, type_map)."""
        type_counts = Counter()
        type_map = {} # Maps global_id -> (type_id, local_id)
        local_type_counters = defaultdict(int)

        # Iterate through nodes sorted by global ID to ensure consistent local IDs
        for nid in sorted(self.node_types.keys()):
            ntype = self.node_types[nid]
            local_id = local_type_counters[ntype]
            type_map[nid] = (ntype, local_id)
            local_type_counters[ntype] += 1
            type_counts[ntype] += 1
            
        self.nodes = {
            'count': dict(type_counts),
            'type_map': type_map # Keep as dict for direct gid lookup
        }

    def _load_graph_links(self):
        """Loads the structural links (from train.dat) for GNN message passing."""
        path = os.path.join(self.data_dir, 'train.dat')
        if not os.path.exists(path):
            print(f"Warning: Structural link file train.dat not found at {path}. GNN message passing will be limited.")
            return
            
        count = 0
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3: # Need at least src, tgt, type
                        src, tgt, ltype = int(parts[0]), int(parts[1]), int(parts[2])
                        weight = float(parts[3]) if len(parts) > 3 else 1.0
                        # Ensure nodes exist in our node list before adding link
                        if src in self.id2node and tgt in self.id2node:
                            self.links['graph'][ltype].append((src, tgt, weight))
                            count += 1
            print(f"  > Loaded {count} structural links from train.dat.")
        except Exception as e:
            print(f"Error reading train.dat: {e}")


    def _load_lp_splits(self):
        """Loads all pre-split positive link prediction files."""
        print("Loading pre-split link prediction data...")
        split_names = {
            'train_lp': 'train_lp_links.dat',
            'valid_inductive': 'valid_inductive_links.dat',
            'test_transductive': 'test_transductive_links.dat', # Keep loading if exists
            'test_inductive': 'test_inductive_links.dat',
        }
        for split_key, filename in split_names.items():
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            # Assuming format: src_gid, tgt_gid, [optional type], [optional label]
                            if len(parts) >= 2:
                                self.links[split_key].append(
                                    (int(parts[0]), int(parts[1])) # Store as (src_gid, tgt_gid) tuple
                                )
                    print(f"  > Loaded {len(self.links[split_key])} links for {split_key}")
                    # Create the set version for training links
                    if split_key == 'train_lp':
                        self.links['train_lp_set'] = set(self.links['train_lp'])
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            else:
                print(f"  > Info: Link file not found for split '{split_key}' at {path}. Split will be empty.")

    def _build_local_to_global_map(self):
        """Builds mapping from (type, local_id) back to global_id."""
        self.local_to_global_map = {ntype: {} for ntype in self.nodes['count']}
        for global_id, (ntype, local_id) in self.nodes['type_map'].items():
             # Check if ntype is valid before assignment
            if ntype in self.local_to_global_map:
                self.local_to_global_map[ntype][local_id] = global_id
            else:
                 print(f"Warning: Node {global_id} has type {ntype} not found in self.nodes['count']. Skipping local_to_global mapping.")


    def _build_link_type_lookup(self):
        """Builds mapping from (src_type, tgt_type) to link_type_id based on info.dat."""
        self.link_type_lookup = {}
        if 'link.dat' not in self.info:
             print("Warning: 'link.dat' metadata not found in info.dat. Cannot build link type lookup.")
             return

        print("DEBUG: link.dat contents from info ->", self.info['link.dat']) # Debug print

        try:
            for ltype_str, type_info in self.info['link.dat'].items():
                ltype = int(ltype_str)
                src_name, tgt_name = type_info[0], type_info[1]
                
                # Check if names exist in name2type mapping
                if src_name not in self.node_name2type or tgt_name not in self.node_name2type:
                     print(f"Warning: Node names '{src_name}' or '{tgt_name}' for link type {ltype} not found in node types. Skipping this link type lookup.")
                     continue
                     
                src_type = self.node_name2type[src_name]
                tgt_type = self.node_name2type[tgt_name]
                
                # Store lookup for both directions if symmetrical type names used
                self.link_type_lookup[(src_type, tgt_type)] = ltype
                # Add reverse only if source and target types are different,
                # or if explicitly defined in info.dat (assumed not for simplicity here)
                if src_type != tgt_type:
                    self.link_type_lookup[(tgt_type, src_type)] = ltype # Assumes symmetry if not explicitly reversed
            print("  > Built link type lookup based on info.dat.")
        except Exception as e:
            print(f"Error building link type lookup from info.dat: {e}")

    def summary(self):
        """Prints a summary of the loaded dataset."""
        print("\n--- PRELUDE Dataset Summary ---")
        if not self.nodes or not self.nodes.get('count'):
             print("No node data loaded.")
             return
             
        print(f"\nNodes ({self.nodes.get('total', sum(self.nodes.get('count', {}).values()))} total):")
        for ntype_id, count in sorted(self.nodes.get('count', {}).items()):
            ntype_name = self.node_type2name.get(ntype_id, f"Type {ntype_id}")
            print(f"  - {ntype_name.capitalize()} (Type {ntype_id}): {count} nodes")

        print(f"\nStructural Links (for GNN message passing from train.dat):")
        total_graph_links = sum(len(edges) for edges in self.links['graph'].values())
        print(f"  > Total structural links loaded: {total_graph_links}")

        print(f"\nLink Prediction Splits:")
        print(f"  - Training LP Set:       {len(self.links.get('train_lp', []))} links")
        print(f"  - Validation (Inductive): {len(self.links.get('valid_inductive', []))} links")
        print(f"  - Test (Transductive):    {len(self.links.get('test_transductive', []))} links")
        print(f"  - Test (Inductive):       {len(self.links.get('test_inductive', []))} links")
        print("---------------------------------")

    # --- Methods removed ---
    # _load_cell_features is removed.