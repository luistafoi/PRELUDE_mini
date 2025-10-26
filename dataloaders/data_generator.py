# dataloaders/data_generator.py

import os
import numpy as np
import random
import pickle
from collections import defaultdict
import json
import sys

class DataGenerator:
    def __init__(self, dataset_dir):
        """Initializes the DataGenerator by loading graph structure and type info."""
        self.dataset_dir = dataset_dir
        self._load_node_info()
        self._load_type_mappings()
        self._load_link_data()
        self._build_neighbor_dicts() # Builds neighbors excluding direct C-D links
        self.train_neighbors = None # Will be populated by load_train_neighbors

    def _load_node_info(self):
        """Loads node IDs, names, and types from node.dat."""
        self.node2id = {}
        self.id2node = {}
        self.node_type = {}
        node_file = os.path.join(self.dataset_dir, "node.dat")
        if not os.path.exists(node_file):
             print(f"FATAL ERROR: Node file not found at {node_file}")
             sys.exit(1)
        with open(node_file, "r") as f:
            for line in f:
                try:
                    nid, name, ntype = line.strip().split("\t")
                    nid = int(nid)
                    ntype = int(ntype)
                    self.node2id[name] = nid
                    self.id2node[nid] = name
                    self.node_type[nid] = ntype
                except ValueError:
                    print(f"Warning: Skipping malformed line in node.dat: {line.strip()}")
        print(f"  > Loaded info for {len(self.id2node)} nodes.")


    def _load_type_mappings(self):
        """Loads node type names from info.dat."""
        self.node_name2type = {}
        self.node_type2name = {}
        info_path = os.path.join(self.dataset_dir, "info.dat")
        try:
            with open(info_path, 'r') as f:
                info_data = json.load(f)
            # Assuming the simpler format
            node_mapping = info_data["node.dat"]
            for type_id_str, type_info in node_mapping.items():
                type_id = int(type_id_str)
                type_name = type_info[0]
                self.node_name2type[type_name] = type_id
                self.node_type2name[type_id] = type_name
            print(f"  > Loaded type mappings: {self.node_name2type}")
        except Exception as e:
            print(f"FATAL ERROR loading or parsing info.dat: {e}")
            sys.exit(1)

    def _load_link_data(self):
        """Loads link information from link.dat."""
        self.links = defaultdict(list)
        link_file = os.path.join(self.dataset_dir, "link.dat")
        if not os.path.exists(link_file):
             print(f"FATAL ERROR: Link file not found at {link_file}")
             sys.exit(1)
        with open(link_file, "r") as f:
            for line in f:
                try:
                    src, tgt, rtype, weight = line.strip().split("\t")
                    self.links[int(rtype)].append((int(src), int(tgt), float(weight)))
                except ValueError:
                    print(f"Warning: Skipping malformed line in link.dat: {line.strip()}")
        print(f"  > Loaded links for {len(self.links)} relation types.")

    def _build_neighbor_dicts(self):
        """Builds neighbor dicts for message passing, excluding direct C-D links."""
        print("Building neighbor dictionaries (excluding direct drug-cell links)...")
        self.neighbors = defaultdict(lambda: defaultdict(list))

        cell_type_id = self.node_name2type.get('cell', -1)
        drug_type_id = self.node_name2type.get('drug', -1)

        if cell_type_id == -1 or drug_type_id == -1:
             print("FATAL ERROR: 'cell' or 'drug' type not found in info.dat mappings.")
             sys.exit(1)

        link_count = 0
        skipped_count = 0
        for rtype, triples in self.links.items():
            for src, tgt, _ in triples:
                link_count += 1
                src_type = self.node_type.get(src, -2)
                tgt_type = self.node_type.get(tgt, -2)

                is_cell_drug = (src_type == cell_type_id and tgt_type == drug_type_id)
                is_drug_cell = (src_type == drug_type_id and tgt_type == cell_type_id)

                if is_cell_drug or is_drug_cell:
                    skipped_count +=1
                    continue # Skip adding this link to neighbors
                
                self.neighbors[src][rtype].append(tgt)
        print(f"Finished building neighbor dictionaries. Processed {link_count} links, skipped {skipped_count} direct C-D links.")

    def load_train_neighbors(self, file_path):
        """Loads the pre-generated training neighbors from a file."""
        print(f"Info: Loading pre-generated training neighbors from {file_path}")
        if not os.path.exists(file_path):
            print(f"FATAL ERROR: Training neighbor file not found at {file_path}")
            sys.exit(1)
            
        self.train_neighbors = defaultdict(list)
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        center_node_str, neigh_list_str = parts
                        if neigh_list_str: # Avoid adding empty lists for nodes with no neighbors listed
                             self.train_neighbors[center_node_str] = neigh_list_str.split(',')
            print(f"Info: Finished loading {len(self.train_neighbors)} nodes with neighbors.")
        except Exception as e:
             print(f"FATAL ERROR reading neighbor file {file_path}: {e}")
             sys.exit(1)
        return self # Return self for potential chaining

    # --- Utility Functions (Kept) ---

    def sample_neighbors(self, node_id, rtype, n_samples=5):
        """Samples neighbors for a given node and relation type."""
        neighbors = self.neighbors[node_id].get(rtype, [])
        if len(neighbors) <= n_samples:
            return neighbors
        return random.sample(neighbors, n_samples)

    def get_positive_pairs(self, relation_type):
        """Returns a list of (src, tgt) positive pairs for the given relation type."""
        return [(src, tgt) for src, tgt, _ in self.links.get(relation_type, [])]

    def build_edge_set(self):
        """Build a set of (src, tgt, rtype) tuples for fast negative sampling."""
        self.edge_set = set()
        for rtype, triples in self.links.items():
            for src, tgt, _ in triples:
                self.edge_set.add((src, tgt, rtype))
        print("  > Built edge set for negative sampling.")

    def sample_negative_pairs(self, relation_type, num_samples):
        """Randomly sample (src, tgt) pairs that are NOT in the graph for that relation."""
        if not self.links.get(relation_type):
            print(f"Warning: No links found for relation type {relation_type} to sample negatives from.")
            return []
        
        # Determine source and target types based on the first link of this type
        example_src, example_tgt, _ = self.links[relation_type][0]
        src_type = self.node_type.get(example_src)
        tgt_type = self.node_type.get(example_tgt)
        
        if src_type is None or tgt_type is None:
             print(f"Warning: Could not determine node types for relation {relation_type}. Cannot sample negatives.")
             return []

        src_candidates = [nid for nid, ntype in self.node_type.items() if ntype == src_type]
        tgt_candidates = [nid for nid, ntype in self.node_type.items() if ntype == tgt_type]

        if not src_candidates or not tgt_candidates:
             print(f"Warning: No candidate nodes found for types {src_type} or {tgt_type}.")
             return []

        negative_samples = set()
        attempts = 0
        max_attempts = num_samples * 20 # Increased attempts

        if not hasattr(self, 'edge_set'):
            self.build_edge_set()

        while len(negative_samples) < num_samples and attempts < max_attempts:
            src = random.choice(src_candidates)
            tgt = random.choice(tgt_candidates)
            if (src, tgt, relation_type) not in self.edge_set:
                negative_samples.add((src, tgt))
            attempts += 1
            
        if len(negative_samples) < num_samples:
            print(f"Warning: Could only generate {len(negative_samples)}/{num_samples} negative samples for relation {relation_type} after {max_attempts} attempts.")

        return list(negative_samples)

    # --- RW Specific Functions (Removed) ---
    # generate_rw_triples(...) is removed.
    # gen_train_neighbors_with_restart(...) is removed.