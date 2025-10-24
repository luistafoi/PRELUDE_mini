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
        self.dataset_dir = dataset_dir
        self._load_node_info()
        self._load_type_mappings() # <-- Add this call
        self._load_link_data()
        self._build_neighbor_dicts()

        self.train_neighbors = None 

    def _load_node_info(self):
        self.node2id = {}
        self.id2node = {}
        self.node_type = {}

        with open(os.path.join(self.dataset_dir, "node.dat"), "r") as f:
            for line in f:
                nid, name, ntype = line.strip().split("\t")
                nid = int(nid)
                ntype = int(ntype)
                self.node2id[name] = nid
                self.id2node[nid] = name
                self.node_type[nid] = ntype

    def _load_type_mappings(self):
        """Loads node type names from info.dat."""
        self.node_name2type = {}
        self.node_type2name = {}
        info_path = os.path.join(self.dataset_dir, "info.dat")
        try:
            with open(info_path, 'r') as f:
                info_data = json.load(f)
            # Assuming the simpler format based on previous discussions
            node_mapping = info_data["node.dat"]
            for type_id_str, type_info in node_mapping.items():
                type_id = int(type_id_str)
                type_name = type_info[0]
                self.node_name2type[type_name] = type_id
                self.node_type2name[type_id] = type_name
            print(f"  > Loaded type mappings: {self.node_name2type}")
        except Exception as e:
            print(f"Error loading or parsing info.dat: {e}")
            # Handle error appropriately, maybe exit or use defaults
            sys.exit(1) # Or raise an error

    def _load_link_data(self):
        self.links = defaultdict(list)
        with open(os.path.join(self.dataset_dir, "link.dat"), "r") as f:
            for line in f:
                src, tgt, rtype, weight = line.strip().split("\t")
                self.links[int(rtype)].append((int(src), int(tgt), float(weight)))

    # The entire _load_features() method has been REMOVED from this file.
    
    # The entire get_feature() method has been REMOVED from this file.

    def _build_neighbor_dicts(self):
        print("Building neighbor dictionaries (excluding direct drug-cell links)...")
        self.neighbors = defaultdict(lambda: defaultdict(list))

        # --- vvv Get type IDs vvv ---
        cell_type_id = self.node_name2type.get('cell', -1)
        drug_type_id = self.node_name2type.get('drug', -1)
        # --- ^^^ ---

        for rtype, triples in self.links.items():
            for src, tgt, _ in triples:
                
                # --- vvv Add this check vvv ---
                src_type = self.node_type.get(src, -2)
                tgt_type = self.node_type.get(tgt, -2)

                is_cell_drug = (src_type == cell_type_id and tgt_type == drug_type_id)
                is_drug_cell = (src_type == drug_type_id and tgt_type == cell_type_id)

                if is_cell_drug or is_drug_cell:
                    continue # Skip adding this link to neighbors
                # --- ^^^ End check ^^^ ---
                
                self.neighbors[src][rtype].append(tgt)
        print("Finished building neighbor dictionaries.")

    def load_train_neighbors(self, file_path):
        """Loads the pre-generated training neighbors from a file."""
        print(f"Info: Loading pre-generated training neighbors from {file_path}")
        self.train_neighbors = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) != 2: continue
                
                center_node_str, neigh_list_str = parts
                self.train_neighbors[center_node_str] = neigh_list_str.split(',')
        print("Info: Finished loading training neighbors.")
        return self
    
    def gen_train_neighbors_with_restart(self, out_file, num_samples_per_type, restart_prob=0.5):
        """
        Generates a high-quality neighbor list for GNN training using random walks with restart.
        This is a key part of the original HetGNN logic.
        """
        print(f"Info: Generating training neighbors with restart. Writing to {out_file}...")
        
        # Pre-generate global_id -> (type, local_id) and type_name mappings
        type_map = {gid: (self.node_type[gid], lid) for gid, (tid, lid) in self.dataset.nodes['type_map'].items()}
        type_name_map = {v: k for k, v in self.dataset.node_name2type.items()}
        
        with open(out_file, "w") as f:
            for global_id in self.id2node.keys():
                node_type, local_id = type_map[global_id]
                node_type_name = type_name_map[node_type]
                center_node_str = f"{node_type_name}{local_id}"
                
                # Perform walks from this node
                sampled_neighbors = []
                for _ in range(num_samples_per_type[node_type]):
                    current_node_gid = global_id
                    while True:
                        if random.random() < restart_prob:
                            # Add the sampled neighbor and stop this walk
                            if current_node_gid != global_id:
                                neigh_type, neigh_local_id = type_map[current_node_gid]
                                neigh_type_name = type_name_map[neigh_type]
                                sampled_neighbors.append(f"{neigh_type_name}{neigh_local_id}")
                            break
                        
                        # Get all neighbors of the current node
                        possible_next_steps = []
                        if current_node_gid in self.neighbors:
                            for rtype in self.neighbors[current_node_gid]:
                                possible_next_steps.extend(self.neighbors[current_node_gid][rtype])
                        
                        if not possible_next_steps:
                            # Dead end, stop this walk
                            break
                        
                        # Jump to a random neighbor
                        current_node_gid = random.choice(possible_next_steps)

                if sampled_neighbors:
                    f.write(f"{center_node_str}:{','.join(sampled_neighbors)}\n")

        print("Info: Finished generating training neighbors.")

    def sample_neighbors(self, node_id, rtype, n_samples=5):
        neighbors = self.neighbors[node_id][rtype]
        if len(neighbors) <= n_samples:
            return neighbors
        return random.sample(neighbors, n_samples)

    def get_positive_pairs(self, relation_type):
        """Returns a list of (src, tgt) positive pairs for the given relation type"""
        return [(src, tgt) for src, tgt, _ in self.links[relation_type]]

    def build_edge_set(self):
        """Build a set of (src, tgt, rtype) tuples for fast negative sampling"""
        self.edge_set = set()
        for rtype, triples in self.links.items():
            for src, tgt, _ in triples:
                self.edge_set.add((src, tgt, rtype))

    def sample_negative_pairs(self, relation_type, num_samples):
        """Randomly sample (src, tgt) pairs that are NOT in the graph for that relation"""
        if not self.links[relation_type]:
            print(f"Warning: No links found for relation type {relation_type} to sample negatives from.")
            return []
        
        example_edge = self.links[relation_type][0]
        src_type = self.node_type[example_edge[0]]
        tgt_type = self.node_type[example_edge[1]]
        
        src_candidates = [nid for nid, ntype in self.node_type.items() if ntype == src_type]
        tgt_candidates = [nid for nid, ntype in self.node_type.items() if ntype == tgt_type]

        if not src_candidates or not tgt_candidates:
            return []

        negative_samples = set()
        attempts = 0
        max_attempts = num_samples * 10  # prevent infinite loop

        # build_edge_set must be called before this function
        if not hasattr(self, 'edge_set') or not self.edge_set:
            print("Warning: edge_set not found. Building it for negative sampling.")
            self.build_edge_set()

        while len(negative_samples) < num_samples and attempts < max_attempts:
            src = random.choice(src_candidates)
            tgt = random.choice(tgt_candidates)
            if (src, tgt, relation_type) not in self.edge_set:
                negative_samples.add((src, tgt))
            attempts += 1

        return list(negative_samples)
    
    def generate_rw_triples(self, walk_length=10, window_size=5, num_walks=10):
        """
        Generate random walk-based skip-gram triples (center, positive) for self-supervised loss.
        """
        print("Generating random walk skip-gram triples...")
        
        all_neighbors = defaultdict(list)
        for src_id, rtype_map in self.neighbors.items():
            for tgt_list in rtype_map.values():
                all_neighbors[src_id].extend(tgt_list)

        triples = []
        all_nodes = list(self.node2id.values())
        
        for _ in range(num_walks):
            random.shuffle(all_nodes)
            for node_id in all_nodes:
                walk = [node_id]
                while len(walk) < walk_length:
                    cur = walk[-1]
                    neighbors = all_neighbors.get(cur, [])
                    if not neighbors:
                        break
                    walk.append(random.choice(neighbors))

                for i in range(len(walk)):
                    center = walk[i]
                    start = max(0, i - window_size)
                    end = min(len(walk), i + window_size + 1)
                    for j in range(start, end):
                        if i != j:
                            triples.append((center, walk[j]))

        print(f"Generated {len(triples)} random walk triples.")
        return triples