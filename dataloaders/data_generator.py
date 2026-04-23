# dataloaders/data_generator.py

import os
import random
from collections import defaultdict
import json
import sys
import pickle


class DataGenerator:
    def __init__(self, dataset_dir, include_cell_drug=False, dedup_symmetric=False):
        self.dataset_dir = dataset_dir
        self._load_node_info()
        self._load_type_mappings()
        self._load_link_data()
        self._build_neighbor_dicts(include_cell_drug=include_cell_drug,
                                   dedup_symmetric=dedup_symmetric)
        self.train_neighbors = None

    def _load_node_info(self):
        self.node2id = {}
        self.id2node = {}
        self.node_type = {}
        node_file = os.path.join(self.dataset_dir, "node.dat")
        if not os.path.exists(node_file):
            sys.exit(f"FATAL: {node_file} not found")
        with open(node_file, "r") as f:
            for line in f:
                try:
                    nid, name, ntype = line.strip().split("\t")
                    nid, ntype = int(nid), int(ntype)
                    self.node2id[name] = nid
                    self.id2node[nid] = name
                    self.node_type[nid] = ntype
                except ValueError:
                    pass

    def _load_type_mappings(self):
        self.node_name2type = {}
        info_path = os.path.join(self.dataset_dir, "info.dat")
        try:
            with open(info_path, 'r') as f:
                info_data = json.load(f)
            for type_id_str, type_info in info_data["node.dat"].items():
                self.node_name2type[type_info[0]] = int(type_id_str)
        except Exception as e:
            sys.exit(f"FATAL info.dat: {e}")

    def _load_link_data(self):
        """Load structural links from train.dat (NOT link.dat) to prevent leakage."""
        self.links = defaultdict(list)
        self.edge_weights = defaultdict(list)  # Store weights per edge type
        self.adj_list = defaultdict(list)  # For random walks (train-only)

        # Use train.dat instead of link.dat to prevent test leakage
        link_file = os.path.join(self.dataset_dir, "train.dat")
        if not os.path.exists(link_file):
            print(f"Warning: train.dat not found, falling back to link.dat")
            link_file = os.path.join(self.dataset_dir, "link.dat")

        with open(link_file, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split("\t")
                    src, tgt, rtype = int(parts[0]), int(parts[1]), int(parts[2])
                    weight = float(parts[3]) if len(parts) > 3 else 1.0
                    self.links[rtype].append((src, tgt))
                    self.edge_weights[rtype].append(weight)
                    # Bidirectional adjacency for random walks
                    self.adj_list[src].append(tgt)
                    self.adj_list[tgt].append(src)
                except ValueError:
                    pass

    def _build_neighbor_dicts(self, include_cell_drug=False, dedup_symmetric=False):
        """Build BIDIRECTIONAL neighbor dicts.

        Args:
            include_cell_drug: If True, Cell-Drug edges are included in GNN
                neighbors (required for v2 runs with dynamic isolation).
                If False (default), they are excluded as before.
            dedup_symmetric: If True, skip rows already seen in either direction.
                Use when link.dat stores symmetric edges twice (e.g. cell-cell,
                drug-drug, gene-gene) to avoid 2x doubling of neighbor lists.
        """
        self.neighbors = defaultdict(lambda: defaultdict(list))
        self.neighbor_weights = defaultdict(lambda: defaultdict(list))
        cell_id = self.node_name2type.get('cell')
        drug_id = self.node_name2type.get('drug')

        n_cell_drug = 0
        n_skipped_dup = 0
        seen_edges = set() if dedup_symmetric else None

        for rtype, pairs in self.links.items():
            weights = self.edge_weights[rtype]
            for i, (src, tgt) in enumerate(pairs):
                st = self.node_type.get(src)
                tt = self.node_type.get(tgt)
                is_cell_drug = (st == cell_id and tt == drug_id) or (st == drug_id and tt == cell_id)
                if is_cell_drug and not include_cell_drug:
                    continue
                if dedup_symmetric:
                    key_fwd = (src, tgt, rtype)
                    key_rev = (tgt, src, rtype)
                    if key_fwd in seen_edges or key_rev in seen_edges:
                        n_skipped_dup += 1
                        continue
                    seen_edges.add(key_fwd)
                if is_cell_drug:
                    n_cell_drug += 1
                w = weights[i] if i < len(weights) else 1.0
                # BIDIRECTIONAL: add both src→tgt and tgt→src
                self.neighbors[src][rtype].append(tgt)
                self.neighbor_weights[src][rtype].append(w)
                self.neighbors[tgt][rtype].append(src)
                self.neighbor_weights[tgt][rtype].append(w)

        if include_cell_drug:
            print(f"  > Included {n_cell_drug} Cell-Drug edges in neighbor dicts")
        if dedup_symmetric:
            print(f"  > Deduplicated {n_skipped_dup} symmetric edges")

    def load_train_neighbors(self, file_path):
        """Loads the training neighbors from a file (supports .pkl or .txt)."""
        print(f"Info: Loading pre-generated training neighbors from {file_path}")
        if not os.path.exists(file_path):
            print(f"FATAL ERROR: Training neighbor file not found at {file_path}")
            sys.exit(1)

        self.train_neighbors = defaultdict(list)

        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    self.train_neighbors = pickle.load(f)
                print(f"Info: Loaded binary neighbors for {len(self.train_neighbors)} nodes.")
            else:
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(':')
                        if len(parts) == 2 and parts[1]:
                            self.train_neighbors[parts[0]] = parts[1].split(',')
                print(f"Info: Loaded text neighbors for {len(self.train_neighbors)} nodes.")

        except Exception as e:
            print(f"FATAL ERROR reading neighbor file {file_path}: {e}")
            sys.exit(1)

        return self

    # --- Random Walk Logic ---
    def sample_random_walk_pairs(self, batch_nodes, walk_length, num_negatives):
        """
        Performs random walks starting from batch_nodes.
        Returns: (center_nodes, context_nodes, negative_nodes)
        """
        centers = []
        contexts = []
        negatives = []

        all_nodes = list(self.node_type.keys())
        # Filter out excluded nodes (e.g., inductive cells) for negative sampling
        if hasattr(self, '_rw_negative_pool') and self._rw_negative_pool:
            neg_pool = self._rw_negative_pool
        else:
            neg_pool = all_nodes

        for start_node in batch_nodes:
            curr = start_node
            for _ in range(walk_length):
                if not self.adj_list[curr]:
                    break
                curr = random.choice(self.adj_list[curr])

                centers.append(start_node)
                contexts.append(curr)

                negs = []
                for _ in range(num_negatives):
                    negs.append(random.choice(neg_pool))
                negatives.append(negs)

        return centers, contexts, negatives
