# dataloaders/data_loader.py

import os
import json
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

class PRELUDEDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.info = self._load_info()
        
        # This dictionary will now hold the graph structure and all pre-split links
        self.links = {
            'graph': defaultdict(list),
            'train_lp': [],
            'valid_inductive': [],
            'test_transductive': [],
            'test_inductive': [],
        }

        self._load_nodes()

        # Create the node mapping dictionaries first
        type_counts = defaultdict(int)
        type_map = {}
        local_type_counters = defaultdict(int)
        for nid in sorted(self.node_types.keys()):
            ntype = self.node_types[nid]
            local_id = local_type_counters[ntype]
            type_map[nid] = (ntype, local_id)
            local_type_counters[ntype] += 1
            type_counts[ntype] += 1
        self.nodes = {
            'count': dict(type_counts),
            'type_map': dict(type_map)
        }

        # Load all the different link files
        self._load_graph_links()
        self._load_lp_splits()

        # Load cell features
        self.cell_features_raw = None
        self.cell_global_id_to_feature_idx = None
        self._load_cell_features()

        self.node_name2type = {"cell": 0, "drug": 1, "gene": 2}
        self.node_type2name = {v: k for k, v in self.node_name2type.items()}
        
        self.local_to_global_map = {ntype: {} for ntype in self.nodes['count']}
        for global_id, (ntype, local_id) in self.nodes['type_map'].items():
            self.local_to_global_map[ntype][local_id] = global_id

        self.link_type_lookup = {}

        print("DEBUG: link.dat contents ->", self.info['link.dat'])

        for ltype, (src_name, tgt_name, *_ ) in self.info['link.dat'].items():
            src_type = self.node_name2type[src_name]
            tgt_type = self.node_name2type[tgt_name]
            self.link_type_lookup[(src_type, tgt_type)] = int(ltype)
            self.link_type_lookup[(tgt_type, src_type)] = int(ltype)

    def _load_info(self):
        path = os.path.join(self.data_dir, 'info.dat')
        if not os.path.exists(path): return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _load_nodes(self):
        path = os.path.join(self.data_dir, 'node.dat')
        self.node2id = {}
        self.id2node = {}
        self.node_types = {}
        with open(path, 'r') as f:
            for line in f:
                nid, name, ntype = line.strip().split('\t')
                nid = int(nid)
                ntype = int(ntype)
                self.node2id[name] = nid
                self.id2node[nid] = name
                self.node_types[nid] = ntype

    def _load_graph_links(self):
        """Loads the structural links (from train.dat) for GNN message passing."""
        path = os.path.join(self.data_dir, 'train.dat')
        if not os.path.exists(path): return
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4: continue
                src, tgt, ltype, weight = parts
                self.links['graph'][int(ltype)].append((int(src), int(tgt), float(weight)))

    def _load_lp_splits(self):
        """Loads all pre-split positive and negative link files."""
        print("Loading pre-split link prediction data...")
        split_names = {
            'train_lp': 'train_lp_links.dat',
            'valid_inductive': 'valid_inductive_links.dat',
            'test_transductive': 'test_transductive_links.dat',
            'test_inductive': 'test_inductive_links.dat',
        }
        for split_key, filename in split_names.items():
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        # The format is src, tgt, type, label
                        if len(parts) >= 2:
                            self.links[split_key].append(
                                (int(parts[0]), int(parts[1])) # (src_gid, tgt_gid)
                            )
            print(f"  > Loaded {len(self.links[split_key])} links for {split_key}")

    def _load_cell_features(self):
        print("\nLoading VAE-compatible raw cell expression features...")
        # Fallback to different possible filenames for robustness
        expression_file_paths = [
            'data/embeddings/OmicsExpressionProteinCodingGenesTPMLogp1.csv',
            'data/misc/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
        ]
        
        EXPRESSION_FILE = None
        for path in expression_file_paths:
            if os.path.exists(path):
                EXPRESSION_FILE = path
                break
        
        if EXPRESSION_FILE is None:
            print("  > Warning: No expression file found. Skipping cell feature loading.")
            return

        node_cells = { name.upper(): nid for nid, name in self.id2node.items() if self.node_types[nid] == 0 }
        
        df_expr = pd.read_csv(EXPRESSION_FILE)
        # The first column contains the cell IDs (DepMap IDs)
        expr_ids = df_expr.iloc[:, 0].astype(str).str.upper().tolist()
        expr_id_to_idx = {dep_id: i for i, dep_id in enumerate(expr_ids)}

        common_cells = sorted(set(node_cells.keys()).intersection(expr_id_to_idx.keys()))
        print(f"  > Matched {len(common_cells)} cell lines with expression data.")

        if not common_cells:
            print("  > No common cells found. Skipping feature extraction.")
            return

        idxs = [expr_id_to_idx[cid] for cid in common_cells]
        expression_array = df_expr.iloc[idxs, 1:].astype(np.float32).to_numpy()

        self.cell_features_raw = torch.tensor(expression_array, dtype=torch.float32)
        self.cell_global_id_to_feature_idx = { self.node2id[cid.upper()]: i for i, cid in enumerate(common_cells) }
        self.valid_cell_ids = set(self.cell_global_id_to_feature_idx.keys())
        self.valid_cell_local_ids = [ self.nodes['type_map'][gid][1] for gid in self.valid_cell_ids ]

        print(f"  > Feature shape: {self.cell_features_raw.shape}")
        print("Cell feature loading complete.\n")

    def summary(self):
        print("--- PRELUDE Dataset Summary ---")
        print(f"\nNodes:")
        for ntype_id, count in sorted(self.nodes['count'].items()):
            ntype_name = self.node_type2name.get(ntype_id, f"Type {ntype_id}")
            print(f"  - {ntype_name.capitalize()} (Type {ntype_id}): {count} nodes")

        print(f"\nStructural Links (for GNN message passing from train.dat):")
        total_graph_links = sum(len(edges) for edges in self.links['graph'].values())
        print(f"  > Total structural links for GNN: {total_graph_links}")

        print(f"\nLink Prediction Splits:")
        print(f"  - Training LP Set:       {len(self.links['train_lp'])} links")
        print(f"  - Validation (Inductive): {len(self.links['valid_inductive'])} links")
        print(f"  - Test (Transductive):    {len(self.links['test_transductive'])} links")
        print(f"  - Test (Inductive):       {len(self.links['test_inductive'])} links")
        print("---------------------------------")