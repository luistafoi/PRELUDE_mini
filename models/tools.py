# models/tools.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import pickle # Added
from collections import defaultdict
import sys

# Custom module imports
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator # Still passed but not used for neighbors
from models.layers import RnnGnnLayer

# This MUST match the value in generate_neighbors.py
MAX_NEIGHBORS = 10
PAD_VALUE = -1

class HetAgg(nn.Module):
    def __init__(self, args, dataset: PRELUDEDataset, feature_loader: FeatureLoader, device):
        super(HetAgg, self).__init__()
        self.args = args
        self.device = device
        self.embed_d = args.embed_d
        self.dataset = dataset
        self.feature_loader = feature_loader # FeatureLoader now holds static features

        self.node_types = sorted(self.dataset.nodes['count'].keys())
        cell_type_id = self.dataset.node_name2type['cell']
        drug_type_id = self.dataset.node_name2type['drug']
        gene_type_id = self.dataset.node_name2type['gene']

        # --- Feature Projection Layers ---
        self.feat_proj = nn.ModuleDict()
        try:
            # Drugs
            if drug_type_id in self.dataset.nodes['count'] and self.feature_loader.drug_features.numel() > 0:
                drug_feat_dim = self.feature_loader.drug_features.shape[1]
                self.feat_proj[str(drug_type_id)] = nn.Linear(drug_feat_dim, self.embed_d).to(device)
                print(f"  > Initialized Linear projection for Drugs (Dim: {drug_feat_dim} -> {self.embed_d})")
            # Genes
            if gene_type_id in self.dataset.nodes['count'] and self.feature_loader.gene_features.numel() > 0:
                gene_feat_dim = self.feature_loader.gene_features.shape[1]
                self.feat_proj[str(gene_type_id)] = nn.Linear(gene_feat_dim, self.embed_d).to(device)
                print(f"  > Initialized Linear projection for Genes (Dim: {gene_feat_dim} -> {self.embed_d})")
            # Cells
            if cell_type_id in self.dataset.nodes['count'] and self.feature_loader.cell_features.numel() > 0:
                cell_feat_dim = self.feature_loader.cell_features.shape[1]
                self.feat_proj[str(cell_type_id)] = nn.Linear(cell_feat_dim, self.embed_d).to(device)
                print(f"  > Initialized Linear projection for Cells (Dim: {cell_feat_dim} -> {self.embed_d}) - Using Static VAE")
        except Exception as e:
             print(f"FATAL ERROR during feature projection setup: {e}")
             sys.exit(1)

        # --- GNN Layers ---
        self.gnn_layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.gnn_layers.append(
                RnnGnnLayer(self.embed_d, self.embed_d, self.node_types)
            )

        # --- Link Prediction Head ---
        self.lp_bilinear = None
        self.drug_type_name = None
        self.cell_type_name = None
        
        # --- *** START NEW BLOCK: Load Pre-Processed Neighbors *** ---
        neighbor_pkl_path = os.path.join(args.data_dir, "train_neighbors_preprocessed.pkl")
        print(f"  > Loading pre-processed neighbors from {neighbor_pkl_path}...")
        if not os.path.exists(neighbor_pkl_path):
            print(f"FATAL ERROR: {neighbor_pkl_path} not found.")
            print("Please re-run 'scripts/generate_neighbors.py' before training.")
            sys.exit(1)
        
        try:
            with open(neighbor_pkl_path, "rb") as f:
                self.precomputed_neighbors = pickle.load(f)
            print("  > Pre-processed neighbors loaded successfully.")
        except Exception as e:
            print(f"FATAL ERROR loading neighbor pickle file: {e}")
            sys.exit(1)
            
        # Create a fallback/empty neighbor set for nodes not in the dict
        self.empty_neighbors_by_type = {
            nt_id: [PAD_VALUE] * MAX_NEIGHBORS for nt_id in self.node_types
        }
        # --- *** END NEW BLOCK *** ---


    def setup_link_prediction(self, drug_type_name, cell_type_name):
        """Sets up the bilinear layer for link prediction."""
        self.drug_type_name = drug_type_name
        self.cell_type_name = cell_type_name
        factor = 2 if self.args.use_skip_connection else 1
        self.lp_bilinear = nn.Bilinear(self.embed_d * factor, self.embed_d * factor, 1).to(self.device)
        print(f"INFO: Setup link prediction head (Skip Connection: {self.args.use_skip_connection}).")


    def conteng_agg(self, local_id_batch, node_type):
        """Gets the initial node features after projection."""
        
        # --- This is our robust fix from before ---
        if isinstance(local_id_batch, torch.Tensor):
            if local_id_batch.numel() == 0:
                return torch.tensor([], device=self.device)
        else:
            if not local_id_batch:
                return torch.tensor([], device=self.device)
        # --- End fix ---

        # Select the correct feature tensor from FeatureLoader based on type
        if node_type == self.dataset.node_name2type['cell']:
            raw_features = self.feature_loader.cell_features[local_id_batch]
        elif node_type == self.dataset.node_name2type['drug']:
            raw_features = self.feature_loader.drug_features[local_id_batch]
        elif node_type == self.dataset.node_name2type['gene']:
            raw_features = self.feature_loader.gene_features[local_id_batch]
        else:
            print(f"Warning: Unknown node type {node_type} in conteng_agg. Returning zeros.")
            # Check if input is a tensor or list to determine batch size
            batch_len = local_id_batch.size(0) if isinstance(local_id_batch, torch.Tensor) else len(local_id_batch)
            return torch.zeros(batch_len, self.embed_d, device=self.device)

        # Project the raw features
        if str(node_type) in self.feat_proj:
             return self.feat_proj[str(node_type)](raw_features)
        else:
             print(f"Warning: No projection layer for type {node_type}.")
             return raw_features


    def node_het_agg(self, id_batch_local, node_type, data_generator: DataGenerator, excluded_link_types=None):
        """Performs GNN message passing using pre-processed neighbors."""
        # Note: data_generator and excluded_link_types are no longer needed here,
        # as this logic is now baked into the pre-processed file.
        
        current_embeds = self.conteng_agg(id_batch_local, node_type)
        batch_size = len(id_batch_local)

        for layer in self.gnn_layers:
            final_agg_neighbors_for_layer = {}
            
            # --- *** START VECTORIZED NEIGHBOR LOOKUP *** ---
            
            for nt_id in self.node_types: # Loop over neighbor *types* (e.g., cell, drug, gene)
                
                # 1. Look up pre-padded neighbor LIDs for the whole batch
                # This is a fast Python list comprehension
                batch_neighbor_lids_list = [
                    self.precomputed_neighbors.get(
                        (node_type, lid.item()), self.empty_neighbors_by_type
                    )[nt_id] 
                    for lid in id_batch_local
                ]
                
                # 2. Convert to a tensor of shape (batch_size, MAX_NEIGHBORS)
                # This is fast.
                batch_neighbor_lids_tensor = torch.tensor(
                    batch_neighbor_lids_list, dtype=torch.long, device=self.device
                ) # Shape: [6000, 10]
                
                # 3. Get features for all neighbors at once
                # Flatten to shape [60000]
                batch_neighbor_lids_tensor_flat = batch_neighbor_lids_tensor.flatten()
                
                # Create a mask for valid LIDs (all except PAD_VALUE)
                valid_mask = (batch_neighbor_lids_tensor_flat != PAD_VALUE)
                
                # Get LIDs that are not padding
                valid_lids = batch_neighbor_lids_tensor_flat[valid_mask]
                
                # Create a feature tensor of all zeros
                # Shape: [60000, embed_d]
                final_neighbor_features_flat = torch.zeros(
                    batch_neighbor_lids_tensor_flat.numel(), 
                    self.embed_d, 
                    device=self.device
                )
                
                # If there are any valid neighbors at all...
                if valid_lids.numel() > 0:
                    # ...get their features. This is a single, batched call.
                    neighbor_features = self.conteng_agg(valid_lids, nt_id)
                    # ...and scatter them into the correct positions.
                    final_neighbor_features_flat[valid_mask] = neighbor_features
                
                # 4. Reshape back to (batch_size, MAX_NEIGHBORS, embed_d)
                final_agg_neighbors_for_layer[nt_id] = final_neighbor_features_flat.view(
                    batch_size, MAX_NEIGHBORS, self.embed_d
                )
            
            # --- *** END VECTORIZED NEIGHBOR LOOKUP *** ---

            # Pass aggregated neighbors to the GNN layer
            current_embeds = layer(current_embeds, final_agg_neighbors_for_layer)
        
        return current_embeds


    def get_combined_embedding(self, id_batch_local, node_type, data_generator, excluded_link_types=None):
        """Concatenates initial features with final GNN embeddings if skip connection is enabled."""
        initial_embeds = self.conteng_agg(id_batch_local, node_type)
        final_embeds = self.node_het_agg(id_batch_local, node_type, data_generator, excluded_link_types)
        
        if self.args.use_skip_connection:
            return torch.cat([initial_embeds, final_embeds], dim=1)
        else:
            return final_embeds

    def link_prediction_loss(self, drug_indices_local, cell_indices_local, labels, data_generator, isolation_ratio=0.0):
        """Calculates link prediction loss."""
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]

        # Get embeddings. Note: data_generator is no longer used by node_het_agg
        drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
        cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
        
        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return F.binary_cross_entropy_with_logits(scores, labels.float())


    def link_prediction_forward(self, drug_indices_local, cell_indices_local, data_generator):
        """Performs forward pass for link prediction during evaluation."""
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]
        
        drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
        cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
        
        if self.lp_bilinear is None:
             raise RuntimeError("Link prediction head not initialized. Call setup_link_prediction first.")
             
        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return torch.sigmoid(scores)
