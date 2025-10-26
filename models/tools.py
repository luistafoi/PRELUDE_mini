# models/tools.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from collections import defaultdict
import sys

# Custom module imports
# Make sure these paths are correct relative to where you run train.py
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.layers import RnnGnnLayer
# VAE import removed

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
        # Create linear layers to project static features to embed_d
        self.feat_proj = nn.ModuleDict()

        try:
            # Drugs
            if drug_type_id in self.dataset.nodes['count'] and self.feature_loader.drug_features.numel() > 0:
                drug_feat_dim = self.feature_loader.drug_features.shape[1]
                self.feat_proj[str(drug_type_id)] = nn.Linear(drug_feat_dim, self.embed_d).to(device)
                # Initialize weights from loaded features (optional but good)
                # self.feat_proj[str(drug_type_id)].weight.data.copy_(self.feature_loader.drug_features.T) # Example
                print(f"  > Initialized Linear projection for Drugs (Dim: {drug_feat_dim} -> {self.embed_d})")
            else:
                 print(f"  > Warning: No drug nodes or features found. Skipping drug projection.")


            # Genes
            if gene_type_id in self.dataset.nodes['count'] and self.feature_loader.gene_features.numel() > 0:
                gene_feat_dim = self.feature_loader.gene_features.shape[1]
                self.feat_proj[str(gene_type_id)] = nn.Linear(gene_feat_dim, self.embed_d).to(device)
                print(f"  > Initialized Linear projection for Genes (Dim: {gene_feat_dim} -> {self.embed_d})")
            else:
                 print(f"  > Warning: No gene nodes or features found. Skipping gene projection.")


            # Cells (Using Static Embeddings)
            if cell_type_id in self.dataset.nodes['count'] and self.feature_loader.cell_features.numel() > 0:
                cell_feat_dim = self.feature_loader.cell_features.shape[1]
                self.feat_proj[str(cell_type_id)] = nn.Linear(cell_feat_dim, self.embed_d).to(device)
                print(f"  > Initialized Linear projection for Cells (Dim: {cell_feat_dim} -> {self.embed_d}) - Using Static VAE")
            else:
                 print(f"  > Warning: No cell nodes or features found. Skipping cell projection.")

        except AttributeError as e:
            print(f"FATAL ERROR: FeatureLoader might be missing expected feature tensors ({e})")
            sys.exit(1)
        except Exception as e:
             print(f"FATAL ERROR during feature projection setup: {e}")
             sys.exit(1)

        # --- GNN Layers ---
        self.gnn_layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.gnn_layers.append(
                RnnGnnLayer(self.embed_d, self.embed_d, self.node_types) # Assumes RnnGnnLayer uses self.embed_d
            )

        # --- Link Prediction Head ---
        self.lp_bilinear = None
        self.drug_type_name = None
        self.cell_type_name = None
        
        # Optional: Add custom weight initialization if desired
        # self._init_weights()

    # _init_weights can be kept if RnnGnnLayer doesn't handle it
    def _init_weights(self):
        """Initializes weights using Xavier Uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                     nn.init.zeros_(module.bias)
            # Add initialization for RNNs if needed and not done in RnnGnnLayer

    def setup_link_prediction(self, drug_type_name, cell_type_name):
        """Sets up the bilinear layer for link prediction."""
        self.drug_type_name = drug_type_name
        self.cell_type_name = cell_type_name
        # Dimension depends on whether skip connection is used
        factor = 2 if self.args.use_skip_connection else 1
        self.lp_bilinear = nn.Bilinear(self.embed_d * factor, self.embed_d * factor, 1).to(self.device)
        print(f"INFO: Setup link prediction head (Skip Connection: {self.args.use_skip_connection}).")


    def conteng_agg(self, local_id_batch, node_type):
        """Gets the initial node features after projection."""
        if not local_id_batch:
            return torch.empty(0, self.embed_d, device=self.device)

        # Select the correct feature tensor from FeatureLoader based on type
        if node_type == self.dataset.node_name2type['cell']:
            raw_features = self.feature_loader.cell_features[local_id_batch]
        elif node_type == self.dataset.node_name2type['drug']:
            raw_features = self.feature_loader.drug_features[local_id_batch]
        elif node_type == self.dataset.node_name2type['gene']:
            raw_features = self.feature_loader.gene_features[local_id_batch]
        else:
            # Fallback for unexpected types - might indicate an error
            print(f"Warning: Unknown node type {node_type} in conteng_agg. Returning zeros.")
            # Determine expected feature dim dynamically or use self.embed_d
            return torch.zeros(len(local_id_batch), self.embed_d, device=self.device)

        # Project the raw features using the corresponding linear layer
        if str(node_type) in self.feat_proj:
             return self.feat_proj[str(node_type)](raw_features)
        else:
             # Handle cases where projection layer might be missing (e.g., if features weren't loaded)
             print(f"Warning: No projection layer found for type {node_type}. Returning raw features (check dimensions!).")
             # This might cause dimension mismatch errors later if raw_features dim != self.embed_d
             return raw_features # Or return zeros: torch.zeros(len(local_id_batch), self.embed_d, device=self.device)


    def node_het_agg(self, id_batch_local, node_type, data_generator: DataGenerator, excluded_link_types=None):
        """Performs GNN message passing using pre-generated neighbors."""
        if excluded_link_types is None:
            excluded_link_types = set()
            
        if data_generator.train_neighbors is None:
            raise RuntimeError("Training neighbors not loaded. Call load_train_neighbors() first.")

        current_embeds = self.conteng_agg(id_batch_local, node_type)
        node_type_name = self.dataset.node_type2name.get(node_type, f"Type{node_type}") # Safer lookup
        
        for layer in self.gnn_layers:
            # This dictionary will hold sampled neighbor features for each type
            neigh_embeds_by_type = defaultdict(list) # Stores lists of feature tensors
            
            # --- Neighbor Sampling Logic (Simplified & Corrected) ---
            max_neighbors_per_type = 10 # Should be configurable via args
            
            all_neighbor_lists_for_batch = [] # Collect all neighbors first
            for i, local_id in enumerate(id_batch_local):
                center_node_str = f"{node_type_name}{local_id}"
                neighbor_strings = data_generator.train_neighbors.get(center_node_str, [])
                
                # Parse neighbors into {type_id: [local_id, ...]}
                parsed_neighbors_for_node = defaultdict(list)
                for neigh_str in neighbor_strings:
                     # Find the type and local ID for the neighbor string
                     matched = False
                     for nt_name, nt_id in self.dataset.node_name2type.items():
                         if neigh_str.startswith(nt_name):
                             try:
                                 neigh_local_id = int(neigh_str[len(nt_name):])
                                 # Check if this link type should be excluded
                                 current_link_type = self.dataset.link_type_lookup.get((node_type, nt_id), -1)
                                 if current_link_type not in excluded_link_types:
                                     parsed_neighbors_for_node[nt_id].append(neigh_local_id)
                                 matched = True
                                 break # Found the type, move to next neighbor string
                             except ValueError:
                                 continue # Skip if ID part is not an integer
                     # if not matched: print(f"Warning: Could not parse neighbor string: {neigh_str}")
                all_neighbor_lists_for_batch.append(parsed_neighbors_for_node)

            # --- Aggregate across the batch for each neighbor type ---
            final_agg_neighbors_for_layer = {}
            for nt_id in self.node_types:
                 batch_neighbor_features = [] # List to hold tensors for each node in batch
                 for node_neighbors in all_neighbor_lists_for_batch:
                     neigh_list = node_neighbors.get(nt_id, [])
                     
                     # Sample/Pad
                     if len(neigh_list) > max_neighbors_per_type:
                         neigh_list = random.sample(neigh_list, max_neighbors_per_type)
                     elif len(neigh_list) < max_neighbors_per_type:
                         num_to_pad = max_neighbors_per_type - len(neigh_list)
                         num_nodes_of_type = self.dataset.nodes['count'].get(nt_id, 0)
                         if num_nodes_of_type > 0:
                              padding = [random.randint(0, num_nodes_of_type - 1) for _ in range(num_to_pad)]
                              neigh_list.extend(padding)
                         else: # Handle case where neighbor type has 0 nodes
                              neigh_list.extend([0] * num_to_pad) # Pad with index 0, might need adjustment

                     # Get features if neighbors exist
                     if neigh_list:
                         node_neigh_features = self.conteng_agg(neigh_list, nt_id) # Shape: (max_neighbors, embed_d)
                     else: # No neighbors after sampling/padding
                         node_neigh_features = torch.zeros(max_neighbors_per_type, self.embed_d, device=self.device)
                         
                     batch_neighbor_features.append(node_neigh_features)
                     
                 # Stack features for the whole batch for this neighbor type
                 if batch_neighbor_features:
                      final_agg_neighbors_for_layer[nt_id] = torch.stack(batch_neighbor_features) # Shape: (batch_size, max_neighbors, embed_d)
                 else: # Should not happen if id_batch_local is not empty
                      final_agg_neighbors_for_layer[nt_id] = torch.empty(0, max_neighbors_per_type, self.embed_d, device=self.device)

            # Pass aggregated neighbors to the GNN layer
            current_embeds = layer(current_embeds, final_agg_neighbors_for_layer)
        
        return current_embeds


    def get_combined_embedding(self, id_batch_local, node_type, data_generator, excluded_link_types=None):
        """Concatenates initial features with final GNN embeddings if skip connection is enabled."""
        initial_embeds = self.conteng_agg(id_batch_local, node_type)
        final_embeds = self.node_het_agg(id_batch_local, node_type, data_generator, excluded_link_types)
        
        if self.args.use_skip_connection:
            # Ensure dimensions match before concatenating if projection happened
            # Note: This assumes conteng_agg and node_het_agg output self.embed_d
            return torch.cat([initial_embeds, final_embeds], dim=1)
        else:
            return final_embeds

    def link_prediction_loss(self, drug_indices_local, cell_indices_local, labels, data_generator, isolation_ratio=0.0):
        """Calculates link prediction loss. Includes simplified node isolation logic."""
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]

        # Use combined embeddings (handles skip connection internally)
        drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
        cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
        
        # --- Simplified Node Isolation (applied conceptually during embedding generation if flag was used) ---
        # The node isolation logic involving masks is removed as it's complex and we aim for simplicity.
        # The curriculum learning for isolation ratio is handled in train.py by passing 0.0.
        # If args.use_node_isolation was true AND isolation_ratio > 0 was passed,
        # get_combined_embedding would need more complex logic to handle it.
        # Since we removed the curriculum, isolation_ratio will be 0.0, disabling it.

        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return F.binary_cross_entropy_with_logits(scores, labels.float())


    def link_prediction_forward(self, drug_indices_local, cell_indices_local, data_generator):
        """Performs forward pass for link prediction during evaluation."""
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]
        
        # Use combined embeddings (handles skip connection internally)
        drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
        cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
        
        if self.lp_bilinear is None:
             raise RuntimeError("Link prediction head not initialized. Call setup_link_prediction first.")
             
        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return torch.sigmoid(scores)

    # --- Methods Removed ---
    # self_supervised_rw_loss(...) is removed.
    # get_embeddings_for_global_ids(...) is removed.
    # train(mode=True) override is removed.