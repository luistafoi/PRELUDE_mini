# models/tools.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from collections import defaultdict

# Custom module imports
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.layers import RnnGnnLayer
from scripts.cell_vae import CellLineVAE


class HetAgg(nn.Module):
    def __init__(self, args, dataset: PRELUDEDataset, feature_loader: FeatureLoader, device):
        super(HetAgg, self).__init__()
        self.args = args
        self.device = device
        self.embed_d = args.embed_d
        self.dataset = dataset
        self.feature_loader = feature_loader
        
        self.node_types = sorted(self.dataset.nodes['count'].keys())
        cell_type_id = self.dataset.node_name2type['cell']
        drug_type_id = self.dataset.node_name2type['drug']
        gene_type_id = self.dataset.node_name2type['gene']
        
        # --- 1. Feature Projection Layers (Instead of nn.Embedding) ---
        self.feat_proj = nn.ModuleDict()
        
        drug_feat_dim = self.feature_loader.drug_features.shape[1]
        self.feat_proj[str(drug_type_id)] = nn.Linear(drug_feat_dim, self.embed_d).to(device)
        
        gene_feat_dim = self.feature_loader.gene_features.shape[1]
        self.feat_proj[str(gene_type_id)] = nn.Linear(gene_feat_dim, self.embed_d).to(device)

        # For cells, we use the VAE encoder, feature handling depends on the chosen mode
        if args.use_vae_encoder:
            print("INFO: Initializing VAE encoder for DYNAMIC cell features.")
            vae_dims = list(map(int, args.vae_dims.split(',')))
            full_vae = CellLineVAE(vae_dims).to(device)
            if os.path.exists(args.vae_checkpoint):
                full_vae.load_state_dict(torch.load(args.vae_checkpoint, weights_only=True))
                print(f"Loaded VAE weights from: {args.vae_checkpoint}")
            self.cell_encoder = full_vae.encoder
        
        elif args.use_static_cell_embeddings:
            print("INFO: Using STATIC pre-computed VAE embeddings for cell features.")
            cell_feat_dim = self.feature_loader.cell_features.shape[1]
            self.feat_proj[str(cell_type_id)] = nn.Linear(cell_feat_dim, self.embed_d).to(device)

        else: # Default 'random' mode
            print("INFO: Using learnable nn.Embedding for RANDOM cell features.")
            num_cells = self.dataset.nodes['count'][cell_type_id]
            self.feat_proj[str(cell_type_id)] = nn.Embedding(num_cells, self.embed_d).to(device)

        # --- 2. GNN Layers (Restored from old model) ---
        self.gnn_layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.gnn_layers.append(
                RnnGnnLayer(self.embed_d, self.embed_d, self.node_types)
            )

        # --- 3. Link Prediction Head ---
        self.lp_bilinear = None
        self.drug_type_name = None
        self.cell_type_name = None

    def train(self, mode=True):
        """
        Overrides the default train method to ensure the VAE encoder always
        stays in evaluation mode (as it's a pre-trained feature extractor).
        """
        super(HetAgg, self).train(mode)
        
        if hasattr(self, 'cell_encoder'):
            # Always keep the VAE encoder in evaluation mode
            self.cell_encoder.eval()
        return self

    def setup_link_prediction(self, drug_type_name, cell_type_name):
        self.drug_type_name = drug_type_name
        self.cell_type_name = cell_type_name
        factor = 2 if self.args.use_skip_connection else 1
        self.lp_bilinear = nn.Bilinear(self.embed_d * factor, self.embed_d * factor, 1).to(self.device)

    def conteng_agg(self, local_id_batch, node_type):
        """Gets the initial node features (before message passing)."""
        if not local_id_batch:
            return torch.empty(0, self.embed_d, device=self.device)
            
        cell_type_id = self.dataset.node_name2type['cell']

        # --- This is the main block for cell feature logic ---
        if node_type == cell_type_id:
            if self.args.use_vae_encoder:
                # Mode 1: Dynamic VAE (slower)
                # Runs the VAE encoder on-the-fly for the batch.
                global_id_batch = [self.dataset.local_to_global_map[node_type][lid] for lid in local_id_batch]
                final_encoded_features = torch.zeros(len(local_id_batch), self.embed_d, device=self.device)
                
                valid_gids, original_indices = [], []
                for i, gid in enumerate(global_id_batch):
                    if gid in self.dataset.cell_global_id_to_feature_idx:
                        valid_gids.append(gid)
                        original_indices.append(i)
                
                if valid_gids:
                    feature_indices = [self.dataset.cell_global_id_to_feature_idx[gid] for gid in valid_gids]
                    raw_features = self.dataset.cell_features_raw[feature_indices].to(self.device)
                    encoded_features = self.cell_encoder(raw_features)
                    final_encoded_features[original_indices] = encoded_features
                
                return final_encoded_features

            elif self.args.use_static_cell_embeddings:
                # Mode 2: Static VAE (faster)
                # Looks up pre-computed embeddings from the FeatureLoader and projects them.
                raw_features = self.feature_loader.cell_features[local_id_batch]
                return self.feat_proj[str(node_type)](raw_features)
            
            else: 
                # Mode 3: Default 'random' mode
                # Uses a standard, learnable nn.Embedding layer.
                local_indices_tensor = torch.LongTensor(local_id_batch).to(self.device)
                return self.feat_proj[str(node_type)](local_indices_tensor)
        
        # --- Logic for Drug and Gene Features (this part is the same for all modes) ---
        elif node_type == self.dataset.node_name2type['drug']:
            raw_features = self.feature_loader.drug_features[local_id_batch]
            return self.feat_proj[str(node_type)](raw_features)
            
        elif node_type == self.dataset.node_name2type['gene']:
            raw_features = self.feature_loader.gene_features[local_id_batch]
            return self.feat_proj[str(node_type)](raw_features)

    def node_het_agg(self, id_batch_local, node_type, data_generator: DataGenerator, excluded_link_types=None):
        """Performs full GNN message passing using the pre-generated neighbor list."""

        if excluded_link_types is None:
            excluded_link_types = set()
        
        if data_generator.train_neighbors is None:
            raise RuntimeError("Training neighbors not loaded in DataGenerator. Call load_train_neighbors() first.")

        current_embeds = self.conteng_agg(id_batch_local, node_type)
        node_type_name = self.dataset.node_type2name[node_type]
        
        for layer in self.gnn_layers:
            neigh_embeds_by_type = defaultdict(list)
            
            for i, local_id in enumerate(id_batch_local):
                center_node_str = f"{node_type_name}{local_id}"
                neighbor_strings = data_generator.train_neighbors.get(center_node_str, [])
                
                parsed_neighbors = defaultdict(list)
                for neigh_str in neighbor_strings:
                    for nt_name, nt_id in self.dataset.node_name2type.items():
                        if neigh_str.startswith(nt_name):
                            try:
                                neigh_local_id = int(neigh_str[len(nt_name):])
                                
                                # --- START OF FIX ---
                                # Find the link type for this specific neighbor interaction
                                neighbor_node_type = self.dataset.node_name2type[nt_name]
                                current_link_type = self.dataset.link_type_lookup.get((node_type, neighbor_node_type), -1)

                                # Only add the neighbor if its link type is not excluded
                                if current_link_type not in excluded_link_types:
                                    parsed_neighbors[nt_id].append(neigh_local_id)
                                # --- END OF FIX ---
                                
                                break 
                            except ValueError:
                                continue

                for nt in self.node_types:
                    num_samples = 10 # This should be configured in args eventually
                    neigh_list = parsed_neighbors.get(nt, [])
                    
                    if len(neigh_list) > num_samples:
                        neigh_list = random.sample(neigh_list, num_samples)
                    elif len(neigh_list) < num_samples:
                        num_to_pad = num_samples - len(neigh_list)
                        cell_type_id = self.dataset.node_name2type['cell']
                        if nt == cell_type_id:
                            padding = [random.choice(self.dataset.valid_cell_local_ids) for _ in range(num_to_pad)]
                        else:
                            num_nodes_of_type = self.dataset.nodes['count'][nt]
                            padding = [random.randint(0, num_nodes_of_type - 1) for _ in range(num_to_pad)]
                        neigh_list.extend(padding)
                    
                    neigh_embeds_by_type[nt].append(neigh_list)
            
            aggregated_neighbors = {}
            for nt, batched_neigh_ids in neigh_embeds_by_type.items():
                flat_ids = [item for sublist in batched_neigh_ids for item in sublist]
                num_samples = 10 # Must match the sampling number above
                if flat_ids:
                    neigh_feats = self.conteng_agg(flat_ids, nt)
                    aggregated_neighbors[nt] = neigh_feats.view(len(id_batch_local), num_samples, self.embed_d)
                else:
                    aggregated_neighbors[nt] = torch.zeros(len(id_batch_local), num_samples, self.embed_d, device=self.device)

            current_embeds = layer(current_embeds, aggregated_neighbors)
        
        return current_embeds

    def get_combined_embedding(self, id_batch_local, node_type, data_generator, excluded_link_types=None):
        """Concatenates initial features with final GNN embeddings for skip connections."""
        initial_embeds = self.conteng_agg(id_batch_local, node_type)
        final_embeds = self.node_het_agg(id_batch_local, node_type, data_generator, excluded_link_types)
        
        if self.args.use_skip_connection:
            return torch.cat([initial_embeds, final_embeds], dim=1)
        else:
            return final_embeds

    def link_prediction_loss(self, drug_indices_local, cell_indices_local, labels, data_generator, isolation_ratio=0.0):
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]

        # --- New Curriculum Node Isolation Logic ---
        if self.args.use_node_isolation and isolation_ratio > 0 and self.training:
            # 1. Identify unique cells in the batch and select which ones to isolate
            unique_cell_ids = list(set(cell_indices_local))
            num_to_isolate = int(len(unique_cell_ids) * isolation_ratio)
            if num_to_isolate == 0:
                # Fallback to original logic if no cells would be isolated
                drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
                cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
                scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
                return F.binary_cross_entropy_with_logits(scores, labels.float())

            cells_to_isolate = set(random.sample(unique_cell_ids, k=num_to_isolate))
            
            # 2. Create masks for isolated and non-isolated links in the batch
            is_isolated_mask = torch.tensor([c in cells_to_isolate for c in cell_indices_local], device=self.device)
            is_not_isolated_mask = ~is_isolated_mask

            total_loss = 0.0
            
            # 3. Calculate loss for non-isolated links (standard message passing)
            if torch.any(is_not_isolated_mask):
                non_isolated_drugs = [d for i, d in enumerate(drug_indices_local) if is_not_isolated_mask[i]]
                non_isolated_cells = [c for i, c in enumerate(cell_indices_local) if is_not_isolated_mask[i]]
                non_isolated_labels = labels[is_not_isolated_mask]

                drug_embeds1 = self.get_combined_embedding(non_isolated_drugs, drug_type_id, data_generator)
                cell_embeds1 = self.get_combined_embedding(non_isolated_cells, cell_type_id, data_generator)
                
                scores1 = self.lp_bilinear(drug_embeds1, cell_embeds1).squeeze(-1)
                total_loss += F.binary_cross_entropy_with_logits(scores1, non_isolated_labels.float())

            # 4. Calculate loss for isolated links (cells are drug-ignorant)
            if torch.any(is_isolated_mask):
                isolated_drugs = [d for i, d in enumerate(drug_indices_local) if is_isolated_mask[i]]
                isolated_cells = [c for i, c in enumerate(cell_indices_local) if is_isolated_mask[i]]
                isolated_labels = labels[is_isolated_mask]

                # Get drug embeddings normally
                drug_embeds2 = self.get_combined_embedding(isolated_drugs, drug_type_id, data_generator)
                
                # Get cell embeddings with drug links excluded for message passing
                cell_drug_link_type = self.dataset.link_type_lookup.get((cell_type_id, drug_type_id), -1)
                if cell_drug_link_type == -1:
                    raise ValueError("Could not find cell-drug link type in link_type_lookup")

                cell_embeds2 = self.get_combined_embedding(
                    isolated_cells, 
                    cell_type_id, 
                    data_generator, 
                    excluded_link_types={cell_drug_link_type}
                )
                
                scores2 = self.lp_bilinear(drug_embeds2, cell_embeds2).squeeze(-1)
                total_loss += F.binary_cross_entropy_with_logits(scores2, isolated_labels.float())
                
            return total_loss

        else: # Original logic if isolation is off or not in training mode
            drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
            cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
            scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
            return F.binary_cross_entropy_with_logits(scores, labels.float())

    def link_prediction_forward(self, drug_indices_local, cell_indices_local, data_generator):
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]
        
        drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
        cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
        
        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return torch.sigmoid(scores)

    def get_embeddings_for_global_ids(self, global_ids, data_generator):
        """Gathers final GNN embeddings for a heterogeneous batch of global node IDs."""
        if not global_ids:
            return torch.empty(0, self.embed_d, device=self.device)

        ids_by_type = defaultdict(list)
        indices_by_type = defaultdict(list)

        for i, gid in enumerate(global_ids):
            node_type, local_id = self.dataset.nodes['type_map'][gid]
            ids_by_type[node_type].append(local_id)
            indices_by_type[node_type].append(i)

        factor = 2 if self.args.use_skip_connection else 1
        output_embeds = torch.zeros(len(global_ids), self.embed_d * factor, device=self.device)

        for n_type, id_batch in ids_by_type.items():
            if id_batch:
                embeds = self.get_combined_embedding(id_batch, n_type, data_generator)
                original_indices = torch.tensor(indices_by_type[n_type], device=self.device, dtype=torch.long)
                output_embeds.index_copy_(0, original_indices, embeds)

        return output_embeds

    def self_supervised_rw_loss(self, triple_list_batch, data_generator):
        """Calculates skip-gram loss for random walk triples."""
        center_ids = [t[0] for t in triple_list_batch]
        pos_ids = [t[1] for t in triple_list_batch]
        neg_ids = [t[2] for t in triple_list_batch]
        
        c_embed = self.get_embeddings_for_global_ids(center_ids, data_generator)
        p_embed = self.get_embeddings_for_global_ids(pos_ids, data_generator)
        n_embed = self.get_embeddings_for_global_ids(neg_ids, data_generator)
        
        p_score = torch.sum(torch.mul(c_embed, p_embed), dim=1)
        n_score = torch.sum(torch.mul(c_embed, n_embed), dim=1)
        
        return -torch.mean(F.logsigmoid(p_score) + F.logsigmoid(-n_score))