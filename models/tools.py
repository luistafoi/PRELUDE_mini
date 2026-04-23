# models/tools.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
import sys
import argparse

# Custom module imports
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.layers import HetGNNLayer

PAD_VALUE = -1


@dataclass
class SubgraphInfo:
    """Holds subgraph topology for mini-batch GNN propagation."""
    node_sets: Dict[int, torch.Tensor]               # {type: (S,) LongTensor of local IDs}
    lid_to_pos: Dict[int, torch.Tensor]               # {type: (N_full,) LongTensor} full lid -> subgraph pos (-1 if absent)
    sub_neighbor_lids: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)   # {ct: {nt: (S_ct, M)}}
    sub_neighbor_weights: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    sub_neighbor_masks: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)


class HetAgg(nn.Module):
    def __init__(self, args, dataset: PRELUDEDataset, feature_loader: FeatureLoader, device):
        super(HetAgg, self).__init__()
        self.args = args
        self.device = device
        self.embed_d = args.embed_d
        self.max_neighbors = getattr(args, 'max_neighbors', 10)
        self.dataset = dataset
        self.feature_loader = feature_loader

        self.node_types = sorted(self.dataset.nodes['count'].keys())
        cell_type_id = self.dataset.node_name2type['cell']
        drug_type_id = self.dataset.node_name2type['drug']
        gene_type_id = self.dataset.node_name2type['gene']

        # --- Feature Projection Layers (with ReLU and Dropout) ---
        self.feat_proj = nn.ModuleDict()
        dropout_rate = args.dropout
        try:
            # Drugs
            if drug_type_id in self.dataset.nodes['count'] and self.feature_loader.drug_features.numel() > 0:
                drug_feat_dim = self.feature_loader.drug_features.shape[1]
                self.feat_proj[str(drug_type_id)] = nn.Sequential(
                    nn.Linear(drug_feat_dim, self.embed_d),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate)
                ).to(device)
                print(f"  > Initialized Linear->ReLU->Dropout projection for Drugs (Dim: {drug_feat_dim} -> {self.embed_d})")

            # Genes
            if gene_type_id in self.dataset.nodes['count'] and self.feature_loader.gene_features.numel() > 0:
                gene_feat_dim = self.feature_loader.gene_features.shape[1]
                self.feat_proj[str(gene_type_id)] = nn.Sequential(
                    nn.Linear(gene_feat_dim, self.embed_d),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate)
                ).to(device)
                print(f"  > Initialized Linear->ReLU->Dropout projection for Genes (Dim: {gene_feat_dim} -> {self.embed_d})")

            # Cells
            if cell_type_id in self.dataset.nodes['count'] and self.feature_loader.cell_features.numel() > 0:
                cell_feat_dim = self.feature_loader.cell_features.shape[1]
                gene_enc_dim = getattr(args, 'gene_encoder_dim', 0)

                cell_feat_source = getattr(args, 'cell_feature_source', 'vae')
                use_gene_enc = gene_enc_dim > 0 and cell_feat_source in ('multiomic', 'hybrid')

                if use_gene_enc:
                    # Option B: per-gene MLP encoder for multiomic features
                    # For hybrid: first vae_dim dims are VAE, rest is multiomic (n_genes*4 + 2)
                    n_channels = 4
                    if cell_feat_source == 'hybrid':
                        self.hybrid_vae_dim = 512
                        multiomic_dim = cell_feat_dim - self.hybrid_vae_dim
                    else:
                        self.hybrid_vae_dim = 0
                        multiomic_dim = cell_feat_dim
                    self.n_target_genes = (multiomic_dim - 2) // n_channels  # 964
                    self.gene_encoder = nn.Sequential(
                        nn.Linear(n_channels, gene_enc_dim),
                        nn.ReLU(),
                        nn.Linear(gene_enc_dim, gene_enc_dim),
                    ).to(device)
                    # flat_dim = gene_encoded + flags + (VAE if hybrid)
                    flat_dim = self.n_target_genes * gene_enc_dim + 2 + self.hybrid_vae_dim
                    self.feat_proj[str(cell_type_id)] = nn.Sequential(
                        nn.Linear(flat_dim, self.embed_d),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_rate)
                    ).to(device)
                    vae_note = f" + VAE({self.hybrid_vae_dim})" if self.hybrid_vae_dim > 0 else ""
                    print(f"  > Initialized Gene Encoder MLP({n_channels}->{gene_enc_dim}) + projection for Cells "
                          f"({self.n_target_genes} genes × {gene_enc_dim} + 2 flags{vae_note} = {flat_dim} -> {self.embed_d})")
                else:
                    self.feat_proj[str(cell_type_id)] = nn.Sequential(
                        nn.Linear(cell_feat_dim, self.embed_d),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_rate)
                    ).to(device)
                    print(f"  > Initialized Linear->ReLU->Dropout projection for Cells (Dim: {cell_feat_dim} -> {self.embed_d})")

            # --- Cross-Attention Gene Encoder (Option D) ---
            if getattr(args, 'use_cross_attention', False):
                ca_dim = getattr(args, 'cross_attn_dim', 32)
                n_channels = 4
                total_feat_dim = self.feature_loader.cell_features.shape[1]
                vae_dim = getattr(self, 'hybrid_vae_dim', 0)
                ca_n_genes = (total_feat_dim - vae_dim - 2) // n_channels
                self.ca_n_genes = ca_n_genes
                self.ca_dim = ca_dim
                self.ca_gene_encoder = nn.Sequential(
                    nn.Linear(n_channels, ca_dim),
                    nn.ReLU(),
                    nn.Linear(ca_dim, ca_dim),
                ).to(device)
                print(f"  > Initialized Cross-Attention gene encoder: MLP({n_channels}->{ca_dim}) "
                      f"for {ca_n_genes} genes")

        except Exception as e:
             print(f"FATAL ERROR during feature projection setup: {e}")
             sys.exit(1)

        # --- Weighted aggregation pairs (parsed from --weighted_agg_pairs) ---
        # Maps center_type_id -> set(neigh_type_id) for pairs using proper weighted mean.
        self.weighted_agg_pairs = {}
        pairs_str = getattr(args, 'weighted_agg_pairs', '') or ''
        if pairs_str.strip():
            for tok in pairs_str.split(','):
                tok = tok.strip()
                if ':' not in tok:
                    continue
                ct_name, nt_name = tok.split(':', 1)
                ct_name, nt_name = ct_name.strip(), nt_name.strip()
                ct_id = self.dataset.node_name2type.get(ct_name)
                nt_id = self.dataset.node_name2type.get(nt_name)
                if ct_id is None or nt_id is None:
                    print(f"  > WARN: unknown type in weighted_agg_pairs: '{tok}' — skipped")
                    continue
                self.weighted_agg_pairs.setdefault(ct_id, set()).add(nt_id)
            print(f"  > Weighted-mean aggregation enabled for pairs: "
                  f"{ {self.dataset.node_type2name[ct]: sorted(self.dataset.node_type2name[nt] for nt in nts) for ct, nts in self.weighted_agg_pairs.items()} }")

        # --- GNN Layers ---
        self.gnn_layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.gnn_layers.append(
                HetGNNLayer(self.embed_d, self.embed_d, self.node_types, dropout_rate=args.dropout)
            )

        # --- Link Prediction Head ---
        self.lp_bilinear = None
        self.drug_type_name = None
        self.cell_type_name = None

        # --- Load Pre-Processed Neighbors & Build Index Tensors ---
        neighbor_pkl_path = os.path.join(args.data_dir, "train_neighbors_preprocessed.pkl")
        print(f"  > Loading pre-processed neighbors from {neighbor_pkl_path}...")
        if not os.path.exists(neighbor_pkl_path):
            print(f"FATAL ERROR: {neighbor_pkl_path} not found.")
            sys.exit(1)

        try:
            with open(neighbor_pkl_path, "rb") as f:
                self.precomputed_neighbors = pickle.load(f)
            print("  > Pre-processed neighbors loaded successfully.")
        except Exception as e:
            print(f"FATAL ERROR loading neighbor pickle file: {e}")
            sys.exit(1)

        self.empty_neighbors_by_type = {
            nt_id: {'ids': [PAD_VALUE] * self.max_neighbors, 'weights': [0.0] * self.max_neighbors}
            for nt_id in self.node_types
        }

        # --- Pre-build neighbor index tensors for full-graph forward ---
        self._build_neighbor_tensors()

    def _build_neighbor_tensors(self):
        """Convert precomputed_neighbors dict into GPU tensors for vectorized lookup.

        Creates self.neighbor_lids[center_type][neigh_type] -> (N, M) LongTensor
        and self.neighbor_weights[center_type][neigh_type] -> (N, M) FloatTensor
        and self.neighbor_masks[center_type][neigh_type] -> (N, M) BoolTensor
        where N = num nodes of center_type, M = max_neighbors.
        """
        self.neighbor_lids = {}
        self.neighbor_weights = {}
        self.neighbor_masks = {}

        for ct in self.node_types:
            n_nodes = self.dataset.nodes['count'][ct]
            self.neighbor_lids[ct] = {}
            self.neighbor_weights[ct] = {}
            self.neighbor_masks[ct] = {}

            for nt in self.node_types:
                lids = torch.full((n_nodes, self.max_neighbors), PAD_VALUE, dtype=torch.long)
                weights = torch.zeros(n_nodes, self.max_neighbors, dtype=torch.float)

                for lid in range(n_nodes):
                    entry = self.precomputed_neighbors.get((ct, lid), self.empty_neighbors_by_type)
                    neigh_data = entry[nt]
                    for j in range(self.max_neighbors):
                        lids[lid, j] = neigh_data['ids'][j]
                        weights[lid, j] = neigh_data['weights'][j]

                mask = (lids != PAD_VALUE)
                # Replace PAD with 0 for safe indexing (masked out later)
                safe_lids = lids.clone()
                safe_lids[~mask] = 0

                self.neighbor_lids[ct][nt] = safe_lids.to(self.device)
                self.neighbor_weights[ct][nt] = weights.to(self.device)
                self.neighbor_masks[ct][nt] = mask.to(self.device)

        total_entries = sum(
            self.neighbor_lids[ct][nt].numel()
            for ct in self.node_types for nt in self.node_types
        )
        print(f"  > Pre-built neighbor index tensors: {total_entries} entries across {len(self.node_types)**2} pairs")

    def setup_link_prediction(self, drug_type_name, cell_type_name):
        """Sets up the bilinear layer for link prediction."""
        self.drug_type_name = drug_type_name
        self.cell_type_name = cell_type_name

        # --- Residual GNN parameters (M15) ---
        self.residual_scale_params = None
        self.residual_scale_fixed = None
        if getattr(self.args, 'use_residual_gnn', False):
            if self.args.residual_scale > 0:
                self.residual_scale_fixed = self.args.residual_scale
                print(f"INFO: Residual GNN with FIXED scale = {self.residual_scale_fixed}")
            else:
                self.residual_scale_params = nn.ParameterDict({
                    str(nt_id): nn.Parameter(torch.zeros(1, device=self.device))
                    for nt_id in self.node_types
                })
                print(f"INFO: Residual GNN with LEARNED per-type scale (init sigmoid(0)=0.5).")

        if getattr(self.args, 'use_node_gate', False):
            # Per-node adaptive skip gate: MLP(cat(projection, gnn_output)) -> scalar alpha
            self.node_gate_mlp = nn.Sequential(
                nn.Linear(self.embed_d * 2, self.embed_d),
                nn.ReLU(),
                nn.Linear(self.embed_d, 1),
            ).to(self.device)
            print(f"INFO: Setup PER-NODE adaptive skip gate MLP (input: {self.embed_d * 2}, output: 1).")
        elif self.args.use_skip_connection:
            # Per-type gated skip: separate learnable gate for each node type
            self.skip_gates = nn.ParameterDict({
                str(nt_id): nn.Parameter(torch.zeros(self.embed_d, device=self.device))
                for nt_id in self.node_types
            })
            print(f"INFO: Setup PER-TYPE gated skip ({len(self.node_types)} gates, initialized at 0.5).")

        # Compute pos_weight from training LP links for class-balanced BCE
        if getattr(self.args, 'regression', False):
            self.lp_pos_weight = torch.tensor([1.0], device=self.device)
            print("INFO: Regression mode — using MAE loss (no pos_weight)")
        else:
            train_lp = self.dataset.links.get('train_lp', [])
            if train_lp:
                n_pos = sum(1 for _, _, l in train_lp if l > 0.5)
                n_neg = len(train_lp) - n_pos
                self.lp_pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=self.device)
                print(f"INFO: BCE pos_weight = {self.lp_pos_weight.item():.2f} (neg={n_neg}, pos={n_pos})")
            else:
                self.lp_pos_weight = torch.tensor([1.0], device=self.device)

        # --- Cross-Attention scoring head (Option D) ---
        if getattr(self.args, 'use_cross_attention', False):
            ca_dim = self.ca_dim
            embed_dim = self.embed_d
            # Drug query projection: drug_embed (embed_d) -> query (ca_dim)
            self.ca_drug_query = nn.Linear(embed_dim, ca_dim).to(self.device)
            # Scoring MLP: concat(drug_embed, cell_context) -> score
            self.ca_score_head = nn.Sequential(
                nn.Linear(embed_dim + ca_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(p=self.args.dropout),
                nn.Linear(embed_dim, 1),
            ).to(self.device)
            self.lp_bilinear = None
            print(f"INFO: Setup Cross-Attention scoring head (drug query: {embed_dim}->{ca_dim}, "
                  f"score MLP: {embed_dim + ca_dim}->1)")

        embed_dim = self.embed_d
        if getattr(self.args, 'use_mlp_head', False):
            self.lp_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(p=self.args.dropout),
                nn.Linear(embed_dim, 1),
            ).to(self.device)
            self.lp_bilinear = None
            print(f"INFO: Setup MLP link prediction head (input: {embed_dim * 2}).")
        else:
            self.lp_bilinear = nn.Bilinear(embed_dim, embed_dim, 1).to(self.device)
            print(f"INFO: Setup bilinear link prediction head (Dim: {embed_dim}).")

        # Dual head: separate inductive scoring head
        self.lp_bilinear_ind = None
        if getattr(self.args, 'dual_head', False):
            self.lp_bilinear_ind = nn.Bilinear(embed_dim, embed_dim, 1).to(self.device)
            print(f"INFO: Setup DUAL HEAD — inductive bilinear head (Dim: {embed_dim}).")


    def _score_pairs(self, drug_embeds, cell_embeds, use_inductive_head=False):
        """Score drug-cell pairs using either bilinear or MLP head. Returns (B,) logits."""
        if use_inductive_head and self.lp_bilinear_ind is not None:
            return self.lp_bilinear_ind(drug_embeds, cell_embeds).squeeze(-1)
        if self.lp_bilinear is not None:
            return self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        else:
            return self.lp_mlp(torch.cat([drug_embeds, cell_embeds], dim=-1)).squeeze(-1)

    def _get_cell_gene_tokens(self, cell_lids):
        """Get per-gene token embeddings for cells (cross-attention scoring).

        Args:
            cell_lids: (B,) tensor of cell local IDs

        Returns:
            (B, n_genes, ca_dim) gene token embeddings
        """
        raw = self.feature_loader.cell_features[cell_lids]  # (B, feat_dim)
        vae_dim = getattr(self, 'hybrid_vae_dim', 0)
        multiomic = raw[:, vae_dim:] if vae_dim > 0 else raw
        gene_data = multiomic[:, :-2]  # (B, n_genes*4) — strip flags
        B = gene_data.shape[0]
        gene_data = gene_data.view(B, self.ca_n_genes, 4)  # (B, 964, 4)
        return self.ca_gene_encoder(gene_data)  # (B, 964, ca_dim)

    def _cross_attention_score(self, drug_embeds, cell_lids):
        """Score drug-cell pairs via cross-attention.

        Drug embedding attends over cell's gene tokens to produce
        a drug-specific cell context, then scores via MLP.

        Args:
            drug_embeds: (B, embed_d) drug embeddings (from projection or GNN)
            cell_lids: (B,) cell local IDs

        Returns:
            (B,) logit scores
        """
        import math
        gene_tokens = self._get_cell_gene_tokens(cell_lids)  # (B, 964, ca_dim)
        drug_query = self.ca_drug_query(drug_embeds)  # (B, ca_dim)

        # Attention: (B, 964)
        attn_scores = torch.bmm(
            gene_tokens, drug_query.unsqueeze(-1)
        ).squeeze(-1) / math.sqrt(self.ca_dim)  # (B, 964)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, 964)

        # Weighted sum of gene tokens
        cell_context = torch.bmm(
            attn_weights.unsqueeze(1), gene_tokens
        ).squeeze(1)  # (B, ca_dim)

        # Score
        combined = torch.cat([drug_embeds, cell_context], dim=-1)  # (B, embed_d + ca_dim)
        return self.ca_score_head(combined).squeeze(-1)  # (B,)

    def _encode_cell_features(self, raw):
        """Apply per-gene MLP encoder to multiomic cell features if configured.

        Input:  (B, feat_dim) raw cell features (multiomic or hybrid)
        Output: (B, encoded_dim) encoded features ready for projection
        """
        if not hasattr(self, 'gene_encoder'):
            return raw

        vae_dim = getattr(self, 'hybrid_vae_dim', 0)

        # For hybrid: split off VAE prefix
        if vae_dim > 0:
            vae_part = raw[:, :vae_dim]                   # (B, 512)
            multiomic = raw[:, vae_dim:]                   # (B, n_genes*4 + 2)
        else:
            vae_part = None
            multiomic = raw

        # Split off flags (last 2 dims of multiomic)
        flags = multiomic[:, -2:]                          # (B, 2)
        gene_data = multiomic[:, :-2]                      # (B, n_genes*4)

        B = gene_data.shape[0]
        gene_data = gene_data.view(B, self.n_target_genes, 4)  # (B, 964, 4)
        encoded = self.gene_encoder(gene_data)                  # (B, 964, gene_enc_dim)
        encoded = encoded.reshape(B, -1)                        # (B, 964*gene_enc_dim)

        parts = [encoded, flags]
        if vae_part is not None:
            parts.append(vae_part)
        return torch.cat(parts, dim=-1)

    def _project_all(self, node_type):
        """Get projected features for ALL nodes of a given type. Returns (N, embed_d)."""
        if node_type == self.dataset.node_name2type['cell']:
            raw = self.feature_loader.cell_features
        elif node_type == self.dataset.node_name2type['drug']:
            raw = self.feature_loader.drug_features
        elif node_type == self.dataset.node_name2type['gene']:
            raw = self.feature_loader.gene_features
        else:
            n = self.dataset.nodes['count'][node_type]
            return torch.zeros(n, self.embed_d, device=self.device)

        if str(node_type) in self.feat_proj:
            if node_type == self.dataset.node_name2type['cell']:
                raw = self._encode_cell_features(raw)
            return self.feat_proj[str(node_type)](raw)
        return raw

    def conteng_agg(self, local_id_batch, node_type):
        """Gets the initial node features *after* projection (batch version)."""

        if isinstance(local_id_batch, torch.Tensor):
            if local_id_batch.numel() == 0:
                return torch.zeros(0, self.embed_d, device=self.device)
        else:
            if not local_id_batch:
                return torch.zeros(0, self.embed_d, device=self.device)

        if node_type == self.dataset.node_name2type['cell']:
            raw_features = self.feature_loader.cell_features[local_id_batch]
        elif node_type == self.dataset.node_name2type['drug']:
            raw_features = self.feature_loader.drug_features[local_id_batch]
        elif node_type == self.dataset.node_name2type['gene']:
            raw_features = self.feature_loader.gene_features[local_id_batch]
        else:
            batch_len = local_id_batch.size(0) if isinstance(local_id_batch, torch.Tensor) else len(local_id_batch)
            return torch.zeros(batch_len, self.embed_d, device=self.device)

        if str(node_type) in self.feat_proj:
             if node_type == self.dataset.node_name2type['cell']:
                 raw_features = self._encode_cell_features(raw_features)
             return self.feat_proj[str(node_type)](raw_features)
        else:
             return raw_features


    def create_isolation_masks(self, isolated_cell_lids):
        """Create temporary masks that hide Cell-Drug edges for isolated cells.

        For isolated cells: zero out cell→drug neighbors (cell can't see drugs).
        For drugs: zero out drug→cell entries pointing to isolated cells.
        Cell-Cell, Cell-Gene, Drug-Gene all remain untouched.

        Args:
            isolated_cell_lids: list or tensor of cell local IDs to isolate

        Returns:
            masks: dict[center_type][neigh_type] -> (N, M) bool tensor
        """
        cell_type = self.dataset.node_name2type['cell']
        drug_type = self.dataset.node_name2type['drug']

        # Clone base masks (only the two we need to modify)
        masks = {}
        for ct in self.node_types:
            masks[ct] = {}
            for nt in self.node_types:
                if (ct == cell_type and nt == drug_type) or (ct == drug_type and nt == cell_type):
                    masks[ct][nt] = self.neighbor_masks[ct][nt].clone()
                else:
                    masks[ct][nt] = self.neighbor_masks[ct][nt]  # no clone, unchanged

        if not isinstance(isolated_cell_lids, torch.Tensor):
            isolated_cell_lids = torch.tensor(isolated_cell_lids, dtype=torch.long, device=self.device)

        if isolated_cell_lids.numel() == 0:
            return masks

        # 1. Isolated cells can't see drugs
        masks[cell_type][drug_type][isolated_cell_lids] = False

        # 2. Drugs can't see isolated cells
        n_cells = self.dataset.nodes['count'][cell_type]
        isolated_flag = torch.zeros(n_cells, dtype=torch.bool, device=self.device)
        isolated_flag[isolated_cell_lids] = True

        drug_cell_neigh_lids = self.neighbor_lids[drug_type][cell_type]  # (N_drugs, M)
        # For each drug neighbor slot, check if the cell there is isolated
        is_isolated_neigh = isolated_flag[drug_cell_neigh_lids]  # (N_drugs, M)
        masks[drug_type][cell_type][is_isolated_neigh] = False

        return masks

    def _gather_neighbor_features(self, center_type, neigh_type, embedding_table, masks_override=None):
        """Gather neighbor features from embedding table using pre-built index tensors.

        Args:
            center_type: type ID of center nodes
            neigh_type: type ID of neighbor nodes
            embedding_table: (N_neigh, embed_d) tensor of current embeddings for neigh_type
            masks_override: optional dict[ct][nt] of dynamic masks (from create_isolation_masks)

        Returns:
            features: (N_center, M, embed_d) weighted neighbor features
            mask: (N_center, M) boolean mask of valid neighbors
            weight_sum: (N_center,) sum(weight*mask) for this pair if (center, neigh) is in
                weighted_agg_pairs, else None.
        """
        lids = self.neighbor_lids[center_type][neigh_type]       # (N, M)
        weights = self.neighbor_weights[center_type][neigh_type]  # (N, M)

        if masks_override is not None:
            mask = masks_override[center_type][neigh_type]
        else:
            mask = self.neighbor_masks[center_type][neigh_type]  # (N, M)

        # Gather: index into embedding_table with safe_lids
        gathered = embedding_table[lids]  # (N, M, D)
        # Apply edge weights
        gathered = gathered * weights.unsqueeze(-1)  # (N, M, D)
        # Zero out padded entries
        gathered = gathered * mask.unsqueeze(-1).float()

        # Weight-sum denominator for proper weighted mean (only for requested pairs)
        weight_sum = None
        if neigh_type in self.weighted_agg_pairs.get(center_type, ()):
            weight_sum = (weights * mask.float()).sum(dim=1)  # (N,)

        return gathered, mask, weight_sum

    def compute_all_embeddings(self, masks_override=None):
        """Full-graph multi-hop GNN forward pass.

        Each layer uses the PREVIOUS layer's embeddings for neighbors,
        enabling true multi-hop message passing.

        Args:
            masks_override: optional dynamic masks from create_isolation_masks().
                When provided, overrides self.neighbor_masks for Cell-Drug edges.

        Returns:
            embedding_tables: dict {node_type: (N, embed_d) tensor}
        """
        # Layer 0: projected features for all node types
        embedding_tables = {}
        for nt in self.node_types:
            embedding_tables[nt] = self._project_all(nt)

        # Multi-hop: each layer reads from previous layer's embeddings
        for layer in self.gnn_layers:
            new_tables = {}
            for ct in self.node_types:
                center_embeds = embedding_tables[ct]  # (N_ct, D)
                neigh_embeds_by_type = {}
                neigh_masks_by_type = {}
                neigh_weight_sums_by_type = {}

                for nt in self.node_types:
                    gathered, mask, wsum = self._gather_neighbor_features(
                        ct, nt, embedding_tables[nt], masks_override=masks_override
                    )
                    neigh_embeds_by_type[nt] = gathered
                    neigh_masks_by_type[nt] = mask
                    if wsum is not None:
                        neigh_weight_sums_by_type[nt] = wsum

                new_tables[ct] = layer(
                    center_embeds, neigh_embeds_by_type, neigh_masks_by_type,
                    neigh_weight_sums_by_type=(neigh_weight_sums_by_type or None),
                )

            embedding_tables = new_tables

        return embedding_tables

    # ==========================================
    # M14: Mini-Batch GNN Propagation
    # ==========================================

    def expand_to_k_hop(self, seed_lids_by_type, k, neighbor_sample_size=0, masks_override=None):
        """Expand seed nodes to k-hop neighborhood, return SubgraphInfo.

        Args:
            seed_lids_by_type: {node_type: (S,) LongTensor of local IDs}
            k: number of hops (typically n_layers)
            neighbor_sample_size: max neighbors per node per type per hop (0=all)
            masks_override: optional dynamic masks from create_isolation_masks()

        Returns:
            SubgraphInfo with subgraph topology
        """
        # Initialize node_sets with seeds
        node_sets = {}
        for nt in self.node_types:
            if nt in seed_lids_by_type and seed_lids_by_type[nt].numel() > 0:
                node_sets[nt] = seed_lids_by_type[nt].unique()
            else:
                node_sets[nt] = torch.zeros(0, dtype=torch.long, device=self.device)

        # Expand k hops
        for _hop in range(k):
            new_nodes = {nt: [] for nt in self.node_types}
            for ct in self.node_types:
                if node_sets[ct].numel() == 0:
                    continue
                for nt in self.node_types:
                    # Get neighbors of current frontier for this type pair
                    lids = self.neighbor_lids[ct][nt][node_sets[ct]]     # (S_ct, M)
                    if masks_override is not None:
                        mask = masks_override[ct][nt][node_sets[ct]]
                    else:
                        mask = self.neighbor_masks[ct][nt][node_sets[ct]]    # (S_ct, M)

                    # Optional neighbor sampling
                    if neighbor_sample_size > 0 and mask.any():
                        # Vectorized sampling: for each node, keep at most neighbor_sample_size valid neighbors
                        n_valid = mask.float().sum(dim=1)  # (S_ct,)
                        needs_sample = n_valid > neighbor_sample_size
                        if needs_sample.any():
                            # Generate random scores, set invalid to -inf, take topk
                            rand_scores = torch.rand_like(mask.float())
                            rand_scores[~mask] = -1.0
                            _, topk_idx = rand_scores.topk(
                                min(neighbor_sample_size, mask.shape[1]), dim=1
                            )
                            # Build new mask: only keep topk positions
                            sample_mask = torch.zeros_like(mask)
                            sample_mask.scatter_(1, topk_idx, True)
                            # Only apply sampling to nodes that need it
                            mask = torch.where(needs_sample.unsqueeze(1), sample_mask & mask, mask)

                    # Extract valid neighbor LIDs
                    valid_neigh = lids[mask]  # flat 1D
                    if valid_neigh.numel() > 0:
                        new_nodes[nt].append(valid_neigh)

            # Merge new nodes with existing, dedup
            for nt in self.node_types:
                parts = [node_sets[nt]]
                parts.extend(new_nodes[nt])
                if len(parts) > 1:
                    node_sets[nt] = torch.cat(parts).unique()

        return self._build_subgraph_info(node_sets, masks_override=masks_override)

    def _build_subgraph_info(self, node_sets, masks_override=None):
        """Build SubgraphInfo from node_sets: create lid_to_pos mapping and slice neighbor tensors.

        Args:
            node_sets: {node_type: (S,) LongTensor of local IDs in subgraph}

        Returns:
            SubgraphInfo with remapped neighbor tensors
        """
        # Build lid_to_pos: full-size tensor mapping full lid -> subgraph position
        lid_to_pos = {}
        for nt in self.node_types:
            n_full = self.dataset.nodes['count'][nt]
            mapping = torch.full((n_full,), -1, dtype=torch.long, device=self.device)
            if node_sets[nt].numel() > 0:
                mapping[node_sets[nt]] = torch.arange(
                    node_sets[nt].shape[0], dtype=torch.long, device=self.device
                )
            lid_to_pos[nt] = mapping

        # Slice and remap neighbor tensors for subgraph nodes
        sub_neighbor_lids = {}
        sub_neighbor_weights = {}
        sub_neighbor_masks = {}

        for ct in self.node_types:
            sub_neighbor_lids[ct] = {}
            sub_neighbor_weights[ct] = {}
            sub_neighbor_masks[ct] = {}

            if node_sets[ct].numel() == 0:
                for nt in self.node_types:
                    sub_neighbor_lids[ct][nt] = torch.zeros(0, self.max_neighbors, dtype=torch.long, device=self.device)
                    sub_neighbor_weights[ct][nt] = torch.zeros(0, self.max_neighbors, dtype=torch.float, device=self.device)
                    sub_neighbor_masks[ct][nt] = torch.zeros(0, self.max_neighbors, dtype=torch.bool, device=self.device)
                continue

            ct_nodes = node_sets[ct]  # (S_ct,)
            for nt in self.node_types:
                # Slice: only rows for subgraph center nodes
                sliced_lids = self.neighbor_lids[ct][nt][ct_nodes]       # (S_ct, M)
                sliced_weights = self.neighbor_weights[ct][nt][ct_nodes]  # (S_ct, M)
                if masks_override is not None:
                    sliced_mask = masks_override[ct][nt][ct_nodes]
                else:
                    sliced_mask = self.neighbor_masks[ct][nt][ct_nodes]  # (S_ct, M)

                # Remap neighbor LIDs to subgraph positions
                # Use lid_to_pos[nt] to map; neighbors not in subgraph get -1
                safe_lids = sliced_lids.clamp(min=0)  # safe index (PAD was already 0)
                remapped = lid_to_pos[nt][safe_lids]  # (S_ct, M)

                # Update mask: valid AND neighbor is in subgraph
                in_subgraph = remapped >= 0
                final_mask = sliced_mask & in_subgraph

                # Replace out-of-subgraph with 0 for safe indexing
                remapped = remapped.clamp(min=0)

                sub_neighbor_lids[ct][nt] = remapped
                sub_neighbor_weights[ct][nt] = sliced_weights
                sub_neighbor_masks[ct][nt] = final_mask

        return SubgraphInfo(
            node_sets=node_sets,
            lid_to_pos=lid_to_pos,
            sub_neighbor_lids=sub_neighbor_lids,
            sub_neighbor_weights=sub_neighbor_weights,
            sub_neighbor_masks=sub_neighbor_masks,
        )

    def compute_batch_embeddings(self, subgraph_info):
        """Mini-batch GNN forward pass on subgraph only.

        Same structure as compute_all_embeddings() but operates on the subgraph
        defined by subgraph_info.

        Args:
            subgraph_info: SubgraphInfo from expand_to_k_hop()

        Returns:
            embedding_tables: dict {node_type: (S_ct, embed_d) tensor}
        """
        sg = subgraph_info

        # Layer 0: projected features for subgraph nodes only
        embedding_tables = {}
        for nt in self.node_types:
            if sg.node_sets[nt].numel() > 0:
                embedding_tables[nt] = self.conteng_agg(sg.node_sets[nt], nt)
            else:
                embedding_tables[nt] = torch.zeros(0, self.embed_d, device=self.device)

        # GNN layers: same logic as full-graph but using subgraph neighbor tensors
        for layer in self.gnn_layers:
            new_tables = {}
            for ct in self.node_types:
                center_embeds = embedding_tables[ct]  # (S_ct, D)
                if center_embeds.shape[0] == 0:
                    new_tables[ct] = center_embeds
                    continue

                neigh_embeds_by_type = {}
                neigh_masks_by_type = {}
                neigh_weight_sums_by_type = {}

                for nt in self.node_types:
                    lids = sg.sub_neighbor_lids[ct][nt]       # (S_ct, M)
                    weights = sg.sub_neighbor_weights[ct][nt]  # (S_ct, M)
                    mask = sg.sub_neighbor_masks[ct][nt]       # (S_ct, M)

                    # Gather from subgraph embedding table
                    if embedding_tables[nt].shape[0] > 0:
                        gathered = embedding_tables[nt][lids]  # (S_ct, M, D)
                    else:
                        gathered = torch.zeros(
                            lids.shape[0], lids.shape[1], self.embed_d,
                            device=self.device
                        )
                    gathered = gathered * weights.unsqueeze(-1)
                    gathered = gathered * mask.unsqueeze(-1).float()

                    neigh_embeds_by_type[nt] = gathered
                    neigh_masks_by_type[nt] = mask

                    if nt in self.weighted_agg_pairs.get(ct, ()):
                        neigh_weight_sums_by_type[nt] = (weights * mask.float()).sum(dim=1)  # (S_ct,)

                new_tables[ct] = layer(
                    center_embeds, neigh_embeds_by_type, neigh_masks_by_type,
                    neigh_weight_sums_by_type=(neigh_weight_sums_by_type or None),
                )

            embedding_tables = new_tables

        return embedding_tables

    # --- EMA Teacher ---

    def init_ema_teacher(self):
        """Initialize EMA teacher as a copy of current trainable parameters.

        Stores only projection and GNN layer weights (not neighbor tensors).
        Called once from train.py after model setup.
        """
        self._ema_params = {}
        for name, param in self.named_parameters():
            self._ema_params[name] = param.data.clone()
        print(f"INFO: EMA teacher initialized ({len(self._ema_params)} parameter tensors, "
              f"momentum={self.args.ema_momentum})")

    @torch.no_grad()
    def update_ema_teacher(self):
        """Update teacher weights: θ_t = m*θ_t + (1-m)*θ_s."""
        m = self.args.ema_momentum
        for name, param in self.named_parameters():
            if name in self._ema_params:
                self._ema_params[name].mul_(m).add_(param.data, alpha=1.0 - m)

    def ema_teacher_loss(self, embedding_tables, subgraph_info=None):
        """MSE between current GNN embeddings and teacher's GNN embeddings.

        Produces teacher embeddings by temporarily swapping in EMA weights,
        running a forward pass (no grad), then restoring original weights.
        """
        # Save current params, load teacher params
        saved = {}
        for name, param in self.named_parameters():
            saved[name] = param.data.clone()
            if name in self._ema_params:
                param.data.copy_(self._ema_params[name])

        # Compute teacher embeddings (no grad, eval mode for deterministic dropout)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if subgraph_info is not None:
                teacher_tables = self.compute_batch_embeddings(subgraph_info)
            else:
                teacher_tables = self.compute_all_embeddings()
        if was_training:
            self.train()

        # Restore student params
        for name, param in self.named_parameters():
            param.data.copy_(saved[name])

        # MSE between student and teacher embeddings
        total_loss = 0.0
        total_nodes = 0
        for nt in self.node_types:
            if nt not in embedding_tables:
                continue
            student = embedding_tables[nt]
            teacher = teacher_tables[nt]
            total_loss = total_loss + F.mse_loss(student, teacher, reduction='sum')
            total_nodes += student.shape[0]

        if total_nodes == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / total_nodes

    def alignment_loss(self, embedding_tables, subgraph_info=None, node_type_names=None):
        """Cosine alignment loss between projected and GNN embeddings.

        Penalizes drift so GNN output stays in the same angular space as
        projection-only embeddings. This keeps the bilinear head calibrated
        for both GNN and projection-only (inductive) inputs.

        Args:
            embedding_tables: dict from compute_all/batch_embeddings()
            subgraph_info: SubgraphInfo if mini-batch, None if full-graph
            node_type_names: list of type names to align (e.g. ['cell']),
                             or None for all types

        Returns:
            scalar loss: mean(1 - cos_sim(proj, gnn)) over selected node types
        """
        if node_type_names is not None:
            target_types = {self.dataset.node_name2type[n] for n in node_type_names}
        else:
            target_types = set(self.node_types)

        total_loss = 0.0
        total_nodes = 0

        for nt in self.node_types:
            if nt not in target_types:
                continue
            if subgraph_info is not None:
                node_lids = subgraph_info.node_sets[nt]
                if node_lids.numel() == 0:
                    continue
                gnn_embeds = embedding_tables[nt]  # already (S, D) indexed by subgraph pos
            else:
                n = self.dataset.nodes['count'][nt]
                if n == 0:
                    continue
                node_lids = torch.arange(n, device=self.device)
                gnn_embeds = embedding_tables[nt]  # (N, D)

            projected = self.conteng_agg(node_lids, nt)  # (S, D) or (N, D)

            # Cosine similarity: 1 = identical direction, 0 = orthogonal
            cos_sim = F.cosine_similarity(projected, gnn_embeds, dim=-1)  # (S,)
            # Loss = mean(1 - cos_sim), so 0 when perfectly aligned
            total_loss = total_loss + (1.0 - cos_sim).sum()
            total_nodes += cos_sim.shape[0]

        if total_nodes == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / total_nodes

    def _get_final_embed(self, local_ids, node_type, embedding_tables=None, subgraph_info=None):
        """Get final embeddings: projection + optional GNN + skip gate blend.

        Args:
            local_ids: (B,) tensor of local node IDs
            node_type: integer type ID
            embedding_tables: dict from compute_all_embeddings(), or None to use projection only

        Returns:
            (B, embed_d) tensor of final embeddings
        """
        if local_ids.numel() == 0:
            return torch.zeros(0, self.embed_d, device=self.device)

        projected = self.conteng_agg(local_ids, node_type)

        # Freeze cell GNN: cells always stay in projection-space
        if getattr(self.args, 'freeze_cell_gnn', False) and self.cell_type_name is not None:
            cell_type_id = self.dataset.node_name2type[self.cell_type_name]
            if node_type == cell_type_id:
                return projected

        if embedding_tables is not None and node_type in embedding_tables:
            if subgraph_info is not None:
                # Mini-batch: remap local_ids -> subgraph positions via lid_to_pos
                sg_pos = subgraph_info.lid_to_pos[node_type][local_ids]  # (B,)
                gnn_embeds = embedding_tables[node_type][sg_pos]
            else:
                # Full-graph: direct indexing
                gnn_embeds = embedding_tables[node_type][local_ids]
        else:
            gnn_embeds = projected  # fallback: no GNN, use projection

        # Residual GNN (M15): proj + scale * (gnn - proj)
        if getattr(self.args, 'use_residual_gnn', False):
            delta = gnn_embeds - projected
            if self.residual_scale_params is not None:
                scale = torch.sigmoid(self.residual_scale_params[str(node_type)])
            else:
                scale = self.residual_scale_fixed
            return projected + scale * delta

        # Skip gate blending
        elif getattr(self.args, 'use_node_gate', False) and hasattr(self, 'node_gate_mlp'):
            # Per-node gate: MLP(cat(proj, gnn)) -> sigmoid -> alpha per node
            gate_input = torch.cat([projected, gnn_embeds], dim=-1)  # (B, 2D)
            alpha = torch.sigmoid(self.node_gate_mlp(gate_input))   # (B, 1)
            return alpha * projected + (1 - alpha) * gnn_embeds
        elif self.args.use_skip_connection and hasattr(self, 'skip_gates'):
            # Per-type gate (legacy)
            alpha = torch.sigmoid(self.skip_gates[str(node_type)])  # (D,)
            return alpha * projected + (1 - alpha) * gnn_embeds
        else:
            return gnn_embeds

    def link_prediction_loss(self, drug_indices_local, cell_indices_local, labels, data_generator,
                             isolation_ratio=0.0, embedding_tables=None, subgraph_info=None,
                             isolated_cell_set=None):
        """Calculates link prediction loss.

        When embedding_tables is provided, uses pre-computed full-graph embeddings.
        isolation_ratio is now handled at the GNN level via dynamic masking
        in compute_all_embeddings(), not here.

        Args:
            isolated_cell_set: set of cell local IDs that were isolated this batch
                (for dual head routing). If None, all pairs use default head.
        """
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]

        drug_embeds = self._get_final_embed(drug_indices_local, drug_type_id, embedding_tables, subgraph_info)

        # Cross-attention scoring: drug attends over cell gene tokens
        if getattr(self.args, 'use_cross_attention', False):
            scores = self._cross_attention_score(drug_embeds, cell_indices_local)
            if getattr(self.args, 'regression', False):
                return F.l1_loss(scores, labels.float())
            return F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=self.lp_pos_weight)

        cell_embeds = self._get_final_embed(cell_indices_local, cell_type_id, embedding_tables, subgraph_info)

        # Dual head: route isolated cells to inductive head, rest to transductive head
        if self.lp_bilinear_ind is not None and isolated_cell_set is not None and self.training:
            iso_mask = torch.tensor(
                [lid.item() in isolated_cell_set for lid in cell_indices_local],
                dtype=torch.bool, device=self.device
            )
            n_iso = iso_mask.sum().item()
            n_trans = (~iso_mask).sum().item()

            total_loss = 0.0
            # Inductive head loss (isolated cells)
            if n_iso > 0:
                scores_ind = self._score_pairs(drug_embeds[iso_mask], cell_embeds[iso_mask],
                                               use_inductive_head=True)
                if getattr(self.args, 'regression', False):
                    loss_ind = F.l1_loss(scores_ind, labels[iso_mask].float())
                else:
                    loss_ind = F.binary_cross_entropy_with_logits(
                        scores_ind, labels[iso_mask].float(), pos_weight=self.lp_pos_weight)
                ind_weight = getattr(self.args, 'inductive_loss_weight', 1.0)
                total_loss = total_loss + ind_weight * loss_ind

            # Transductive head loss (non-isolated cells)
            if n_trans > 0:
                scores_trans = self._score_pairs(drug_embeds[~iso_mask], cell_embeds[~iso_mask],
                                                 use_inductive_head=False)
                if getattr(self.args, 'regression', False):
                    loss_trans = F.l1_loss(scores_trans, labels[~iso_mask].float())
                else:
                    loss_trans = F.binary_cross_entropy_with_logits(
                        scores_trans, labels[~iso_mask].float(), pos_weight=self.lp_pos_weight)
                total_loss = total_loss + loss_trans

            return total_loss

        # Single head (default)
        scores = self._score_pairs(drug_embeds, cell_embeds)
        if getattr(self.args, 'regression', False):
            return F.l1_loss(scores, labels.float())
        return F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=self.lp_pos_weight)


    def projection_lp_loss(self, drug_indices_local, cell_indices_local, labels):
        """LP loss using projection-only embeddings (no GNN, no gate).

        Forces the bilinear head to make good predictions from raw projected
        features alone — exactly what inductive cells will have at inference.
        Both drug AND cell sides use projection-only, so the head can't hide
        structural signal on either side.

        Args:
            drug_indices_local: (B,) drug local IDs
            cell_indices_local: (B,) cell local IDs
            labels: (B,) binary labels

        Returns:
            scalar BCE loss
        """
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]

        drug_proj = self.conteng_agg(drug_indices_local, drug_type_id)

        # Cross-attention: drug attends over cell gene tokens (projection-only for drug side)
        if getattr(self.args, 'use_cross_attention', False):
            scores = self._cross_attention_score(drug_proj, cell_indices_local)
            return F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=self.lp_pos_weight)

        cell_proj = self.conteng_agg(cell_indices_local, cell_type_id)

        scores = self._score_pairs(drug_proj, cell_proj)
        if getattr(self.args, 'regression', False):
            return F.l1_loss(scores, labels.float())
        return F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=self.lp_pos_weight)

    def link_prediction_forward(self, drug_indices_local, cell_indices_local, data_generator,
                                embedding_tables=None, subgraph_info=None, use_inductive_head=False):
        """Performs forward pass for link prediction during evaluation.

        Args:
            use_inductive_head: if True and dual_head is enabled, use the inductive scoring head.
                Use this for inductive evaluation / Sanger cross-dataset.
        """
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]

        drug_embeds = self._get_final_embed(drug_indices_local, drug_type_id, embedding_tables, subgraph_info)

        # Cross-attention scoring
        if getattr(self.args, 'use_cross_attention', False):
            scores = self._cross_attention_score(drug_embeds, cell_indices_local)
            return torch.sigmoid(scores)

        cell_embeds = self._get_final_embed(cell_indices_local, cell_type_id, embedding_tables, subgraph_info)

        if self.lp_bilinear is None and not hasattr(self, 'lp_mlp'):
             raise RuntimeError("Link prediction head not initialized. Call setup_link_prediction first.")

        scores = self._score_pairs(drug_embeds, cell_embeds, use_inductive_head=use_inductive_head)
        if getattr(self.args, 'regression', False):
            return scores
        return torch.sigmoid(scores)

    def get_embeddings_from_gids(self, gids_tensor, data_generator, embedding_tables=None, subgraph_info=None):
        """
        Gets the final embeddings for a batch of GIDs.
        Uses embedding_tables from compute_all_embeddings() if provided.
        """
        lids_by_type = defaultdict(list)
        original_indices_by_type = defaultdict(list)

        for i, gid in enumerate(gids_tensor.tolist()):
            try:
                ntype, lid = self.dataset.nodes['type_map'][gid]
                lids_by_type[ntype].append(lid)
                original_indices_by_type[ntype].append(i)
            except KeyError:
                pass

        final_embeds = torch.zeros(len(gids_tensor), self.embed_d, device=self.device)

        for ntype, lids in lids_by_type.items():
            if lids:
                lid_tensor = torch.tensor(lids, dtype=torch.long, device=self.device)
                embeds_for_type = self._get_final_embed(lid_tensor, ntype, embedding_tables, subgraph_info)

                original_indices = torch.tensor(original_indices_by_type[ntype], dtype=torch.long, device=self.device)
                final_embeds.index_copy_(0, original_indices, embeds_for_type)

        return final_embeds

    def self_supervised_triplet_loss(self, anchor_gids, pos_gids, neg_gids, data_generator,
                                     embedding_tables=None, subgraph_info=None):
        """
        Calculates the Triplet Loss based on feature similarity.

        Supports two modes:
        - Hard-margin (default): max(0, d_pos - d_neg + margin)
        - Soft-margin (--use_soft_margin_triplet): log(1 + exp(d_pos - d_neg))
        """
        anchor_embeds = self.get_embeddings_from_gids(anchor_gids, data_generator, embedding_tables, subgraph_info)
        pos_embeds = self.get_embeddings_from_gids(pos_gids, data_generator, embedding_tables, subgraph_info)
        neg_embeds = self.get_embeddings_from_gids(neg_gids, data_generator, embedding_tables, subgraph_info)

        if getattr(self.args, 'use_soft_margin_triplet', False):
            d_pos = F.pairwise_distance(anchor_embeds, pos_embeds, p=2)
            d_neg = F.pairwise_distance(anchor_embeds, neg_embeds, p=2)
            loss = F.softplus(d_pos - d_neg).mean()
        else:
            loss = F.triplet_margin_loss(
                anchor_embeds, pos_embeds, neg_embeds,
                margin=self.args.triplet_margin, p=2
            )

        return loss
