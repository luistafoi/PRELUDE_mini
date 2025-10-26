# dataloaders/feature_loader.py

import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle

class FeatureLoader:
    """
    Handles loading pre-computed static node features for cells, drugs, and genes.
    Creates feature tensors aligned by local node ID.
    """
    def __init__(self, dataset, device):
        print("Initializing FeatureLoader...")
        self.dataset = dataset # Expects an initialized PRELUDEDataset object
        self.device = device
        
        # Load features for each type
        self.cell_features = self._load_cell_features().to(self.device)
        self.drug_features = self._load_drug_features().to(self.device)
        self.gene_features = self._load_gene_features().to(self.device)

        print("FeatureLoader initialization complete.")

    def _load_drug_features(self):
        """Loads drug embeddings from CSV and aligns by local ID."""
        drug_type_id = self.dataset.node_name2type.get("drug", -1)
        if drug_type_id == -1 or drug_type_id not in self.dataset.nodes['count']:
            print("  > Warning: Drug type not found or has no nodes. Returning empty tensor.")
            return torch.empty(0)
            
        num_drugs = self.dataset.nodes['count'][drug_type_id]
        
        # Adjust path relative to project root
        path = "data/embeddings/drugs_with_embeddings.csv"
        if not os.path.exists(path):
             print(f"  > FATAL ERROR: Drug embedding file not found at {path}")
             sys.exit(1)
             
        print(f"  > Loading drug features from: {path}")
        df = pd.read_csv(path)
        
        drug_name_to_feat = {
            # Use iloc for potentially unnamed 'Drug' column if index_col=0 wasn't used
            row.iloc[0].upper(): np.array(row[1:].values.astype(np.float32))
            for _, row in df.iterrows()
        }
        
        if not drug_name_to_feat:
             print(f"  > FATAL ERROR: No drug features loaded from {path}")
             sys.exit(1)
             
        feature_dim = len(next(iter(drug_name_to_feat.values())))
        aligned_drug_feats = torch.zeros(num_drugs, feature_dim, dtype=torch.float32)
        
        loaded_count = 0
        for global_id, (ntype, local_id) in self.dataset.nodes['type_map'].items():
            if ntype == drug_type_id:
                # Need id2node from PRELUDEDataset
                node_name = self.dataset.id2node.get(global_id, "").upper()
                feature_vec = drug_name_to_feat.get(node_name)
                if feature_vec is not None:
                    # Ensure local_id is within bounds
                    if 0 <= local_id < num_drugs:
                         aligned_drug_feats[local_id] = torch.from_numpy(feature_vec)
                         loaded_count += 1
                    else:
                         print(f"  > Warning: Drug local ID {local_id} out of bounds (max: {num_drugs-1}).")
                else:
                    print(f"  > Warning: No feature found for drug '{node_name}' (global ID: {global_id}). Using zeros.")
                    
        print(f"  > Aligned drug features tensor: {aligned_drug_feats.shape}. Loaded {loaded_count}/{num_drugs}.")
        return aligned_drug_feats

    def _load_gene_features(self):
        """Loads gene embeddings from pickle and aligns by local ID."""
        gene_type_id = self.dataset.node_name2type.get("gene", -1)
        if gene_type_id == -1 or gene_type_id not in self.dataset.nodes['count']:
            print("  > Warning: Gene type not found or has no nodes. Returning empty tensor.")
            return torch.empty(0)
            
        num_genes = self.dataset.nodes['count'][gene_type_id]

        # Adjust path relative to project root
        path = "data/embeddings/gene_embeddings_esm_by_symbol.pkl"
        if not os.path.exists(path):
             print(f"  > FATAL ERROR: Gene embedding file not found at {path}")
             sys.exit(1)
             
        print(f"  > Loading gene features from: {path}")
        try:
             with open(path, "rb") as f:
                 # Ensure keys are strings and uppercase for robust matching
                 gene_name_to_feat = {str(k).upper(): v for k, v in pickle.load(f).items()}
        except Exception as e:
             print(f"  > FATAL ERROR: Could not load pickle file {path}: {e}")
             sys.exit(1)

        if not gene_name_to_feat:
             print(f"  > FATAL ERROR: No gene features loaded from {path}")
             sys.exit(1)
             
        feature_dim = len(next(iter(gene_name_to_feat.values())))
        aligned_gene_feats = torch.zeros(num_genes, feature_dim, dtype=torch.float32)
        
        loaded_count = 0
        for global_id, (ntype, local_id) in self.dataset.nodes['type_map'].items():
            if ntype == gene_type_id:
                node_name = self.dataset.id2node.get(global_id, "").upper() # Match case
                feature_vec = gene_name_to_feat.get(node_name)
                if feature_vec is not None:
                     if 0 <= local_id < num_genes:
                         # Ensure feature_vec is a numpy array before converting
                         if not isinstance(feature_vec, np.ndarray):
                              feature_vec = np.array(feature_vec)
                         aligned_gene_feats[local_id] = torch.from_numpy(feature_vec.astype(np.float32))
                         loaded_count += 1
                     else:
                          print(f"  > Warning: Gene local ID {local_id} out of bounds (max: {num_genes-1}).")
                else:
                     print(f"  > Warning: No feature found for gene '{node_name}' (global ID: {global_id}). Using zeros.")
        
        print(f"  > Aligned gene features tensor: {aligned_gene_feats.shape}. Loaded {loaded_count}/{num_genes}.")
        return aligned_gene_feats
    
    def _load_cell_features(self):
        """
        Loads the pre-computed static VAE embeddings from the .npy file.
        Assumes the .npy file is already aligned by local cell ID.
        """
        cell_type_id = self.dataset.node_name2type.get("cell", -1)
        if cell_type_id == -1 or cell_type_id not in self.dataset.nodes['count']:
            print("  > Warning: Cell type not found or has no nodes. Returning empty tensor.")
            return torch.empty(0)
            
        num_cells = self.dataset.nodes['count'][cell_type_id]
        
        # Adjust path relative to project root
        path = "data/embeddings/final_vae_cell_embeddings.npy"
        
        if not os.path.exists(path):
            print(f"  > FATAL ERROR: Pre-computed cell embeddings not found at {path}.")
            # Decide if returning zeros is acceptable or should exit
            # return torch.zeros(num_cells, 256, dtype=torch.float32) # Assuming 256 if file missing
            sys.exit(1)

        print(f"  > Loading static cell features from: {path}")
        try:
            vae_embeds = np.load(path)
        except Exception as e:
            print(f"  > FATAL ERROR: Could not load numpy file {path}: {e}")
            sys.exit(1)

        # --- Simplified Logic: Assume direct alignment ---
        if vae_embeds.shape[0] != num_cells:
            print(f"  > WARNING: Mismatch! Expected {num_cells} cell embeddings based on node.dat, but found {vae_embeds.shape[0]} in {path}.")
            # Handle mismatch: either pad, truncate, or error out
            # For simplicity, let's just use the loaded embeddings, but this might cause errors later if sizes mismatch GNN layers
            # A safer approach might be to resize/pad:
            feature_dim = vae_embeds.shape[1]
            aligned_cell_feats = torch.zeros(num_cells, feature_dim, dtype=torch.float32)
            common_count = min(num_cells, vae_embeds.shape[0])
            aligned_cell_feats[:common_count] = torch.from_numpy(vae_embeds[:common_count].astype(np.float32))
            print(f"  > Aligned cell features tensor created with shape: {aligned_cell_feats.shape} (padded/truncated).")

        else:
             # If sizes match, directly convert
             aligned_cell_feats = torch.from_numpy(vae_embeds.astype(np.float32))
             print(f"  > Loaded static cell features tensor: {aligned_cell_feats.shape}")
             
        return aligned_cell_feats