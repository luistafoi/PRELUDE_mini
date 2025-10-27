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
        Loads cell embeddings from .npy and .txt files and aligns by local ID.
        This version uses the .txt file to map cell names to embedding rows.
        """
        cell_type_id = self.dataset.node_name2type.get("cell", -1)
        if cell_type_id == -1 or cell_type_id not in self.dataset.nodes['count']:
            print("  > Warning: Cell type not found or has no nodes. Returning empty tensor.")
            return torch.empty(0)

        num_cells = self.dataset.nodes['count'][cell_type_id]
        print(f"  > Aligning features for {num_cells} 'cell' nodes found in graph.")

        # --- Define Paths ---
        embed_path = "data/embeddings/final_vae_cell_embeddings.npy"
        names_path = "data/embeddings/final_vae_cell_names.txt"

        # --- Load Embeddings ---
        if not os.path.exists(embed_path):
            print(f"  > FATAL ERROR: Cell embedding file not found at {embed_path}")
            sys.exit(1)
        try:
            embeds_data = np.load(embed_path)
        except Exception as e:
            print(f"  > FATAL ERROR: Could not load numpy file {embed_path}: {e}")
            sys.exit(1)

        # --- Load Names ---
        if not os.path.exists(names_path):
            print(f"  > FATAL ERROR: Cell names file not found at {names_path}")
            sys.exit(1)
        try:
            with open(names_path, 'r') as f:
                names_data = [line.strip().upper() for line in f if line.strip()]
        except Exception as e:
            print(f"  > FATAL ERROR: Could not load names file {names_path}: {e}")
            sys.exit(1)

        # --- VALIDATION ---
        if len(names_data) != embeds_data.shape[0]:
            print(f"  > FATAL ERROR: Mismatch between names and embeddings!")
            print(f"  > Found {len(names_data)} names in {names_path}")
            print(f"  > Found {embeds_data.shape[0]} embeddings in {embed_path}")
            print(f"  > Please re-run 'scripts/cell_vae.py' to generate matching files.")
            sys.exit(1)
            
        if not names_data:
            print(f"  > FATAL ERROR: No cell names loaded from {names_path}.")
            sys.exit(1)

        # --- Create Name-to-Feature Map ---
        feature_dim = embeds_data.shape[1]
        cell_name_to_feat = {name: embeds_data[i] for i, name in enumerate(names_data)}
        print(f"  > Loaded {len(cell_name_to_feat)} cell features from source files (Dim: {feature_dim}).")

        # --- Align Features to Graph Node IDs ---
        aligned_cell_feats = torch.zeros(num_cells, feature_dim, dtype=torch.float32)
        loaded_count = 0
        
        for global_id, (ntype, local_id) in self.dataset.nodes['type_map'].items():
            if ntype == cell_type_id:
                # Get the node's original name (e.g., 'ACH-000123')
                node_name = self.dataset.id2node.get(global_id, "").upper()
                feature_vec = cell_name_to_feat.get(node_name)
                
                if feature_vec is not None:
                    # Ensure local_id is valid
                    if 0 <= local_id < num_cells:
                        aligned_cell_feats[local_id] = torch.from_numpy(feature_vec.astype(np.float32))
                        loaded_count += 1
                    else:
                        print(f"  > Warning: Cell local ID {local_id} out of bounds (max: {num_cells-1}).")
                else:
                    print(f"  > Warning: No feature found for cell '{node_name}' (global ID: {global_id}). Using zeros.")

        print(f"  > Aligned cell features tensor: {aligned_cell_feats.shape}. Loaded {loaded_count}/{num_cells}.")
        
        if loaded_count != num_cells:
             print(f"  > WARNING: Mismatch! Graph expected {num_cells} cells, but features were only found for {loaded_count}.")
             print(f"  > This suggests the graph (node.dat) was built with a different cell list than the feature files.")

        return aligned_cell_feats