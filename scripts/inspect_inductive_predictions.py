# scripts/inspect_inductive_predictions.py

import sys
import os
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg

def main():
    # 1. Setup
    args = read_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"--- Inspecting INDUCTIVE Predictions (New Patients) ---")

    # 2. Load Data
    print(f"Loading data from {args.data_dir}...")
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir)
    generator.load_train_neighbors(os.path.join(args.data_dir, "train_neighbors_preprocessed.pkl"))

    # 3. Load Inductive Test Links
    # We manually load this specific file to be safe
    inductive_file = os.path.join(args.data_dir, "test_inductive_links.dat")
    if not os.path.exists(inductive_file):
        print(f"FATAL: {inductive_file} not found.")
        sys.exit(1)
        
    print(f"Loading New Patient links from: {inductive_file}")
    target_links = []
    with open(inductive_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            target_links.append((int(parts[0]), int(parts[1])))
    
    print(f"  > Found {len(target_links)} pairs to evaluate.")

    # 4. Load Model
    print(f"Loading model: {args.load_path}")
    model_args_ns = argparse.Namespace(**vars(args))
    if not hasattr(model_args_ns, 'use_skip_connection'): model_args_ns.use_skip_connection = False
    
    model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    
    checkpoint = torch.load(args.load_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    model.eval()

    # 5. Run Predictions
    results = []
    
    # Batch processing
    batch_size = args.mini_batch_s
    
    # Load metadata if available
    metadata_map = {}
    if args.metadata_file and os.path.exists(args.metadata_file):
        df_meta = pd.read_csv(args.metadata_file)
        # Assume cols: cell_name, tissue_type
        metadata_map = df_meta.set_index('cell_name')['tissue_type'].to_dict()

    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(target_links), batch_size)):
            batch = target_links[i : i + batch_size]
            
            # Prepare tensors
            # Note: In the file, src=Cell, tgt=Drug (usually)
            # We need to map GID -> LID
            
            drug_lids = []
            cell_lids = []
            valid_indices = []
            
            for idx, (src, tgt) in enumerate(batch):
                src_type = dataset.nodes['type_map'][src][0]
                tgt_type = dataset.nodes['type_map'][tgt][0]
                
                drug_gid, cell_gid = None, None
                
                if src_type == dataset.node_name2type['drug']:
                    drug_gid, cell_gid = src, tgt
                else:
                    drug_gid, cell_gid = tgt, src
                
                # Get LIDs
                d_lid = dataset.nodes['type_map'][drug_gid][1]
                c_lid = dataset.nodes['type_map'][cell_gid][1]
                
                drug_lids.append(d_lid)
                cell_lids.append(c_lid)
                valid_indices.append(idx)

            if not drug_lids: continue

            d_tensor = torch.tensor(drug_lids, dtype=torch.long, device=device)
            c_tensor = torch.tensor(cell_lids, dtype=torch.long, device=device)
            
            # Predict
            scores = model.link_prediction_forward(d_tensor, c_tensor, generator)
            scores_np = scores.cpu().numpy()
            
            # Store
            for j, score in enumerate(scores_np):
                original_idx = valid_indices[j]
                src, tgt = batch[original_idx]
                
                # Resolve names
                n1_name = dataset.id2node[src]
                n2_name = dataset.id2node[tgt]
                
                # Identify which is cell for metadata
                cell_name = n1_name if dataset.nodes['type_map'][src][0] == dataset.node_name2type['cell'] else n2_name
                drug_name = n2_name if cell_name == n1_name else n1_name
                tissue = metadata_map.get(cell_name, "Unknown")
                
                results.append({
                    "Cell": cell_name,
                    "Drug": drug_name,
                    "Tissue": tissue,
                    "Score": score,
                    "Label": 1 # These are all positive links in the test file
                })

    # 6. Save Report
    out_file = os.path.join(args.save_dir, f"{args.model_name}_inductive_inspection.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"\nReport saved to {out_file}")

if __name__ == "__main__":
    main()