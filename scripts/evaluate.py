# scripts/evaluate.py

import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
from scripts.train import LinkPredictionDataset # Re-use the dataset class
from utils.evaluation import evaluate_model # Use the centralized evaluation function

def main():
    args = read_args()
    
    if not args.load_path or not os.path.exists(args.load_path):
        print(f"Error: Must provide a valid path to a trained model checkpoint using --load_path. Path provided: '{args.load_path}'")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Common Components ---
    print(f"Loading dataset from: {args.data_dir}")
    try:
        dataset = PRELUDEDataset(args.data_dir)
        feature_loader = FeatureLoader(dataset, device)
        # Load neighbors - required by evaluate_model as it calls model.link_prediction_forward
        neighbor_file = os.path.join(args.data_dir, "train_neighbors.txt")
        if not os.path.exists(neighbor_file):
             print(f"FATAL ERROR: Neighbor file not found at {neighbor_file}. Cannot evaluate.")
             sys.exit(1)
        generator = DataGenerator(args.data_dir).load_train_neighbors(neighbor_file)
    except Exception as e:
         print(f"FATAL ERROR during data loading: {e}")
         sys.exit(1)

    # --- Initialize Model ---
    print("Initializing model...")
    try:
        model = HetAgg(args, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    except Exception as e:
         print(f"FATAL ERROR during model initialization: {e}")
         sys.exit(1)

    # --- Load Trained Weights ---
    print(f"Loading trained model weights from: {args.load_path}")
    try:
        model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
    except Exception as e:
         print(f"FATAL ERROR loading model weights: {e}")
         sys.exit(1)
         
    model.eval() # Set model to evaluation mode

    # --- Evaluate on Inductive Test Set ---
    print("\n--- Evaluating on Inductive Test Set ---")
    test_inductive_pos = dataset.links.get('test_inductive')

    if test_inductive_pos:
        # Use LinkPredictionDataset to handle positive links and generate negatives
        test_inductive_dataset = LinkPredictionDataset(test_inductive_pos, dataset, neg_sample_ratio=1) # 1:1 neg sampling
        test_inductive_loader = DataLoader(test_inductive_dataset, batch_size=args.mini_batch_s)

        print(f"  > Evaluating on {len(test_inductive_dataset)} links ({len(test_inductive_pos)} positive).")
        
        # Call the evaluation function from utils
        inductive_metrics = evaluate_model(model, test_inductive_loader, generator, device, dataset)

        print("\n--- Inductive Test Set Performance ---")
        # Loss is NaN as it's not calculated in the simplified evaluate_model
        # print(f"  Loss:     {inductive_metrics.get('Val_Loss', 'N/A'):.4f}")
        print(f"  ROC-AUC:  {inductive_metrics['ROC-AUC']:.4f}")
        print(f"  F1-Score: {inductive_metrics['F1-Score']:.4f}")
        print(f"  MRR:      {inductive_metrics['MRR']:.4f}")
        print("------------------------------------")
    else:
        print("  > No inductive test links found ('test_inductive_links.dat'). Skipping evaluation.")

if __name__ == "__main__":
    main()