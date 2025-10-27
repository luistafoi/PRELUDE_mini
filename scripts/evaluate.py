# scripts/evaluate.py

import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse # Import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
# Re-use the dataset class from train.py for consistency in creating loaders
from scripts.train import LinkPredictionDataset
# Use the centralized evaluation function from utils
from utils.evaluation import evaluate_model

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
        # Load generator - needed by evaluate_model which calls model.link_prediction_forward
        generator = DataGenerator(args.data_dir)
        # Note: If using preprocessed neighbors, generator doesn't need explicit neighbor loading here
        # as the model loads the .pkl file itself.

    except Exception as e:
         print(f"FATAL ERROR during data loading: {e}")
         sys.exit(1)

    # --- Initialize Model ---
    print("Initializing model...")
    try:
        # Create a namespace from args to ensure model gets all needed params
        model_args_ns = argparse.Namespace(**vars(args))
        # Ensure consistency with training (e.g., skip connection)
        if not hasattr(model_args_ns, 'use_skip_connection'): model_args_ns.use_skip_connection = False

        model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    except Exception as e:
         print(f"FATAL ERROR during model initialization: {e}")
         sys.exit(1)

    # --- Load Trained Weights ---
    print(f"Loading trained model weights from: {args.load_path}")
    try:
        # Use weights_only=True for security
        model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
    except Exception as e:
         print(f"FATAL ERROR loading model weights: {e}")
         if 'size mismatch' in str(e):
              print("\nHint: Ensure evaluation arguments (e.g., --use_skip_connection, --embed_d) MATCH the arguments used during training.")
         sys.exit(1)

    model.eval() # Set model to evaluation mode

    # --- START FIX: Determine Which Test Set to Use ---
    test_pos = None
    test_set_name = "None"
    test_set_key = "None"

    # Prioritize Transductive Test Set
    if dataset.links.get('test_transductive'):
        test_pos = dataset.links['test_transductive']
        test_set_name = "Transductive Test Set"
        test_set_key = 'test_transductive'
    # Fallback to Inductive Test Set
    elif dataset.links.get('test_inductive'):
        test_pos = dataset.links['test_inductive']
        test_set_name = "Inductive Test Set"
        test_set_key = 'test_inductive'

    print(f"\n--- Evaluating on {test_set_name} ---")
    # --- END FIX ---

    if test_pos:
        # Use LinkPredictionDataset to handle positive links and generate negatives
        # Using sample_ratio=1.0 to get all positive links for evaluation
        test_dataset = LinkPredictionDataset(test_pos, dataset, sample_ratio=1.0, neg_sample_ratio=1) # 1:1 neg sampling
        # Use num_workers from args for consistency
        test_loader = DataLoader(test_dataset, batch_size=args.mini_batch_s, num_workers=args.num_workers)

        print(f"  > Evaluating on {len(test_dataset)} links ({len(test_pos)} positive from '{test_set_key}_links.dat').")

        # Call the evaluation function from utils
        test_metrics = evaluate_model(model, test_loader, generator, device, dataset)

        print(f"\n--- {test_set_name} Performance ---")
        # Loss is NaN as it's not calculated in evaluate_model
        print(f"  ROC-AUC:  {test_metrics['ROC-AUC']:.4f}")
        print(f"  F1-Score: {test_metrics['F1-Score']:.4f}")
        print(f"  MRR:      {test_metrics['MRR']:.4f}")
        print("------------------------------------")
    else:
        # Update warning message
        print(f"  > No test links found ('test_transductive_links.dat' or 'test_inductive_links.dat'). Skipping evaluation.")

if __name__ == "__main__":
    main()