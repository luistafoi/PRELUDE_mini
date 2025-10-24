# scripts/evaluate.py

import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from collections import defaultdict
import random

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
    
    if not args.load_path:
        print("Error: Must provide a path to a trained model checkpoint using --load_path")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Common Components ---
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir).load_train_neighbors(os.path.join(args.data_dir, "train_neighbors.txt"))
    
    model = HetAgg(args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    
    print(f"\nLoading trained model weights from: {args.load_path}")
    model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
    
    # --- Test 1: Transductive Evaluation ---
    print("\n--- Evaluating: Test Set 1 (Transductive Edge Prediction) ---")
    test_1_pos = dataset.links['test_transductive']
    if test_1_pos:
        test_1_dataset = LinkPredictionDataset(test_1_pos, dataset, neg_sample_ratio=1)
        test_1_loader = DataLoader(test_1_dataset, batch_size=args.mini_batch_s)
        
        print(f"  > Evaluating on {len(test_1_dataset)} links ({len(test_1_pos)} positive).")
        metrics_1 = evaluate_model(model, test_1_loader, generator, device, dataset)

        print("\n--- Test Set 1 Performance ---")
        print(f"  Loss:     {metrics_1.get('Val_Loss', 'N/A'):.4f}")
        print(f"  ROC-AUC:  {metrics_1['ROC-AUC']:.4f}")
        print(f"  F1-Score: {metrics_1['F1-Score']:.4f}")
        print(f"  MRR:      {metrics_1['MRR']:.4f}")
        print("----------------------------------")
    else:
        print("  > No transductive test links found. Skipping.")

    # --- Test 2: Inductive Evaluation ---
    print("\n--- Evaluating: Test Set 2 (Inductive Node Isolation) ---")
    test_2_pos = dataset.links['test_inductive']
    if test_2_pos:
        test_2_dataset = LinkPredictionDataset(test_2_pos, dataset, neg_sample_ratio=1)
        test_2_loader = DataLoader(test_2_dataset, batch_size=args.mini_batch_s)

        print(f"  > Evaluating on {len(test_2_dataset)} links ({len(test_2_pos)} positive).")
        metrics_2 = evaluate_model(model, test_2_loader, generator, device, dataset)

        print("\n--- Test Set 2 Performance ---")
        print(f"  Loss:     {metrics_2.get('Val_Loss', 'N/A'):.4f}")
        print(f"  ROC-AUC:  {metrics_2['ROC-AUC']:.4f}")
        print(f"  F1-Score: {metrics_2['F1-Score']:.4f}")
        print(f"  MRR:      {metrics_2['MRR']:.4f}")
        print("----------------------------------")
    else:
        print("  > No inductive test links found. Skipping.")

if __name__ == "__main__":
    main()
