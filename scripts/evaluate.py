# scripts/evaluate.py

import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tmux/headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
from scripts.train import LinkPredictionDataset
from utils.evaluation import evaluate_model


# --- PLOTTING FUNCTIONS ---

def save_roc_curve_plot(true_labels, pred_probs, test_metrics, test_set_name, save_dir, model_base_name, test_set_key):
    """Calculates and saves the ROC curve plot."""
    binary_labels = (true_labels > 0.5).astype(int)
    if binary_labels.size == 0 or pred_probs.size == 0 or len(np.unique(binary_labels)) < 2:
        print("Skipping ROC plot generation due to insufficient data or labels.")
        return

    try:
        fpr, tpr, thresholds = roc_curve(binary_labels, pred_probs)
        roc_auc_value = test_metrics.get('ROC-AUC', auc(fpr, tpr))

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {test_set_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.5)

        plot_filename = os.path.join(save_dir, f"{model_base_name}_{test_set_key}_roc_curve.pdf")
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        print(f"ROC curve plot saved to: {plot_filename}")
        plt.close()

    except Exception as e:
        print(f"Error generating or saving ROC plot: {e}")


def save_auc_boxplots(results_df, dataset, metadata_file, save_dir, model_base_name, test_set_key="test"):
    """Calculates and saves AUC boxplots per tissue type and per drug."""
    print("\n--- Generating AUC Boxplots ---")

    try:
        if not os.path.exists(metadata_file):
            print(f"Metadata file not found at {metadata_file}. Skipping boxplots.")
            return

        metadata_df = pd.read_csv(metadata_file)
        metadata_map = metadata_df.set_index('cell_name')['tissue_type'].to_dict()

        gid_to_cell_name = {gid: name for gid, name in dataset.id2node.items() if dataset.node_types.get(gid) == 0}
        gid_to_drug_name = {gid: name for gid, name in dataset.id2node.items() if dataset.node_types.get(gid) == 1}

        results_df['cell_name'] = results_df['cell_gid'].map(gid_to_cell_name)
        results_df['drug_name'] = results_df['drug_gid'].map(gid_to_drug_name)
        results_df['tissue_type'] = results_df['cell_name'].map(lambda x: metadata_map.get(x, 'Unknown'))

    except Exception as e:
        print(f"Error merging metadata for boxplots: {e}. Skipping plots.")
        return

    # Plot AUC by Tissue Type
    try:
        auc_by_tissue = []
        for tissue, group in results_df.groupby('tissue_type'):
            binary_labels = (group['true_label'] > 0.5).astype(int)
            if len(binary_labels.unique()) > 1:
                tissue_auc = roc_auc_score(binary_labels, group['pred_prob'])
                auc_by_tissue.append({'tissue_type': tissue, 'AUC': tissue_auc})

        if auc_by_tissue:
            auc_tissue_df = pd.DataFrame(auc_by_tissue).sort_values(by='AUC', ascending=False)
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='AUC', y='tissue_type', data=auc_tissue_df, orient='h', palette='viridis')
            sns.stripplot(x='AUC', y='tissue_type', data=auc_tissue_df, orient='h', color=".25")
            plt.title(f'Test AUC per Cancer Type - {model_base_name}')
            plt.xlabel('ROC-AUC Score')
            plt.ylabel('Cancer Type')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()

            plot_filename = os.path.join(save_dir, f"{model_base_name}_{test_set_key}_auc_by_tissue.pdf")
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"AUC by tissue plot saved to: {plot_filename}")
            plt.close()
    except Exception as e:
        print(f"Error generating tissue boxplot: {e}")

    # Plot AUC by Drug
    try:
        auc_by_drug = []
        for drug, group in results_df.groupby('drug_name'):
            binary_labels = (group['true_label'] > 0.5).astype(int)
            if len(binary_labels.unique()) > 1:
                drug_auc = roc_auc_score(binary_labels, group['pred_prob'])
                auc_by_drug.append({'drug_name': drug, 'AUC': drug_auc, 'num_links': len(group)})

        if auc_by_drug:
            auc_drug_df = pd.DataFrame(auc_by_drug)
            auc_drug_df = auc_drug_df[auc_drug_df['num_links'] > 10].sort_values(by='AUC', ascending=False)

            top_15 = auc_drug_df.head(15)
            bottom_15 = auc_drug_df.tail(15)
            plot_df = pd.concat([top_15, bottom_15])

            plt.figure(figsize=(12, 10))
            sns.barplot(x='AUC', y='drug_name', data=plot_df, palette='coolwarm')
            plt.title(f'Test AUC per Drug (Top/Bottom 15) - {model_base_name}')
            plt.xlabel('ROC-AUC Score')
            plt.ylabel('Drug')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()

            plot_filename = os.path.join(save_dir, f"{model_base_name}_{test_set_key}_auc_by_drug.pdf")
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
            print(f"AUC by drug plot saved to: {plot_filename}")
            plt.close()

    except Exception as e:
        print(f"Error generating drug plot: {e}")


# --- MAIN EXECUTION ---

def main(args, use_inductive):
    if not args.load_path or not os.path.exists(args.load_path):
        print(f"Error: Must provide a valid model checkpoint using --load_path.")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Components ---
    print(f"Loading dataset from: {args.data_dir}")
    try:
        dataset = PRELUDEDataset(args.data_dir)
        feature_loader = FeatureLoader(dataset, device)
        generator = DataGenerator(args.data_dir)
    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}")
        sys.exit(1)

    # --- Initialize Model ---
    print("Initializing model...")
    try:
        model_args_ns = argparse.Namespace(**vars(args))
        model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        sys.exit(1)

    # --- Load Trained Weights ---
    print(f"Loading trained model weights from: {args.load_path}")
    try:
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            print("Unknown checkpoint format.")
    except Exception as e:
        print(f"FATAL ERROR loading model weights: {e}")
        sys.exit(1)

    model.eval()

    # --- Determine Which Test Set to Use ---
    if use_inductive:
        print(">>> Mode: INDUCTIVE Evaluation (New Patients)")
        test_links = dataset.links.get('test_inductive', [])
        if not test_links:
            print("FATAL ERROR: 'test_inductive_links.dat' not found or empty.")
            sys.exit(1)
        test_set_name = "Inductive Test Set (Unseen Cells)"
        test_set_key = 'test_inductive'
    else:
        print(">>> Mode: TRANSDUCTIVE Evaluation (Old Patients)")
        test_links = dataset.links.get('test_transductive', [])
        if not test_links:
            print("FATAL ERROR: 'test_transductive_links.dat' not found.")
            sys.exit(1)
        test_set_name = "Transductive Test Set (Seen Cells)"
        test_set_key = 'test_transductive'

    print(f"\n--- Evaluating on {test_set_name} ---")

    # Uses actual labels from split files (no random negative generation)
    test_dataset = LinkPredictionDataset(test_links, dataset, sample_ratio=1.0)
    test_loader = DataLoader(test_dataset, batch_size=args.mini_batch_s, num_workers=args.num_workers)

    print(f"  > Evaluating on {len(test_dataset)} links.")

    # --- Run Evaluation ---
    test_metrics, results_df = evaluate_model(model, test_loader, generator, device, dataset)

    # --- Print Results ---
    print(f"\n--- Final Results ({test_set_name}) ---")
    print(f"  ROC-AUC:  {test_metrics['ROC-AUC']:.4f}")
    print(f"  F1-Score: {test_metrics['F1-Score']:.4f}")
    print(f"  MRR:      {test_metrics['MRR']:.4f}")
    print("------------------------------------------")

    # --- Plotting ---
    save_dir = args.save_dir if hasattr(args, 'save_dir') else 'checkpoints'
    model_base_name = args.model_name if hasattr(args, 'model_name') else os.path.splitext(os.path.basename(args.load_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    true_labels = results_df['true_label'].values
    pred_probs = results_df['pred_prob'].values

    save_roc_curve_plot(true_labels, pred_probs,
                        test_metrics, test_set_name, save_dir, model_base_name, test_set_key)

    metadata_file = args.metadata_file if hasattr(args, 'metadata_file') else 'data/misc/cell_line_metadata.csv'
    save_auc_boxplots(results_df, dataset, metadata_file, save_dir, model_base_name, test_set_key)


if __name__ == "__main__":
    use_inductive = False
    if '--inductive' in sys.argv:
        use_inductive = True
        sys.argv.remove('--inductive')

    args = read_args()
    main(args, use_inductive)
