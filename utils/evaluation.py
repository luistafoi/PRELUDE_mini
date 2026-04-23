# utils/evaluation.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator


def evaluate_model(model, data_loader, generator: DataGenerator, device, dataset: PRELUDEDataset,
                   embedding_tables=None, regression=False):
    """
    Evaluates the model on a given dataloader (validation or test).
    Classification mode: ROC-AUC, F1-Score, MRR.
    Regression mode: Spearman, Pearson, MAE + binary metrics via median threshold.

    Args:
        embedding_tables: optional dict from model.compute_all_embeddings().
        regression: if True, compute regression metrics (Spearman, Pearson, MAE).

    Returns:
    1. metrics (dict): Scalar metrics.
    2. results_df (pd.DataFrame): DataFrame with all predictions and metadata.
    """
    model.eval()

    all_pred_probs = []
    all_true_labels = []
    all_cell_gids = []
    all_drug_gids = []

    cell_type_id = dataset.node_name2type.get('cell', -1)
    drug_type_id = dataset.node_name2type.get('drug', -1)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)

        for u_lids_batch, v_lids_batch, labels_batch, u_types_batch, u_gids_batch, v_gids_batch in pbar:

            u_lids_batch = u_lids_batch.to(device)
            v_lids_batch = v_lids_batch.to(device)
            labels_batch = labels_batch.to(device)
            u_types_batch = u_types_batch.to(device)

            if u_lids_batch.numel() == 0:
                continue

            u_type_id = u_types_batch[0].item()
            assert (u_types_batch == u_type_id).all(), "Mixed node types in eval batch"

            if u_type_id == drug_type_id:
                drug_lids, cell_lids = u_lids_batch, v_lids_batch
                drug_gids, cell_gids = u_gids_batch, v_gids_batch
            else:
                drug_lids, cell_lids = v_lids_batch, u_lids_batch
                drug_gids, cell_gids = v_gids_batch, u_gids_batch

            preds = model.link_prediction_forward(
                drug_lids, cell_lids, generator,
                embedding_tables=embedding_tables
            )

            all_pred_probs.extend(preds.cpu().numpy())
            all_true_labels.extend(labels_batch.cpu().numpy())
            all_cell_gids.extend(cell_gids.numpy())
            all_drug_gids.extend(drug_gids.numpy())

    # --- Create Results DataFrame ---
    results_df = pd.DataFrame({
        'cell_gid': all_cell_gids,
        'drug_gid': all_drug_gids,
        'true_label': all_true_labels,
        'pred_prob': all_pred_probs
    })

    # --- Calculate Metrics ---
    true_labels_np = results_df['true_label'].values
    pred_np = results_df['pred_prob'].values

    if regression:
        metrics = {"Val_Loss": np.nan, "Spearman": 0.0, "Pearson": 0.0, "MAE": 0.0,
                   "ROC-AUC": 0.0, "F1-Score": 0.0}

        # Filter out NaN labels if any
        valid_mask = ~np.isnan(true_labels_np)
        n_dropped = int((~valid_mask).sum())
        if n_dropped > 0:
            print(f"Warning: Dropped {n_dropped}/{len(true_labels_np)} NaN labels from eval")
        true_valid = true_labels_np[valid_mask]
        pred_valid = pred_np[valid_mask]

        if len(true_valid) < 10:
            print("Warning: Too few valid labels for regression metrics.")
            return metrics, results_df

        try:
            metrics["Spearman"] = spearmanr(true_valid, pred_valid).statistic
            metrics["Pearson"] = pearsonr(true_valid, pred_valid).statistic
            metrics["MAE"] = np.mean(np.abs(true_valid - pred_valid))
        except Exception as e:
            print(f"Warning: Could not calculate regression metrics: {e}")

        # Also compute binary metrics using median split for reference
        median_true = np.median(true_valid)
        binary_true = (true_valid > median_true).astype(int)
        if len(np.unique(binary_true)) == 2:
            try:
                metrics["ROC-AUC"] = roc_auc_score(binary_true, pred_valid)
                pred_binary = (pred_valid > np.median(pred_valid)).astype(int)
                metrics["F1-Score"] = f1_score(binary_true, pred_binary)
            except ValueError:
                pass

        return metrics, results_df

    # --- Classification mode ---
    metrics = {"Val_Loss": np.nan, "ROC-AUC": 0.0, "F1-Score": 0.0, "MRR": 0.0}

    # Binarize soft labels for metrics (>0.5 = positive)
    binary_labels_np = (true_labels_np > 0.5).astype(int)

    if binary_labels_np.size == 0 or len(np.unique(binary_labels_np)) < 2:
        print("Warning: Not enough data or distinct labels found for metric calculation.")
        return metrics, results_df

    try:
        metrics["ROC-AUC"] = roc_auc_score(binary_labels_np, pred_np)
        metrics["F1-Score"] = f1_score(binary_labels_np, pred_np > 0.5)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC/F1: {e}")

    # MRR: For each cell, rank ALL drugs by predicted score,
    # compute reciprocal rank of each true positive, average across all positives
    reciprocal_ranks = []
    for cell_gid, group in results_df.groupby('cell_gid'):
        positives = group[group['true_label'] > 0.5]
        if len(positives) == 0:
            continue
        # Sort all drugs for this cell by predicted score (descending)
        ranked = group.sort_values(by='pred_prob', ascending=False).reset_index(drop=True)
        # Find rank of EACH true positive (not just first hit)
        for _, pos_row in positives.iterrows():
            rank_idx = ranked.index[ranked['drug_gid'] == pos_row['drug_gid']]
            if len(rank_idx) > 0:
                reciprocal_ranks.append(1.0 / (rank_idx[0] + 1))

    metrics["MRR"] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return metrics, results_df
