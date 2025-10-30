# utils/evaluation.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from collections import defaultdict
from tqdm import tqdm # Added for progress bar

# --- Import necessary classes for type hinting (optional but good practice) ---
# Add these if not already present, adjust paths as needed
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
# from models.tools import HetAgg # Model class needed for type hint

def evaluate_model(model, data_loader, generator: DataGenerator, device, dataset: PRELUDEDataset):
    """
    Evaluates the model on a given dataloader (validation or test).
    Calculates ROC-AUC, F1-Score, and MRR. Assumes dataloader yields (u_lids, v_lids, labels, u_types).
    NOW RETURNS: metrics dict, true labels array, predicted probabilities array
    """
    model.eval()
    all_preds = []
    all_labels = []
    # --- For MRR, we need GLOBAL IDs to group by head node ---
    # We need to map LIDs back to GIDs. Let's build that map once.
    lid_to_gid_map = {
        (ntype, lid): gid for gid, (ntype, lid) in dataset.nodes['type_map'].items()
    }
    preds_by_head_gid = defaultdict(list) # Use GID for grouping
    # --- End MRR setup ---


    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)

        # Unpack 4 items now: u_lids, v_lids, labels, u_types
        for u_lids_batch, v_lids_batch, labels_batch, u_types_batch in pbar:

            # Move data to device
            u_lids_batch, v_lids_batch, labels_batch = u_lids_batch.to(device), v_lids_batch.to(device), labels_batch.to(device)
            u_types_batch = u_types_batch.to(device) # Also move types to device

            # Check for empty batch
            if u_lids_batch.numel() == 0:
                continue

            # Determine drug/cell order using LIDs and Types
            # Get type of first node in u_lids batch
            u_type_id = u_types_batch[0].item()
            drug_type_id = dataset.node_name2type.get('drug', -1)
            cell_type_id = dataset.node_name2type.get('cell', -1) # Need cell type ID for MRR head

            # --- Determine head node GIDs for MRR grouping ---
            head_lids_batch = None
            head_type_id = -1
            if u_type_id == drug_type_id:
                 drug_lids, cell_lids = u_lids_batch, v_lids_batch
                 head_lids_batch = cell_lids # Assuming cell is the head for MRR
                 head_type_id = cell_type_id
            else: # Assume u_type is cell if not drug
                 drug_lids, cell_lids = v_lids_batch, u_lids_batch
                 head_lids_batch = cell_lids # Cell is still the head
                 head_type_id = cell_type_id

            # Map head LIDs back to GIDs for grouping
            head_gids_list = [lid_to_gid_map.get((head_type_id, lid.item()), -1) for lid in head_lids_batch]
            # --- End MRR head GID determination ---


            # Forward pass using LIDs
            # Pass generator, as the current tools.py forward pass requires it (even if unused)
            preds = model.link_prediction_forward(drug_lids, cell_lids, generator)

            # Store results
            batch_preds_cpu = preds.cpu().numpy()
            batch_labels_cpu = labels_batch.cpu().numpy() # Ensure labels are on CPU for numpy operations
            all_preds.extend(batch_preds_cpu)
            all_labels.extend(batch_labels_cpu)

            # Group for MRR calculation using the head node's global ID
            for i in range(len(head_gids_list)):
                head_gid = head_gids_list[i]
                if head_gid != -1: # Check if mapping was successful
                    preds_by_head_gid[head_gid].append((batch_preds_cpu[i], batch_labels_cpu[i]))

    # --- Calculate Metrics ---
    metrics = {"Val_Loss": np.nan, "ROC-AUC": 0.0, "F1-Score": 0.0, "MRR": 0.0} # Initialize with defaults

    # Convert lists to numpy arrays once
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    if all_labels_np.size == 0 or len(np.unique(all_labels_np)) < 2:
        print("Warning: Not enough data or distinct labels found for metric calculation.")
        # Return empty arrays along with default metrics
        return metrics, all_labels_np, all_preds_np

    try:
        metrics["ROC-AUC"] = roc_auc_score(all_labels_np, all_preds_np)
        metrics["F1-Score"] = f1_score(all_labels_np, all_preds_np > 0.5)
    except ValueError as e:
         print(f"Warning: Could not calculate AUC/F1: {e}")


    # Calculate MRR
    reciprocal_ranks = []
    for head_gid, predictions in preds_by_head_gid.items():
        # Check if there are any true positive links for this head node
        if any(label == 1.0 for score, label in predictions):
            # Sort predictions by score (descending)
            predictions.sort(key=lambda x: x[0], reverse=True)
            # Find the rank of the first true positive
            for rank, (score, label) in enumerate(predictions):
                if label == 1.0:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break # Only consider the rank of the first hit

    metrics["MRR"] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Loss is no longer calculated here, return NaN
    metrics["Val_Loss"] = np.nan

    # Return the collected labels and predictions along with the metrics
    return metrics, all_labels_np, all_preds_np

