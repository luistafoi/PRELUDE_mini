# utils/evaluation.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from collections import defaultdict
from tqdm import tqdm # Added for progress bar

# Note: Removed generator import if not needed by the simplified forward pass

def evaluate_model(model, dataloader, generator, device, dataset): # Kept generator for now, might remove later
    """
    Evaluates the model on a given dataloader (validation or test).
    Calculates ROC-AUC, F1-Score, and MRR. Loss calculation is removed for simplicity.
    """
    model.eval()
    all_preds = []
    all_labels = []
    preds_by_cell = defaultdict(list) # For MRR

    with torch.no_grad():
        # Add tqdm progress bar
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for u_gids_batch, v_gids_batch, labels_batch in pbar:
            # Convert global IDs to local IDs needed by the model
            # This assumes the dataloader yields global IDs
            try:
                u_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in u_gids_batch]
                v_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in v_gids_batch]
            except KeyError as e:
                print(f"Error: Node ID {e} not found in dataset.nodes['type_map']. Check data consistency.")
                continue # Skip batch if IDs are invalid

            labels = labels_batch # Labels are already floats, move to device later if needed
            
            # Determine which list contains cells and drugs based on the first node's type
            u_type = dataset.nodes['type_map'].get(u_gids_batch[0].item(), [None])[0]
            drug_type_id = dataset.node_name2type.get('drug', -1)

            if u_type == drug_type_id:
                drug_lids, cell_lids = u_lids, v_lids
                # For MRR, we need the global IDs of the 'head' node (assuming cell is the head for MRR)
                head_gids = v_gids_batch.numpy()
            else: # Assume u_type is cell if not drug
                drug_lids, cell_lids = v_lids, u_lids
                head_gids = u_gids_batch.numpy()

            # Move tensors to device right before model call
            # Note: The model expects local IDs
            # Ensure drug_lids and cell_lids are lists of integers
            if not drug_lids or not cell_lids: continue # Skip empty batches

            # Run model inference
            # Pass generator, as the current tools.py forward pass requires it
            preds = model.link_prediction_forward(drug_lids, cell_lids, generator)

            # Store results
            batch_preds_cpu = preds.cpu().numpy()
            batch_labels_cpu = labels.cpu().numpy() # Ensure labels are on CPU for numpy operations
            all_preds.extend(batch_preds_cpu)
            all_labels.extend(batch_labels_cpu)

            # Group for MRR calculation using the head node's global ID
            for i in range(len(head_gids)):
                head_gid = head_gids[i]
                preds_by_cell[head_gid].append((batch_preds_cpu[i], batch_labels_cpu[i]))

    # --- Calculate Metrics ---
    metrics = {"Val_Loss": np.nan, "ROC-AUC": 0.0, "F1-Score": 0.0, "MRR": 0.0} # Initialize with defaults
    
    if not all_labels or len(np.unique(all_labels)) < 2:
        print("Warning: Not enough data or labels for metric calculation.")
        return metrics

    try:
        metrics["ROC-AUC"] = roc_auc_score(all_labels, all_preds)
        # Use a threshold of 0.5 for F1 score calculation
        metrics["F1-Score"] = f1_score(np.array(all_labels), np.array(all_preds) > 0.5)
    except ValueError as e:
         print(f"Warning: Could not calculate AUC/F1: {e}")


    # Calculate MRR
    reciprocal_ranks = []
    for head_gid, predictions in preds_by_cell.items():
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

    return metrics