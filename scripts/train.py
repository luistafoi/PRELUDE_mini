# scripts/train.py

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse # Added for Namespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args # Assumes simplified args.py
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg # Assumes simplified tools.py
from utils.evaluation import evaluate_model

# --- Helper: Memory-Efficient Dataset for Link Prediction ---
class LinkPredictionDataset(Dataset):
    """
    Custom Dataset for link prediction.
    Randomly samples a ratio of positive links and generates negatives each time it's created.
    """
    def __init__(self, all_positive_links, full_dataset: PRELUDEDataset, sample_ratio=0.8, neg_sample_ratio=1):
        self.full_dataset = full_dataset
        self.neg_sample_ratio = neg_sample_ratio
        self.sample_ratio = sample_ratio # Ratio of positives to sample

        # --- Sample positive links for this instance ---
        num_pos_to_sample = int(len(all_positive_links) * self.sample_ratio)
        # Ensure at least one positive link is sampled if possible
        num_pos_to_sample = max(1, num_pos_to_sample) if all_positive_links else 0

        if num_pos_to_sample > 0:
             # Make sure to sample *from* the list, not sample the list itself
             self.current_epoch_positive_links = random.sample(list(all_positive_links), num_pos_to_sample)
        else:
             self.current_epoch_positive_links = []

        # Get drug node IDs for negative sampling
        drug_type_id = self.full_dataset.node_name2type.get('drug', -1)
        # Ensure node_types attribute exists before accessing
        # Use nodes['type_map'] which is guaranteed by _build_node_structure_dict
        self.all_drug_gids = [gid for gid, (ntype, lid) in self.full_dataset.nodes['type_map'].items() if ntype == drug_type_id]
        if not self.all_drug_gids:
             print("Warning: No drug GIDs found. Cannot perform negative sampling.")


        # Use the full set of training positives for negative sampling check
        # Assuming train_lp_set was added in PRELUDEDataset
        self.full_train_pos_set = self.full_dataset.links.get('train_lp_set', set())

        # Generate negative samples based on the CURRENT positive sample
        self.negative_links = self._generate_negative_samples()

        # Combine and create final arrays
        u_nodes = [p[0] for p in self.current_epoch_positive_links] + [n[0] for n in self.negative_links]
        v_nodes = [p[1] for p in self.current_epoch_positive_links] + [n[1] for n in self.negative_links]
        labels = [1.0] * len(self.current_epoch_positive_links) + [0.0] * len(self.negative_links)

        self.u_nodes_gid = np.array(u_nodes, dtype=np.int64)
        self.v_nodes_gid = np.array(v_nodes, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.float32)

        # --- Pre-compute LIDs and Type IDs to speed up training loop ---
        print("  > Pre-computing LIDs and Type IDs for dataset...") # Generic message
        self.u_lids = np.zeros_like(self.u_nodes_gid)
        self.v_lids = np.zeros_like(self.v_nodes_gid)
        self.u_types = np.zeros_like(self.u_nodes_gid)

        try:
            for i in range(len(self.u_nodes_gid)):
                u_gid = self.u_nodes_gid[i]
                v_gid = self.v_nodes_gid[i]

                u_type_id, u_lid = self.full_dataset.nodes['type_map'][u_gid]
                _, v_lid = self.full_dataset.nodes['type_map'][v_gid]

                self.u_lids[i] = u_lid
                self.v_lids[i] = v_lid
                self.u_types[i] = u_type_id
        except KeyError as e:
            print(f"FATAL ERROR in LinkPredictionDataset: GID {e} not found in nodes['type_map'].")
            print("This can happen if link files are out of sync with node.dat.")
            sys.exit(1)

        print("  > Pre-computing complete.")

    def _generate_negative_samples(self):
        neg_links = []
        # Generate negatives based on the *sampled* positives for this epoch
        num_neg_samples = int(len(self.current_epoch_positive_links) * self.neg_sample_ratio) # Cast to int

        # Check if positive links or drug candidates exist
        if not self.current_epoch_positive_links or not self.all_drug_gids:
            return []

        attempts = 0
        max_attempts = num_neg_samples * 20 # Increased limit

        while len(neg_links) < num_neg_samples and attempts < max_attempts:
            # Sample cell from the current epoch's positive links
            cell_gid, _ = random.choice(self.current_epoch_positive_links)
            # Sample a random drug ID
            neg_drug_gid = random.choice(self.all_drug_gids)

            # Check against the FULL set of known training positives
            if (cell_gid, neg_drug_gid) not in self.full_train_pos_set:
                neg_links.append((cell_gid, neg_drug_gid))
            attempts += 1

        if len(neg_links) < num_neg_samples:
            print(f"  > Warning: Only generated {len(neg_links)}/{num_neg_samples} negative samples.")

        return neg_links

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return pre-computed LIDs and Type IDs
        return (torch.tensor(self.u_lids[idx], dtype=torch.long),
                torch.tensor(self.v_lids[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float),
                torch.tensor(self.u_types[idx], dtype=torch.long))


# --- Main Training Function ---
def main():
    args = read_args() # Assumes args.py is simplified

    # --- Setup ---
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Use save_dir for checkpoints, create if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created checkpoint directory: {args.save_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Components ---
    print(f"Loading dataset from: {args.data_dir}")
    try:
        dataset = PRELUDEDataset(args.data_dir)
        feature_loader = FeatureLoader(dataset, device) # Loads static features

        # NOTE: DataGenerator is now only used inside the loss/forward pass
        # It's loaded there if needed. We don't need to load neighbors explicitly here
        # IF using the pre-processed neighbor file via models/tools.py.
        # Let's keep the generator initialization for now as evaluate_model might need it.
        generator = DataGenerator(args.data_dir)

    except Exception as e:
         print(f"FATAL ERROR during data/feature loading: {e}")
         sys.exit(1)

    # --- Initialize Model ---
    print("Initializing model...")
    try:
        # Create a dummy args namespace if HetAgg expects specific attributes not in read_args()
        model_args_ns = argparse.Namespace(**vars(args))
        # Ensure necessary args for HetAgg are present (e.g., use_skip_connection)
        if not hasattr(model_args_ns, 'use_skip_connection'): model_args_ns.use_skip_connection = False # Default if missing

        model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    except Exception as e:
         print(f"FATAL ERROR during model initialization: {e}")
         sys.exit(1)


    # --- Prepare DataLoaders ONCE ---
    print("\nPreparing dataloaders...")

    # --- Training Loader (Created Once) ---
    all_train_pos = dataset.links.get('train_lp', [])
    if not all_train_pos:
         print("FATAL ERROR: No training LP links found ('train_lp_links.dat'). Cannot train.")
         sys.exit(1)

    # The sampling/negative sampling happens ONCE here
    if args.train_fraction < 1.0:
        print(f"  > WARNING: Using only {args.train_fraction * 100:.0f}% of training data for speed.")

    train_dataset = LinkPredictionDataset(all_train_pos, dataset, sample_ratio=args.train_fraction, neg_sample_ratio=1)
    train_loader = DataLoader(train_dataset, batch_size=args.mini_batch_s, shuffle=True, num_workers=args.num_workers) # Use args.num_workers
    print(f"  > Created training loader with {len(train_dataset)} links.")

    # --- Fixed Validation Loader (Created Once) ---
    # --- START FIX ---
    # Use the generic 'validation' key set by the updated PRELUDEDataset
    valid_pos = dataset.links.get('validation', [])
    # --- END FIX ---
    if not valid_pos:
        print("Warning: No validation links found in dataset.links['validation']. Validation will be skipped.")
        valid_loader = None
    else:
        # Use LinkPredictionDataset with sample_ratio=1.0 for validation
        valid_dataset = LinkPredictionDataset(valid_pos, dataset, sample_ratio=1.0, neg_sample_ratio=1)
        valid_loader = DataLoader(valid_dataset, batch_size=args.mini_batch_s, num_workers=args.num_workers) # Use args.num_workers
        print(f"  > Created validation loader with {len(valid_dataset)} links.")

    # --- Training Loop & Logging Setup ---
    best_valid_auc = 0.0
    patience_counter = 0
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")
    log_path = os.path.join(args.save_dir, f"{args.model_name}_log.csv")
    # Construct log directory using the suffix if provided
    log_dir = f'runs/{args.model_name}'
    if hasattr(args, 'log_dir_suffix') and args.log_dir_suffix:
        log_dir += f'_{args.log_dir_suffix}'
    writer = SummaryWriter(log_dir=log_dir)


    print(f"\n--- Starting Model Training for {args.epochs} Epochs ---")
    print(f" > Best model will be saved to: {save_path}")
    print(f" > Logs will be saved to: {log_path} and TensorBoard {log_dir}")

    try: # Wrap training loop in try-except for cleaner exit
        with open(log_path, 'w', newline='') as log_file:
            csv_writer = csv.writer(log_file)
            # Simplified CSV header
            csv_writer.writerow(['epoch', 'lp_loss', 'total_loss', 'val_loss', 'val_auc', 'val_f1', 'val_mrr', 'lr', 'grad_norm'])

            # --- Main Epoch Loop ---
            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch+1}/{args.epochs}")
                model.train()
                total_lp_loss_epoch = 0
                total_grad_norm_epoch = 0
                num_batches_processed = 0

                # --- Link Prediction Training Phase ---
                lp_iterator = tqdm(train_loader, desc="Link Prediction Training", leave=False)
                # We now get LIDs and types directly from the parallel loader
                for i, (u_lids_batch, v_lids_batch, labels_batch, u_types_batch) in enumerate(lp_iterator):
                    # Move data to device
                    u_lids, v_lids, labels = u_lids_batch.to(device), v_lids_batch.to(device), labels_batch.to(device)
                    u_types = u_types_batch.to(device) # Also move types

                    # Determine drug/cell order
                    u_type_id = u_types[0].item()
                    drug_type_id = dataset.node_name2type.get('drug', -1)

                    if u_type_id == drug_type_id:
                        drug_lids, cell_lids = u_lids, v_lids
                    else:
                        drug_lids, cell_lids = v_lids, u_lids

                    # --- Forward, Backward, Optimize ---
                    optimizer.zero_grad()
                    # Pass generator - required by evaluate_model, might be removable from loss later
                    loss = model.link_prediction_loss(drug_lids, cell_lids, labels, generator, isolation_ratio=0.0)
                    loss.backward()

                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    total_grad_norm_epoch += grad_norm.item() if torch.isfinite(grad_norm) else 0.0

                    optimizer.step()

                    # --- Accumulate Loss ---
                    total_lp_loss_epoch += loss.item()
                    num_batches_processed += 1
                    lp_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})

                # --- End of Epoch Calculations ---
                avg_lp_loss_epoch = total_lp_loss_epoch / num_batches_processed if num_batches_processed > 0 else 0
                avg_grad_norm_epoch = total_grad_norm_epoch / num_batches_processed if num_batches_processed > 0 else 0
                current_lr = optimizer.param_groups[0]['lr']

                print(f"  Avg LP Loss: {avg_lp_loss_epoch:.4f}")

                # --- Validation & Early Stopping ---
                val_metrics = {"Val_Loss": np.nan, "ROC-AUC": np.nan, "F1-Score": np.nan, "MRR": np.nan}
                if valid_loader is not None and (epoch + 1) % args.val_freq == 0:
                    print("  Running Validation...")
                    # Pass generator to evaluate_model
                    val_metrics = evaluate_model(model, valid_loader, generator, device, dataset)
                    print(f"  Validation | AUC: {val_metrics['ROC-AUC']:.4f}, F1: {val_metrics['F1-Score']:.4f}, MRR: {val_metrics['MRR']:.4f}")

                    if val_metrics['ROC-AUC'] > best_valid_auc:
                        best_valid_auc = val_metrics['ROC-AUC']
                        patience_counter = 0
                        torch.save(model.state_dict(), save_path)
                        print(f"  ✨ New best model saved to {save_path} (AUC: {best_valid_auc:.4f})")
                    else:
                        patience_counter += 1
                        print(f"  > Validation AUC did not improve. Patience: {patience_counter}/{args.patience}")
                        if patience_counter >= args.patience:
                            print(f"  Stopping early due to lack of validation improvement.")
                            break # Exit epoch loop

                # --- Logging ---
                avg_total_loss_epoch = avg_lp_loss_epoch # Total loss is just LP loss
                log_row = [
                    epoch + 1, avg_lp_loss_epoch, avg_total_loss_epoch,
                    val_metrics['Val_Loss'], val_metrics['ROC-AUC'], val_metrics['F1-Score'], val_metrics['MRR'],
                    current_lr, avg_grad_norm_epoch
                ]
                csv_writer.writerow(log_row)

                writer.add_scalar('Loss/Train_LP', avg_lp_loss_epoch, epoch + 1)
                writer.add_scalar('Loss/Train_Total', avg_total_loss_epoch, epoch + 1)
                writer.add_scalar('Training/Learning_Rate', current_lr, epoch + 1)
                writer.add_scalar('Training/Gradient_Norm', avg_grad_norm_epoch, epoch + 1)

                # Log validation metrics if calculated
                if not np.isnan(val_metrics['ROC-AUC']):
                    writer.add_scalar('Validation/AUC', val_metrics['ROC-AUC'], epoch + 1)
                    writer.add_scalar('Validation/F1-Score', val_metrics['F1-Score'], epoch + 1)
                    writer.add_scalar('Validation/MRR', val_metrics['MRR'], epoch + 1)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # --- Clean Up ---
        writer.close()
        print("\n--- Training Loop Finished ---")
        print(f"Best validation AUC achieved: {best_valid_auc:.4f}")
        print(f"Logs saved to: {log_path}")
        print(f"TensorBoard logs saved to: {log_dir}") # Use log_dir variable
        if os.path.exists(save_path):
             print(f"Best model checkpoint saved to: {save_path}")
        else:
             print("Warning: No best model was saved (or training was interrupted before first save).")


if __name__ == "__main__":
    main()