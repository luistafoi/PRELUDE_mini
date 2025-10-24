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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
from utils.evaluation import evaluate_model # <-- Import from the new utility file

# --- Helper: Memory-Efficient Dataset for Link Prediction ---
class LinkPredictionDataset(Dataset):
    """
    Custom Dataset for link prediction. Handles on-the-fly negative sampling.
    """
    def __init__(self, positive_links, full_dataset: PRELUDEDataset, neg_sample_ratio=1):
        self.positive_links = positive_links
        self.full_dataset = full_dataset
        self.neg_sample_ratio = neg_sample_ratio

        # Get all possible drug nodes for negative sampling
        drug_type_id = self.full_dataset.node_name2type['drug']
        self.all_drug_gids = [gid for gid, (ntype, _) in self.full_dataset.nodes['type_map'].items() if ntype == drug_type_id]

        # Create a set of existing positive links for fast lookup during negative sampling
        self.existing_pos_links = set(positive_links)

        # Generate negative samples
        self.negative_links = self._generate_negative_samples()

        # Combine and create final arrays
        u_nodes = [p[0] for p in self.positive_links] + [n[0] for n in self.negative_links]
        v_nodes = [p[1] for p in self.positive_links] + [n[1] for n in self.negative_links]
        labels = [1.0] * len(self.positive_links) + [0.0] * len(self.negative_links)

        self.u_nodes = np.array(u_nodes, dtype=np.int64)
        self.v_nodes = np.array(v_nodes, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.float32)

    def _generate_negative_samples(self):
        neg_links = []
        num_neg_samples = len(self.positive_links) * self.neg_sample_ratio
        
        while len(neg_links) < num_neg_samples:
            # Get a cell from a random positive link
            cell_gid, _ = random.choice(self.positive_links)
            # Get a random drug
            neg_drug_gid = random.choice(self.all_drug_gids)

            # Check if this is a true negative
            if (cell_gid, neg_drug_gid) not in self.existing_pos_links:
                neg_links.append((cell_gid, neg_drug_gid))
        
        return neg_links

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.u_nodes[idx], self.v_nodes[idx], self.labels[idx]

# --- The old evaluate() function is now removed from this file ---

# --- Main Training Function ---
def main():
    args = read_args()

    # --- Setup ---
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Components ---
    NEIGHBOR_FILE = os.path.join(args.data_dir, "train_neighbors.txt")
    if not os.path.exists(NEIGHBOR_FILE):
        print(f"Error: Neighbor file not found at '{NEIGHBOR_FILE}'")
        print("Please run 'python scripts/generate_neighbors.py' first.")
        return

    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir).load_train_neighbors(NEIGHBOR_FILE)

    model = HetAgg(args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Prepare DataLoaders using pre-defined splits ---
    print("\nLoading data using new splits...")
    
    # Training data
    train_pos = dataset.links['train_lp']
    train_dataset = LinkPredictionDataset(train_pos, dataset, neg_sample_ratio=1)
    train_loader = DataLoader(train_dataset, batch_size=args.mini_batch_s, shuffle=True)
    print(f"  > Created training loader with {len(train_dataset)} links ({len(train_pos)} positive).")
    
    # Validation data (Inductive)
    valid_pos = dataset.links['valid_inductive']
    valid_dataset = LinkPredictionDataset(valid_pos, dataset, neg_sample_ratio=1)
    valid_loader = DataLoader(valid_dataset, batch_size=args.mini_batch_s)
    print(f"  > Created validation loader with {len(valid_dataset)} links ({len(valid_pos)} positive).")

    # --- Training Loop & Logging Setup ---
    best_valid_auc = 0.0
    patience_counter = 0
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")
    log_path = os.path.join(args.save_dir, f"{args.model_name}_log.csv")
    writer = SummaryWriter(log_dir=f'runs/{args.model_name}')

    with open(log_path, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        # Updated header for the new metrics
        csv_writer.writerow(['epoch', 'lp_loss', 'rw_loss', 'val_loss', 'val_auc', 'val_f1', 'val_mrr', 'lr', 'grad_norm'])

        print("\n--- Starting Model Training ---")

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            model.train()
            total_lp_loss = 0
            total_rw_loss = 0
            total_grad_norm = 0

            # Curriculum learning setup
            max_isolation_ratio = 0.5
            current_isolation_ratio = max_isolation_ratio * (epoch / (args.epochs - 1)) if args.epochs > 1 else 0
            
            current_lambda = 1000.0
            if args.use_lp_curriculum:
                max_lambda = args.lp_loss_lambda
                current_lambda = max_lambda * (epoch / (args.epochs - 1)) if args.epochs > 1 else max_lambda

            # Phase 1: Supervised Link Prediction
            lp_iterator = tqdm(train_loader, desc="Phase 1: Link Prediction")
            for u_gids, v_gids, labels in lp_iterator:
                optimizer.zero_grad()

                u_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in u_gids]
                v_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in v_gids]
                labels = labels.to(device)

                u_type = dataset.nodes['type_map'][u_gids[0].item()][0]
                drug_type_id = dataset.node_name2type['drug']

                if u_type == drug_type_id:
                    drug_lids, cell_lids = u_lids, v_lids
                else:
                    drug_lids, cell_lids = v_lids, u_lids

                loss = model.link_prediction_loss(drug_lids, cell_lids, labels, generator, current_isolation_ratio)
                weighted_loss = current_lambda * loss
                weighted_loss.backward()
                
                # Calculate gradient norm before optimizer step
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                total_grad_norm += grad_norm.item()

                optimizer.step()
                total_lp_loss += loss.item()
                lp_iterator.set_postfix({"Loss": total_lp_loss / (lp_iterator.n + 1)})

            # Phase 2: Self-Supervised Random Walk
            if args.use_rw_loss:
                rw_pairs = generator.generate_rw_triples(walk_length=args.walk_length, window_size=args.window_size, num_walks=args.num_walks)

                if rw_pairs:
                    all_node_ids = list(dataset.id2node.keys())
                    rw_batch = [(c, p, random.choice(all_node_ids)) for c, p in rw_pairs]

                    rw_iterator = tqdm(range(0, len(rw_batch), args.mini_batch_s), desc="Phase 2: Random Walk")
                    for i in rw_iterator:
                        optimizer.zero_grad()
                        batch = rw_batch[i : i + args.mini_batch_s]
                        if not batch: continue
                        loss_rw = model.self_supervised_rw_loss(batch, generator)
                        loss_rw.backward()
                        optimizer.step()
                        total_rw_loss += loss_rw.item()
                        rw_iterator.set_postfix({"Loss": total_rw_loss / (rw_iterator.n + 1)})

            # Calculate average metrics for the epoch
            avg_lp_loss = total_lp_loss / len(train_loader) if train_loader else 0
            avg_grad_norm = total_grad_norm / len(train_loader) if train_loader else 0
            current_lr = optimizer.param_groups[0]['lr']
            avg_rw_loss = 0
            if args.use_rw_loss and rw_pairs:
                num_rw_batches = (len(rw_batch) + args.mini_batch_s - 1) // args.mini_batch_s
                avg_rw_loss = total_rw_loss / num_rw_batches if num_rw_batches > 0 else 0

            # --- Validation & Early Stopping ---
            val_metrics = {"Val_Loss": np.nan, "ROC-AUC": np.nan, "F1-Score": np.nan, "MRR": np.nan}
            if (epoch + 1) % args.val_freq == 0:
                val_metrics = evaluate_model(model, valid_loader, generator, device, dataset)
                print(f"  Validation | Loss: {val_metrics['Val_Loss']:.4f}, AUC: {val_metrics['ROC-AUC']:.4f}, F1: {val_metrics['F1-Score']:.4f}, MRR: {val_metrics['MRR']:.4f}")

                if val_metrics['ROC-AUC'] > best_valid_auc:
                    best_valid_auc = val_metrics['ROC-AUC']
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                    print(f"  ✨ New best model saved to {save_path} (AUC: {best_valid_auc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Stopping early as validation AUC has not improved for {patience_counter} checks.")
                        break
            
            # Log all metrics to CSV and TensorBoard
            log_row = [
                epoch + 1, avg_lp_loss, avg_rw_loss, 
                val_metrics['Val_Loss'], val_metrics['ROC-AUC'], val_metrics['F1-Score'], val_metrics['MRR'],
                current_lr, avg_grad_norm
            ]
            csv_writer.writerow(log_row)
            
            writer.add_scalar('Loss/Train_LP', avg_lp_loss, epoch + 1)
            writer.add_scalar('Training/Learning_Rate', current_lr, epoch + 1)
            writer.add_scalar('Training/Gradient_Norm', avg_grad_norm, epoch + 1)
            if args.use_rw_loss:
                writer.add_scalar('Loss/Train_RW', avg_rw_loss, epoch + 1)
            
            if not np.isnan(val_metrics['ROC-AUC']):
                writer.add_scalar('Loss/Validation_LP', val_metrics['Val_Loss'], epoch + 1)
                writer.add_scalar('Validation/AUC', val_metrics['ROC-AUC'], epoch + 1)
                writer.add_scalar('Validation/F1-Score', val_metrics['F1-Score'], epoch + 1)
                writer.add_scalar('Validation/MRR', val_metrics['MRR'], epoch + 1)
    
    writer.close()
    print("\n--- Training Complete ---")
    print(f"Best validation AUC: {best_valid_auc:.4f}")

if __name__ == "__main__":
    main()
