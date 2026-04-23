# scripts/train.py

import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
from utils.evaluation import evaluate_model
from utils.loss_manager import DynamicLossManager


def compute_mad(embedding_tables, neighbor_lids, neighbor_masks, node_types, max_sample=2000):
    """Compute Mean Average Distance (MAD) per node type.

    MAD = (1/N) * sum_i [ (1/|N(i)|) * sum_{j in N(i)} ||h_i - h_j||_2 ]

    A low MAD means neighbors have very similar embeddings (over-smoothing).
    Computed on a random subsample for speed.

    Returns:
        dict: {type_name: mad_value} for each node type, plus 'overall'
    """
    type_names = {0: 'cell', 1: 'drug', 2: 'gene'}
    mad_results = {}

    total_mad, total_count = 0.0, 0
    for ct in node_types:
        embeds = embedding_tables[ct]  # (N, D)
        n_nodes = embeds.shape[0]
        if n_nodes == 0:
            continue

        # Subsample for speed
        if n_nodes > max_sample:
            idx = torch.randperm(n_nodes, device=embeds.device)[:max_sample]
        else:
            idx = torch.arange(n_nodes, device=embeds.device)

        node_embeds = embeds[idx]  # (S, D)
        mad_sum = 0.0
        count = 0

        for nt in node_types:
            lids = neighbor_lids[ct][nt][idx]   # (S, M)
            mask = neighbor_masks[ct][nt][idx]   # (S, M)
            neigh_embeds = embedding_tables[nt][lids]  # (S, M, D)

            # ||h_i - h_j||_2 for each neighbor
            diffs = node_embeds.unsqueeze(1) - neigh_embeds  # (S, M, D)
            dists = diffs.norm(dim=-1)  # (S, M)
            dists = dists * mask.float()

            # Mean over valid neighbors per node
            n_valid = mask.float().sum(dim=1).clamp(min=1.0)  # (S,)
            per_node = dists.sum(dim=1) / n_valid  # (S,)

            # Only count nodes that have at least 1 valid neighbor of this type
            has_neigh = mask.any(dim=1)
            if has_neigh.any():
                mad_sum += per_node[has_neigh].sum().item()
                count += has_neigh.sum().item()

        if count > 0:
            type_mad = mad_sum / count
            mad_results[type_names.get(ct, str(ct))] = type_mad
            total_mad += mad_sum
            total_count += count

    mad_results['overall'] = total_mad / max(1, total_count)
    return mad_results


# --- Mini-Batch Seed Collection Helpers ---

def collect_seed_lids_from_lp_batch(drug_lids, cell_lids, dataset):
    """Collect seed local IDs from a link prediction batch.

    Args:
        drug_lids: (B,) tensor of drug local IDs
        cell_lids: (B,) tensor of cell local IDs
        dataset: PRELUDEDataset instance

    Returns:
        {node_type_id: (S,) unique LongTensor}
    """
    drug_type_id = dataset.node_name2type['drug']
    cell_type_id = dataset.node_name2type['cell']
    return {
        drug_type_id: drug_lids.unique(),
        cell_type_id: cell_lids.unique(),
    }


def collect_seed_lids_from_triplet_batch(anc_gids, pos_gids, neg_gids, dataset):
    """Collect seed local IDs from a triplet batch (GIDs -> LIDs by type).

    Args:
        anc_gids, pos_gids, neg_gids: (B,) tensors of global IDs
        dataset: PRELUDEDataset instance

    Returns:
        {node_type_id: (S,) unique LongTensor}
    """
    seeds = defaultdict(list)
    for gid in torch.cat([anc_gids, pos_gids, neg_gids]).unique().tolist():
        ntype, lid = dataset.nodes['type_map'][gid]
        seeds[ntype].append(lid)

    device = anc_gids.device
    return {
        nt: torch.tensor(lids, dtype=torch.long, device=device).unique()
        for nt, lids in seeds.items()
    }


def merge_seed_lids(*seed_dicts):
    """Merge multiple seed dicts, deduplicating per type.

    Returns:
        {node_type_id: (S,) unique LongTensor}
    """
    merged = defaultdict(list)
    for d in seed_dicts:
        for nt, lids in d.items():
            merged[nt].append(lids)

    return {
        nt: torch.cat(parts).unique()
        for nt, parts in merged.items()
        if parts
    }


# --- LinkPredictionDataset ---
class LinkPredictionDataset(Dataset):
    """Dataset for link prediction using pre-labeled (src, tgt, label) tuples."""

    def __init__(self, labeled_links, full_dataset: PRELUDEDataset, sample_ratio=1.0):
        """
        Args:
            labeled_links: list of (src_gid, tgt_gid, label) tuples where label is a float in [0,1]
            full_dataset: PRELUDEDataset instance
            sample_ratio: fraction of links to use per epoch
        """
        self.full_dataset = full_dataset

        # Optionally subsample
        num_to_sample = int(len(labeled_links) * sample_ratio)
        if num_to_sample < len(labeled_links):
            self.links = random.sample(labeled_links, num_to_sample)
        else:
            self.links = list(labeled_links)

        n_pos = sum(1 for _, _, l in self.links if l > 0.5)
        n_neg = sum(1 for _, _, l in self.links if l <= 0.5)
        n_soft = sum(1 for _, _, l in self.links if 0 < l < 1)
        extra = f", soft={n_soft}" if n_soft > 0 else ""
        print(f"    LP Dataset: {len(self.links)} links (pos={n_pos}, neg={n_neg}{extra})")

        # Pre-compute LIDs, types, and GIDs
        self.u_lids = np.zeros(len(self.links), dtype=np.int64)
        self.v_lids = np.zeros(len(self.links), dtype=np.int64)
        self.u_types = np.zeros(len(self.links), dtype=np.int64)
        self.u_gids = np.zeros(len(self.links), dtype=np.int64)
        self.v_gids = np.zeros(len(self.links), dtype=np.int64)
        self.labels = np.zeros(len(self.links), dtype=np.float32)

        if len(self.links) == 0:
            return

        try:
            for i, (u_gid, v_gid, label) in enumerate(self.links):
                u_type_id, u_lid = self.full_dataset.nodes['type_map'][u_gid]
                v_type_id, v_lid = self.full_dataset.nodes['type_map'][v_gid]
                self.u_lids[i] = u_lid
                self.v_lids[i] = v_lid
                self.u_types[i] = u_type_id
                self.u_gids[i] = u_gid
                self.v_gids[i] = v_gid
                self.labels[i] = float(label)
        except KeyError as e:
            print(f"FATAL ERROR in LinkPredictionDataset: GID {e} not found in nodes['type_map'].")
            sys.exit(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.u_lids[idx], dtype=torch.long),
                torch.tensor(self.v_lids[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float),
                torch.tensor(self.u_types[idx], dtype=torch.long),
                torch.tensor(self.u_gids[idx], dtype=torch.long),
                torch.tensor(self.v_gids[idx], dtype=torch.long))


# --- Triplet Dataset (Cell Sim) ---
class TripletDataset(IterableDataset):
    def __init__(self, triplet_map, num_pos, num_neg, name2gid):
        """
        Args:
            triplet_map: dict keyed by cell NAME -> {'pos': [names], 'neg': [names]}
            name2gid: dict mapping cell name (uppercase) -> global node ID
        """
        self.num_pos = num_pos
        self.num_neg = num_neg

        # Convert name-keyed triplet map to GID-keyed
        self.gid_triplet_map = {}
        skipped = 0
        for name, info in triplet_map.items():
            anchor_gid = name2gid.get(name.upper())
            if anchor_gid is None:
                skipped += 1
                continue
            pos_gids = [name2gid[p.upper()] for p in info.get('pos', []) if p.upper() in name2gid]
            neg_gids = [name2gid[n.upper()] for n in info.get('neg', []) if n.upper() in name2gid]
            if pos_gids and neg_gids:
                self.gid_triplet_map[anchor_gid] = {'pos': pos_gids, 'neg': neg_gids}

        self.anchor_gids = list(self.gid_triplet_map.keys())
        print(f"  > TripletDataset created with {len(self.anchor_gids)} anchor cells (skipped {skipped}).")

    def __iter__(self):
        random.shuffle(self.anchor_gids)
        for anchor_gid in self.anchor_gids:
            triplet_info = self.gid_triplet_map[anchor_gid]
            pos_gids = triplet_info['pos']
            neg_gids = triplet_info['neg']

            sampled_pos_gids = [pos_gids[i] for i in torch.randint(0, len(pos_gids), (self.num_pos,))]
            sampled_neg_gids = [neg_gids[i] for i in torch.randint(0, len(neg_gids), (self.num_neg,))]

            for pos_gid in sampled_pos_gids:
                for neg_gid in sampled_neg_gids:
                    yield (torch.tensor(anchor_gid, dtype=torch.long),
                           torch.tensor(pos_gid, dtype=torch.long),
                           torch.tensor(neg_gid, dtype=torch.long))


# --- Main Training Function ---
def main():
    args = read_args()

    # --- Setup ---
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created checkpoint directory: {args.save_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Components ---
    print(f"Loading dataset from: {args.data_dir}")
    try:
        dataset = PRELUDEDataset(args.data_dir, regression=getattr(args, 'regression', False))
        feature_loader = FeatureLoader(dataset, device,
                                       cell_feature_source=getattr(args, 'cell_feature_source', 'vae'))
        generator = DataGenerator(args.data_dir)
    except Exception as e:
        print(f"FATAL ERROR during data/feature loading: {e}")
        sys.exit(1)

    # --- Initialize Model ---
    print("Initializing model...")
    try:
        model_args_ns = argparse.Namespace(**vars(args))
        model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
        if getattr(args, 'ema_lambda', 0) > 0:
            model.init_ema_teacher()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # LR scheduler: optional linear warmup then cosine decay
        warmup_epochs = getattr(args, 'lr_warmup_epochs', 0)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
        )
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
            )
            print(f"INFO: LR warmup for {warmup_epochs} epochs (lr/10 -> lr), then cosine decay.")
        else:
            scheduler = cosine_scheduler
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        sys.exit(1)

    # --- Prepare DataLoaders ---
    print("\nPreparing dataloaders...")

    # 1. Link Prediction (LP) Loader
    all_train_lp = dataset.links.get('train_lp', [])
    if not all_train_lp:
        print("FATAL ERROR: No training LP links found. Cannot train.")
        sys.exit(1)

    train_dataset_lp = LinkPredictionDataset(all_train_lp, dataset, sample_ratio=args.train_fraction)
    train_loader_lp = DataLoader(train_dataset_lp, batch_size=args.mini_batch_s, shuffle=True,
                                 num_workers=args.num_workers)
    print(f"  > Created training loader (LP) with {len(train_dataset_lp)} links.")

    # 2. Triplet Loader
    train_loader_triplet = None
    if args.use_triplet_loss:
        triplet_map_file = os.path.join(args.data_dir, "cell_triplet_map.pkl")
        if os.path.exists(triplet_map_file):
            with open(triplet_map_file, "rb") as f:
                triplet_map = pickle.load(f)
            name2gid = {name.upper(): gid for name, gid in dataset.node2id.items()}
            train_dataset_triplet = TripletDataset(triplet_map, args.triplet_num_pos, args.triplet_num_neg, name2gid)
            triplet_batch_size = max(1, args.mini_batch_s // (args.triplet_num_pos * args.triplet_num_neg))
            train_loader_triplet = DataLoader(train_dataset_triplet, batch_size=triplet_batch_size, num_workers=0)
            print(f"  > Enabled Triplet Loss (Batch size: {triplet_batch_size}).")
        else:
            print("  > Warning: Triplet map not found. Skipping Triplet Loss.")

    # 3. Validation Loaders — DUAL: both inductive AND transductive
    valid_ind_links = dataset.links.get('valid_inductive', [])
    valid_trans_links = dataset.links.get('valid_transductive', [])

    valid_loader_ind = None
    valid_loader_trans = None

    if valid_ind_links:
        valid_dataset_ind = LinkPredictionDataset(valid_ind_links, dataset, sample_ratio=1.0)
        valid_loader_ind = DataLoader(valid_dataset_ind, batch_size=args.mini_batch_s, num_workers=args.num_workers)
        print(f"  > Created INDUCTIVE validation loader with {len(valid_dataset_ind)} links.")
    else:
        print("  > Warning: No inductive validation links found.")

    if valid_trans_links:
        valid_dataset_trans = LinkPredictionDataset(valid_trans_links, dataset, sample_ratio=1.0)
        valid_loader_trans = DataLoader(valid_dataset_trans, batch_size=args.mini_batch_s, num_workers=args.num_workers)
        print(f"  > Created TRANSDUCTIVE validation loader with {len(valid_dataset_trans)} links.")
    else:
        print("  > Warning: No transductive validation links found.")

    # --- Training Loop & Logging Setup ---
    best_valid_auc = 0.0
    patience_counter = 0
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")
    log_path = os.path.join(args.save_dir, f"{args.model_name}_log.csv")

    writer = SummaryWriter(log_dir=f'runs/{args.model_name}')

    # --- Dynamic Loss Manager ---
    loss_manager = None
    if args.use_dynamic_loss:
        target_ratios = {"lp": args.loss_target_lp}
        if args.use_triplet_loss:
            target_ratios["triplet"] = args.loss_target_triplet
        loss_manager = DynamicLossManager(
            target_ratios=target_ratios,
            ema_decay=args.loss_ema_decay,
            warmup_batches=args.loss_warmup_batches,
            max_weight=args.loss_max_weight,
        )
        print(f"  > DynamicLossManager enabled (targets: {target_ratios}, warmup: {args.loss_warmup_batches} batches, max_weight: {args.loss_max_weight})")

    freeze_after = getattr(args, 'freeze_after_epoch', 0)
    do_mad = getattr(args, 'compute_mad', False)
    use_minibatch = getattr(args, 'use_minibatch_gnn', False)
    neighbor_sample_size = getattr(args, 'neighbor_sample_size', 0)

    print(f"\n--- Starting Model Training for {args.epochs} Epochs ---")
    print(f"L1: {args.l1_lambda > 0} | CellSim: {args.use_triplet_loss} | Node Gate: {getattr(args, 'use_node_gate', False)}")
    print(f"Dynamic Loss: {args.use_dynamic_loss} | Soft Triplet: {getattr(args, 'use_soft_margin_triplet', False)}")
    print(f"Dual Validation: ind={valid_loader_ind is not None} trans={valid_loader_trans is not None}")
    print(f"Freeze after epoch: {freeze_after if freeze_after > 0 else 'disabled'} | MAD: {do_mad}")
    print(f"Mini-batch GNN: {use_minibatch} | Neighbor sample size: {neighbor_sample_size if neighbor_sample_size > 0 else 'all'}")
    print(f"Alignment loss: lambda={args.align_lambda}" if args.align_lambda > 0 else "Alignment loss: disabled")
    print(f"Dual-path LP: weight={args.proj_loss_weight}" if args.proj_loss_weight > 0 else "Dual-path LP: disabled")
    print(f"Residual GNN: {getattr(args, 'use_residual_gnn', False)} (scale={args.residual_scale})" if getattr(args, 'use_residual_gnn', False) else "Residual GNN: disabled")
    is_regression = getattr(args, 'regression', False)
    checkpoint_metric = 'Spearman' if is_regression else 'Inductive AUC'
    print(f"Regression mode: {is_regression}" if is_regression else "Classification mode")
    print(f"Checkpoint on: {checkpoint_metric}")

    try:
        with open(log_path, 'w', newline='') as log_file:
            csv_writer = csv.writer(log_file)
            if is_regression:
                header = ['epoch', 'lp_loss', 'triplet_loss', 'total_loss',
                         'val_ind_spearman', 'val_ind_pearson', 'val_ind_mae',
                         'val_trans_spearman', 'val_trans_pearson', 'val_trans_mae', 'lr']
            else:
                header = ['epoch', 'lp_loss', 'triplet_loss', 'total_loss',
                         'val_ind_auc', 'val_ind_f1', 'val_trans_auc', 'val_trans_f1', 'lr']
            if do_mad:
                header.extend(['mad_cell', 'mad_drug', 'mad_gene', 'mad_overall'])
            if getattr(args, 'use_node_gate', False) or getattr(args, 'use_residual_gnn', False):
                header.extend(['gate_cell', 'gate_drug', 'gate_gene'])
            csv_writer.writerow(header)

            for epoch in range(args.epochs):
                # --- Freeze check: set LR to near-zero after target epoch ---
                if freeze_after > 0 and (epoch + 1) > freeze_after:
                    for pg in optimizer.param_groups:
                        pg['lr'] = 1e-8
                    if (epoch + 1) == freeze_after + 1:
                        print(f"  >>> FREEZE: LR set to 1e-8 after epoch {freeze_after}")

                model.train()
                metrics = defaultdict(float)
                num_batches = 0

                # Build triplet iterator if enabled
                triplet_iter = None
                if train_loader_triplet is not None:
                    triplet_iter = iter(train_loader_triplet)

                pbar = tqdm(train_loader_lp, desc=f"Ep {epoch + 1}", leave=False)

                for lp_batch in pbar:
                    optimizer.zero_grad()

                    # --- Unpack LP batch ---
                    u_lids_b, v_lids_b, labels_b, u_types_b, _, _ = lp_batch
                    u_lids = u_lids_b.to(device)
                    v_lids = v_lids_b.to(device)
                    labels = labels_b.to(device)
                    u_types = u_types_b.to(device)
                    assert (u_types == u_types[0]).all(), "Mixed node types in LP batch"

                    if u_types[0].item() == dataset.node_name2type.get('drug', -1):
                        drug_lids, cell_lids = u_lids, v_lids
                    else:
                        drug_lids, cell_lids = v_lids, u_lids

                    # --- Get triplet batch (needed for seed collection in mini-batch) ---
                    triplet_batch_data = None
                    if triplet_iter is not None:
                        try:
                            triplet_batch_data = next(triplet_iter)
                        except StopIteration:
                            triplet_iter = iter(train_loader_triplet)
                            triplet_batch_data = next(triplet_iter)

                    # --- Compute embeddings: mini-batch or full-graph ---
                    subgraph_info = None
                    if use_minibatch:
                        # Collect seed LIDs from LP batch
                        lp_seeds = collect_seed_lids_from_lp_batch(drug_lids, cell_lids, dataset)

                        # Collect seed LIDs from triplet batch if available
                        if triplet_batch_data is not None:
                            anc, pos, neg = [x.to(device) for x in triplet_batch_data]
                            triplet_seeds = collect_seed_lids_from_triplet_batch(anc, pos, neg, dataset)
                            all_seeds = merge_seed_lids(lp_seeds, triplet_seeds)
                        else:
                            all_seeds = lp_seeds

                        # Move seeds to device
                        all_seeds = {nt: lids.to(device) for nt, lids in all_seeds.items()}

                        # Expand to k-hop and compute subgraph embeddings
                        subgraph_info = model.expand_to_k_hop(
                            all_seeds, k=args.n_layers, neighbor_sample_size=neighbor_sample_size
                        )
                        embedding_tables = model.compute_batch_embeddings(subgraph_info)
                    else:
                        # Full-graph forward: compute all embeddings (with gradients)
                        embedding_tables = model.compute_all_embeddings()

                    # --- 1. Link Prediction Loss (dual-path) ---
                    loss_lp_gnn = model.link_prediction_loss(
                        drug_lids, cell_lids, labels, generator,
                        isolation_ratio=args.isolation_ratio,
                        embedding_tables=embedding_tables,
                        subgraph_info=subgraph_info,
                    )

                    if args.proj_loss_weight > 0:
                        loss_lp_proj = model.projection_lp_loss(drug_lids, cell_lids, labels)
                        loss_lp = (1 - args.proj_loss_weight) * loss_lp_gnn + args.proj_loss_weight * loss_lp_proj
                        metrics['lp_proj'] += loss_lp_proj.item()
                    else:
                        loss_lp = loss_lp_gnn

                    metrics['lp'] += loss_lp.item()

                    # --- 2. Triplet Loss ---
                    loss_triplet = None
                    if triplet_batch_data is not None:
                        if not use_minibatch:
                            anc, pos, neg = [x.to(device) for x in triplet_batch_data]
                        loss_triplet = model.self_supervised_triplet_loss(
                            anc, pos, neg, generator,
                            embedding_tables=embedding_tables,
                            subgraph_info=subgraph_info,
                        )
                        metrics['triplet'] += loss_triplet.item()

                    # --- 3. Alignment Loss ---
                    loss_align = None
                    if args.align_lambda > 0:
                        align_types = args.align_types if args.align_types else None
                        loss_align = model.alignment_loss(
                            embedding_tables, subgraph_info, node_type_names=align_types
                        )
                        metrics['align'] += loss_align.item()

                    # --- 3b. EMA Teacher Loss ---
                    loss_ema = None
                    if getattr(args, 'ema_lambda', 0) > 0:
                        loss_ema = model.ema_teacher_loss(embedding_tables, subgraph_info)
                        metrics['ema'] += loss_ema.item()

                    # --- 4. Combine losses ---
                    total_loss_batch = 0.0
                    if loss_manager is not None:
                        raw_losses = {"lp": loss_lp}
                        if loss_triplet is not None:
                            raw_losses["triplet"] = loss_triplet
                        total_loss_batch = loss_manager.combine(raw_losses)
                    else:
                        total_loss_batch = loss_lp
                        if loss_triplet is not None:
                            total_loss_batch = total_loss_batch + (args.triplet_loss_weight * loss_triplet)

                    # Alignment loss (added outside loss manager — direct regularizer)
                    if loss_align is not None:
                        total_loss_batch = total_loss_batch + (args.align_lambda * loss_align)

                    # EMA teacher loss (direct regularizer)
                    if loss_ema is not None:
                        total_loss_batch = total_loss_batch + (args.ema_lambda * loss_ema)

                    # --- 5. L1 Regularization ---
                    if args.l1_lambda > 0:
                        l1 = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                        total_loss_batch = total_loss_batch + (args.l1_lambda * l1)

                    total_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if getattr(args, 'ema_lambda', 0) > 0:
                        model.update_ema_teacher()
                    metrics['total'] += total_loss_batch.item()
                    num_batches += 1

                    # Update Progress Bar
                    desc = f"LP: {metrics['lp'] / max(1, num_batches):.3f}"
                    if args.proj_loss_weight > 0:
                        desc += f" | ProjLP: {metrics['lp_proj'] / max(1, num_batches):.3f}"
                    if args.use_triplet_loss:
                        desc += f" | Trip: {metrics['triplet'] / max(1, num_batches):.3f}"
                    if args.align_lambda > 0:
                        desc += f" | Align: {metrics['align'] / max(1, num_batches):.3f}"
                    if getattr(args, 'ema_lambda', 0) > 0:
                        desc += f" | EMA: {metrics['ema'] / max(1, num_batches):.3f}"
                    pbar.set_postfix_str(desc)

                # --- End of Epoch ---
                if freeze_after == 0 or (epoch + 1) <= freeze_after:
                    scheduler.step()
                div = max(1, num_batches)
                current_lr = optimizer.param_groups[0]['lr']
                epoch_summary = f"  Avg Loss -> LP: {metrics['lp'] / div:.4f} | Total: {metrics['total'] / div:.4f} | LR: {current_lr:.2e}"
                if args.align_lambda > 0:
                    epoch_summary += f" | Align: {metrics['align'] / div:.4f}"
                if getattr(args, 'ema_lambda', 0) > 0:
                    epoch_summary += f" | EMA: {metrics['ema'] / div:.4f}"
                print(epoch_summary)

                # Dynamic loss manager: log and step
                if loss_manager is not None:
                    loss_manager.step_epoch()
                    lm_info = loss_manager.epoch_summary()
                    parts = []
                    for k in ["lp", "triplet"]:
                        if f"ema_{k}" in lm_info and lm_info[f"ema_{k}"] > 0:
                            parts.append(f"{k}: ema={lm_info[f'ema_{k}']:.4f} w={lm_info[f'weight_{k}']:.3f} stale={lm_info[f'stale_{k}']}")
                    print(f"  LossManager -> {' | '.join(parts)}")

                # --- Dual Validation ---
                val_ind_auc, val_ind_f1 = 0.0, 0.0
                val_trans_auc, val_trans_f1 = 0.0, 0.0
                val_ind_mae, val_trans_mae = 0.0, 0.0

                if (epoch + 1) % args.val_freq == 0:
                    # Compute embeddings once for all validation
                    with torch.no_grad():
                        eval_tables = model.compute_all_embeddings()

                    if valid_loader_ind:
                        ind_metrics, _ = evaluate_model(
                            model, valid_loader_ind, generator, device, dataset,
                            embedding_tables=eval_tables, regression=is_regression
                        )
                        if is_regression:
                            val_ind_auc = ind_metrics['Spearman']
                            val_ind_f1 = ind_metrics['Pearson']
                            val_ind_mae = ind_metrics['MAE']
                            print(f"  Val Inductive   -> Spearman: {val_ind_auc:.4f} | Pearson: {val_ind_f1:.4f} | MAE: {val_ind_mae:.4f}")
                        else:
                            val_ind_auc = ind_metrics['ROC-AUC']
                            val_ind_f1 = ind_metrics['F1-Score']
                            print(f"  Val Inductive   -> AUC: {val_ind_auc:.4f} | F1: {val_ind_f1:.4f}")

                    if valid_loader_trans:
                        trans_metrics, _ = evaluate_model(
                            model, valid_loader_trans, generator, device, dataset,
                            embedding_tables=eval_tables, regression=is_regression
                        )
                        if is_regression:
                            val_trans_auc = trans_metrics['Spearman']
                            val_trans_f1 = trans_metrics['Pearson']
                            val_trans_mae = trans_metrics['MAE']
                            print(f"  Val Transductive -> Spearman: {val_trans_auc:.4f} | Pearson: {val_trans_f1:.4f} | MAE: {val_trans_mae:.4f}")
                        else:
                            val_trans_auc = trans_metrics['ROC-AUC']
                            val_trans_f1 = trans_metrics['F1-Score']
                            print(f"  Val Transductive -> AUC: {val_trans_auc:.4f} | F1: {val_trans_f1:.4f}")

                    # MAD diagnostic
                    mad_cell, mad_drug, mad_gene, mad_overall = 0.0, 0.0, 0.0, 0.0
                    if do_mad:
                        mad = compute_mad(
                            eval_tables, model.neighbor_lids, model.neighbor_masks,
                            model.node_types
                        )
                        mad_cell = mad.get('cell', 0.0)
                        mad_drug = mad.get('drug', 0.0)
                        mad_gene = mad.get('gene', 0.0)
                        mad_overall = mad.get('overall', 0.0)
                        print(f"  MAD -> cell: {mad_cell:.4f} | drug: {mad_drug:.4f} | gene: {mad_gene:.4f} | overall: {mad_overall:.4f}")

                    # Gate alpha diagnostic
                    gate_alphas = {}
                    if getattr(args, 'use_node_gate', False) and hasattr(model, 'node_gate_mlp'):
                        type_names = {0: 'cell', 1: 'drug', 2: 'gene'}
                        parts = []
                        for nt in model.node_types:
                            n = dataset.nodes['count'][nt]
                            lids = torch.arange(n, device=device)
                            proj = model.conteng_agg(lids, nt)
                            gnn = eval_tables[nt][lids]
                            gate_in = torch.cat([proj, gnn], dim=-1)
                            alpha = torch.sigmoid(model.node_gate_mlp(gate_in)).squeeze(-1)
                            name = type_names.get(nt, str(nt))
                            gate_alphas[name] = alpha.mean().item()
                            parts.append(f"{name}: {alpha.mean():.3f}")
                        print(f"  Gate α -> {' | '.join(parts)}  (1=proj, 0=GNN)")

                    elif getattr(args, 'use_residual_gnn', False) and hasattr(model, 'residual_scale_params') and model.residual_scale_params is not None:
                        type_names = {0: 'cell', 1: 'drug', 2: 'gene'}
                        parts = []
                        for nt in model.node_types:
                            scale_val = torch.sigmoid(model.residual_scale_params[str(nt)]).item()
                            name = type_names.get(nt, str(nt))
                            gate_alphas[name] = scale_val
                            parts.append(f"{name}: {scale_val:.3f}")
                        print(f"  Residual scale -> {' | '.join(parts)}  (0=proj, 1=full delta)")

                    # Checkpoint on primary metric
                    checkpoint_val = val_ind_auc if valid_loader_ind else val_trans_auc
                    if checkpoint_val > best_valid_auc:
                        best_valid_auc = checkpoint_val
                        patience_counter = 0
                        torch.save(model.state_dict(), save_path)
                        metric_name = "Spearman" if is_regression else "Ind AUC"
                        print(f"  Saved new best model ({metric_name}: {best_valid_auc:.4f})")
                    else:
                        patience_counter += 1
                        print(f"  > No improvement. Patience: {patience_counter}/{args.patience}")
                        if patience_counter >= args.patience:
                            print("  Stopping early.")
                            break

                    # Log to CSV
                    if is_regression:
                        row = [
                            epoch + 1, metrics['lp'] / div, metrics['triplet'] / div,
                            metrics['total'] / div,
                            val_ind_auc, val_ind_f1, val_ind_mae,
                            val_trans_auc, val_trans_f1, val_trans_mae,
                            current_lr
                        ]
                    else:
                        row = [
                            epoch + 1, metrics['lp'] / div, metrics['triplet'] / div,
                            metrics['total'] / div,
                            val_ind_auc, val_ind_f1, val_trans_auc, val_trans_f1,
                            current_lr
                        ]
                    if do_mad:
                        row.extend([mad_cell, mad_drug, mad_gene, mad_overall])
                    if gate_alphas:
                        row.extend([gate_alphas.get('cell', 0), gate_alphas.get('drug', 0), gate_alphas.get('gene', 0)])
                    csv_writer.writerow(row)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        writer.close()
        metric_name = "Inductive Spearman" if is_regression else "Inductive AUC"
        print(f"Best {metric_name}: {best_valid_auc:.4f}")


if __name__ == "__main__":
    main()
