# scripts/tune.py — Optuna hyperparameter tuning for PRELUDE HetGNN

import sys
import os
import json
import torch
import torch.optim as optim
import numpy as np
import random
import argparse
import pickle
import optuna
from torch.utils.data import DataLoader
from collections import defaultdict
from itertools import zip_longest
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
from utils.evaluation import evaluate_model
from scripts.train import LinkPredictionDataset, TripletDataset, RandomWalkDataset

# Globals — loaded once, reused across trials
dataset = None
feature_loader = None
generator = None
triplet_map = None
device = None


def objective(trial):
    global dataset, feature_loader, generator, triplet_map, device

    args = read_args()

    # --- Tunable hyperparameters (14) ---
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    args.l1_lambda = trial.suggest_float("l1_lambda", 1e-7, 1e-3, log=True)
    args.embed_d = trial.suggest_categorical("embed_d", [128, 256])
    args.n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
    args.triplet_loss_weight = trial.suggest_float("triplet_loss_weight", 0.05, 1.0, log=True)
    args.triplet_margin = trial.suggest_float("triplet_margin", 0.3, 1.5)
    args.triplet_num_pos = trial.suggest_int("triplet_num_pos", 1, 5)
    args.rw_loss_weight = trial.suggest_float("rw_loss_weight", 0.01, 0.5, log=True)
    args.walk_length = trial.suggest_categorical("walk_length", [3, 5, 7])
    args.isolation_ratio = trial.suggest_float("isolation_ratio", 0.05, 0.4)
    args.mini_batch_s = trial.suggest_categorical("mini_batch_s", [1024, 2048, 4096])
    args.rw_neg_samples = trial.suggest_categorical("rw_neg_samples", [3, 5])

    # --- Fixed parameters ---
    args.use_skip_connection = True
    args.use_triplet_loss = True
    args.use_rw_loss = True
    args.max_neighbors = 20
    args.validate_inductive = True
    args.train_fraction = 0.5
    args.triplet_num_neg = 5
    args.epochs = 15
    args.patience = 7

    # --- Seed for reproducibility within trial ---
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # --- Initialize Model ---
    model_args_ns = argparse.Namespace(**vars(args))
    model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
    model.setup_link_prediction("drug", "cell")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- LP Loader ---
    all_train_lp = dataset.links.get('train_lp', [])
    train_dataset_lp = LinkPredictionDataset(all_train_lp, dataset, sample_ratio=args.train_fraction)
    train_loader_lp = DataLoader(train_dataset_lp, batch_size=args.mini_batch_s, shuffle=True, num_workers=0)

    # --- Triplet Loader ---
    train_loader_triplet = None
    if triplet_map is not None:
        name2gid = {name.upper(): gid for name, gid in dataset.node2id.items()}
        train_dataset_triplet = TripletDataset(triplet_map, args.triplet_num_pos, args.triplet_num_neg, name2gid)
        triplet_batch_size = max(1, args.mini_batch_s // (args.triplet_num_pos * args.triplet_num_neg))
        train_loader_triplet = DataLoader(train_dataset_triplet, batch_size=triplet_batch_size, num_workers=0)

    # --- RW Loader ---
    inductive_cells = set()
    for s, t, l in dataset.links.get('test_inductive', []):
        inductive_cells.add(s)
        inductive_cells.add(t)
    for s, t, l in dataset.links.get('valid_inductive', []):
        inductive_cells.add(s)
        inductive_cells.add(t)
    all_nodes = [n for n in generator.node_type.keys() if n not in inductive_cells]
    generator._rw_negative_pool = all_nodes
    rw_dataset = RandomWalkDataset(all_nodes, generator, args.walk_length, args.rw_neg_samples)
    train_loader_rw = DataLoader(rw_dataset, batch_size=args.mini_batch_s, num_workers=0)

    # --- Validation Loader ---
    valid_links = dataset.links.get('valid_inductive', [])
    if not valid_links:
        valid_links = dataset.links.get('valid_transductive', [])
    valid_dataset = LinkPredictionDataset(valid_links, dataset, sample_ratio=1.0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.mini_batch_s, num_workers=0)

    # --- Training Loop ---
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        num_batches = 0

        iterators = [train_loader_lp, train_loader_triplet, train_loader_rw]
        train_iterator = zip_longest(*iterators, fillvalue=None)

        for batch_data in train_iterator:
            optimizer.zero_grad()
            total_loss_batch = 0.0

            lp_batch = batch_data[0]
            triplet_batch = batch_data[1]
            rw_batch = batch_data[2]

            # 1. Link Prediction Loss
            if lp_batch is not None:
                u_lids_b, v_lids_b, labels_b, u_types_b, _, _ = lp_batch
                u_lids = u_lids_b.to(device)
                v_lids = v_lids_b.to(device)
                labels = labels_b.to(device)
                u_types = u_types_b.to(device)

                if u_types[0].item() == dataset.node_name2type.get('drug', -1):
                    drug_lids, cell_lids = u_lids, v_lids
                else:
                    drug_lids, cell_lids = v_lids, u_lids

                loss_lp = model.link_prediction_loss(drug_lids, cell_lids, labels, generator,
                                                     isolation_ratio=args.isolation_ratio)
                total_loss_batch += loss_lp

            # 2. Triplet Loss
            if triplet_batch is not None:
                anc, pos, neg = [x.to(device) for x in triplet_batch]
                loss_triplet = model.self_supervised_triplet_loss(anc, pos, neg, generator)
                total_loss_batch += (args.triplet_loss_weight * loss_triplet)

            # 3. Random Walk Loss
            if rw_batch is not None:
                center_gids, ctx_gids, neg_gids = [x.to(device) for x in rw_batch]
                center_emb = model.get_embeddings_from_gids(center_gids, generator)
                ctx_emb = model.get_embeddings_from_gids(ctx_gids, generator)

                pos_loss = -torch.log(
                    torch.sigmoid(torch.sum(center_emb * ctx_emb, dim=1)) + 1e-15
                ).mean()
                flat_negs = neg_gids.view(-1)
                neg_emb = model.get_embeddings_from_gids(flat_negs, generator).view(
                    neg_gids.shape[0], neg_gids.shape[1], -1
                )
                neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
                neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-15).sum(dim=1).mean()

                loss_rw = pos_loss + neg_loss
                total_loss_batch += (args.rw_loss_weight * loss_rw)

            # 4. L1 Regularization
            if args.l1_lambda > 0:
                l1 = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                total_loss_batch += (args.l1_lambda * l1)

            if total_loss_batch > 0:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                num_batches += 1

        scheduler.step()

        # --- Validation ---
        val_metrics, _ = evaluate_model(model, valid_loader, generator, device, dataset)
        auc = val_metrics['ROC-AUC']

        # Report to Optuna for pruning
        trial.report(auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping within trial
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    return best_auc


if __name__ == "__main__":
    args = read_args()

    # --- Global Data Loading (once) ---
    print("=" * 50)
    print("PRELUDE HetGNN — Optuna Hyperparameter Tuning (M10)")
    print("=" * 50)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading dataset...")
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir)
    generator.load_train_neighbors(os.path.join(args.data_dir, "train_neighbors_preprocessed.pkl"))

    # Load triplet map
    triplet_map_file = os.path.join(args.data_dir, "cell_triplet_map.pkl")
    if os.path.exists(triplet_map_file):
        with open(triplet_map_file, "rb") as f:
            triplet_map = pickle.load(f)
        print(f"Loaded triplet map with {len(triplet_map)} anchors.")
    else:
        print("WARNING: Triplet map not found. Triplet loss will be disabled.")
        triplet_map = None

    # --- Create Optuna Study ---
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///prelude_tuning.db",
        study_name="prelude_M10_opt",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    n_trials = 100
    print(f"\nStarting optimization: {n_trials} trials")
    print(f"  Objective: maximize inductive validation AUC")
    print(f"  Pruner: MedianPruner (startup=5, warmup=3)")
    print(f"  Training: 15 epochs, 50% data, patience=7")
    print()

    study.optimize(objective, n_trials=n_trials)

    # --- Results ---
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trials: {len(study.trials)} total, {complete} complete, {pruned} pruned")
    print(f"\nBest Inductive AUC: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best params to JSON
    output = {
        "best_auc": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    json_path = "best_params_M10.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBest params saved to: {json_path}")
