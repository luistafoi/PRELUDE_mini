"""Optuna hyperparameter tuning for M25-full (dual head + dynamic isolation).

Tunes: embed_d, n_layers, isolation_ratio, dropout, lr, align_lambda, neighbor_sample_size.
Fixed: VAE features, dual_head, include_cell_drug, full data, use_minibatch_gnn.
Objective: DepMap inductive AUC.

Usage:
    python scripts/tune_optuna.py --gpu 0 --n_trials 100
    python scripts/tune_optuna.py --gpu 0 --n_trials 100 --study_name my_study
"""

import sys
import os
import argparse
import random
import csv
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.exceptions import TrialPruned

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg
from utils.evaluation import evaluate_model
from scripts.train import LinkPredictionDataset, TripletDataset
from scripts.train import collect_seed_lids_from_lp_batch, collect_seed_lids_from_triplet_batch, merge_seed_lids


def objective(trial, args):
    """Single Optuna trial: train model with sampled hyperparameters, return best ind AUC."""

    # --- Sample hyperparameters ---
    embed_d = trial.suggest_categorical('embed_d', [128, 256, 512])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    isolation_ratio = trial.suggest_float('isolation_ratio', 0.2, 0.7, step=0.1)
    dropout = trial.suggest_float('dropout', 0.1, 0.3, step=0.05)
    lr = trial.suggest_float('lr', 5e-5, 1e-3, log=True)
    align_lambda = trial.suggest_float('align_lambda', 0.0, 2.0, step=0.5)
    neighbor_sample_size = trial.suggest_categorical('neighbor_sample_size', [3, 5, 10])

    # Fixed params
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Seed for reproducibility within trial
    seed = 42 + trial.number
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Load data (shared across trials via args) ---
    dataset = args._dataset
    feature_loader = args._feature_loader
    generator = args._generator

    # --- Build model ---
    model_args = argparse.Namespace(
        data_dir=args.data_dir,
        embed_d=embed_d,
        n_layers=n_layers,
        max_neighbors=10,
        dropout=dropout,
        use_skip_connection=True,
        use_node_gate=True,
        cell_feature_source='vae',
        gene_encoder_dim=0,
        use_cross_attention=False,
        cross_attn_dim=32,
        freeze_cell_gnn=False,
        regression=False,
        use_residual_gnn=False,
        residual_scale=0.0,
        ema_lambda=0.0,
        ema_momentum=0.999,
        include_cell_drug=True,
        dual_head=True,
        inductive_loss_weight=1.0,
        backbone_lr_scale=1.0,
    )

    try:
        model = HetAgg(model_args, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    except Exception as e:
        print(f"  Trial {trial.number}: Model init failed: {e}")
        raise TrialPruned()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Dataloaders ---
    all_train_lp = dataset.links.get('train_lp', [])
    train_dataset = LinkPredictionDataset(all_train_lp, dataset, sample_ratio=0.5)
    train_loader = DataLoader(train_dataset, batch_size=args.mini_batch_s, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)

    valid_ind_links = dataset.links.get('valid_inductive', [])
    valid_dataset_ind = LinkPredictionDataset(valid_ind_links, dataset, sample_ratio=1.0)
    valid_loader_ind = DataLoader(valid_dataset_ind, batch_size=args.mini_batch_s, shuffle=False,
                                  num_workers=args.num_workers)

    # Triplet loader
    train_loader_triplet = None
    triplet_map_path = os.path.join(args.data_dir, "cell_drug_triplet_map.pkl")
    if os.path.exists(triplet_map_path):
        import pickle
        with open(triplet_map_path, 'rb') as f:
            triplet_data = pickle.load(f)
        triplet_map = triplet_data.get('triplet_map', {})
        if triplet_map:
            cell_type_id = dataset.node_name2type['cell']
            name2gid = {}
            for gid, (ntype, lid) in dataset.nodes['type_map'].items():
                if ntype == cell_type_id:
                    name2gid[dataset.id2node.get(gid, '')] = gid
            train_dataset_triplet = TripletDataset(triplet_map, num_pos=3, num_neg=3, name2gid=name2gid)
            train_loader_triplet = DataLoader(train_dataset_triplet, batch_size=51, num_workers=0)

    # --- Identify training cells for isolation ---
    cell_type_id = dataset.node_name2type['cell']
    training_cell_lids = set()
    for src, tgt, _ in dataset.links.get('train_lp', []):
        src_type = dataset.nodes['type_map'].get(src, (None, None))[0]
        tgt_type = dataset.nodes['type_map'].get(tgt, (None, None))[0]
        if src_type == cell_type_id:
            training_cell_lids.add(dataset.nodes['type_map'][src][1])
        elif tgt_type == cell_type_id:
            training_cell_lids.add(dataset.nodes['type_map'][tgt][1])
    training_cell_lids = sorted(training_cell_lids)

    # --- Training loop ---
    best_ind_auc = 0.0
    patience_counter = 0
    patience = args.patience

    for epoch in range(args.epochs):
        model.train()
        triplet_iter = iter(train_loader_triplet) if train_loader_triplet else None

        for batch in train_loader:
            u_lids, v_lids, labels, u_types, u_gids, v_gids = [x.to(device) for x in batch]

            drug_type_id = dataset.node_name2type['drug']
            u_type_id = u_types[0].item()
            if u_type_id == drug_type_id:
                drug_lids, cell_lids = u_lids, v_lids
            else:
                drug_lids, cell_lids = v_lids, u_lids

            # Triplet batch
            triplet_batch_data = None
            if triplet_iter is not None:
                try:
                    triplet_batch_data = next(triplet_iter)
                except StopIteration:
                    triplet_iter = iter(train_loader_triplet)
                    triplet_batch_data = next(triplet_iter)

            # Dynamic isolation
            n_isolate = max(1, int(len(training_cell_lids) * isolation_ratio))
            isolated = random.sample(training_cell_lids, n_isolate)
            isolated_cell_set = set(isolated)
            isolation_masks = model.create_isolation_masks(isolated)

            # Mini-batch GNN
            optimizer.zero_grad()
            lp_seeds = collect_seed_lids_from_lp_batch(drug_lids, cell_lids, dataset)
            if triplet_batch_data is not None:
                anc, pos, neg = [x.to(device) for x in triplet_batch_data]
                triplet_seeds = collect_seed_lids_from_triplet_batch(anc, pos, neg, dataset)
                all_seeds = merge_seed_lids(lp_seeds, triplet_seeds)
            else:
                all_seeds = lp_seeds
            all_seeds = {nt: lids.to(device) for nt, lids in all_seeds.items()}

            subgraph_info = model.expand_to_k_hop(
                all_seeds, k=n_layers, neighbor_sample_size=neighbor_sample_size,
                masks_override=isolation_masks
            )
            embedding_tables = model.compute_batch_embeddings(subgraph_info)

            # LP loss
            loss_lp = model.link_prediction_loss(
                drug_lids, cell_lids, labels, generator,
                embedding_tables=embedding_tables,
                subgraph_info=subgraph_info,
                isolated_cell_set=isolated_cell_set,
            )

            # Triplet loss
            loss_triplet = torch.tensor(0.0, device=device)
            if triplet_batch_data is not None:
                loss_triplet = model.self_supervised_triplet_loss(
                    anc, pos, neg, generator,
                    embedding_tables=embedding_tables,
                    subgraph_info=subgraph_info,
                )

            # Alignment loss
            loss_align = torch.tensor(0.0, device=device)
            if align_lambda > 0:
                loss_align = model.alignment_loss(embedding_tables, subgraph_info)

            # Combine
            total_loss = loss_lp + loss_triplet + align_lambda * loss_align

            # L1
            l1 = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
            total_loss = total_loss + 1e-5 * l1

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # --- Validation ---
        with torch.no_grad():
            eval_tables = model.compute_all_embeddings()
        ind_metrics, _ = evaluate_model(
            model, valid_loader_ind, generator, device, dataset,
            embedding_tables=eval_tables, use_inductive_head=True
        )
        val_ind_auc = ind_metrics['ROC-AUC']

        # Report to Optuna for pruning
        trial.report(val_ind_auc, epoch)
        if trial.should_prune():
            raise TrialPruned()

        if val_ind_auc > best_ind_auc:
            best_ind_auc = val_ind_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        print(f"  Trial {trial.number} Ep {epoch+1}: Ind AUC={val_ind_auc:.4f} (best={best_ind_auc:.4f})")

    return best_ind_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--study_name', type=str, default='M25_tuning')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Max epochs per trial (shorter than full training for speed)')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--mini_batch_s', type=int, default=10240)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Pre-load data once (shared across all trials)
    print("Loading dataset (shared across all trials)...")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args._dataset = PRELUDEDataset(args.data_dir)
    args._feature_loader = FeatureLoader(args._dataset, device)
    args._generator = DataGenerator(args.data_dir, include_cell_drug=True)
    print("Data loaded.\n")

    # Create Optuna study
    db_path = f"sqlite:///optuna_{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=db_path,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )

    print(f"Starting Optuna study: {args.study_name}")
    print(f"  Trials: {args.n_trials}, Max epochs/trial: {args.epochs}")
    print(f"  DB: {db_path}")
    print(f"  Pruner: MedianPruner (startup=10, warmup=3)\n")

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    # --- Results ---
    print("\n" + "=" * 60)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best Ind AUC: {study.best_trial.value:.4f}")
    print(f"Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save top 10 trials
    results_path = f"results/optuna_{args.study_name}_results.csv"
    os.makedirs("results", exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False)
    trials_df.to_csv(results_path, index=False)
    print(f"\nAll trials saved to: {results_path}")

    # Print top 10
    print("\nTop 10 trials:")
    top10 = trials_df.head(10)
    for _, row in top10.iterrows():
        params = {k.replace('params_', ''): row[k] for k in row.index if k.startswith('params_')}
        print(f"  Trial {int(row['number']):3d}: AUC={row['value']:.4f} | {params}")


if __name__ == '__main__':
    main()
