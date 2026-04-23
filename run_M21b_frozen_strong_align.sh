#!/bin/bash
# M21b: Frozen cells + strong alignment on drugs/genes
# Combines M20 (freeze_cell_gnn) with strong alignment (lambda=10).
# Cells: always projection-only (no divergence by construction).
# Drugs/genes: GNN-refined but tightly constrained near projection space.
# No isolation_ratio needed (cells always frozen).

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M21b_frozen_strong_align \
    --cell_feature_source vae \
    --freeze_cell_gnn \
    --embed_d 256 \
    --n_layers 2 \
    --max_neighbors 10 \
    --use_node_gate \
    --use_skip_connection \
    --use_triplet_loss \
    --use_soft_margin_triplet \
    --use_dynamic_loss \
    --loss_target_lp 0.85 \
    --loss_target_triplet 0.15 \
    --lr 0.0003 \
    --l1_lambda 1e-5 \
    --weight_decay 1e-4 \
    --epochs 30 \
    --patience 12 \
    --dropout 0.2 \
    --mini_batch_s 256 \
    --train_fraction 0.5 \
    --compute_mad \
    --use_minibatch_gnn \
    --neighbor_sample_size 5 \
    --align_lambda 10.0 \
    --align_types drug gene \
    --gpu 0 \
    --num_workers 4
