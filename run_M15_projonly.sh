#!/bin/bash
# M15 Projection-Only Baseline
# Pure projection -> bilinear head, NO GNN layers (n_layers=0).
# Establishes the ceiling for what features alone can achieve.
# If ind AUC ~= 0.70-0.72, features are the bottleneck, not architecture.
# If ind AUC >> 0.72, GNN is actively hurting inductive performance.

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M15_projonly \
    --embed_d 256 \
    --n_layers 0 \
    --max_neighbors 10 \
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
    --gpu 0 \
    --num_workers 4
