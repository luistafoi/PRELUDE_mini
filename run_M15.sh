#!/bin/bash
# M15: Residual GNN Architecture
# final = proj + scale * (gnn - proj)
# Projection IS the base representation; GNN learns additive correction.
# When GNN unavailable (inductive), delta=0 -> returns proj exactly.
# No alignment/dual-path needed — spaces aligned by construction.
# Learned per-type scale via sigmoid (init 0.5).

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M15_residual \
    --embed_d 256 \
    --n_layers 2 \
    --max_neighbors 10 \
    --use_residual_gnn \
    --use_skip_connection \
    --use_triplet_loss \
    --use_soft_margin_triplet \
    --use_dynamic_loss \
    --loss_target_lp 0.85 \
    --loss_target_triplet 0.15 \
    --isolation_ratio 0.2 \
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
    --gpu 0 \
    --num_workers 4
