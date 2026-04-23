#!/bin/bash
# M14d: Stronger cell+drug alignment loss
# M14c showed lambda=0.2 over all types was too weak (gene-dominated, 0.054 effective).
# Fix: lambda=1.0, cell+drug alignment (the two types the bilinear head consumes).
#   - 17K genes no longer dilute the signal
#   - Both sides of the bilinear product stay in compatible space
#   - Alignment gradient ~1.0 * 0.27 = 0.27 vs LP ~0.85 (meaningful ratio)

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M14d_align10_cd \
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
    --isolation_ratio 0.4 \
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
    --align_lambda 1.0 \
    --align_types cell drug \
    --gpu 0 \
    --num_workers 4
