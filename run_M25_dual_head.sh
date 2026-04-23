#!/bin/bash
# M25: Dual scoring heads + dynamic Cell-Drug isolation
# Two bilinear heads: inductive (trains on isolated cells only) and
# transductive (trains on non-isolated cells with Cell-Drug in GNN).
# The inductive head is structurally guaranteed to never see Cell-Drug
# signal, so it can't be corrupted by transductive calibration drift.
# Shared GNN backbone trains from both heads' gradients.
#
# REQUIRES: regenerated neighbor pickle with --include_cell_drug

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M25_dual_head_full \
    --cell_feature_source vae \
    --include_cell_drug \
    --dual_head \
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
    --epochs 50 \
    --patience 20 \
    --dropout 0.2 \
    --mini_batch_s 10240 \
    --train_fraction 1.0 \
    --compute_mad \
    --use_minibatch_gnn \
    --neighbor_sample_size 5 \
    --align_lambda 1.0 \
    --align_types cell drug gene \
    --gpu 0 \
    --num_workers 4
