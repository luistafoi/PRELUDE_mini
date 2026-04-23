#!/bin/bash
# M14b: Mini-Batch GNN Propagation with 0.6 Isolation Ratio
# Based on M13 (elastic net) + mini-batch subgraph computation:
#   - Only compute k-hop subgraph per batch (not full 20K nodes)
#   - Stochastic neighbor sampling (5 per type per hop) for regularization
#   - Expect: faster per-batch, less overshoot, more stable ind AUC curve

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M14b_ir06_minibatch \
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
    --isolation_ratio 0.6 \
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
