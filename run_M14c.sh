#!/bin/bash
# M14c: Mini-Batch GNN + Projection-GNN Alignment Loss
# Based on M14 + cosine alignment loss (lambda=0.2):
#   - Keeps GNN output in same angular space as projections
#   - Prevents gate from bypassing GNN entirely (alpha ~0.95 in M14)
#   - Bilinear head stays calibrated for both GNN and projection-only inputs
#   - Expect: ind AUC plateau instead of decline, gate alphas closer to 0.5

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M14c_align02 \
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
    --align_lambda 0.2 \
    --gpu 0 \
    --num_workers 4
