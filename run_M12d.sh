#!/bin/bash
# M12d: Same as M12 but with n_layers=1 (1 true hop)
# Control experiment: is 1 real hop enough, or does
# the 2nd hop cause over-smoothing that hurts inductive?

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M12d_1layer \
    --embed_d 256 \
    --n_layers 1 \
    --max_neighbors 10 \
    --use_node_gate \
    --use_skip_connection \
    --use_triplet_loss \
    --use_soft_margin_triplet \
    --use_dynamic_loss \
    --loss_target_lp 0.85 \
    --loss_target_triplet 0.15 \
    --epochs 30 \
    --patience 12 \
    --lr 0.001 \
    --dropout 0.2 \
    --mini_batch_s 256 \
    --train_fraction 0.5 \
    --gpu 0 \
    --num_workers 4
