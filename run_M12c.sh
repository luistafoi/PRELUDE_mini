#!/bin/bash
# M12c: Same as M12 but with LR warmup (5 epochs)
# Tests whether slower initial learning stabilizes
# the inductive AUC volatility seen in M12.

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M12c_warmup \
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
    --lr_warmup_epochs 5 \
    --epochs 30 \
    --patience 12 \
    --lr 0.001 \
    --dropout 0.2 \
    --mini_batch_s 256 \
    --train_fraction 0.5 \
    --gpu 0 \
    --num_workers 4
