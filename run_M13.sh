#!/bin/bash
# M13: Elastic Net + Lower LR
# Based on M12b (best variant) + M12check findings:
#   - 2 layers (more = worse for inductive)
#   - isolation 0.4 (sweet spot)
#   - LR 3e-4 (M12check proved 1e-3 overshoots after epoch 3)
#   - Elastic net: L1=1e-5 (sparsity) + L2=1e-4 (weight decay)
#   - Still 0.5 fraction — validate curve stability before full data

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M13_elasticnet \
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
    --gpu 0 \
    --num_workers 4
