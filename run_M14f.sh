#!/bin/bash
# M14f: Dual-Path LP Loss + Alignment
# Both sides of bilinear head trained on projection-only embeddings (30% weight).
# Combined with alignment loss (lambda=0.2) on all types.
#   - proj_loss_weight=0.3: LP = 0.7*gnn_lp + 0.3*proj_lp
#   - align_lambda=0.2: gentle space alignment (complementary)
#   - isolation_ratio=0.4: still useful for cell-side diversity
# Expect: head learns decision boundary that works for projections,
#   ind AUC sustains instead of declining after epoch 3.

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M14f_dualpath \
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
    --proj_loss_weight 0.3 \
    --gpu 0 \
    --num_workers 4
