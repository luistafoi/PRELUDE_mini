#!/bin/bash
# M12check: Diagnostic run based on M12b (best variant)
#
# Tests:
# 1. FREEZE TEST: Train normally for 3 epochs, then LR -> 1e-8.
#    If ind AUC holds steady -> optimization issue (LR too aggressive post ep3).
#    If ind AUC drops -> over-smoothing or structural issue.
#
# 2. FULL-BATCH VALIDATION: Already the default (compute_all_embeddings).
#    No sampling artifacts to worry about.
#
# 3. MAD CHECK: Logs Mean Average Distance per epoch per type.
#    If MAD < 0.1 by epoch 5 -> over-smoothing confirmed.

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M12check_freeze3_mad \
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
    --freeze_after_epoch 3 \
    --compute_mad \
    --epochs 20 \
    --patience 20 \
    --lr 0.001 \
    --dropout 0.2 \
    --mini_batch_s 256 \
    --train_fraction 0.5 \
    --gpu 0 \
    --num_workers 4
