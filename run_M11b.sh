#!/bin/bash
# run_M11b.sh — M11b: Stabilized dynamic loss + anti-overfitting
# DIAGNOSTIC RUN — fixes from M11a:
#   1. Weight cap (10x) — prevents triplet runaway (was 67x)
#   2. Rebalanced ratios — LP=0.85, trip=0.10, rw=0.05 (aux as regularizers)
#   3. Halved LR (3.38e-5) — M11a peaked at ep1 then overfit
#   4. Stronger L1/L2 (~2x) — fight overfitting
#
# M11a results: Ind=0.717, Trans=0.863, val peaked 0.844 at ep1
# M09 re-run:   Ind=0.622, Trans=0.948
set -e

NAME="M11b_StableDynLoss"
DATA_DIR="data/processed"
MAX_NEIGHBORS=20

ARGS="--data_dir $DATA_DIR \
      --model_name $NAME \
      --validate_inductive \
      --use_skip_connection \
      --n_layers 4 \
      --max_neighbors $MAX_NEIGHBORS \
      --embed_d 256 \
      --lr 0.0000338 \
      --dropout 0.37 \
      --weight_decay 0.0000712 \
      --l1_lambda 0.0000062 \
      --use_triplet_loss \
      --triplet_num_pos 2 \
      --use_rw_loss \
      --walk_length 5 \
      --isolation_ratio 0.2 \
      --epochs 30 \
      --patience 12 \
      --mini_batch_s 2048 \
      --num_workers 8 \
      --train_fraction 0.5 \
      --use_dynamic_loss \
      --use_soft_margin_triplet \
      --use_rw_normalize \
      --loss_target_lp 0.85 \
      --loss_target_triplet 0.10 \
      --loss_target_rw 0.05 \
      --loss_max_weight 10.0"

echo "=========================================="
echo "DIAGNOSTIC RUN: $NAME"
echo "=========================================="
echo "Fixes from M11a:"
echo "  1. Weight cap 10x (was uncapped, hit 67x)"
echo "  2. Ratios: LP=0.85 Trip=0.10 RW=0.05 (was 0.6/0.3/0.1)"
echo "  3. LR halved: 3.38e-5 (was 6.76e-5)"
echo "  4. L1/L2 doubled (fight overfitting)"
echo ""
echo "Baseline: M11a Ind=0.717 | M09 Ind=0.622"
echo "=========================================="

# Train
echo ""
echo "--- Training ---"
python scripts/train.py $ARGS

# Inductive evaluation
echo ""
echo "--- Inductive Evaluation ---"
python scripts/evaluate.py $ARGS --load_path "checkpoints/${NAME}.pth" --inductive

# Transductive evaluation
echo ""
echo "--- Transductive Evaluation ---"
python scripts/evaluate.py $ARGS --load_path "checkpoints/${NAME}.pth"

echo ""
echo "=== DIAGNOSTIC COMPLETE: $NAME ==="
echo "Check: triplet weight stays <10x, val AUC doesn't peak at ep1"
