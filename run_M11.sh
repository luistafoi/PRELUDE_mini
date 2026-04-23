#!/bin/bash
# run_M11.sh — M11: Dynamic Loss Manager + Soft-Margin Triplet + RW Normalization
# DIAGNOSTIC RUN — short training to verify loss fixes work before full convergence
#
# What we're checking:
#   1. Triplet loss does NOT flatline to 0 (soft-margin keeps gradient alive)
#   2. RW loss DECREASES over epochs (L2 normalization unfreezes it)
#   3. DynamicLossManager weights adapt and stabilize after warmup
#   4. AUC doesn't regress from M09 baseline (0.817 inductive)
set -e

NAME="M11_DynamicLoss_SoftTriplet_RWNorm"
DATA_DIR="data/processed"
MAX_NEIGHBORS=20

# --- PARAMETERS (M09 foundation + M11 features) ---
ARGS="--data_dir $DATA_DIR \
      --model_name $NAME \
      --validate_inductive \
      --use_skip_connection \
      --n_layers 4 \
      --max_neighbors $MAX_NEIGHBORS \
      --embed_d 256 \
      --lr 0.0000676 \
      --dropout 0.37 \
      --weight_decay 0.0000356 \
      --l1_lambda 0.0000031 \
      --use_triplet_loss \
      --triplet_num_pos 2 \
      --use_rw_loss \
      --walk_length 5 \
      --isolation_ratio 0.2 \
      --epochs 30 \
      --patience 10 \
      --mini_batch_s 2048 \
      --num_workers 8 \
      --train_fraction 0.5 \
      --use_dynamic_loss \
      --use_soft_margin_triplet \
      --use_rw_normalize \
      --loss_target_lp 0.6 \
      --loss_target_triplet 0.3 \
      --loss_target_rw 0.1"

echo "=========================================="
echo "DIAGNOSTIC RUN: $NAME"
echo "=========================================="
echo "Checking:"
echo "  1. Triplet stays alive (soft-margin)"
echo "  2. RW decreases (L2 norm)"
echo "  3. Weights adapt (DynamicLossManager)"
echo "  4. AUC vs M09 baseline (0.817 ind)"
echo ""
echo "30 epochs, patience 10 — graph reuse from M09"
echo "=========================================="

# Skip steps 0-4: graph/splits/neighbors unchanged since M09

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
echo "Check logs for: triplet != 0, RW decreasing, weight adaptation"
