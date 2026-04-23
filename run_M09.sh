#!/bin/bash
# run_M09.sh — M09: Per-type gates + class-balanced BCE + denser Cell-Cell
# Architecture: gated skip (per node type), pos_weight BCE, cosine LR
# Data: Cell-Cell top 5%, 20 neighbors/type, 4 GNN layers
set -e

NAME="M09_PerTypeGate_BalancedBCE"
DATA_DIR="data/processed"
MAX_NEIGHBORS=20

# --- PARAMETERS ---
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
      --triplet_loss_weight 0.24 \
      --triplet_margin 0.90 \
      --triplet_num_pos 2 \
      --use_rw_loss \
      --rw_loss_weight 0.03 \
      --walk_length 5 \
      --isolation_ratio 0.2 \
      --epochs 70 \
      --patience 20 \
      --mini_batch_s 2048 \
      --num_workers 8 \
      --train_fraction 1.0"

echo "=========================================="
echo "RUNNING MODEL: $NAME"
echo "=========================================="
echo "New in M09:"
echo "  1. Per-type gated skip (separate gate for cell/drug/gene)"
echo "  2. Class-balanced BCE (pos_weight ~4.9, auto-computed)"
echo "  3. Cell-Cell top 5% (~75 neighbors, was 3%/45)"
echo "  4. Isolation 0.2 (gate handles projected/GNN balance)"
echo "  5. Cosine LR annealing + patience 20"
echo ""
echo "Previous results:"
echo "  M08c:             Ind AUC=0.6169  Trans AUC=0.9866"
echo "  M08d (no gate):   Ind AUC=0.6688  Trans AUC=0.9743"
echo "  M08d (gate 0.4):  Ind AUC=0.7157  Trans AUC=0.9239"
echo "  M08d (gate 0.2):  Ind AUC=0.7431  Trans AUC=0.9239  (val peak, ep1)"
echo "=========================================="

# 0. Precompute similarity (top 5%)
echo ""
echo "--- Step 0: Generating Similarity Map (top 5%, K=2) ---"
python scripts/precompute_cell_similarity.py --top-pct 0.05 --num-pos 2

# 1. Rebuild graph
echo ""
echo "--- Step 1: Rebuilding Graph ---"
python scripts/build_graph_files.py --raw-dir data/raw --output-dir $DATA_DIR

# 2. Create splits
echo ""
echo "--- Step 2: Creating Splits ---"
python scripts/create_splits.py

# 3. Regenerate neighbors
echo ""
echo "--- Step 3: Regenerating Neighbors (max_neighbors=$MAX_NEIGHBORS) ---"
python scripts/generate_neighbors.py --data_dir $DATA_DIR --max_neighbors $MAX_NEIGHBORS

# 4. Integrity test
echo ""
echo "--- Step 4: Graph Integrity Check ---"
python scripts/test_graph_integrity.py --data_dir $DATA_DIR

# 5. Train
echo ""
echo "--- Step 5: Training ---"
python scripts/train.py $ARGS

# 6. Inductive evaluation
echo ""
echo "--- Step 6: Inductive Evaluation ---"
python scripts/evaluate.py $ARGS --load_path "checkpoints/${NAME}.pth" --inductive

# 7. Transductive evaluation
echo ""
echo "--- Step 7: Transductive Evaluation ---"
python scripts/evaluate.py $ARGS --load_path "checkpoints/${NAME}.pth"

echo ""
echo "=== RUN COMPLETE: $NAME ==="
