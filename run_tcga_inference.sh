#!/bin/bash
# Run TCGA inference with all 3 modes
# Usage: bash run_tcga_inference.sh [model_name]
# Run in tmux: tmux new -s tcga && bash run_tcga_inference.sh

set -e

MODEL_NAME="${1:-v2_M03_drug_sim}"
DATA_DIR="data/processed_v2"

echo "=========================================="
echo "TCGA INFERENCE: $MODEL_NAME"
echo "=========================================="
echo "Started at: $(date)"
echo ""

conda run -n hgb_env --no-capture-output python scripts/inference_tcga.py \
    --model_name $MODEL_NAME \
    --data_dir $DATA_DIR \
    --gpu 0

echo ""
echo "Finished at: $(date)"
echo "Results in: results/$MODEL_NAME/"
