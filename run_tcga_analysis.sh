#!/bin/bash
# Comprehensive TCGA analysis with multiple normalization strategies
# Run in tmux: tmux new -s tcga_analysis && bash run_tcga_analysis.sh

set -e

MODEL_NAME="${1:-v2_M03_drug_sim}"

echo "=========================================="
echo "TCGA COMPREHENSIVE ANALYSIS: $MODEL_NAME"
echo "Started: $(date)"
echo "=========================================="

conda run -n hgb_env --no-capture-output python scripts/analyze_tcga.py \
    --model_name $MODEL_NAME \
    --gpu 0

echo ""
echo "Finished: $(date)"
echo "Results in: results/$MODEL_NAME/"
