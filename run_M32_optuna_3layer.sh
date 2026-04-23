#!/bin/bash
# M32: Optuna best 3-layer config — based on M25 architecture
# Trial 46: AUC=0.873
# Key: n_layers=3, align_lambda=0.0, lr=0.0001, dropout=0.1, neighbor_sample=5
# Deeper model may learn more abstract/transferable representations for Sanger.
# Full data, dual head, include_cell_drug

set -e

MODEL_NAME="M32_optuna_3layer"

echo "=== Training $MODEL_NAME ==="
conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name $MODEL_NAME \
    --cell_feature_source vae \
    --include_cell_drug \
    --dual_head \
    --embed_d 256 \
    --n_layers 3 \
    --max_neighbors 10 \
    --use_node_gate \
    --use_skip_connection \
    --use_triplet_loss \
    --use_soft_margin_triplet \
    --use_dynamic_loss \
    --loss_target_lp 0.85 \
    --loss_target_triplet 0.15 \
    --isolation_ratio 0.3 \
    --lr 0.0001 \
    --l1_lambda 1e-5 \
    --weight_decay 1e-4 \
    --epochs 50 \
    --patience 20 \
    --dropout 0.1 \
    --mini_batch_s 10240 \
    --train_fraction 1.0 \
    --compute_mad \
    --use_minibatch_gnn \
    --neighbor_sample_size 5 \
    --align_lambda 0.0 \
    --gpu 0 \
    --num_workers 4

echo ""
echo "=== Sanger Evaluation ==="
for scenario in S1 S2 S3 S4; do
    echo "--- $scenario ---"
    conda run -n hgb_env --no-capture-output python scripts/inference_sanger.py \
        --data_dir data/sanger_processed \
        --checkpoint checkpoints/${MODEL_NAME}.pth \
        --scenario $scenario \
        --embed_d 256 --n_layers 3 --max_neighbors 10 \
        --use_node_gate --use_skip_connection --dropout 0.1 \
        --gpu 0 2>&1 | grep -E "Overall|AUC=|Pairs:|Predictions saved"
done

echo ""
echo "=== Overall Sanger AUC ==="
conda run -n hgb_env --no-capture-output python -c "
import pandas as pd
from sklearn.metrics import roc_auc_score
for s in ['S1', 'S2', 'S3', 'S4']:
    try:
        df = pd.read_csv(f'results/sanger_gnn/sanger_{s}_gnn_preds.csv')
        binary = (df['true_label'] > 0.5).astype(int)
        if binary.nunique() == 2:
            print(f'  {s}: AUC = {roc_auc_score(binary, df[\"pred_prob\"]):.4f} ({len(df)} pairs)')
        else:
            print(f'  {s}: Only one class ({len(df)} pairs)')
    except FileNotFoundError:
        print(f'  {s}: No predictions file')
"

echo "Done."
