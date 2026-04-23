#!/bin/bash
# M15 MLP Head Baseline (projection-only + MLP instead of bilinear)
# MLP(cat(drug_proj, cell_proj)) -> score
# No GNN (n_layers=0), no bilinear interaction.
# Tests whether bilinear structure matters or MLP can learn the same.

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M15_mlp \
    --embed_d 256 \
    --n_layers 0 \
    --max_neighbors 10 \
    --use_mlp_head \
    --use_triplet_loss \
    --use_soft_margin_triplet \
    --use_dynamic_loss \
    --loss_target_lp 0.85 \
    --loss_target_triplet 0.15 \
    --lr 0.0003 \
    --l1_lambda 1e-5 \
    --weight_decay 1e-4 \
    --epochs 30 \
    --patience 12 \
    --dropout 0.2 \
    --mini_batch_s 256 \
    --train_fraction 0.5 \
    --gpu 0 \
    --num_workers 4
