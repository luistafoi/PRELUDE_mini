#!/bin/bash
# M17b: Multi-omic + per-gene MLP encoder (Option B)
# Shared MLP(4->8) processes each gene's 4 channels together,
# learning cross-channel interactions (expression × dependency).
# Then flatten + project to 256-dim.
# Compare to M17 (flat multiomic, best ind=0.831) and M15_projonly (VAE, best ind=0.816).

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M17b_gene_encoder \
    --cell_feature_source multiomic \
    --gene_encoder_dim 8 \
    --embed_d 256 \
    --n_layers 0 \
    --max_neighbors 10 \
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
