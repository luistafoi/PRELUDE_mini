#!/bin/bash
# M17d: Cross-Attention Scoring (Option D)
# Drug embedding attends over cell's per-gene tokens to produce
# drug-specific cell representation. Each drug "asks" about its target genes.
# No GNN (n_layers=0) — isolates the cross-attention contribution.
# gene_encoder_dim=8 for Option B cell projection (still needed for GNN path),
# cross_attn_dim=32 for the attention gene tokens (separate encoder).

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M17d_cross_attn \
    --cell_feature_source multiomic \
    --gene_encoder_dim 8 \
    --use_cross_attention \
    --cross_attn_dim 32 \
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
