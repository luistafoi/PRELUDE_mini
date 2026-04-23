#!/bin/bash
# M17: Multi-omic cell features (Option B) — projection-only baseline
# Replaces 512-dim VAE with 3858-dim drug-target multi-omic features:
#   964 genes × 4 channels (expression, CRISPR dependency, copy number, mutation)
#   + 2 binary flags (has_crispr, has_cn)
# No GNN (n_layers=0) to isolate feature quality improvement.
# Compare directly to M15_projonly (VAE features, same architecture).

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M17_multiomic_projonly \
    --cell_feature_source multiomic \
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
