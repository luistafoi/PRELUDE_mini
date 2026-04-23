#!/bin/bash
# M30: VAE + dual head + full data + diff LR (0.1x backbone)
# Combines M25-full (VAE, full data, best DepMap) with M26 (diff LR, best Sanger).
# Same as M29 but VAE instead of scGPT.

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M30_vae_full_diffLR \
    --cell_feature_source vae \
    --include_cell_drug \
    --dual_head \
    --backbone_lr_scale 0.1 \
    --inductive_loss_weight 3.0 \
    --embed_d 256 \
    --n_layers 2 \
    --max_neighbors 10 \
    --use_node_gate \
    --use_skip_connection \
    --use_triplet_loss \
    --use_soft_margin_triplet \
    --use_dynamic_loss \
    --loss_target_lp 0.85 \
    --loss_target_triplet 0.15 \
    --isolation_ratio 0.4 \
    --lr 0.0003 \
    --l1_lambda 1e-5 \
    --weight_decay 1e-4 \
    --epochs 50 \
    --patience 20 \
    --dropout 0.2 \
    --mini_batch_s 10240 \
    --train_fraction 1.0 \
    --compute_mad \
    --use_minibatch_gnn \
    --neighbor_sample_size 5 \
    --align_lambda 1.0 \
    --align_types cell drug gene \
    --gpu 0 \
    --num_workers 4
