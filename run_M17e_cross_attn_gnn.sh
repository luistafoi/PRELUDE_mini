#!/bin/bash
# M17e: Cross-Attention Scoring + GNN backbone (M14e settings)
# Combines Option D (drug attends over cell gene tokens) with 2-layer GNN.
# GNN enriches drug/cell embeddings with structural signal,
# then cross-attention scores using drug-specific gene-level cell representation.
# Key question: does GNN + cross-attention scoring avoid the ind AUC decline?

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M17e_cross_attn_gnn \
    --cell_feature_source multiomic \
    --gene_encoder_dim 8 \
    --use_cross_attention \
    --cross_attn_dim 32 \
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
    --patience 15 \
    --dropout 0.2 \
    --mini_batch_s 256 \
    --train_fraction 0.5 \
    --compute_mad \
    --use_minibatch_gnn \
    --neighbor_sample_size 5 \
    --align_lambda 1.0 \
    --align_types cell drug gene \
    --gpu 0 \
    --num_workers 4
