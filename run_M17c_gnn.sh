#!/bin/bash
# M17c: Multi-omic + gene encoder + GNN (M14e settings)
# Same features as M17b but with 2-layer GNN, node gate, alignment loss.
# The key test: do richer features sustain the GNN boost past epoch 2-3?

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M17c_gnn \
    --cell_feature_source multiomic \
    --gene_encoder_dim 8 \
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
    --epochs 30 \
    --patience 12 \
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
