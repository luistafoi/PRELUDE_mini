#!/bin/bash
# M24b: Dynamic isolation with Cell-Drug edges in GNN with ratio 0.8
# Key change: Cell-Drug edges are now IN the GNN neighbor graph.
# Each batch, isolation_ratio fraction of training cells have their
# Cell-Drug edges dynamically masked (both directions).
# This simulates the inductive condition at the GNN level while
# keeping Cell-Cell and Cell-Gene messages intact.
#
# At eval: training cells naturally have Cell-Drug in GNN,
# inductive cells naturally don't (edges not in train.dat).
#
# REQUIRES: regenerated neighbor pickle with --include_cell_drug

set -e

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir data/processed \
    --save_dir checkpoints \
    --model_name M24b_dynamic_highisolation \
    --cell_feature_source vae \
    --include_cell_drug \
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
    --isolation_ratio 0.8 \
    --lr 0.0003 \
    --l1_lambda 1e-5 \
    --weight_decay 1e-4 \
    --epochs 30 \
    --patience 12 \
    --dropout 0.2 \
    --mini_batch_s 2560 \
    --train_fraction 0.5 \
    --compute_mad \
    --use_minibatch_gnn \
    --neighbor_sample_size 5 \
    --align_lambda 1.0 \
    --align_types cell drug gene \
    --gpu 0 \
    --num_workers 4
