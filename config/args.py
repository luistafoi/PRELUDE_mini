# config/args.py

import argparse

def read_args():
    parser = argparse.ArgumentParser()
    
    # --- Data & Path Arguments ---
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing the processed graph data.')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save the best model checkpoint.')
    parser.add_argument('--model_name', type=str, default='prelude_model',
                        help='Name for the saved model file.')
    
    # --- Inference / Restore ---
    parser.add_argument('--load_path', type=str, default='',
                        help='Full path to a saved model checkpoint to load for evaluation.')
    parser.add_argument('--metadata_file', type=str, default='data/misc/cell_line_metadata.csv', 
                        help='Path to cell line metadata CSV.')
    
    # --- Hardware ---
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers.')

    # --- Model Architecture ---
    parser.add_argument('--embed_d', type=int, default=256,
                        help='Embedding dimension for all node types.')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GNN layers.')
    parser.add_argument('--max_neighbors', type=int, default=10,
                        help='Max neighbors per type sampled for GNN aggregation. '
                             'Must match generate_neighbors.py value.')
    parser.add_argument('--use_skip_connection', action='store_true',
                        help='Enable skip connections in final embedding.')
    parser.add_argument('--use_node_gate', action='store_true',
                        help='Per-node adaptive skip gate MLP (instead of per-type). '
                             'Overrides --use_skip_connection with a learned per-node alpha.')
    parser.add_argument('--use_static_cell_embeddings', action='store_true',
                        help='Use pre-computed VAE embeddings.')
    parser.add_argument('--freeze_cell_gnn', action='store_true',
                        help='Cells always use projection-only (no GNN refinement). '
                             'GNN only refines drug/gene nodes. Eliminates proj/GNN space divergence for inductive cells.')
    parser.add_argument('--include_cell_drug', action='store_true',
                        help='Include Cell-Drug edges in GNN neighbor data (requires regenerated neighbor pickle). '
                             'Used with isolation_ratio for dynamic masking during training.')
    parser.add_argument('--dual_head', action='store_true',
                        help='Use separate scoring heads for inductive (isolated) and transductive (non-isolated) cells. '
                             'Requires --include_cell_drug. Prevents transductive signal from corrupting inductive head.')
    parser.add_argument('--inductive_loss_weight', type=float, default=1.0,
                        help='Weight multiplier for inductive (isolated) cells loss. >1 biases backbone toward inductive. '
                             'E.g. 3.0 = isolated cells contribute 3x more to backbone gradients.')
    parser.add_argument('--backbone_lr_scale', type=float, default=1.0,
                        help='LR multiplier for GNN backbone + projections (relative to --lr). '
                             '<1 slows backbone drift. E.g. 0.1 = backbone trains at 10x lower LR.')
    parser.add_argument('--cell_feature_source', type=str, default='vae',
                        choices=['vae', 'multiomic', 'hybrid', 'scgpt'],
                        help='Cell feature source: vae (512-dim), multiomic (3858-dim), hybrid (both concatenated), or scgpt (512-dim foundation model).')
    parser.add_argument('--gene_encoder_dim', type=int, default=0,
                        help='Per-gene MLP hidden dim for multiomic features (0=disabled, use flat projection). '
                             'E.g. 8: shared MLP(4->8) per gene, then flatten.')
    parser.add_argument('--use_cross_attention', action='store_true',
                        help='Option D: drug attends over cell gene tokens for scoring. '
                             'Requires --cell_feature_source multiomic and --gene_encoder_dim > 0.')
    parser.add_argument('--cross_attn_dim', type=int, default=32,
                        help='Dimension of gene tokens and drug query in cross-attention scoring.')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    
    # --- THIS WAS MISSING ---
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout rate (0.0 to 1.0) to prevent overfitting.') 
    # ------------------------

    parser.add_argument('--mini_batch_s', type=int, default=256)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='Number of epochs for linear LR warmup (0=disabled). '
                             'LR ramps from lr/10 to lr, then cosine decay takes over.')
    parser.add_argument('--freeze_after_epoch', type=int, default=0,
                        help='Freeze LR to 1e-8 after this epoch (0=disabled). '
                             'Diagnostic: tests whether post-peak decline is optimization vs structural.')
    parser.add_argument('--compute_mad', action='store_true',
                        help='Compute Mean Average Distance (MAD) per epoch to diagnose over-smoothing.')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--random_seed', default=42, type=int)

    # ==========================================
    # EXPERIMENTAL MATRIX FLAGS (REGULARIZATION)
    # ==========================================

    # 1. L1 Regularization (Sparsity)
    parser.add_argument('--l1_lambda', type=float, default=0.0,
                        help='L1 Regularization weight (Lasso). Pushes weights to zero for sparsity.')

    # 2. L2 Regularization (Weight Decay)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 Regularization weight (Ridge). Prevents large weights for stability.')

    # 3. Cell Similarity (Triplet) Loss
    parser.add_argument('--use_triplet_loss', action='store_true',
                        help='Enable self-supervised Cell Similarity loss.')
    parser.add_argument('--triplet_loss_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=1.0)
    parser.add_argument('--triplet_num_pos', type=int, default=1, 
                        help='Number of positive sister cells to sample per anchor.')
    parser.add_argument('--triplet_num_neg', type=int, default=5)

    # 4. Random Walk Loss (Topology Regularization)
    parser.add_argument('--use_rw_loss', action='store_true', 
                        help='Enable Random Walk topology loss.')
    parser.add_argument('--rw_loss_weight', type=float, default=1.0)
    parser.add_argument('--walk_length', type=int, default=5, 
                        help='Length of random walk per node.')
    parser.add_argument('--rw_neg_samples', type=int, default=5, 
                        help='Number of negative samples per RW step.')
    
    # 5. Per-Cell Isolation Training (Cold-Start Simulation)
    parser.add_argument('--isolation_ratio', type=float, default=0.0,
                        help='Fraction of unique cells per batch to isolate (mask all drug-edges). '
                             'Simulates inductive cold-start during training. 0.0=disabled, 0.2=20%% of cells.')

    # 6. Inductive Validation
    parser.add_argument('--validate_inductive', action='store_true',
                        help='If set, uses Inductive (New Patient) set for validation during training.')

    # ==========================================
    # M11: DYNAMIC LOSS MANAGER + LOSS FIXES
    # ==========================================

    # Dynamic loss balancing
    parser.add_argument('--use_dynamic_loss', action='store_true',
                        help='Enable adaptive loss balancing (DynamicLossManager).')
    parser.add_argument('--loss_target_lp', type=float, default=0.6,
                        help='Target ratio for LP (BCE) loss.')
    parser.add_argument('--loss_target_triplet', type=float, default=0.3,
                        help='Target ratio for triplet loss.')
    parser.add_argument('--loss_target_rw', type=float, default=0.1,
                        help='Target ratio for random walk loss.')
    parser.add_argument('--loss_ema_decay', type=float, default=0.99,
                        help='EMA decay for loss magnitude tracking.')
    parser.add_argument('--loss_warmup_batches', type=int, default=50,
                        help='Batches before adaptive weighting activates.')
    parser.add_argument('--loss_max_weight', type=float, default=10.0,
                        help='Max adaptive weight for any aux loss (relative to LP=1.0). Prevents runaway.')

    # Soft-margin triplet
    parser.add_argument('--use_soft_margin_triplet', action='store_true',
                        help='Use soft-margin triplet loss (always provides gradient).')

    # RW embedding normalization
    parser.add_argument('--use_rw_normalize', action='store_true',
                        help='L2-normalize embeddings in RW skip-gram loss (cosine similarity).')

    # ==========================================
    # M14: MINI-BATCH GNN PROPAGATION
    # ==========================================
    parser.add_argument('--use_minibatch_gnn', action='store_true',
                        help='Enable mini-batch GNN propagation (only compute subgraph per batch).')
    parser.add_argument('--neighbor_sample_size', type=int, default=0,
                        help='Random neighbors per node per type per hop (0=all neighbors).')
    parser.add_argument('--align_lambda', type=float, default=0.0,
                        help='Weight for projection-GNN cosine alignment loss (0=disabled). '
                             'Penalizes drift between projected and GNN embeddings.')
    parser.add_argument('--align_types', nargs='+', default=None,
                        help='Node types to align (e.g. cell drug). Default: all types.')
    parser.add_argument('--proj_loss_weight', type=float, default=0.0,
                        help='Weight for projection-only LP loss (dual-path training). '
                             'Total LP = (1-w)*gnn_lp + w*proj_lp. 0=disabled, 0.3=recommended.')

    # ==========================================
    # REGRESSION MODE
    # ==========================================
    parser.add_argument('--regression', action='store_true',
        help='Regression mode: predict continuous drug response (negated LMFI) instead of binary labels. '
             'Uses MAE loss, Spearman correlation for checkpointing.')

    # ==========================================
    # EMA TEACHER REGULARIZATION
    # ==========================================
    parser.add_argument('--ema_lambda', type=float, default=0.0,
        help='Weight for EMA teacher regularization loss (0=disabled). '
             'Penalizes GNN embeddings drifting from slow-moving teacher.')
    parser.add_argument('--ema_momentum', type=float, default=0.999,
        help='EMA momentum for teacher update: θ_t = m*θ_t + (1-m)*θ_s. '
             'Higher = slower teacher update.')

    # ==========================================
    # M15: RESIDUAL GNN ARCHITECTURE
    # ==========================================
    parser.add_argument('--use_mlp_head', action='store_true',
        help='Use MLP head instead of bilinear: MLP(cat(drug, cell)) -> score.')
    parser.add_argument('--use_residual_gnn', action='store_true',
        help='Residual GNN: final = proj + scale*(gnn-proj). Proj is the base; GNN learns correction.')
    parser.add_argument('--residual_scale', type=float, default=0.0,
        help='Fixed residual scale (>0=fixed, 0=learned per-type via sigmoid). Only with --use_residual_gnn.')

    # ==========================================
    # WEIGHTED AGGREGATION PER (center_type:neighbor_type) PAIR
    # ==========================================
    parser.add_argument('--weighted_agg_pairs', type=str, default='',
        help='Comma-separated (center:neighbor) type pairs to use proper weighted-mean aggregation '
             '(denominator = sum of edge weights instead of neighbor count). '
             'E.g. "drug:drug" for Tanimoto-weighted drug-drug aggregation. '
             'Default (empty) = count-based biased mean everywhere.')

    return parser.parse_args()