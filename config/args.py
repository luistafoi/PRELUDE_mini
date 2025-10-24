import argparse

def read_args():
    parser = argparse.ArgumentParser()
    
    # --- Data & Path Arguments ---
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing the processed graph data and neighbor files.')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save the best model checkpoint.')
    parser.add_argument('--model_name', type=str, default='prelude_model',
                        help='Name for the saved model file (e.g., my_experiment).')
    parser.add_argument('--load_path', type=str, default='',
                        help='Full path to a saved model checkpoint to load for evaluation.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use.')

    # --- Model Architecture Arguments ---
    parser.add_argument('--embed_d', type=int, default=256,
                        help='Embedding dimension for all node types.')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GNN layers.')

    # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=150,
                        help='Maximum number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Adam optimizer weight decay.')
    parser.add_argument('--mini_batch_s', type=int, default=256,
                        help='Mini-batch size for link prediction training.')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping (in validation checks).')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='How often to run validation (in epochs).')
    parser.add_argument('--random_seed', default=42, type=int)

    # --- Feature & VAE Arguments ---
    parser.add_argument('--use_skip_connection', action='store_true',
                        help='Enable skip connections in final embedding.')
    parser.add_argument('--use_node_isolation', action='store_true',
                        help='Enable curriculum learning via node isolation.')
    parser.add_argument('--use_vae_encoder', action='store_true',
                        help='Use VAE encoder for cell features.')
    parser.add_argument('--use_rw_loss', action='store_true',
                        help='Enable self-supervised RW skip-gram loss.')
    parser.add_argument('--vae_checkpoint', type=str, default='data/embeddings/cell_vae_weights.pth',
                        help='Path to trained VAE weights.')
    parser.add_argument('--vae_dims', type=str, default='19193,10000,5000,1000,500,256',
                        help='Comma-separated dims for VAE layers.')
    parser.add_argument('--use_lp_curriculum', action='store_true',
                        help='Enable curriculum learning for the LP loss weight.')
    parser.add_argument('--lp_loss_lambda', type=float, default=100.0,
                        help='The maximum weight for the LP loss when curriculum is enabled.')
    parser.add_argument('--use_static_cell_embeddings', action='store_true',
                        help='(Faster) Use pre-computed VAE embeddings from a file for cell feature.')


    # --- Random Walk (for self-supervised loss) ---
    parser.add_argument('--walk_length', type=int, default=10, help='Length of each random walk.')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for skip-gram.')
    parser.add_argument('--num_walks', type=int, default=2, help='Number of walks per node per epoch.')

    return parser.parse_args()