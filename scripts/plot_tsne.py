# scripts/plot_tsne.py

import sys
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
# Import the dataset class from train.py to generate neg samples
from scripts.train import LinkPredictionDataset

def generate_cell_tsne_plot(args, dataset, feature_loader, generator, model):
    """
    Loads a trained model, generates 'before' and 'after' GNN embeddings
    for all cells, runs t-SNE, and saves a comparative plot.
    The 'After' plot *only* shows nodes present in the training graph.
    """
    print("\n--- Generating Cell t-SNE Plot ---")
    
    if not args.metadata_file or not os.path.exists(args.metadata_file):
        print(f"Error: Cell metadata file not found at {args.metadata_file}. Skipping cell t-SNE plot.")
        return
        
    print(f"Loading metadata from: {args.metadata_file}")
    metadata_df = pd.read_csv(args.metadata_file)
    metadata_map = metadata_df.set_index('cell_name')['tissue_type'].to_dict()

    print("Getting all cell nodes...")
    cell_type_id = dataset.node_name2type.get('cell', -1)
    if cell_type_id == -1:
        print("Error: 'cell' node type not found. Skipping cell t-SNE plot.")
        return

    cell_data = []
    train_node_keys = set(model.precomputed_neighbors.keys())

    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == cell_type_id:
            cell_name = dataset.id2node.get(gid)
            if cell_name:
                is_train_node = (ntype, lid) in train_node_keys
                cell_data.append({
                    'gid': gid,
                    'lid': lid,
                    'name': cell_name,
                    'tissue_type': metadata_map.get(cell_name, 'Unknown'),
                    'is_train_node': is_train_node
                })
    
    cell_df = pd.DataFrame(cell_data)
    cell_lids_tensor = torch.tensor(cell_df['lid'].values, dtype=torch.long).to(model.device)
    
    print(f"Found {len(cell_df)} total cell nodes ({cell_df['is_train_node'].sum()} in training graph). Generating embeddings...")

    with torch.no_grad():
        initial_embeds = model.conteng_agg(cell_lids_tensor, cell_type_id).cpu().numpy()
        final_embeds = model.node_het_agg(cell_lids_tensor, cell_type_id, generator).cpu().numpy()

    print("Cell embeddings generated. Running t-SNE...")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    
    initial_tsne = tsne.fit_transform(initial_embeds)
    final_tsne = tsne.fit_transform(final_embeds)
    
    cell_df['initial_tsne_1'] = initial_tsne[:, 0]
    cell_df['initial_tsne_2'] = initial_tsne[:, 1]
    cell_df['final_tsne_1'] = final_tsne[:, 0]
    cell_df['final_tsne_2'] = final_tsne[:, 1]

    print("Plotting cell t-SNE...")
    
    top_tissues = cell_df['tissue_type'].value_counts().nlargest(10).index
    
    plot_df_before = cell_df[cell_df['tissue_type'].isin(top_tissues)]
    plot_df_after = cell_df[cell_df['tissue_type'].isin(top_tissues) & cell_df['is_train_node']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    sns.scatterplot(
        data=plot_df_before, x='initial_tsne_1', y='initial_tsne_2', hue='tissue_type',
        palette='viridis', ax=ax1, s=50, alpha=0.7
    )
    ax1.set_title('Initial Embeddings (All Cells, Static Projected)', fontsize=16)
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    sns.scatterplot(
        data=plot_df_after, x='final_tsne_1', y='final_tsne_2', hue='tissue_type',
        palette='viridis', ax=ax2, s=50, alpha=0.7
    )
    ax2.set_title('Final Embeddings (TRAINING Cells Only, After GNN+RW)', fontsize=16)
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.get_legend().remove()

    fig.suptitle('t-SNE Visualization of Cell Line Embeddings (Inductive Model)', fontsize=20, y=1.02)
    plt.tight_layout()
    
    model_base_name = args.model_name if hasattr(args, 'model_name') else os.path.splitext(os.path.basename(args.load_path))[0]
    output_filename = f"tsne_plot_cells_{model_base_name}.pdf"
    save_dir = args.save_dir if hasattr(args, 'save_dir') else 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, output_filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight') 
    print(f"\nCell t-SNE plot saved successfully to: {output_path}")
    plt.close()

def generate_edge_tsne_plot(args, dataset, feature_loader, generator, model):
    """
    Generates 'before' and 'after' GNN embeddings for cell-drug edges
    (from the validation set) and plots a t-SNE, colored by label.
    Samples 10k positive and 10k negative links for speed.
    """
    print("\n--- Generating Edge t-SNE Plot (Inductive Validation Set) ---")
    
    valid_pos = dataset.links.get('validation', [])
    if not valid_pos:
        print("Error: No validation links found. Skipping edge t-SNE plot.")
        return

    print("Loading validation links and generating negative samples...")
    valid_dataset = LinkPredictionDataset(valid_pos, dataset, sample_ratio=1.0, neg_sample_ratio=1)
    
    u_lids_all = torch.tensor(valid_dataset.u_lids, dtype=torch.long)
    v_lids_all = torch.tensor(valid_dataset.v_lids, dtype=torch.long)
    labels_all = torch.tensor(valid_dataset.labels, dtype=torch.long)
    u_types_all = torch.tensor(valid_dataset.u_types, dtype=torch.long)

    print(f"Processing {len(labels_all)} total edges ({len(valid_pos)} positive, {len(labels_all)-len(valid_pos)} negative)...")

    SAMPLE_SIZE = 10000
    pos_indices = torch.where(labels_all == 1)[0]
    neg_indices = torch.where(labels_all == 0)[0]

    if len(pos_indices) > SAMPLE_SIZE:
        pos_indices_sampled = pos_indices[torch.randperm(len(pos_indices))[:SAMPLE_SIZE]]
    else:
        pos_indices_sampled = pos_indices
    if len(neg_indices) > SAMPLE_SIZE:
        neg_indices_sampled = neg_indices[torch.randperm(len(neg_indices))[:SAMPLE_SIZE]]
    else:
        neg_indices_sampled = neg_indices

    sampled_indices = torch.cat([pos_indices_sampled, neg_indices_sampled])
    
    u_lids_tensor = u_lids_all[sampled_indices].to(model.device)
    v_lids_tensor = v_lids_all[sampled_indices].to(model.device)
    labels_tensor = labels_all[sampled_indices]
    u_types_tensor = u_types_all[sampled_indices].to(model.device)
    
    print(f"Sampled {len(labels_tensor)} total edges for t-SNE ({len(pos_indices_sampled)} positive, {len(neg_indices_sampled)} negative)...")

    cell_lids = []
    drug_lids = []
    cell_type_id = dataset.node_name2type.get('cell', -1)
    drug_type_id = dataset.node_name2type.get('drug', -1)
    
    if len(u_types_tensor) == 0:
        print("Error: No links to process after sampling. Skipping edge t-SNE.")
        return
        
    u_type_id_first = u_types_tensor[0].item()
    
    if u_type_id_first == cell_type_id:
        cell_lids = u_lids_tensor
        drug_lids = v_lids_tensor
    elif u_type_id_first == drug_type_id:
        cell_lids = v_lids_tensor
        drug_lids = u_lids_tensor
    else:
        print("Error: First link in sampled set is not a cell-drug link. Skipping edge t-SNE.")
        return

    print("Generating edge embeddings...")
    with torch.no_grad():
        cell_embeds_initial = model.conteng_agg(cell_lids, cell_type_id)
        drug_embeds_initial = model.conteng_agg(drug_lids, drug_type_id)
        
        cell_embeds_final_gnn = model.node_het_agg(cell_lids, cell_type_id, generator)
        drug_embeds_final_gnn = model.node_het_agg(drug_lids, drug_type_id, generator)

        edge_embeds_before = (cell_embeds_initial * drug_embeds_initial).cpu().numpy()
        edge_embeds_after = (cell_embeds_final_gnn * drug_embeds_final_gnn).cpu().numpy()

    print("Edge embeddings generated. Running t-SNE...")

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    
    tsne_before_data = tsne.fit_transform(edge_embeds_before)
    tsne_after_data = tsne.fit_transform(edge_embeds_after)

    df = pd.DataFrame(labels_tensor.cpu().numpy(), columns=['label'])
    df['label_str'] = df['label'].map({1: 'Positive Link', 0: 'Negative Link'})
    df['tsne_before_1'] = tsne_before_data[:, 0]
    df['tsne_before_2'] = tsne_before_data[:, 1]
    df['tsne_after_1'] = tsne_after_data[:, 0]
    df['tsne_after_2'] = tsne_after_data[:, 1]

    print("Plotting edge t-SNE...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    sns.scatterplot(
        data=df, x='tsne_before_1', y='tsne_before_2', hue='label_str',
        palette={'Positive Link':'blue', 'Negative Link':'red'}, ax=ax1,
        s=10, alpha=0.5
    )
    ax1.set_title('Edge Embeddings (Initial, Projected)', fontsize=16)
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    sns.scatterplot(
        data=df, x='tsne_after_1', y='tsne_after_2', hue='label_str',
        palette={'Positive Link':'blue', 'Negative Link':'red'}, ax=ax2,
        s=10, alpha=0.5
    )
    ax2.set_title('Edge Embeddings (After GNN - Unseen Nodes)', fontsize=16)
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.get_legend().remove()
    
    plt.figtext(0.5, 0.01, 
                "Note: The 'After GNN' plot for edges appears identical to the 'Initial' plot.\n"
                "This is expected, as these are *unseen* inductive validation links.\n"
                "The GNN bypasses message passing for unseen nodes, using only their initial projected features for prediction.", 
                ha="center", fontsize=10, style='italic')

    fig.suptitle('t-SNE Visualization of C-D Edge Embeddings (Inductive Validation Set)', fontsize=20, y=1.04)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    model_base_name = args.model_name if hasattr(args, 'model_name') else os.path.splitext(os.path.basename(args.load_path))[0]
    output_filename = f"tsne_plot_edges_{model_base_name}.pdf"
    save_dir = args.save_dir if hasattr(args, 'save_dir') else 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, output_filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\nEdge t-SNE plot saved successfully to: {output_path}")
    plt.close()


def main(args): # <-- Pass args in
    """
    Main function to load shared components and run plotting functions.
    """
    # --- 1. Argument Parsing (MOVED TO __name__ == "__main__") ---
    
    # --- 2. Check for Load Path ---
    if not args.load_path or not os.path.exists(args.load_path):
        print(f"Error: Must provide a valid model checkpoint using --load_path. Path provided: '{args.load_path}'")
        sys.exit(1)
        
    device = torch.device('cpu') # Use CPU for plotting
    print(f"Using device: {device}")

    # --- 3. Load Shared Components ONCE ---
    print(f"Loading dataset from: {args.data_dir}")
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir)
    
    print("Initializing model...")
    model_args_ns = argparse.Namespace(**vars(args))
    model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    
    print(f"Loading trained model weights from: {args.load_path}")
    try:
        model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"FATAL ERROR loading model weights: {e}")
        if 'size mismatch' in str(e):
            print("\nHint: Ensure arguments (e.g., --use_skip_connection, --embed_d, --n_layers, --use_rw_loss) MATCH the trained model.")
        sys.exit(1)
    
    model.eval()

    # --- 4. Run Plotting Functions ---
    generate_cell_tsne_plot(args, dataset, feature_loader, generator, model)
    generate_edge_tsne_plot(args, dataset, feature_loader, generator, model)

if __name__ == "__main__":
    
    # --- START SIMPLIFIED ARGUMENT PARSING ---
    
    # 1. Read all the *known* args from config/args.py
    # This function (read_args) is defined in config/args.py and parses sys.argv
    args = read_args() 
    
    # 2. Manually add our script-specific argument if it's not already in config/args.py
    # (We added it, but this is a safe fallback)
    if not hasattr(args, 'metadata_file'):
         print("Adding default metadata_file path...")
         setattr(args, 'metadata_file', 'data/misc/cell_line_metadata.csv')
    
    # 3. Now `args` contains everything. Call the main function.
    main(args)
    
    # --- END SIMPLIFIED ARGUMENT PARSING ---