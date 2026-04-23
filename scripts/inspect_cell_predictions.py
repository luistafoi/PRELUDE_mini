import sys
import os
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg

# --- Configuration ---
# Default path to the raw gene expression file (as seen in cell_vae.py)
DEFAULT_GENE_EXPRESSION_FILE = 'data/misc/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
BATCH_SIZE = 256 # Batch size for running inference against all drugs

def inspect_single_cell(args, dataset, feature_loader, generator, model):
    """
    Loads a trained model and inspects a single cell:
    1. Prints the Top-K most highly-expressed genes.
    2. Predicts and prints the Top-K drugs the model believes will interact.
    """
    device = model.device
    target_cell_name_upper = args.cell_name.strip().upper()

    # --- 1. Find Top Expressed Genes ---
    print(f"\n--- Task 1: Finding Top {args.top_k} Expressed Genes for {target_cell_name_upper} ---")
    if not os.path.exists(args.gene_expression_file):
        print(f"Warning: Gene expression file not found at {args.gene_expression_file}. Skipping this task.")
    else:
        try:
            print(f"Loading gene expression data from {args.gene_expression_file}...")
            df_expr = pd.read_csv(args.gene_expression_file, index_col=0)
            df_expr.index = df_expr.index.str.upper() # Ensure matching
            
            if target_cell_name_upper in df_expr.index:
                cell_expression_series = df_expr.loc[target_cell_name_upper]
                top_genes = cell_expression_series.sort_values(ascending=False)
                
                print(f"Top {args.top_k} expressed genes (Logp1(TPM)):")
                print(top_genes.head(args.top_k))
            else:
                print(f"Error: Cell name '{target_cell_name_upper}' not found in the gene expression file index.")
        except Exception as e:
            print(f"Error processing gene expression file: {e}")

    # --- 2. Find Target Cell and All Drugs ---
    print(f"\n--- Task 2: Predicting Top {args.top_k} Drugs for {target_cell_name_upper} ---")
    print("Finding target cell and all drugs in the graph...")
    cell_type_id = dataset.node_name2type.get('cell', -1)
    drug_type_id = dataset.node_name2type.get('drug', -1)
    
    target_cell_gid = dataset.node2id.get(target_cell_name_upper, -1)
    if target_cell_gid == -1:
        print(f"Error: Cell '{target_cell_name_upper}' not found in graph's node.dat. Cannot make predictions.")
        return
    
    try:
        target_cell_lid = dataset.nodes['type_map'][target_cell_gid][1]
    except KeyError:
        print(f"Error: Cell GID {target_cell_gid} not found in nodes['type_map'].")
        return

    all_drug_data = [] # List of (lid, name)
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == drug_type_id:
            all_drug_data.append((lid, dataset.id2node[gid]))
            
    if not all_drug_data:
        print("Error: No drug nodes found in the dataset.")
        return
        
    all_drug_lids = torch.tensor([d[0] for d in all_drug_data], dtype=torch.long).to(device)
    all_drug_names = [d[1] for d in all_drug_data]
    num_all_drugs = len(all_drug_data)
    print(f"Found target cell (LID: {target_cell_lid}) and {num_all_drugs} total drugs to test against.")

    # --- 3. Run Predictions in Batches ---
    print("Running predictions for all drugs...")
    all_scores = []
    
    target_cell_lids_full = torch.tensor([target_cell_lid] * num_all_drugs, dtype=torch.long).to(device)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_all_drugs, BATCH_SIZE), desc="Predicting drug batches"):
            drug_lids_batch = all_drug_lids[i : i + BATCH_SIZE]
            cell_lids_batch = target_cell_lids_full[i : i + BATCH_SIZE]
            
            preds = model.link_prediction_forward(drug_lids_batch, cell_lids_batch, generator)
            all_scores.extend(preds.cpu().numpy())
            
    # --- 4. Show Top K Results ---
    print(f"\n--- Top {args.top_k} Predicted Drugs for {target_cell_name_upper} ---")
    
    results = list(zip(all_scores, all_drug_names))
    results.sort(key=lambda x: x[0], reverse=True) # Sort by score, descending
    
    for i in range(args.top_k):
        if i < len(results):
            score, name = results[i]
            print(f"  Rank {i+1:2}: {name:30} (Score: {score:.4f})")
            
    print("\n--- Inspection Complete ---")


def inspect_all_test_cells(args, dataset, feature_loader, generator, model):
    """
    Loads a trained model and, for ALL cells in the test set:
    1. Finds the Top-K most highly-expressed genes.
    2. Predicts the Top-K drugs the model believes will interact.
    3. Saves all results to a comprehensive CSV file.
    """
    device = model.device
    
    # --- 1. Load Gene Expression Data ---
    print(f"\n--- Task 1: Loading Gene Expression Data ---")
    df_expr = None
    if not os.path.exists(args.gene_expression_file):
        print(f"Warning: Gene expression file not found at {args.gene_expression_file}. Top genes will be empty.")
    else:
        try:
            print(f"Loading gene expression data from {args.gene_expression_file}...")
            df_expr = pd.read_csv(args.gene_expression_file, index_col=0)
            df_expr.index = df_expr.index.str.upper() # Ensure matching
        except Exception as e:
            print(f"Error processing gene expression file: {e}")
            
    # --- 2. Load Cell Metadata (for tissue type) ---
    metadata_map = {}
    if os.path.exists(args.metadata_file):
        print(f"Loading metadata from: {args.metadata_file}")
        try:
            metadata_df = pd.read_csv(args.metadata_file)
            metadata_map = metadata_df.set_index('cell_name')['tissue_type'].to_dict()
        except Exception as e:
            print(f"Warning: Could not load metadata file {args.metadata_file}: {e}")
    else:
        print(f"Info: Metadata file not found at {args.metadata_file}. Tissue types will be 'Unknown'.")


    # --- 3. Identify Target Test Cells ---
    print("\n--- Task 2: Identifying Target Cells from Test Set ---")
    cell_type_id = dataset.node_name2type.get('cell', -1)
    drug_type_id = dataset.node_name2type.get('drug', -1)

    # Find the correct test set (transductive or inductive)
    test_pos = None
    if dataset.links.get('test_transductive'):
        test_pos = dataset.links['test_transductive']
        print(f"Found {len(test_pos)} links in 'test_transductive'")
    elif dataset.links.get('test_inductive'):
        test_pos = dataset.links['test_inductive']
        print(f"Found {len(test_pos)} links in 'test_inductive'")
    
    if not test_pos:
        print("Error: No test links found in dataset. Cannot generate report.")
        return

    # Get all unique cell GIDs from the test links
    test_cell_gids = set()
    for gid_a, gid_b in test_pos:
        if dataset.node_types.get(gid_a) == cell_type_id:
            test_cell_gids.add(gid_a)
        if dataset.node_types.get(gid_b) == cell_type_id:
            test_cell_gids.add(gid_b)
            
    print(f"Found {len(test_cell_gids)} unique cells in the test set.")
    
    # --- 4. Get All Drugs ---
    print("Getting all drugs in the graph...")
    all_drug_data = [] # List of (lid, name)
    for gid, (ntype, lid) in dataset.nodes['type_map'].items():
        if ntype == drug_type_id:
            all_drug_data.append((lid, dataset.id2node[gid]))
            
    if not all_drug_data:
        print("Error: No drug nodes found in the dataset.")
        return
        
    all_drug_lids_tensor = torch.tensor([d[0] for d in all_drug_data], dtype=torch.long).to(device)
    all_drug_names = [d[1] for d in all_drug_data]
    num_all_drugs = len(all_drug_data)
    
    # --- 5. Run Inspection Loop for All Test Cells ---
    print(f"\n--- Task 3: Running predictions for {len(test_cell_gids)} test cells ---")
    
    all_results_data = [] # This will hold data for our final CSV
    
    for cell_gid in tqdm(test_cell_gids, desc="Inspecting test cells"):
        try:
            cell_lid = dataset.nodes['type_map'][cell_gid][1]
            cell_name = dataset.id2node[cell_gid]
        except KeyError:
            print(f"Warning: Cell GID {cell_gid} not in maps. Skipping.")
            continue
            
        cell_name_upper = cell_name.upper()
        
        # A. Get Top Genes
        top_genes_list = []
        if df_expr is not None and cell_name_upper in df_expr.index:
            cell_expression_series = df_expr.loc[cell_name_upper]
            top_genes = cell_expression_series.sort_values(ascending=False).head(args.top_k)
            top_genes_list = list(zip(top_genes.index, top_genes.values))
        else:
            # Pad with placeholders
            top_genes_list = [('N/A', 0.0)] * args.top_k

        # B. Get Top Drugs
        all_scores = []
        target_cell_lids_full = torch.tensor([cell_lid] * num_all_drugs, dtype=torch.long).to(device)

        with torch.no_grad():
            for i in range(0, num_all_drugs, BATCH_SIZE):
                drug_lids_batch = all_drug_lids_tensor[i : i + BATCH_SIZE]
                cell_lids_batch = target_cell_lids_full[i : i + BATCH_SIZE]
                
                preds = model.link_prediction_forward(drug_lids_batch, cell_lids_batch, generator)
                all_scores.extend(preds.cpu().numpy())
        
        results = list(zip(all_drug_names, all_scores))
        results.sort(key=lambda x: x[1], reverse=True) # Sort by score, descending
        top_drugs_list = results[:args.top_k]

        # C. Combine and Store Results for CSV
        cell_tissue = metadata_map.get(cell_name, 'Unknown')
        
        for i in range(args.top_k):
            row = {
                'cell_name': cell_name,
                'cell_tissue': cell_tissue,
                'rank': i + 1,
                'top_gene': top_genes_list[i][0],
                'top_gene_expression': top_genes_list[i][1],
                'top_predicted_drug': top_drugs_list[i][0],
                'prediction_score': top_drugs_list[i][1]
            }
            all_results_data.append(row)

    # --- 6. Save Comprehensive Report ---
    print("\n--- Task 4: Saving Comprehensive Report ---")
    if not all_results_data:
        print("Error: No results were generated.")
        return
        
    results_df = pd.DataFrame(all_results_data)
    
    # Ensure save directory exists
    save_dir = os.path.dirname(args.output_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    results_df.to_csv(args.output_file, index=False)
    print(f"\nComprehensive report saved to: {args.output_file}")
    print("\n--- Inspection Complete ---")


if __name__ == "__main__":
    
    # --- Argument Parsing (Fixed) ---
    
    # 1. Create a parser *just* for our new/modified args
    parser = argparse.ArgumentParser(description="Inspect a single cell or all test cells.")
    parser.add_argument('--cell_name', type=str, default=None, # Make optional
                        help="The name of the cell to inspect (e.g., 'ACH-000824'). If not provided, runs on all test cells.")
    parser.add_argument('--top_k', type=int, default=20, 
                        help="Number of top genes/drugs to list.")
    parser.add_argument('--gene_expression_file', type=str, default=DEFAULT_GENE_EXPRESSION_FILE,
                        help="Path to the raw gene expression CSV file.")
    parser.add_argument('--output_file', type=str, default='checkpoints/inspection_report.csv',
                        help="Output path for the full test set inspection report.")
    parser.add_argument('--metadata_file', type=str, default='data/misc/cell_line_metadata.csv', 
                        help='Path to cell line metadata CSV (must have "cell_name" and "tissue_type" columns).')
    
    # Parse *only* the args this parser knows, collecting the rest
    inspect_args, other_argv = parser.parse_known_args()
    
    # Temporarily set sys.argv to the *remaining* args for read_args()
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + other_argv
    
    try:
        args = read_args() # This gets all the model/data args
    finally:
        sys.argv = original_argv # Restore
    
    # Manually add our parsed args to the main args object
    vars(args).update(vars(inspect_args))
    
    # --- Load Model and Data (Shared) ---
    device = torch.device('cpu') # Use CPU for inspection
    print(f"Using device: {device}")
    
    if not args.load_path or not os.path.exists(args.load_path):
        print(f"Error: Must provide a valid model checkpoint using --load_path. Path provided: '{args.load_path}'")
        sys.exit(1)
        
    print(f"Loading dataset from: {args.data_dir}")
    try:
        dataset = PRELUDEDataset(args.data_dir)
        feature_loader = FeatureLoader(dataset, device)
        generator = DataGenerator(args.data_dir)
    except Exception as e:
         print(f"FATAL ERROR during data loading: {e}")
         sys.exit(1)

    print("Initializing model...")
    try:
        model_args_ns = argparse.Namespace(**vars(args))
        model = HetAgg(model_args_ns, dataset, feature_loader, device).to(device)
        model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
        
        print(f"Loading trained model weights from: {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
    except Exception as e:
         print(f"FATAL ERROR loading model: {e}")
         if 'size mismatch' in str(e):
              print("\nHint: Ensure inspection arguments (e.g., --use_skip_connection, --embed_d) MATCH the trained model.")
         sys.exit(1)
    
    model.eval()

    # --- Decide which function to run ---
    if args.cell_name:
        # Run inspection for a single cell
        inspect_single_cell(args, dataset, feature_loader, generator, model)
    else:
        # Run inspection for all test cells and save a report
        inspect_all_test_cells(args, dataset, feature_loader, generator, model)

