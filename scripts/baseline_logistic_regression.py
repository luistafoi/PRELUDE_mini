# scripts/baseline_logistic_regression.py

import sys
import os
import pandas as pd
import numpy as np
import random
import pickle
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
DATA_DIR = "data/processed"
EMBED_DIR = "data/embeddings"
CELL_EMBED_FILE = os.path.join(EMBED_DIR, "final_vae_cell_embeddings.npy")
CELL_NAMES_FILE = os.path.join(EMBED_DIR, "final_vae_cell_names.txt")
DRUG_EMBED_FILE = os.path.join(EMBED_DIR, "drugs_with_embeddings.csv")
NODE_MAP_FILE = os.path.join(DATA_DIR, "node_mappings.json") # GID -> Name map
NODE_FILE = os.path.join(DATA_DIR, "node.dat") # GID -> Type map

# --- Task Files (Inductive) ---
TRAIN_LP_FILE = os.path.join(DATA_DIR, "train_lp_links.dat")
VALID_LP_FILE = os.path.join(DATA_DIR, "valid_inductive_links.dat")
# ---

def load_features():
    """Loads cell and drug features into dictionaries: name -> embedding."""
    print("Loading features...")
    
    # 1. Load Cell Features
    cell_embeds = np.load(CELL_EMBED_FILE)
    with open(CELL_NAMES_FILE, 'r') as f:
        cell_names = [line.strip() for line in f if line.strip()]
    
    # Use .upper() for robust matching
    cell_features = {name.upper(): embed for name, embed in zip(cell_names, cell_embeds)}
    print(f"  > Loaded {len(cell_features)} cell features.")

    # 2. Load Drug Features
    df_drugs = pd.read_csv(DRUG_EMBED_FILE)
    drug_col_name = df_drugs.columns[0]
    drug_features = {}
    for _, row in df_drugs.iterrows():
        name = str(row[drug_col_name]).strip().upper()
        embed = np.array(row[1:].values, dtype=np.float32)
        drug_features[name] = embed
    print(f"  > Loaded {len(drug_features)} drug features.")
    
    return cell_features, drug_features

def load_links_and_nodes():
    """Loads node maps and link files."""
    print("Loading links and node maps...")
    
    # 1. Load GID -> Name map
    try:
        with open(NODE_MAP_FILE, 'r') as f:
            gid_to_name = json.load(f)
        # Invert map: GID (as int) -> Name (as upper)
        gid_to_name = {int(gid): name.upper() for name, gid in gid_to_name.items()}
    except FileNotFoundError:
        print(f"  > {NODE_MAP_FILE} not found, falling back to node.dat")
        gid_to_name = {}
        with open(NODE_FILE, 'r') as f:
            for line in f:
                gid, name, ntype = line.strip().split('\t')
                gid_to_name[int(gid)] = name.upper() # Already upper from build_graph
                
    # 2. Load GID -> Type map
    gid_to_type = {}
    with open(NODE_FILE, 'r') as f:
        for line in f:
            gid, name, ntype = line.strip().split('\t')
            gid_to_type[int(gid)] = int(ntype)
    
    print(f"  > Loaded mappings for {len(gid_to_name)} nodes.")

    # 3. Load Positive Links
    def _read_links(filepath):
        links = []
        if not os.path.exists(filepath):
            print(f"  > Warning: Link file not found: {filepath}")
            return links
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    links.append((int(parts[0]), int(parts[1])))
        return links

    train_pos_links = _read_links(TRAIN_LP_FILE)
    valid_pos_links = _read_links(VALID_LP_FILE)
    
    print(f"  > Loaded {len(train_pos_links)} train positive links.")
    print(f"  > Loaded {len(valid_pos_links)} valid positive links.")
    
    return gid_to_name, gid_to_type, train_pos_links, valid_pos_links

# --- START FIX 1: Update function signature ---
def create_dataset(pos_links, gid_to_name, full_gid_to_type, cell_features, drug_features, neg_sampling_drug_gids, neg_sample_ratio=1):
# --- END FIX 1 ---
    """Creates a concatenated feature dataset (X, y) from links."""
    
    X = []
    y = []
    
    cell_type_id = 0 # Assuming 0=cell
    drug_type_id = 1 # Assuming 1=drug
    
    # --- START FIX 2: Use the provided drug GID list for sampling ---
    all_drug_gids = neg_sampling_drug_gids
    if not all_drug_gids:
    # --- END FIX 2 ---
        print("  > ERROR: No drug nodes found for negative sampling.")
        return np.array([]), np.array([])
        
    pos_link_set = set(pos_links) # For fast lookup
    
    # --- 1. Process Positive Links ---
    pos_cells_for_neg = [] # Collect cells to use for negative sampling
    for gid_a, gid_b in tqdm(pos_links, desc="  Processing positive links"):
        # Figure out which is cell and which is drug
        gid_cell, gid_drug = None, None
        
        # --- START FIX 3: Use the FULL gid_to_type map here ---
        type_a = full_gid_to_type.get(gid_a)
        type_b = full_gid_to_type.get(gid_b)
        # --- END FIX 3 ---

        if type_a == cell_type_id and type_b == drug_type_id:
            gid_cell, gid_drug = gid_a, gid_b
        elif type_a == drug_type_id and type_b == cell_type_id:
            gid_cell, gid_drug = gid_b, gid_a
        else:
            continue # Skip non C-D links
        
        pos_cells_for_neg.append(gid_cell) # Add cell to list for neg sampling
            
        # Get names
        name_cell = gid_to_name.get(gid_cell)
        name_drug = gid_to_name.get(gid_drug)
        
        # Get features
        feat_cell = cell_features.get(name_cell)
        feat_drug = drug_features.get(name_drug)
        
        if feat_cell is not None and feat_drug is not None:
            X.append(np.concatenate([feat_cell, feat_drug]))
            y.append(1.0)

    # --- 2. Process Negative Links ---
    num_neg_samples = int(len(pos_links) * neg_sample_ratio)
    neg_links_added = 0
    
    # --- START FIX 4: Check the correct list ---
    if not pos_cells_for_neg:
    # --- END FIX 4 ---
        print("  > ERROR: No cells found in positive links to generate negatives.")
        return np.array(X), np.array(y) # Return positive-only data if any

    for _ in tqdm(range(num_neg_samples), desc="  Generating negative links"):
        
        # Get a random cell from the positive link list
        gid_cell = random.choice(pos_cells_for_neg)
        
        # Sample a random drug
        gid_drug_neg = random.choice(all_drug_gids)
        
        # Check if this is a known positive
        if (gid_cell, gid_drug_neg) not in pos_link_set and (gid_drug_neg, gid_cell) not in pos_link_set:
            # Get names
            name_cell = gid_to_name.get(gid_cell)
            name_drug_neg = gid_to_name.get(gid_drug_neg)
            
            # Get features
            feat_cell = cell_features.get(name_cell)
            feat_drug_neg = drug_features.get(name_drug_neg)
            
            if feat_cell is not None and feat_drug_neg is not None:
                X.append(np.concatenate([feat_cell, feat_drug_neg]))
                y.append(0.0)
                neg_links_added += 1

    return np.array(X), np.array(y)


if __name__ == "__main__":
    
    # 1. Load all data
    cell_feats, drug_feats = load_features()
    gid_to_name, gid_to_type, train_pos, valid_pos = load_links_and_nodes()
    
    # --- START FIX 5: Correctly define drug lists for sampling ---
    cell_type_id = 0
    drug_type_id = 1
    
    # For training, we can sample from ALL drugs in the graph
    train_neg_sampling_drugs = [gid for gid, ntype in gid_to_type.items() if ntype == drug_type_id]
    
    # 2. Create Train Dataset (X, y)
    print("\nBuilding Training Dataset (X_train, y_train)...")
    # Pass the full gid_to_type map, and the specific drug list for neg sampling
    X_train, y_train = create_dataset(train_pos, gid_to_name, gid_to_type, cell_feats, drug_feats, train_neg_sampling_drugs)
    # --- END FIX 5 ---
    
    if X_train.size == 0:
        print("FATAL: No training data could be created. Check feature and link files.")
        sys.exit(1)
        
    # 3. Create Validation Dataset (X, y)
    print("\nBuilding Validation Dataset (X_valid, y_valid)...")
    # --- START FIX 6: Correctly define validation drug list ---
    # For inductive validation, negatives must *only* be sampled from *validation* drugs
    valid_drug_gids = set()
    for gid_a, gid_b in valid_pos:
        if gid_to_type.get(gid_a) == drug_type_id: valid_drug_gids.add(gid_a)
        if gid_to_type.get(gid_b) == drug_type_id: valid_drug_gids.add(gid_b)
    
    valid_neg_sampling_drugs = list(valid_drug_gids)
    
    # Pass the full gid_to_type map, and the *validation-only* drug list for neg sampling
    X_valid, y_valid = create_dataset(valid_pos, gid_to_name, gid_to_type, cell_feats, drug_feats, valid_neg_sampling_drugs)
    # --- END FIX 6 ---
    
    if X_valid.size == 0:
        print("FATAL: No validation data could be created. Check feature and link files.")
        sys.exit(1)
    
    print(f"\nTraining set: {X_train.shape}, Validation set: {X_valid.shape}")
    
    # 4. Standardize Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # 5. Train Model
    print("Training Logistic Regression model...")
    # --- START FIX 7: Re-add the model definition line ---
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, C=0.1) 
    # --- END FIX 7 ---
    model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate
    print("Evaluating model...")
    # Get probabilities for AUC
    y_valid_probs = model.predict_proba(X_valid_scaled)[:, 1]
    # Get discrete predictions for F1
    y_valid_preds = model.predict(X_valid_scaled)
    
    auc_val = roc_auc_score(y_valid, y_valid_probs)
    f1_val = f1_score(y_valid, y_valid_preds)
    
    print("\n--- Logistic Regression (Feature-Only) Baseline ---")
    print(f"  Validation AUC: {auc_val:.4f}")
    print(f"  Validation F1:  {f1_val:.4f}")
    print("-------------------------------------------------------")

    # --- Plotting Code (already added) ---
    print("\nGenerating ROC curve plot...")
    try:
        fpr, tpr, thresholds = roc_curve(y_valid, y_valid_probs)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Baseline Logistic Regression (Inductive)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.5)

        # Save the figure
        save_dir = "checkpoints" # Save plot in checkpoints dir
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, "baseline_logistic_regression_roc_curve.pdf")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to: {plot_filename}")
        plt.close()

    except Exception as e:
        print(f"Error generating or saving ROC plot: {e}")