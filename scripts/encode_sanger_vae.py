import sys
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Import your VAE class definition
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cell_vae import CellLineVAE 

# --- CONFIG ---
LATENT_DIM = 512
DROPOUT_RATE = 0.4
BATCH_SIZE = 64

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanger_aligned', type=str, required=True)
    parser.add_argument('--depmap_original', type=str, required=True)
    parser.add_argument('--vae_weights', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='data/sanger_validation/sanger_vae_embeddings.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Fit Scaler on ORIGINAL Data
    print("1. Loading original DepMap data to fit StandardScaler...")
    try:
        df_train = pd.read_csv(args.depmap_original, index_col=0)
        scaler = StandardScaler()
        scaler.fit(df_train.values.astype(np.float32))
        
        # --- THE FIX: HANDLE ZERO VARIANCE ---
        # If std is 0, set it to 1 to avoid Divide-By-Zero NaNs
        scaler.scale_[scaler.scale_ == 0.0] = 1.0
        # -------------------------------------
        
        input_dim = df_train.shape[1]
        print(f"   Scaler fitted on {len(df_train)} cells. Input Dim: {input_dim}")
        del df_train 
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

    # 2. Load and Transform Sanger Data
    print(f"2. Loading Aligned Sanger Data from {args.sanger_aligned}...")
    df_sanger = pd.read_csv(args.sanger_aligned, index_col=0)
    
    # 3. SAFETY CHECK: Fill any accidental NaNs from alignment
    if df_sanger.isnull().values.any():
        print("   Warning: Found NaNs in aligned data. Filling with 0.0...")
        df_sanger = df_sanger.fillna(0.0)

    print("   Normalizing Sanger data...")
    try:
        sanger_scaled = scaler.transform(df_sanger.values.astype(np.float32))
        
        # DOUBLE CHECK: Did the scaler produce NaNs?
        if np.isnan(sanger_scaled).any():
            print("   CRITICAL WARNING: Scaler produced NaNs! Replacing with 0.0...")
            sanger_scaled = np.nan_to_num(sanger_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
    except Exception as e:
        print(f"   Scaler Error: {e}")
        sys.exit(1)

    # 3. Load VAE
    print("3. Initializing VAE...")
    dims = [input_dim, 10000, 5000, 2048, 1024, LATENT_DIM]
    model = CellLineVAE(dims, dropout_rate=DROPOUT_RATE).to(device)
    
    print(f"   Loading weights from {args.vae_weights}...")
    model.load_state_dict(torch.load(args.vae_weights, map_location=device))
    model.eval()

    # 4. Inference Loop
    print("4. Generating Embeddings...")
    sanger_tensor = torch.tensor(sanger_scaled, dtype=torch.float32)
    dataset = TensorDataset(sanger_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            mu, _ = model.encode(batch_x)
            
            # Move to CPU immediately to check for NaNs
            batch_emb = mu.cpu().numpy()
            if np.isnan(batch_emb).any():
                print("   Warning: VAE output contained NaNs! (Likely dead neuron or input issue)")
                batch_emb = np.nan_to_num(batch_emb)
                
            embeddings.append(batch_emb)
            
    final_emb = np.concatenate(embeddings, axis=0)
    
    # 5. Save
    print(f"5. Saving to {args.output_path}...")
    df_out = pd.DataFrame(final_emb, index=df_sanger.index)
    df_out.to_csv(args.output_path)
    print("Done.")

if __name__ == "__main__":
    main()