# scripts/cell_vae.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import sys # Added for path checks

# --- VAE Model Definition ---
class CellLineVAE(nn.Module):
    def __init__(self, dims, dropout_rate=0.4):
        super(CellLineVAE, self).__init__()
        self.encoder = nn.Sequential()
        # Correctly builds encoder layers
        for i in range(len(dims) - 1):
            self.encoder.add_module(f'enc_fc{i}', nn.Linear(dims[i], dims[i + 1]))
            self.encoder.add_module(f'enc_bn{i}', nn.BatchNorm1d(dims[i + 1]))
            self.encoder.add_module(f'enc_relu{i}', nn.ReLU())
            self.encoder.add_module(f'enc_dropout{i}', nn.Dropout(dropout_rate))

        self.fc_mu = nn.Linear(dims[-1], dims[-1])
        self.fc_logvar = nn.Linear(dims[-1], dims[-1])

        self.decoder = nn.Sequential()
        # Correctly builds decoder layers (note: final layer might need adjustment based on input data range)
        for i in range(len(dims) - 1, 0, -1):
            self.decoder.add_module(f'dec_fc{i}', nn.Linear(dims[i], dims[i - 1]))
            self.decoder.add_module(f'dec_bn{i}', nn.BatchNorm1d(dims[i - 1]))
            self.decoder.add_module(f'dec_relu{i}', nn.ReLU())
            self.decoder.add_module(f'dec_dropout{i}', nn.Dropout(dropout_rate))
        # No final activation (like Sigmoid) assumed, which is typical for StandardScaler normalized data

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- VAE Loss Function ---
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Uses scaled MSE and correct KLD term
    mse = F.mse_loss(recon_x, x, reduction='mean') * x.size(1)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Ensure KLD is averaged over batch size for stability if mse is mean
    kld /= x.size(0) # Average KLD per sample in batch
    return mse + beta * kld

if __name__ == '__main__':
    # Define paths relative to project root (assuming script is run from project root)
    EXPRESSION_FILE = 'data/misc/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    OUTPUT_EMBEDDING_FILE = 'data/embeddings/final_vae_cell_embeddings.npy'
    OUTPUT_MODEL_FILE = 'data/embeddings/cell_vae_weights.pth'
    OUTPUT_CELL_LIST_FILE = 'data/embeddings/final_vae_cell_names.txt'
    # Check if input file exists
    if not os.path.exists(EXPRESSION_FILE):
        print(f"FATAL ERROR: Expression file not found at {EXPRESSION_FILE}")
        sys.exit(1)
        
    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDING_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_MODEL_FILE), exist_ok=True)


    # Hyperparameters
    LATENT_DIM = 512
    LEARNING_RATE = 7e-4
    EPOCHS = 120
    BATCH_SIZE = 64
    DROPOUT_RATE = 0.4
    KL_ANNEAL_EPOCHS = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load expression data
    print(f"Loading expression data from {EXPRESSION_FILE}...")
    try:
        df_expr = pd.read_csv(EXPRESSION_FILE, index_col=0)
        expression_data = df_expr.values.astype(np.float32)
        print(f"  > Loaded expression data with shape: {expression_data.shape}")
    except Exception as e:
        print(f"FATAL ERROR loading expression data: {e}")
        sys.exit(1)

    # Normalize
    print("Normalizing data using StandardScaler...")
    scaler = StandardScaler()
    expression_data_scaled = scaler.fit_transform(expression_data)

    # Train/val split
    print("Splitting data into train/validation sets...")
    X_train, X_val = train_test_split(expression_data_scaled, test_size=0.15, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=BATCH_SIZE)
    print(f"  > Train size: {len(X_train)}, Validation size: {len(X_val)}")


    # Model setup
    input_dim = expression_data_scaled.shape[1]
    # Architecture definition
    dims = [input_dim, 10000, 5000, 2048, 1024, LATENT_DIM]
    print(f"Initializing VAE model with architecture: {dims}")
    model = CellLineVAE(dims, dropout_rate=DROPOUT_RATE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=35, gamma=0.1) # LR scheduler

    print("\n--- Training VAE ---")
    best_val_loss = float('inf') # For saving the best model based on validation loss

    for epoch in range(EPOCHS):
        model.train()
        train_loss_epoch = 0
        # Determine beta for KL annealing
        beta = min(1.0, (epoch + 1) / KL_ANNEAL_EPOCHS) # Start annealing from epoch 0

        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()

        scheduler.step() # Step the LR scheduler

        # Validation
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                # Use same beta for validation loss calculation for fair comparison
                val_loss_epoch += loss_function(recon_x, x, mu, logvar, beta=beta).item()

        avg_train_loss = train_loss_epoch / len(train_loader.dataset)
        avg_val_loss = val_loss_epoch / len(val_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Beta: {beta:.3f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
             best_val_loss = avg_val_loss
             torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
             # print(f"  > Saved new best model weights to: {OUTPUT_MODEL_FILE} (Val Loss: {best_val_loss:.4f})")


    print("\nTraining complete. Saving final outputs...")

    # --- Generate and Save Final Embeddings (using the best saved model) ---
    print(f"Reloading best model weights from {OUTPUT_MODEL_FILE} for final embedding generation...")
    # Load the best weights back into the model
    try:
        model.load_state_dict(torch.load(OUTPUT_MODEL_FILE))
    except Exception as e:
         print(f"Warning: Could not reload best model weights. Using final epoch weights. Error: {e}")
         
    model.eval()
    # Use the full dataset (scaled) to generate final embeddings
    full_data_tensor = torch.tensor(expression_data_scaled, dtype=torch.float32).to(device)
    final_embeddings_mu = []
    # Process in batches if dataset is large to avoid OOM
    with torch.no_grad():
        for i in range(0, len(full_data_tensor), BATCH_SIZE):
             batch = full_data_tensor[i:i+BATCH_SIZE]
             mu, _ = model.encode(batch)
             final_embeddings_mu.append(mu.cpu())
             
    final_embeddings_mu = torch.cat(final_embeddings_mu, dim=0).numpy()

    np.save(OUTPUT_EMBEDDING_FILE, final_embeddings_mu)
    print(f"Saved embeddings to: {OUTPUT_EMBEDDING_FILE}")

    # --- vvv ADD THIS BLOCK vvv ---
    # Save the cell names (DepMap IDs) in the same order as the embeddings
    cell_names_in_order = df_expr.index.tolist()
    with open(OUTPUT_CELL_LIST_FILE, 'w') as f:
        for name in cell_names_in_order:
            f.write(f"{name}\n")
    print(f"Saved cell names/IDs to: {OUTPUT_CELL_LIST_FILE}")
    # --- ^^^ END BLOCK ^^^ ---

    # Save model weights
    print(f"Saved model weights to: {OUTPUT_MODEL_FILE}")