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

# --- VAE Model Definition ---
class CellLineVAE(nn.Module):
    def __init__(self, dims, dropout_rate=0.4):
        super(CellLineVAE, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims) - 1):
            self.encoder.add_module(f'enc_fc{i}', nn.Linear(dims[i], dims[i + 1]))
            self.encoder.add_module(f'enc_bn{i}', nn.BatchNorm1d(dims[i + 1]))
            self.encoder.add_module(f'enc_relu{i}', nn.ReLU())
            self.encoder.add_module(f'enc_dropout{i}', nn.Dropout(dropout_rate))

        self.fc_mu = nn.Linear(dims[-1], dims[-1])
        self.fc_logvar = nn.Linear(dims[-1], dims[-1])

        self.decoder = nn.Sequential()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder.add_module(f'dec_fc{i}', nn.Linear(dims[i], dims[i - 1]))
            self.decoder.add_module(f'dec_bn{i}', nn.BatchNorm1d(dims[i - 1]))
            self.decoder.add_module(f'dec_relu{i}', nn.ReLU())
            self.decoder.add_module(f'dec_dropout{i}', nn.Dropout(dropout_rate))

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
    mse = F.mse_loss(recon_x, x, reduction='mean') * x.size(1)  # scale for fair KLD comparison
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld

if __name__ == '__main__':
    # Paths
    EXPRESSION_FILE = 'data/embeddings/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    OUTPUT_EMBEDDING_FILE = 'data/embeddings/final_vae_cell_embeddings.npy'
    OUTPUT_MODEL_FILE = 'data/embeddings/cell_vae_weights.pth'

    # Hyperparameters
    LATENT_DIM = 256
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 64
    DROPOUT_RATE = 0.4
    KL_ANNEAL_EPOCHS = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load expression data
    df_expr = pd.read_csv(EXPRESSION_FILE, index_col=0)
    expression_data = df_expr.values.astype(np.float32)

    # Normalize
    scaler = StandardScaler()
    expression_data_scaled = scaler.fit_transform(expression_data)

    # Train/val split
    X_train, X_val = train_test_split(expression_data_scaled, test_size=0.15, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=BATCH_SIZE)

    # Model setup
    input_dim = expression_data_scaled.shape[1]
    dims = [input_dim, 10000, 5000, 1000, 500, LATENT_DIM]
    model = CellLineVAE(dims, dropout_rate=DROPOUT_RATE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=35, gamma=0.1)

    print("\n--- Training VAE ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)

        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                val_loss += loss_function(recon_x, x, mu, logvar, beta=beta).item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Beta: {beta:.3f} | Train Loss: {train_loss / len(train_loader.dataset):.4f} | Val Loss: {val_loss / len(val_loader.dataset):.4f}")

    print("\nTraining complete. Saving outputs...")

    # Save embeddings (mu)
    model.eval()
    full_data_tensor = torch.tensor(expression_data_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, _ = model.encode(full_data_tensor)
    np.save(OUTPUT_EMBEDDING_FILE, mu.cpu().numpy())
    print(f"Saved embeddings to: {OUTPUT_EMBEDDING_FILE}")

    # Save model weights
    torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
    print(f"Saved model weights to: {OUTPUT_MODEL_FILE}")