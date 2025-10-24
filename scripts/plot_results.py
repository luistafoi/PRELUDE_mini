# scripts/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(log_file_path):
    """
    Reads a training log CSV and plots the training losses and validation AUC.
    """
    print(f"Reading log file from: {log_file_path}")
    df = pd.read_csv(log_file_path)

    # --- Create the Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Training and Validation Metrics', fontsize=16)

    # Subplot 1: Training Losses
    ax1.plot(df['epoch'], df['lp_loss'], 'o-', label='Link Prediction Loss', alpha=0.7)
    # Only plot RW loss if it was used (i.e., not all zeros or NaN)
    if 'rw_loss' in df.columns and df['rw_loss'].notna().any() and df['rw_loss'].sum() > 0:
        ax1.plot(df['epoch'], df['rw_loss'], 's-', label='Random Walk Loss', alpha=0.7)
    
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses per Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Validation AUC
    # Drop NaN values for clean plotting
    val_df = df.dropna(subset=['val_auc'])
    ax2.plot(val_df['epoch'], val_df['val_auc'], 'o-', label='Validation AUC', color='green')
    ax2.scatter(val_df['epoch'], val_df['val_auc'], color='green') # Highlight the points
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC Score')
    ax2.set_title('Validation AUC per Checkpoint')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Improve layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # --- Save the Figure ---
    output_filename = os.path.splitext(log_file_path)[0] + '_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from a log file.")
    parser.add_argument('log_file', type=str, help='Path to the training log CSV file.')
    
    args = parser.parse_args()
    plot_metrics(args.log_file)