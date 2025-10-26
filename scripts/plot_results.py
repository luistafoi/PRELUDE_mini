# scripts/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys # Added for path checks

def plot_metrics(log_file_path):
    """
    Reads a training log CSV and plots the training losses and validation AUC.
    """
    if not os.path.exists(log_file_path):
        print(f"FATAL ERROR: Log file not found at {log_file_path}")
        sys.exit(1)

    print(f"Reading log file from: {log_file_path}")
    try:
        df = pd.read_csv(log_file_path)
    except Exception as e:
        print(f"FATAL ERROR: Could not read log file {log_file_path}: {e}")
        sys.exit(1)

    # Check for essential columns
    if 'epoch' not in df.columns or 'lp_loss' not in df.columns or 'val_auc' not in df.columns:
        print("FATAL ERROR: Log file is missing required columns ('epoch', 'lp_loss', 'val_auc').")
        sys.exit(1)

    # --- Create the Plot ---
    # Use object-oriented interface for better control
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Training and Validation Metrics', fontsize=16)

    # Subplot 1: Training Losses
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['lp_loss'], marker='o', linestyle='-', label='Link Prediction Loss', alpha=0.8)

    # Only plot RW loss if it exists, is not all NaN, and has non-zero values
    plot_rw = False
    if 'rw_loss' in df.columns and df['rw_loss'].notna().any():
         # Check if there are any non-zero rw_loss values, ignoring NaN
         if (df['rw_loss'].dropna() != 0).any():
              ax1.plot(df['epoch'], df['rw_loss'], marker='s', linestyle='-', label='Random Walk Loss', alpha=0.8)
              plot_rw = True

    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses per Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    # Improve y-axis limits if RW loss is very large
    if plot_rw:
        # Potentially set different y-axes if scales differ significantly
        # Or clip large values for better visualization of LP loss trend
        pass


    # Subplot 2: Validation AUC
    ax2 = axes[1]
    # Drop rows where val_auc is NaN (epochs where validation wasn't run)
    val_df = df.dropna(subset=['val_auc'])
    if not val_df.empty:
        ax2.plot(val_df['epoch'], val_df['val_auc'], marker='o', linestyle='-', label='Validation AUC', color='green')
        # ax2.scatter(val_df['epoch'], val_df['val_auc'], color='green') # Scatter is redundant with marker='o'
    else:
        print("Warning: No validation AUC values found in the log file.")

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC Score')
    ax2.set_title('Validation AUC per Checkpoint')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(bottom=max(0.4, val_df['val_auc'].min() - 0.05) if not val_df.empty else 0.4, top=1.0) # Set reasonable y-axis limits

    # Improve layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap

    # --- Save the Figure ---
    # Create the output filename based on the input log file name
    output_filename = os.path.splitext(log_file_path)[0] + '_plot.png'
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved to: {output_filename}")
        # Optionally display the plot
        # plt.show()
    except Exception as e:
        print(f"Error saving plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from a log file.")
    # Make log_file argument required
    parser.add_argument('log_file', type=str, help='Path to the training log CSV file.')

    args = parser.parse_args()

    # Check if the log file exists before proceeding
    if not os.path.exists(args.log_file):
         print(f"FATAL ERROR: Specified log file not found at '{args.log_file}'")
    else:
        plot_metrics(args.log_file)