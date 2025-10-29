import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# --- Configuration ---
CHECKPOINTS_DIR = "checkpoints"
RUNS_TO_PLOT = {
    "Transductive (Baseline)": "prelude_mini_clean_v2", # Assuming this is the transductive run
    "Inductive (l2_d512_lrE-5)": "inductive_l2_d512_lrE-5" # Assuming this is an inductive run
}
OUTPUT_FILENAME = "validation_metrics_comparison.pdf" # Or .pdf, .svg etc.

# Metrics to plot and their column names in the CSV
METRICS_TO_PLOT = {
    "Validation AUC": "val_auc",
    "Validation F1": "val_f1",
    "Validation MRR": "val_mrr"
}

# Plotting styles (optional)
STYLES = ['-', '--', '-.', ':']
COLORS = plt.cm.viridis # Use a colormap for distinct colors

# --- Main Script ---
def plot_metrics():
    print(f"Generating plot comparing runs: {', '.join(RUNS_TO_PLOT.values())}")
    print(f"Reading logs from directory: {CHECKPOINTS_DIR}")

    # Create the plot figure - one subplot for each metric
    num_metrics = len(METRICS_TO_PLOT)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 4 * num_metrics), sharex=True)
    # Ensure axes is always a list/array, even if only one metric
    if num_metrics == 1:
        axes = [axes]

    run_found_count = 0
    plot_colors = COLORS(np.linspace(0, 1, len(RUNS_TO_PLOT)))

    for i, (display_name, run_name) in enumerate(RUNS_TO_PLOT.items()):
        log_file = os.path.join(CHECKPOINTS_DIR, f"{run_name}_log.csv")

        if not os.path.exists(log_file):
            print(f"  > Warning: Log file not found for run '{run_name}' at {log_file}. Skipping.")
            continue

        try:
            df = pd.read_csv(log_file)
            run_found_count += 1
            print(f"  > Plotting data for '{display_name}' ({len(df)} epochs)")

            # Plot each metric in its subplot
            for j, (metric_display_name, metric_col) in enumerate(METRICS_TO_PLOT.items()):
                if metric_col in df.columns:
                    axes[j].plot(
                        df['epoch'],
                        df[metric_col],
                        label=display_name,
                        linestyle=STYLES[i % len(STYLES)],
                        color=plot_colors[i],
                        linewidth=2
                    )
                    axes[j].set_ylabel(metric_display_name)
                    axes[j].grid(True, linestyle='--', alpha=0.6)
                    axes[j].legend()
                else:
                    print(f"    - Warning: Metric column '{metric_col}' not found in {log_file}")

        except Exception as e:
            print(f"  > Error reading or plotting {log_file}: {e}")

    if run_found_count == 0:
        print("\nError: No valid log files found for the specified runs. Cannot generate plot.")
        return

    # Final plot adjustments
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Validation Performance Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

    # Save the figure
    output_path = os.path.join(CHECKPOINTS_DIR, OUTPUT_FILENAME) # Save in checkpoints dir
    try:
        plt.savefig(output_path, dpi=300)
        print(f"\nPlot saved successfully to: {output_path}")
    except Exception as e:
        print(f"\nError saving plot: {e}")

    # Optionally display the plot
    # plt.show()


if __name__ == "__main__":
    # Add project root to path if needed (though not necessary for this script)
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plot_metrics()
