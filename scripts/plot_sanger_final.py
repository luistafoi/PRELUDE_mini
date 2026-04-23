import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import roc_auc_score

# --- CONFIG ---
RESULTS_DIR = "results/final_sanger_validation"
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "Figure_CrossDataset_Validation.pdf")

def get_readable_name(filename):
    if "S1" in filename: return "S1: Known Drugs\n(known cells)"
    if "S2" in filename: return "S2: New Drugs\n(known cells)"
    if "S3" in filename: return "S3: Known Drugs\n(NEW cells)"
    if "S4" in filename: return "S4: New Drugs\n(NEW cells)"
    return filename

def main():
    # 1. Find files
    pred_files = glob.glob(os.path.join(RESULTS_DIR, "*_preds.csv"))
    if not pred_files:
        print(f"No result files found in {RESULTS_DIR}")
        return

    data = []
    print("Calculating final metrics...")

    for f in pred_files:
        try:
            df = pd.read_csv(f)
            
            # --- FIX: Check for correct column name ---
            score_col = 'pred_prob' 
            if 'prediction_prob' in df.columns: score_col = 'prediction_prob'
            
            if score_col not in df.columns:
                print(f"Skipping {f} (Column '{score_col}' not found. Available: {df.columns.tolist()})")
                continue

            # Calculate AUC
            auc = roc_auc_score(df['label'], df[score_col])
            label = get_readable_name(os.path.basename(f))
            
            # Color Logic
            color = "#95a5a6" # Grey (Baseline)
            if "S3" in label: color = "#2ecc71" # Green (The Main Result - Inductive)
            if "S2" in label or "S4" in label: color = "#e74c3c" # Red (Chemical Generalization)
            if "S1" in label: color = "#3498db" # Blue (Transfer Learning)
            
            data.append({"Scenario": label, "AUC": auc, "Color": color})
            print(f"  > {label.replace(chr(10), ' ')}: {auc:.4f}")
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # 2. Plot
    if not data: return
    df_plot = pd.DataFrame(data).sort_values("Scenario")
    
    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    
    # Bar Plot
    bars = plt.bar(df_plot['Scenario'], df_plot['AUC'], color=df_plot['Color'], edgecolor='black', alpha=0.8)
    
    # Reference Line
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label="Random (0.5)")
    
    # Annotate Bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylim(0.40, 0.65) # Zoom slightly to emphasize differences
    plt.ylabel("ROC-AUC Score", fontsize=12)
    plt.title("Cross-Dataset Generalization (Broad → Sanger)", fontsize=14, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"\n✅ Final Figure saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()