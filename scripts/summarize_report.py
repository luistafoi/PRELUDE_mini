# scripts/summarize_report.py

import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_file', type=str, required=True, help='Path to inspection CSV.')
    parser.add_argument('--score_threshold', type=float, default=0.85)
    parser.add_argument('--top_k_filter', type=int, default=10)
    args = parser.parse_args()

    print(f"--- Summarizing Report: {args.report_file} ---")
    
    try:
        df = pd.read_csv(args.report_file)
    except FileNotFoundError:
        print("Report file not found.")
        sys.exit(1)

    if df.empty:
        print("Report is empty.")
        return

    print(f"Total Predictions: {len(df)}")
    print(f"Average Prediction Score: {df['Score'].mean():.4f}")
    
    # 1. High Confidence Hits
    high_conf = df[df['Score'] >= args.score_threshold]
    print(f"\nHigh Confidence Hits (Score >= {args.score_threshold}): {len(high_conf)}")
    
    # 2. Top Drugs for New Patients
    print("\nTop Predicted Drugs for New Patients (by Mean Score):")
    top_drugs = df.groupby('Drug')['Score'].mean().sort_values(ascending=False).head(args.top_k_filter)
    print(top_drugs)

    # 3. Performance by Tissue (if Metadata exists)
    if 'Tissue' in df.columns and 'Unknown' not in df['Tissue'].unique():
        print("\nAverage Score by Tissue Type:")
        print(df.groupby('Tissue')['Score'].mean().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()