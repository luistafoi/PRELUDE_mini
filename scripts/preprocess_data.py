# scripts/preprocess_data.py

"""
Data Preprocessing Script for the PRELUDE Project.

This script performs the initial data cleaning and filtering. It takes raw
drug sensitivity, gene dependency, and cell line metadata files as input,
filters them for high-quality data, and produces a single, cleaned file
of interactions ready for graph construction.

Example usage:
    python scripts/preprocess_data.py \
        --sensitivity-file path/to/sensitivity.csv \
        --dependency-file path/to/dependency.csv \
        --cell-meta-file path/to/cell_meta.csv \
        --output-file data/processed/cleaned_interactions.csv
"""

import pandas as pd
import argparse

def load_and_filter_sensitivity(file_path: str, z_score_threshold: float = -2.0) -> pd.DataFrame:
    """Loads and filters the drug sensitivity data."""
    print(f"Loading drug sensitivity data from {file_path}...")
    df = pd.read_csv(file_path)
    # Example filtering: keep only significant sensitivity scores
    df_filtered = df[df['z_score'] < z_score_threshold].copy()
    print(f"  > Filtered to {len(df_filtered)} significant drug-cell interactions.")
    return df_filtered[['cell_line_name', 'drug_name', 'z_score']]

def load_and_filter_dependencies(file_path: str, dependency_threshold: float = 0.5) -> pd.DataFrame:
    """Loads and filters the gene dependency data."""
    print(f"Loading gene dependency data from {file_path}...")
    df = pd.read_csv(file_path)
    # Example filtering: keep only strong dependencies
    df_filtered = df[df['dependency_score'] > dependency_threshold].copy()
    print(f"  > Filtered to {len(df_filtered)} significant gene-cell interactions.")
    return df_filtered[['cell_line_name', 'gene_name', 'dependency_score']]

def main(args):
    """Main execution function."""
    # Load and process each input file
    df_sens = load_and_filter_sensitivity(args.sensitivity_file)
    df_dep = load_and_filter_dependencies(args.dependency_file)
    
    # In a real script, one might also load cell metadata to filter for specific lineages,
    # or merge other interaction types like gene-drug and gene-gene interactions.
    # For this example, we'll assume the primary goal is combining sensitivity and dependency.
    
    # Standardize column names for concatenation
    df_sens.rename(columns={'drug_name': 'target_name', 'z_score': 'weight'}, inplace=True)
    df_sens['interaction_type'] = 'cell_drug'
    
    df_dep.rename(columns={'gene_name': 'target_name', 'dependency_score': 'weight'}, inplace=True)
    df_dep['interaction_type'] = 'cell_gene'
    
    # Combine into a single dataframe
    df_combined = pd.concat([df_sens, df_dep], ignore_index=True)
    
    # Save the cleaned, combined data
    df_combined.to_csv(args.output_file, index=False)
    print(f"\nSuccessfully saved {len(df_combined)} cleaned interactions to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw data for graph construction.")
    parser.add_argument('--sensitivity-file', required=True, help='Path to the raw drug sensitivity CSV file.')
    parser.add_argument('--dependency-file', required=True, help='Path to the raw gene dependency CSV file.')
    parser.add_argument('--cell-meta-file', required=True, help='Path to the cell line metadata file.')
    parser.add_argument('--output-file', default='data/processed/cleaned_interactions.csv', help='Path to save the cleaned output file.')
    
    args = parser.parse_args()
    main(args)