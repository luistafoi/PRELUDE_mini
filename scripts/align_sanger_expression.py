import pandas as pd
import numpy as np
import os
import re
import argparse

def normalize_string(s):
    """Normalizes strings to uppercase alphanumeric for consistent matching."""
    if pd.isna(s): return ""
    s = str(s).upper()
    return re.sub(r'[^A-Z0-9]', '', s)

def clean_gene_symbol(s):
    """
    Cleans gene symbols.
    Example: 'TP53 (7157)' -> 'TP53'
    Example: 'A1BG' -> 'A1BG'
    """
    if pd.isna(s): return ""
    return str(s).split(' ')[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanger_tpm', type=str, required=True, 
                        help='Path to the Sanger "wide format" RNA-seq CSV')
    parser.add_argument('--depmap_tpm', type=str, required=True, 
                        help='Path to DepMap Expression CSV (Template for gene order)')
    parser.add_argument('--output_path', type=str, default='data/sanger_validation/sanger_aligned_tpm.csv')
    args = parser.parse_args()

    print("1. Loading DepMap Template (Target Gene List)...")
    # Read just the header to get the target gene order
    # Assuming DepMap is (Cells x Genes) where index is cell_id
    df_depmap_template = pd.read_csv(args.depmap_tpm, index_col=0, nrows=2)
    
    # Clean target genes (remove Entrez IDs if present)
    target_genes_raw = df_depmap_template.columns.tolist()
    target_genes_clean = [clean_gene_symbol(g) for g in target_genes_raw]
    
    # Create a map to restore original DepMap column names later if needed
    clean_to_original_map = dict(zip(target_genes_clean, target_genes_raw))
    
    print(f"   Model expects {len(target_genes_clean)} genes.")

    print("\n2. Loading and Parsing Sanger Data...")
    # We read without header because the file has a complex multi-row header
    # Snippet Structure Analysis:
    # Row 0: model_id
    # Row 1: model_name (TARGET CELL NAMES)
    # Row 2: data_source
    # Row 3: headers [gene_symbol, ensembl_gene_id, gene_id, ...]
    # Row 4+: Data
    # Col 0: gene_symbol (TARGET GENES)
    # Col 1: ensembl_gene_id (DROP)
    # Col 2: gene_id (DROP)
    # Col 3+: Expression Values
    
    try:
        # Load entire file as object to handle mixed types in headers
        raw_df = pd.read_csv(args.sanger_tpm, header=None, low_memory=False)
    except Exception as e:
        print(f"FATAL ERROR loading Sanger file: {e}")
        return

    # A. Extract Cell Line Names (from Row 1, starting at Column 3)
    # Normalize them immediately to match your other scripts
    cell_names_raw = raw_df.iloc[1, 3:].values
    cell_names_clean = [normalize_string(x) for x in cell_names_raw]
    
    # B. Extract Gene Symbols (from Col 0, starting at Row 4)
    gene_symbols_raw = raw_df.iloc[4:, 0].values
    gene_symbols_clean = [clean_gene_symbol(x) for x in gene_symbols_raw]
    
    # C. Extract Expression Matrix (Row 4+, Col 3+)
    expr_values = raw_df.iloc[4:, 3:].values
    
    print(f"   Raw Dimensions Extracted:")
    print(f"     - Genes (Rows): {len(gene_symbols_clean)}")
    print(f"     - Cells (Cols): {len(cell_names_clean)}")
    
    print("\n3. Transposing to (Cells x Genes)...")
    # Convert to DataFrame: Rows=Cells, Cols=Genes
    # We must transpose the values matrix
    df_sanger = pd.DataFrame(expr_values.T, index=cell_names_clean, columns=gene_symbols_clean)
    
    # Convert to numeric, coercing errors (handling any potential header artifacts)
    df_sanger = df_sanger.apply(pd.to_numeric, errors='coerce')
    
    # Handle Duplicate Genes in Sanger (if any) by averaging them
    if df_sanger.columns.duplicated().any():
        print("   Warning: Duplicate gene symbols found in Sanger. Averaging...")
        df_sanger = df_sanger.groupby(level=0, axis=1).mean()

    # Handle Duplicate Cells (if any) by averaging them
    if df_sanger.index.duplicated().any():
        print("   Warning: Duplicate cell lines found. Averaging...")
        df_sanger = df_sanger.groupby(level=0).mean()

    print(f"   Transposed Shape: {df_sanger.shape}")

    print("\n4. Aligning Genes (Intersecting with DepMap)...")
    
    # Identify Overlap
    sanger_genes_set = set(df_sanger.columns)
    target_genes_set = set(target_genes_clean)
    
    overlap = sanger_genes_set.intersection(target_genes_set)
    missing_in_sanger = target_genes_set - sanger_genes_set
    extra_in_sanger = sanger_genes_set - target_genes_set
    
    print(f"   ------------------------------------------------")
    print(f"   ALIGNMENT STATISTICS:")
    print(f"   ------------------------------------------------")
    print(f"   Model Required Genes:     {len(target_genes_set)}")
    print(f"   Sanger Available Genes:   {len(sanger_genes_set)}")
    print(f"   ------------------------------------------------")
    print(f"   [FOUND] Intersection:     {len(overlap)} genes ({len(overlap)/len(target_genes_set):.1%} coverage)")
    print(f"   [MISSING] Filled with 0:  {len(missing_in_sanger)} genes")
    print(f"   [DROPPED] Extra genes:    {len(extra_in_sanger)} genes")
    print(f"   ------------------------------------------------")

    # Reindex forces the dataframe to match the target list exactly
    # 1. Fill NaN for missing genes with 0.0
    # 2. Drops extra genes automatically
    # 3. Sorts columns to match target order
    df_aligned = df_sanger.reindex(columns=target_genes_clean, fill_value=0.0)
    
    # Restore original DepMap column names (if they had Entrez IDs)
    # This ensures perfect compatibility with the VAE input expectations
    df_aligned.columns = [clean_to_original_map[c] for c in df_aligned.columns]

    print("\n5. Checking Data Scale...")
    # Check max value to guess if Log2 transform is needed
    max_val = df_aligned.max().max()
    print(f"   Max value in aligned data: {max_val:.4f}")
    
    if max_val > 20:
        print("   > Detected raw TPM values (Max > 20). Applying Log2(TPM+1) transform...")
        df_aligned = np.log2(df_aligned + 1)
    else:
        print("   > Data appears to be already log-transformed (Max <= 20). Keeping as is.")

    print(f"\n6. Saving aligned matrix to {args.output_path}...")
    df_aligned.to_csv(args.output_path)
    print("Done.")

if __name__ == "__main__":
    main()