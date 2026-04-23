"""Step 1: Resolve cell and drug IDs across PRISM, Sanger, and CCLE.

Creates master mapping tables:
  - data/processed_v2/master_cells.csv: ACH ID, names, which datasets, has expression
  - data/processed_v2/master_drugs.csv: canonical name, SMILES, which datasets
  - data/processed_v2/master_genes.csv: HUGO symbol, Entrez ID, has ESM embedding

Usage:
    python scripts/pipeline_v2/step1_resolve_ids.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

MISC = 'data/misc'
OUT = 'data/processed_v2'


def main():
    os.makedirs(OUT, exist_ok=True)

    # ==========================================
    # CELL MAPPING
    # ==========================================
    print("=== STEP 1a: Cell ID Resolution ===\n")

    model = pd.read_csv(f'{MISC}/Model.csv')

    # Build reverse lookups
    stripped_to_ach = {str(r['StrippedCellLineName']).upper(): r['ModelID'] for _, r in model.iterrows()}
    cellname_to_ach = {str(r['CellLineName']).upper(): r['ModelID'] for _, r in model.iterrows()}
    cosmic_to_ach = {}
    for _, r in model.iterrows():
        if pd.notna(r.get('COSMICID')):
            try:
                cosmic_to_ach[int(r['COSMICID'])] = r['ModelID']
            except (ValueError, TypeError):
                pass

    def resolve_cell(name, cosmic_id=None):
        if pd.notna(cosmic_id):
            try:
                ach = cosmic_to_ach.get(int(cosmic_id))
                if ach:
                    return ach
            except (ValueError, TypeError):
                pass
        n = str(name).upper().strip()
        c = n.replace('-', '').replace(' ', '').replace('/', '')
        return cellname_to_ach.get(n) or stripped_to_ach.get(n) or stripped_to_ach.get(c)

    # CCLE expression
    expr_idx = pd.read_csv(f'{MISC}/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv',
                           index_col=0, usecols=[0])
    ccle_cells = set(expr_idx.index)

    # PRISM dose-response
    prism = pd.read_csv(f'{MISC}/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv',
                        usecols=['depmap_id'], low_memory=False)
    prism_cells = set(prism['depmap_id'].dropna().unique())

    # Sanger
    sanger = pd.read_csv(f'{MISC}/PANCANCER_IC_Fri Jan 16 16_28_57 2026.csv')
    sanger_resolved = {}
    for _, row in sanger.iterrows():
        name = row['Cell Line Name']
        cosmic = row['Cosmic ID']
        ach = resolve_cell(name, cosmic)
        if ach:
            sanger_resolved[name] = ach
    sanger_cells = set(sanger_resolved.values())

    # Build master cell table
    all_ach = ccle_cells | prism_cells | sanger_cells
    cell_rows = []
    for ach in sorted(all_ach):
        info = model[model['ModelID'] == ach].iloc[0] if ach in model['ModelID'].values else None
        cell_rows.append({
            'ach_id': ach,
            'cell_line_name': info['CellLineName'] if info is not None else '',
            'stripped_name': info['StrippedCellLineName'] if info is not None else '',
            'cosmic_id': int(info['COSMICID']) if info is not None and pd.notna(info.get('COSMICID')) else '',
            'lineage': info['OncotreeLineage'] if info is not None else '',
            'has_ccle_expression': ach in ccle_cells,
            'in_prism': ach in prism_cells,
            'in_sanger': ach in sanger_cells,
            'in_graph': ach in ccle_cells and (ach in prism_cells or ach in sanger_cells),
        })

    cells_df = pd.DataFrame(cell_rows)
    cells_df.to_csv(f'{OUT}/master_cells.csv', index=False)

    graph_cells = cells_df[cells_df['in_graph']]
    train_cells = cells_df[cells_df['has_ccle_expression'] & cells_df['in_prism']]
    eval_cells = cells_df[cells_df['has_ccle_expression'] & cells_df['in_sanger']]
    both_cells = cells_df[cells_df['has_ccle_expression'] & cells_df['in_prism'] & cells_df['in_sanger']]

    print(f"  Total CCLE cells:    {cells_df['has_ccle_expression'].sum()}")
    print(f"  PRISM cells:         {cells_df['in_prism'].sum()}")
    print(f"  Sanger cells:        {cells_df['in_sanger'].sum()}")
    print(f"  Graph cells:         {len(graph_cells)} (has expression + any drug response)")
    print(f"  Trainable:           {len(train_cells)} (expression + PRISM)")
    print(f"  Evaluable:           {len(eval_cells)} (expression + Sanger)")
    print(f"  Concordance:         {len(both_cells)} (all three)")
    print(f"  Saved: {OUT}/master_cells.csv")

    # Save Sanger name → ACH mapping for later use
    pd.DataFrame([{'sanger_name': k, 'ach_id': v} for k, v in sanger_resolved.items()]).to_csv(
        f'{OUT}/sanger_cell_name_to_ach.csv', index=False)

    # ==========================================
    # DRUG MAPPING
    # ==========================================
    print(f"\n=== STEP 1b: Drug ID Resolution ===\n")

    # PRISM drugs (with SMILES)
    prism_drugs_df = pd.read_csv(
        f'{MISC}/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv',
        usecols=['name', 'broad_id', 'smiles'], low_memory=False
    ).drop_duplicates(subset='name')

    prism_drug_map = {}
    for _, row in prism_drugs_df.iterrows():
        name = str(row['name']).upper().strip()
        prism_drug_map[name] = {
            'canonical_name': name,
            'prism_name': row['name'],
            'broad_id': row['broad_id'],
            'smiles': row['smiles'] if pd.notna(row['smiles']) else None,
        }

    # Sanger drugs (with resolved SMILES from step we already did)
    sanger_smiles_path = f'{MISC}/gdsc_drugs_with_smiles.csv'
    if os.path.exists(sanger_smiles_path):
        sanger_drugs_df = pd.read_csv(sanger_smiles_path)
    else:
        print("  WARNING: gdsc_drugs_with_smiles.csv not found. Run drug resolution first.")
        sanger_drugs_df = pd.DataFrame()

    sanger_drug_map = {}
    for _, row in sanger_drugs_df.iterrows():
        name = str(row['DRUG_NAME']).upper().strip()
        sanger_drug_map[name] = {
            'canonical_name': name,
            'sanger_name': row['DRUG_NAME'],
            'sanger_drug_id': row['DRUG_ID'],
            'smiles': row['SMILES'] if pd.notna(row.get('SMILES')) else None,
        }

    # Build fuzzy matching for overlap detection
    prism_clean = {k.replace('-', '').replace(' ', '').replace('(', '').replace(')', ''): k for k in prism_drug_map}
    sanger_clean = {k.replace('-', '').replace(' ', '').replace('(', '').replace(')', ''): k for k in sanger_drug_map}

    # Unified drug table
    drug_rows = []
    seen = set()

    # PRISM drugs first
    for name, info in prism_drug_map.items():
        clean = name.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
        in_sanger = name in sanger_drug_map or clean in sanger_clean
        sanger_name = sanger_drug_map.get(name, {}).get('sanger_name') or \
                      sanger_drug_map.get(sanger_clean.get(clean, ''), {}).get('sanger_name', '')
        smiles = info['smiles']
        # If no PRISM SMILES, try Sanger
        if not smiles and in_sanger:
            sanger_key = name if name in sanger_drug_map else sanger_clean.get(clean, '')
            smiles = sanger_drug_map.get(sanger_key, {}).get('smiles')

        drug_rows.append({
            'canonical_name': name,
            'prism_name': info['prism_name'],
            'sanger_name': sanger_name,
            'broad_id': info['broad_id'],
            'smiles': smiles,
            'in_prism': True,
            'in_sanger': in_sanger,
            'has_smiles': pd.notna(smiles) and smiles is not None,
        })
        seen.add(name)
        seen.add(clean)

    # Sanger-only drugs
    for name, info in sanger_drug_map.items():
        clean = name.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
        if name in seen or clean in seen:
            continue
        drug_rows.append({
            'canonical_name': name,
            'prism_name': '',
            'sanger_name': info['sanger_name'],
            'broad_id': '',
            'smiles': info['smiles'],
            'in_prism': False,
            'in_sanger': True,
            'has_smiles': info['smiles'] is not None,
        })

    drugs_df = pd.DataFrame(drug_rows)
    drugs_df.to_csv(f'{OUT}/master_drugs.csv', index=False)

    print(f"  Total unique drugs:  {len(drugs_df)}")
    print(f"  With SMILES:         {drugs_df['has_smiles'].sum()}")
    print(f"  In PRISM:            {drugs_df['in_prism'].sum()}")
    print(f"  In Sanger:           {drugs_df['in_sanger'].sum()}")
    print(f"  In both:             {(drugs_df['in_prism'] & drugs_df['in_sanger']).sum()}")
    print(f"  Graph drugs (SMILES): {drugs_df['has_smiles'].sum()}")
    print(f"  Saved: {OUT}/master_drugs.csv")

    # ==========================================
    # GENE MAPPING
    # ==========================================
    print(f"\n=== STEP 1c: Gene ID Resolution ===\n")

    # Collect genes from all edge sources
    # Mutations
    mut_cols = list(pd.read_csv(f'{MISC}/OmicsSomaticMutations_24Q2.csv', nrows=0).columns)
    hugo_col = 'HugoSymbol' if 'HugoSymbol' in mut_cols else None
    if hugo_col:
        mut_genes = set(pd.read_csv(f'{MISC}/OmicsSomaticMutations_24Q2.csv',
                                    usecols=[hugo_col], low_memory=False)[hugo_col].dropna().unique())
    else:
        mut_genes = set()
        print("  WARNING: Could not find HugoSymbol in mutations file")

    # DGIdb
    dgi = pd.read_csv(f'{MISC}/DGIdb_Interactions_Enriched_v2.csv')
    dgi_genes = set(dgi['gene_name'].dropna().unique())

    # Gene-Gene PPI
    gg = pd.read_csv(f'{MISC}/GeneGene_Interactions_EntrezMapped_Filtered.csv')
    gg_genes = set(gg['Gene1'].unique()) | set(gg['Gene2'].unique())

    all_genes = mut_genes | dgi_genes | gg_genes

    # ESM embeddings
    esm_path = 'data/embeddings/gene_embeddings_esm_by_symbol.pkl'
    esm_genes = set()
    if os.path.exists(esm_path):
        with open(esm_path, 'rb') as f:
            esm = pickle.load(f)
        esm_genes = set(esm.keys())

    # Expression column genes
    expr_cols = pd.read_csv(f'{MISC}/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv',
                            nrows=0).columns[1:]  # skip index
    expr_gene_symbols = set()
    for col in expr_cols:
        # Format: "SYMBOL (ENTREZ)"
        symbol = col.split(' (')[0] if ' (' in col else col
        expr_gene_symbols.add(symbol)

    gene_rows = []
    for gene in sorted(all_genes):
        gene_rows.append({
            'hugo_symbol': gene,
            'in_mutations': gene in mut_genes,
            'in_dgidb': gene in dgi_genes,
            'in_ppi': gene in gg_genes,
            'has_esm': gene in esm_genes,
            'in_expression': gene in expr_gene_symbols,
            'in_graph': gene in esm_genes,  # Need ESM embedding for feature
        })

    genes_df = pd.DataFrame(gene_rows)
    genes_df.to_csv(f'{OUT}/master_genes.csv', index=False)

    graph_genes = genes_df[genes_df['in_graph']]
    print(f"  Genes from mutations: {len(mut_genes):,}")
    print(f"  Genes from DGIdb:     {len(dgi_genes):,}")
    print(f"  Genes from PPI:       {len(gg_genes):,}")
    print(f"  Total unique:         {len(all_genes):,}")
    print(f"  With ESM embeddings:  {genes_df['has_esm'].sum():,}")
    print(f"  In expression:        {genes_df['in_expression'].sum():,}")
    print(f"  Graph genes:          {len(graph_genes):,}")
    print(f"  Saved: {OUT}/master_genes.csv")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print(f"MASTER TABLES COMPLETE")
    print(f"{'='*60}")
    print(f"\n  {OUT}/master_cells.csv: {len(cells_df)} cells ({len(graph_cells)} in graph)")
    print(f"  {OUT}/master_drugs.csv: {len(drugs_df)} drugs ({drugs_df['has_smiles'].sum()} with SMILES)")
    print(f"  {OUT}/master_genes.csv: {len(genes_df)} genes ({len(graph_genes)} in graph)")
    print(f"  {OUT}/sanger_cell_name_to_ach.csv: {len(sanger_resolved)} mappings")


if __name__ == '__main__':
    main()
