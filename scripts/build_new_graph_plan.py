"""Compute node and edge lists for rebuilt PRISM+Sanger graph pipeline.

Resolves all IDs, finds maximum overlap, outputs summary of what the new graph looks like.

Usage:
    python scripts/build_new_graph_plan.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict

MISC = 'data/misc'


def load_cell_mapping():
    """Build master cell ID mapping from Model.csv."""
    model = pd.read_csv(f'{MISC}/Model.csv')
    cells = {}
    for _, row in model.iterrows():
        ach = row['ModelID']
        cells[ach] = {
            'ach': ach,
            'cell_line_name': str(row.get('CellLineName', '')),
            'stripped_name': str(row.get('StrippedCellLineName', '')).upper(),
            'cosmic_id': row.get('COSMICID'),
            'sanger_model_id': row.get('SangerModelID'),
            'lineage': row.get('OncotreeLineage', ''),
        }
    # Build reverse lookups
    name_to_ach = {}
    stripped_to_ach = {}
    cosmic_to_ach = {}
    for ach, info in cells.items():
        name_to_ach[info['cell_line_name'].upper()] = ach
        stripped_to_ach[info['stripped_name']] = ach
        if pd.notna(info['cosmic_id']):
            cosmic_to_ach[int(info['cosmic_id'])] = ach

    return cells, name_to_ach, stripped_to_ach, cosmic_to_ach


def resolve_sanger_cell(name, cosmic_id, name_to_ach, stripped_to_ach, cosmic_to_ach):
    """Map Sanger cell line name/cosmic to ACH ID."""
    n = str(name).upper()
    clean = n.replace('-', '').replace(' ', '').replace('/', '')
    # Try cosmic first (most reliable)
    if pd.notna(cosmic_id):
        ach = cosmic_to_ach.get(int(cosmic_id))
        if ach:
            return ach
    # Then name
    return name_to_ach.get(n) or stripped_to_ach.get(n) or stripped_to_ach.get(clean)


def load_drug_mapping():
    """Build master drug mapping across PRISM, Sanger, DGIdb."""
    # PRISM dose-response drugs (with SMILES)
    prism = pd.read_csv(f'{MISC}/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv',
                        usecols=['name', 'broad_id', 'smiles'], low_memory=False).drop_duplicates(subset='name')
    prism_drugs = {}
    for _, row in prism.iterrows():
        name = str(row['name']).upper().strip()
        prism_drugs[name] = {
            'name': row['name'],
            'broad_id': row['broad_id'],
            'smiles': row['smiles'],
            'source': 'prism',
        }

    # Sanger drugs (with resolved SMILES)
    sanger_smiles = pd.read_csv(f'{MISC}/gdsc_drugs_with_smiles.csv')
    sanger_drugs = {}
    for _, row in sanger_smiles.iterrows():
        name = str(row['DRUG_NAME']).upper().strip()
        sanger_drugs[name] = {
            'name': row['DRUG_NAME'],
            'drug_id': row['DRUG_ID'],
            'smiles': row['SMILES'] if pd.notna(row['SMILES']) else None,
            'source': 'sanger',
        }

    # Build unified drug list
    # Start with PRISM drugs (our training set)
    all_drugs = {}
    for name, info in prism_drugs.items():
        if pd.notna(info['smiles']):
            all_drugs[name] = info

    # Add Sanger-only drugs that have SMILES
    prism_upper = set(prism_drugs.keys())
    prism_clean = {k.replace('-', '').replace(' ', ''): k for k in prism_upper}
    for name, info in sanger_drugs.items():
        if info['smiles'] is None:
            continue
        clean = name.replace('-', '').replace(' ', '')
        if name not in prism_upper and clean not in prism_clean:
            all_drugs[name] = info

    return all_drugs, prism_drugs, sanger_drugs


def main():
    print("=" * 70)
    print("PRELUDE Graph Rebuild Plan")
    print("=" * 70)

    # --- Cell Mapping ---
    print("\n--- 1. CELL LINES ---")
    cells, name_to_ach, stripped_to_ach, cosmic_to_ach = load_cell_mapping()

    # CCLE expression
    expr_idx = pd.read_csv(f'{MISC}/24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv',
                           index_col=0, usecols=[0])
    ccle_cells = set(expr_idx.index)

    # PRISM dose-response cells
    prism = pd.read_csv(f'{MISC}/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv',
                        usecols=['depmap_id'], low_memory=False)
    prism_cells = set(prism['depmap_id'].dropna().unique())

    # Sanger cells
    sanger = pd.read_csv(f'{MISC}/PANCANCER_IC_Fri Jan 16 16_28_57 2026.csv')
    sanger['ach'] = sanger.apply(
        lambda r: resolve_sanger_cell(r['Cell Line Name'], r['Cosmic ID'],
                                      name_to_ach, stripped_to_ach, cosmic_to_ach), axis=1)
    sanger_cells = set(sanger['ach'].dropna().unique())

    # Compute sets
    train_cells = ccle_cells & prism_cells  # Have expression + PRISM labels
    eval_cells = ccle_cells & sanger_cells  # Have expression + Sanger labels
    all_graph_cells = ccle_cells & (prism_cells | sanger_cells)  # Any drug response
    overlap_cells = ccle_cells & prism_cells & sanger_cells  # All three

    print(f"  CCLE expression:     {len(ccle_cells):,}")
    print(f"  PRISM dose-response: {len(prism_cells):,}")
    print(f"  Sanger IC50:         {len(sanger_cells):,}")
    print(f"")
    print(f"  Can train (CCLE+PRISM):     {len(train_cells):,}")
    print(f"  Can eval (CCLE+Sanger):     {len(eval_cells):,}")
    print(f"  In both (concordance):      {len(overlap_cells):,}")
    print(f"  Total graph cells:          {len(all_graph_cells):,}")

    # --- Drug Mapping ---
    print("\n--- 2. DRUGS ---")
    all_drugs, prism_drugs, sanger_drugs = load_drug_mapping()

    prism_upper = set(prism_drugs.keys())
    sanger_upper = set(sanger_drugs.keys())

    # Compute drug overlap (accounting for name variants)
    prism_clean = {k.replace('-', '').replace(' ', ''): k for k in prism_upper}
    sanger_clean = {k.replace('-', '').replace(' ', ''): k for k in sanger_upper}
    name_match = prism_upper & sanger_upper
    fuzzy_extra = set(prism_clean.keys()) & set(sanger_clean.keys()) - {d.replace('-', '').replace(' ', '') for d in name_match}
    drug_overlap = len(name_match) + len(fuzzy_extra)

    # Drugs with SMILES
    prism_with_smiles = sum(1 for d in prism_drugs.values() if pd.notna(d['smiles']))
    sanger_with_smiles = sum(1 for d in sanger_drugs.values() if d['smiles'] is not None)

    print(f"  PRISM drugs:         {len(prism_drugs):,} ({prism_with_smiles} with SMILES)")
    print(f"  Sanger drugs:        {len(sanger_drugs):,} ({sanger_with_smiles} with SMILES)")
    print(f"  Overlap:             {drug_overlap}")
    print(f"  Total unique drugs:  {len(all_drugs):,} (with SMILES)")

    # --- Edges: Cell-Drug ---
    print("\n--- 3. CELL-DRUG EDGES (Response Data) ---")

    # PRISM: training labels
    prism_full = pd.read_csv(f'{MISC}/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv',
                             usecols=['depmap_id', 'name', 'auc', 'ic50'], low_memory=False)
    prism_trainable = prism_full[
        prism_full['depmap_id'].isin(train_cells) &
        prism_full['name'].str.upper().isin(all_drugs.keys()) &
        prism_full['auc'].notna()
    ]
    print(f"  PRISM (training):")
    print(f"    Measurements:  {len(prism_trainable):,}")
    print(f"    Cells:         {prism_trainable['depmap_id'].nunique()}")
    print(f"    Drugs:         {prism_trainable['name'].nunique()}")
    print(f"    Unique pairs:  {prism_trainable.groupby(['depmap_id', 'name']).ngroups:,}")

    # Sanger: evaluation labels
    sanger_evaluable = sanger[
        sanger['ach'].isin(eval_cells) &
        sanger['Drug Name'].str.upper().isin({k for k in all_drugs.keys()}) &
        sanger['AUC'].notna()
    ]
    print(f"  Sanger (evaluation):")
    print(f"    Measurements:  {len(sanger_evaluable):,}")
    print(f"    Cells:         {sanger_evaluable['ach'].nunique()}")
    print(f"    Drugs:         {sanger_evaluable['Drug Name'].nunique()}")
    print(f"    Unique pairs:  {sanger_evaluable.groupby(['ach', 'Drug Name']).ngroups:,}")

    # Concordance set
    overlap_prism = prism_full[prism_full['depmap_id'].isin(overlap_cells)]
    overlap_sanger = sanger[sanger['ach'].isin(overlap_cells)]
    print(f"  Concordance set (both PRISM+Sanger):")
    print(f"    Cells: {len(overlap_cells)}")

    # --- Edges: Cell-Gene (Mutations) ---
    print("\n--- 4. CELL-GENE EDGES (Mutations) ---")
    mut = pd.read_csv(f'{MISC}/OmicsSomaticMutations_24Q2.csv',
                      usecols=['ModelID', 'HugoSymbol'] if 'ModelID' in pd.read_csv(f'{MISC}/OmicsSomaticMutations_24Q2.csv', nrows=0).columns else None,
                      nrows=0)
    mut_cols = list(pd.read_csv(f'{MISC}/OmicsSomaticMutations_24Q2.csv', nrows=0).columns)
    model_col = 'ModelID' if 'ModelID' in mut_cols else [c for c in mut_cols if 'model' in c.lower()][0] if any('model' in c.lower() for c in mut_cols) else None
    hugo_col = 'HugoSymbol' if 'HugoSymbol' in mut_cols else [c for c in mut_cols if 'hugo' in c.lower()][0] if any('hugo' in c.lower() for c in mut_cols) else None

    if model_col and hugo_col:
        mut = pd.read_csv(f'{MISC}/OmicsSomaticMutations_24Q2.csv', usecols=[model_col, hugo_col], low_memory=False)
        mut_cells = set(mut[model_col].dropna().unique()) & all_graph_cells
        mut_genes = set(mut[hugo_col].dropna().unique())
        mut_in_graph = mut[mut[model_col].isin(all_graph_cells)]
        print(f"  Cells with mutations: {len(mut_cells)}")
        print(f"  Genes mutated:        {len(mut_genes)}")
        print(f"  Total edges:          {len(mut_in_graph):,}")
    else:
        print(f"  Could not find ModelID/HugoSymbol columns. Columns: {mut_cols[:10]}")

    # --- Edges: Drug-Gene (DGIdb) ---
    print("\n--- 5. DRUG-GENE EDGES (DGIdb) ---")
    dgi = pd.read_csv(f'{MISC}/DGIdb_Interactions_Enriched_v2.csv')
    dgi_drugs_in_graph = dgi[dgi['drug_name'].str.upper().isin(all_drugs.keys())]
    print(f"  Total DGIdb interactions:    {len(dgi):,}")
    print(f"  Matching our drug set:       {len(dgi_drugs_in_graph):,}")
    print(f"  Unique drugs with targets:   {dgi_drugs_in_graph['drug_name'].nunique()}")
    print(f"  Unique target genes:         {dgi_drugs_in_graph['gene_name'].nunique()}")

    # --- Edges: Gene-Gene (PPI) ---
    print("\n--- 6. GENE-GENE EDGES (PPI) ---")
    gg = pd.read_csv(f'{MISC}/GeneGene_Interactions_EntrezMapped_Filtered.csv')
    print(f"  Total PPI edges: {len(gg):,}")
    print(f"  Unique genes:    {len(set(gg['Gene1'].unique()) | set(gg['Gene2'].unique()))}")

    # --- Gene Nodes ---
    print("\n--- 7. GENE NODES ---")
    # Genes from all edge types
    mutation_genes = mut_genes if 'mut_genes' in dir() else set()
    dgi_genes = set(dgi_drugs_in_graph['gene_name'].unique())
    gg_genes = set(gg['Gene1'].unique()) | set(gg['Gene2'].unique())
    all_genes = mutation_genes | dgi_genes | gg_genes

    # Check ESM embedding coverage
    import pickle
    esm_path = 'data/embeddings/gene_embeddings_esm_by_symbol.pkl'
    if os.path.exists(esm_path):
        with open(esm_path, 'rb') as f:
            esm = pickle.load(f)
        esm_genes = set(esm.keys())
        genes_with_esm = all_genes & esm_genes
        print(f"  Genes from mutations:  {len(mutation_genes):,}")
        print(f"  Genes from DGIdb:      {len(dgi_genes):,}")
        print(f"  Genes from PPI:        {len(gg_genes):,}")
        print(f"  Total unique genes:    {len(all_genes):,}")
        print(f"  With ESM embeddings:   {len(genes_with_esm):,} ({100*len(genes_with_esm)/max(1,len(all_genes)):.1f}%)")
    else:
        print(f"  ESM embeddings not found at {esm_path}")

    # --- Cell-Cell Edges ---
    print("\n--- 8. CELL-CELL EDGES ---")
    print(f"  Computed from CCLE expression cosine similarity (top 1%)")
    print(f"  Cells available: {len(all_graph_cells)}")
    print(f"  (Recomputed during graph build)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("FINAL GRAPH SUMMARY")
    print("=" * 70)
    print(f"\n  NODES:")
    print(f"    Cells:  {len(all_graph_cells):,} ({len(train_cells):,} trainable, {len(eval_cells):,} evaluable)")
    print(f"    Drugs:  {len(all_drugs):,}")
    print(f"    Genes:  {len(genes_with_esm) if 'genes_with_esm' in dir() else '?':,}")
    print(f"    Total:  {len(all_graph_cells) + len(all_drugs) + (len(genes_with_esm) if 'genes_with_esm' in dir() else 0):,}")

    print(f"\n  EDGES:")
    print(f"    Cell-Drug (PRISM AUC, training):  {prism_trainable.groupby(['depmap_id', 'name']).ngroups:,} pairs")
    print(f"    Cell-Drug (Sanger, evaluation):   {sanger_evaluable.groupby(['ach', 'Drug Name']).ngroups:,} pairs")
    print(f"    Cell-Gene (mutations):            {len(mut_in_graph) if 'mut_in_graph' in dir() else '?':,}")
    print(f"    Drug-Gene (DGIdb):                {len(dgi_drugs_in_graph):,}")
    print(f"    Gene-Gene (PPI):                  {len(gg):,}")
    print(f"    Cell-Cell (similarity):           (computed at build time)")

    print(f"\n  TRAINING:")
    print(f"    Labels: PRISM dose-response AUC (continuous, 0-1)")
    print(f"    Cells:  {len(train_cells):,}")
    print(f"    Drugs:  {prism_trainable['name'].nunique()}")

    print(f"\n  EVALUATION:")
    print(f"    Labels: Sanger IC50/AUC")
    print(f"    Cells:  {len(eval_cells):,} ({len(overlap_cells)} overlap with training)")
    print(f"    Drugs:  {sanger_evaluable['Drug Name'].nunique()} ({drug_overlap} overlap with training)")

    print(f"\n  CONCORDANCE ANALYSIS:")
    print(f"    Cells in both: {len(overlap_cells)}")
    print(f"    Drugs in both: {drug_overlap}")
    print(f"    Pairs in both: ~{len(overlap_cells) * drug_overlap // 2:,} (estimated)")


if __name__ == '__main__':
    main()
