# scripts/curate_raw_links.py

import pandas as pd
import numpy as np
import re
import json
import pickle
from tqdm import tqdm
import argparse
import os


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def normalize_drug_name(name: str) -> str:
    """Strip all non-alphanumeric characters for fuzzy drug name matching."""
    return re.sub(r'[^A-Z0-9]', '', str(name).upper().strip())


def build_entrez_to_hugo(misc_dir: str) -> dict:
    """Build Entrez GeneID -> HUGO symbol mapping from expression file header
    and mutation file, for maximum coverage."""
    mapping = {}

    # Source 1: Expression file header ("SYMBOL (ENTREZ)")
    expr_file = os.path.join(misc_dir,
                             "24Q2_OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
    if os.path.exists(expr_file):
        cols = pd.read_csv(expr_file, nrows=0).columns
        for col in cols[1:]:
            m = re.match(r'^(.*)\s\((\d+)\)$', col)
            if m:
                mapping[int(m.group(2))] = m.group(1).upper()

    # Source 2: Mutation file (HugoSymbol, EntrezGeneID)
    mut_file = os.path.join(misc_dir, "OmicsSomaticMutations_24Q2.csv")
    if os.path.exists(mut_file):
        df = pd.read_csv(mut_file, usecols=['HugoSymbol', 'EntrezGeneID'],
                         low_memory=False)
        df = df.dropna(subset=['HugoSymbol', 'EntrezGeneID']).drop_duplicates(
            subset=['EntrezGeneID'])
        for _, row in df.iterrows():
            eid = int(row['EntrezGeneID'])
            hugo = str(row['HugoSymbol']).upper().strip()
            if eid not in mapping:
                mapping[eid] = hugo

    return mapping


def load_gene_embeddings(emb_dir: str) -> set:
    """Return set of UPPERCASE HUGO symbols that have ESM embeddings."""
    path = os.path.join(emb_dir, "gene_embeddings_esm_by_symbol.pkl")
    with open(path, 'rb') as f:
        gd = pickle.load(f)
    return set(str(k).upper() for k in gd.keys())


def load_drug_embeddings(emb_dir: str) -> set:
    """Return set of UPPERCASE drug names that have embeddings."""
    path = os.path.join(emb_dir, "drugs_with_embeddings.csv")
    df = pd.read_csv(path, usecols=[0])
    return set(df.iloc[:, 0].str.strip().str.upper().unique())


def build_lmfi_drug_lookup(misc_dir: str):
    """Build lookup structures from the CID-filtered LMFI file.

    Returns
    -------
    name_set : set of UPPERCASE drug names
    cid_to_name : dict  PubChem CID (int) -> UPPERCASE name
    norm_to_name : dict  normalized name -> UPPERCASE name
    """
    path = os.path.join(misc_dir,
                        "CellDrug_24Q2_Interactions_CID_Filtered.csv")
    df = pd.read_csv(path, usecols=['name', 'PubChem_CID']).drop_duplicates(
        subset=['name'])
    df['name_upper'] = df['name'].str.strip().str.upper()
    df['cid_int'] = df['PubChem_CID'].astype(int)

    name_set = set(df['name_upper'])
    cid_to_name = dict(zip(df['cid_int'], df['name_upper']))
    norm_to_name = {normalize_drug_name(n): n for n in name_set}
    return name_set, cid_to_name, norm_to_name


# ---------------------------------------------------------------------------
# Cell-Gene mutation edges  (Phase 0C)
# ---------------------------------------------------------------------------

def curate_cell_gene_links(args, entrez_to_hugo, gene_emb_symbols):
    """Generates cell-gene links from mutations with pathogenicity scoring.

    Inclusion: HIGH impact + MODERATE with pathogenic signal.
    Weight: max(applicable pathogenicity scores) per mutation,
            then max per (cell, gene) pair.
    """
    print("\n--- Curating Cell-Gene Links (from Mutations) ---")

    mutation_file = os.path.join(args.misc_dir,
                                 "OmicsSomaticMutations_24Q2.csv")
    output_file = os.path.join(args.raw_dir, "link_cell_gene_mutation.txt")

    if not os.path.exists(mutation_file):
        print(f"  > Error: Mutation file not found at {mutation_file}.")
        return set()

    # Columns we need for scoring
    use_cols = ['ModelID', 'EntrezGeneID', 'LikelyLoF', 'Hotspot',
                'HessDriver', 'OncogeneHighImpact',
                'TumorSuppressorHighImpact', 'RevelScore', 'AMClass',
                'AMPathogenicity']

    print(f"  > Reading mutation data from: {mutation_file}")
    df = pd.read_csv(mutation_file, usecols=use_cols, low_memory=False)
    df.dropna(subset=['ModelID', 'EntrezGeneID'], inplace=True)
    df['EntrezGeneID'] = df['EntrezGeneID'].astype(int)

    # Map Entrez -> HUGO
    df['gene_name'] = df['EntrezGeneID'].map(entrez_to_hugo)
    df.dropna(subset=['gene_name'], inplace=True)
    df['gene_name'] = df['gene_name'].str.upper()

    # Keep only genes that have embeddings
    df = df[df['gene_name'].isin(gene_emb_symbols)].copy()

    print(f"  > Total mutations after gene mapping: {len(df):,}")

    # --- Compute per-mutation pathogenicity weight ---
    # Start with NaN; only qualifying mutations get a weight.
    df['weight'] = np.nan

    # Binary flags → fixed weights
    flag_weights = {
        'LikelyLoF': 1.0,
        'Hotspot': 0.9,
        'HessDriver': 0.9,
        'OncogeneHighImpact': 0.9,
        'TumorSuppressorHighImpact': 0.9,
    }
    for col, w in flag_weights.items():
        mask = df[col] == True
        df.loc[mask, 'weight'] = df.loc[mask, 'weight'].fillna(0).clip(lower=0)
        df.loc[mask, 'weight'] = np.maximum(df.loc[mask, 'weight'], w)

    # REVEL score >= 0.5 → use score directly
    revel_mask = df['RevelScore'].notna() & (df['RevelScore'] >= 0.5)
    df.loc[revel_mask, 'weight'] = np.fmax(
        df.loc[revel_mask, 'weight'].fillna(0), df.loc[revel_mask, 'RevelScore'])

    # AlphaMissense likely_pathogenic → use AMPathogenicity score
    am_mask = (df['AMClass'] == 'likely_pathogenic') & df['AMPathogenicity'].notna()
    df.loc[am_mask, 'weight'] = np.fmax(
        df.loc[am_mask, 'weight'].fillna(0), df.loc[am_mask, 'AMPathogenicity'])

    # Drop mutations that didn't qualify under any criterion
    df = df.dropna(subset=['weight'])
    print(f"  > Qualifying mutations (pathogenic signal): {len(df):,}")

    # Aggregate per (cell, gene): take the max weight
    agg = df.groupby(['ModelID', 'gene_name'])['weight'].max().reset_index()

    # Round to 4 decimal places for clean output
    agg['weight'] = agg['weight'].round(4)

    # Save
    agg.to_csv(output_file, sep='\t', index=False, header=False)
    print(f"  > Saved {len(agg):,} curated cell-gene links to: {output_file}")
    print(f"  > Unique cells: {agg['ModelID'].nunique():,}, "
          f"Unique genes: {agg['gene_name'].nunique():,}")

    # Return the set of genes that appear in cell-gene edges
    return set(agg['gene_name'].unique())


# ---------------------------------------------------------------------------
# Drug-Gene edges  (Phase 0D)
# ---------------------------------------------------------------------------

def curate_gene_drug_links(args, gene_emb_symbols, drug_emb_names):
    """Curates gene-drug links from DGIdb (inhibitory, with interaction_score)
    plus supplementary PubChem bioassay targets.

    Weight: log-transformed + min-max normalized interaction_score for DGIdb,
            binary 1.0 for PubChem fallback.
    """
    print("\n--- Curating Gene-Drug Links (DGIdb + PubChem) ---")

    dgidb_file = os.path.join(args.misc_dir,
                              "DGIdb_Interactions_Enriched_v2.csv")
    pubchem_file = os.path.join(args.misc_dir,
                                "pubchem_drug_gene_edges.json")
    output_file = os.path.join(args.raw_dir, "link_gene_drug_relation.txt")

    INHIBITORY_TYPES = {'inhibitor', 'blocker', 'negative modulator',
                        'inverse agonist'}

    # --- Build LMFI drug lookup for 3-layer matching ---
    lmfi_names, lmfi_cid_to_name, lmfi_norm_to_name = \
        build_lmfi_drug_lookup(args.misc_dir)

    def match_drug_to_lmfi(drug_name_upper, drug_cid):
        """Try to resolve a DGIdb drug to its LMFI canonical name."""
        # Layer 1: exact name
        if drug_name_upper in lmfi_names:
            return drug_name_upper
        # Layer 2: CID
        if pd.notna(drug_cid):
            cid_int = int(drug_cid)
            if cid_int in lmfi_cid_to_name:
                return lmfi_cid_to_name[cid_int]
        # Layer 3: normalized name
        norm = normalize_drug_name(drug_name_upper)
        if norm in lmfi_norm_to_name:
            return lmfi_norm_to_name[norm]
        return None

    # ------------------------------------------------------------------
    # Source 1: DGIdb
    # ------------------------------------------------------------------
    all_edges = []      # (gene_name, drug_name, weight)
    matched_drugs = set()  # track which LMFI drugs we've matched

    if os.path.exists(dgidb_file):
        print(f"  > Reading DGIdb data from: {dgidb_file}")
        df = pd.read_csv(dgidb_file, usecols=[
            'gene_name', 'drug_name', 'interaction_type',
            'interaction_score', 'drug_pubchem_cid'])

        df.dropna(subset=['interaction_type'], inplace=True)
        df['interaction_list'] = df['interaction_type'].str.lower().str.split(',')
        mask = df['interaction_list'].apply(
            lambda types: any(t.strip() in INHIBITORY_TYPES for t in types))
        df = df[mask].copy()
        print(f"  > Inhibitory interactions: {len(df):,}")

        # Clean names
        df['gene_name'] = df['gene_name'].str.strip().str.upper()
        df['drug_name_raw'] = df['drug_name'].str.strip().str.upper()

        # Filter to genes with embeddings
        df = df[df['gene_name'].isin(gene_emb_symbols)]

        # 3-layer drug matching
        df['lmfi_drug'] = df.apply(
            lambda r: match_drug_to_lmfi(r['drug_name_raw'], r['drug_pubchem_cid']),
            axis=1)
        df = df.dropna(subset=['lmfi_drug'])

        # Also filter: drug must have embeddings
        df = df[df['lmfi_drug'].isin(drug_emb_names)]

        # --- Normalize interaction_score ---
        # Log-transform then min-max to [0, 1]
        scores = df['interaction_score'].copy()
        has_score = scores.notna() & (scores > 0)

        if has_score.any():
            log_scores = np.log1p(scores[has_score])
            s_min, s_max = log_scores.min(), log_scores.max()
            if s_max > s_min:
                df.loc[has_score, 'weight'] = \
                    (log_scores - s_min) / (s_max - s_min)
            else:
                df.loc[has_score, 'weight'] = 1.0
            # Rows without score get weight 1.0 (assume full interaction)
            df['weight'] = df['weight'].fillna(1.0)
        else:
            df['weight'] = 1.0

        # Deduplicate per (gene, drug), taking max weight
        dedup = df.groupby(['gene_name', 'lmfi_drug']).agg(
            weight=('weight', 'max')).reset_index()
        dedup.rename(columns={'lmfi_drug': 'drug_name'}, inplace=True)

        matched_drugs = set(dedup['drug_name'].unique())
        all_edges.extend(
            dedup[['gene_name', 'drug_name', 'weight']].values.tolist())
        print(f"  > DGIdb edges: {len(dedup):,} "
              f"({len(matched_drugs)} drugs, "
              f"{dedup['gene_name'].nunique()} genes)")
    else:
        print(f"  > Warning: DGIdb file not found at {dgidb_file}")

    # ------------------------------------------------------------------
    # Source 2: PubChem bioassay (supplementary)
    # ------------------------------------------------------------------
    pubchem_count = 0
    if os.path.exists(pubchem_file):
        print(f"  > Reading PubChem bioassay targets from: {pubchem_file}")
        with open(pubchem_file) as f:
            pubchem_data = json.load(f)

        for entry in pubchem_data:
            drug_name = entry['name'].strip().upper()
            # Only add if drug was NOT already matched via DGIdb
            if drug_name in matched_drugs:
                continue
            # Must have embeddings
            if drug_name not in drug_emb_names:
                continue
            for gene in entry.get('valid_genes', []):
                gene_upper = gene.strip().upper()
                if gene_upper in gene_emb_symbols:
                    all_edges.append([gene_upper, drug_name, 1.0])
                    pubchem_count += 1
            if entry.get('valid_genes'):
                matched_drugs.add(drug_name)

        print(f"  > PubChem supplementary edges: {pubchem_count:,}")
    else:
        print(f"  > Warning: PubChem file not found at {pubchem_file}")

    # --- Save all edges ---
    if all_edges:
        out_df = pd.DataFrame(all_edges, columns=['gene_name', 'drug_name', 'weight'])
        # Deduplicate again (in case of overlap between sources)
        out_df = out_df.groupby(['gene_name', 'drug_name']).agg(
            weight=('weight', 'max')).reset_index()
        out_df['weight'] = out_df['weight'].round(4)
        out_df.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"  > Saved {len(out_df):,} total gene-drug links to: {output_file}")
        print(f"  > Unique drugs: {out_df['drug_name'].nunique():,}, "
              f"Unique genes: {out_df['gene_name'].nunique():,}")
        return set(out_df['gene_name'].unique())
    else:
        print("  > Warning: No gene-drug edges produced!")
        return set()


# ---------------------------------------------------------------------------
# Gene-Gene edges  (Phase 0E)
# ---------------------------------------------------------------------------

def process_gene_gene_links(args, valid_genes, gene_emb_symbols):
    """Processes gene-gene links, filtering to valid gene set.

    Valid genes = genes that have embeddings AND appear in at least one
    cell-gene or drug-gene edge.
    """
    print("\n--- Processing Gene-Gene Links ---")

    gene_gene_file = os.path.join(
        args.misc_dir, "GeneGene_Interactions_EntrezMapped_Filtered.csv")
    output_file = os.path.join(args.raw_dir, "link_gene_gene.txt")

    if not os.path.exists(gene_gene_file):
        print(f"  > Error: Gene-gene file not found at {gene_gene_file}.")
        return

    print(f"  > Reading gene-gene data from: {gene_gene_file}")
    df = pd.read_csv(gene_gene_file)
    df.rename(columns={'Gene1': 'gene1', 'Gene2': 'gene2'}, inplace=True,
              errors='ignore')
    df.dropna(subset=['gene1', 'gene2'], inplace=True)
    df['gene1'] = df['gene1'].astype(str).str.strip().str.upper()
    df['gene2'] = df['gene2'].astype(str).str.strip().str.upper()

    # Filter: both genes must have embeddings AND be in valid_genes
    # (valid_genes = genes with at least one cell or drug edge)
    allowed = valid_genes & gene_emb_symbols
    print(f"  > Allowed genes (have embeddings + cell/drug edge): {len(allowed):,}")

    df_filtered = df[df['gene1'].isin(allowed) & df['gene2'].isin(allowed)]
    # Remove self-loops
    df_filtered = df_filtered[df_filtered['gene1'] != df_filtered['gene2']]

    # Create bidirectional edges with weight 1
    link_set = set()
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered),
                       desc="  - Creating bidirectional links"):
        g1, g2 = row['gene1'], row['gene2']
        link_set.add(f"{g1}\t{g2}\t1")
        link_set.add(f"{g2}\t{g1}\t1")

    with open(output_file, 'w') as f:
        f.write("\n".join(sorted(link_set)))

    print(f"  > Saved {len(link_set):,} bidirectional gene-gene links to: "
          f"{output_file}")

    # Count unique genes in GG edges
    gg_genes = set()
    for line in link_set:
        parts = line.split('\t')
        gg_genes.add(parts[0])
        gg_genes.add(parts[1])
    print(f"  > Unique genes in Gene-Gene edges: {len(gg_genes):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Curate all raw link files for the PRELUDE graph.")
    parser.add_argument('--misc-dir', default='data/misc',
                        help='Directory with original source data files.')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory to save curated raw link files.')
    parser.add_argument('--emb-dir', default='data/embeddings',
                        help='Directory with node embedding files.')

    args = parser.parse_args()
    os.makedirs(args.raw_dir, exist_ok=True)

    print("=" * 60)
    print("PRELUDE — Raw Link File Curation")
    print("=" * 60)

    # --- Load shared resources ---
    print("\n--- Loading shared resources ---")
    entrez_to_hugo = build_entrez_to_hugo(args.misc_dir)
    print(f"  > Entrez->HUGO mapping: {len(entrez_to_hugo):,} genes")

    gene_emb_symbols = load_gene_embeddings(args.emb_dir)
    print(f"  > Gene embeddings: {len(gene_emb_symbols):,} genes")

    drug_emb_names = load_drug_embeddings(args.emb_dir)
    print(f"  > Drug embeddings: {len(drug_emb_names):,} drugs")

    # --- Curate edges (order matters: CG and DG first, then GG) ---

    # Cell-Gene: returns set of genes appearing in cell-gene edges
    cg_genes = curate_cell_gene_links(args, entrez_to_hugo, gene_emb_symbols)

    # Drug-Gene: returns set of genes appearing in drug-gene edges
    dg_genes = curate_gene_drug_links(args, gene_emb_symbols, drug_emb_names)

    # Valid genes = genes with at least one cell or drug edge
    valid_genes = cg_genes | dg_genes
    print(f"\n--- Valid gene set: {len(valid_genes):,} genes "
          f"(cell-gene: {len(cg_genes):,}, drug-gene: {len(dg_genes):,}) ---")

    # Gene-Gene: filtered to valid genes only
    process_gene_gene_links(args, valid_genes, gene_emb_symbols)

    print("\n" + "=" * 60)
    print("All raw link files have been curated.")
    print("=" * 60)
