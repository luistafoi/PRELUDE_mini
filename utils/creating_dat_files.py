import pandas as pd
import os
import re
import json
from sklearn.model_selection import train_test_split # Still imported if you want to re-enable later
from collections import Counter
from tqdm import tqdm
import numpy as np

# --- Configuration ---
path_base = "/data/luis/HetGNN Data Processing/" # Your base path
# Consider a new output directory name to distinguish this version
output_hgb_dataset_dir = os.path.join(path_base, "CellDrugGene_HGB_Dataset_NoSplit_AugmentedNodes_v2") 

# Input files
file_cell_drug_interactions = os.path.join(path_base, "CellDrug_24Q2_Interactions_CID_Filtered.csv")
file_unique_drug_metadata = os.path.join(path_base, "CellDrug_24Q2_Unique_Drug_Metadata.csv") # Fallback basic metadata
file_drug_fingerprints = os.path.join(path_base, "CellDrug_24Q2_Unique_Drugs_with_PubChemFP.csv") # Preferred metadata with features

# Determine which drug metadata file to load
if os.path.exists(file_drug_fingerprints):
    print(f"Using drug metadata with fingerprints: {file_drug_fingerprints}")
    file_to_load_drug_metadata = file_drug_fingerprints
else:
    print(f"Fingerprint file {file_drug_fingerprints} not found. Using basic drug metadata: {file_unique_drug_metadata}")
    file_to_load_drug_metadata = file_unique_drug_metadata


file_depmap_gene_expression = os.path.join(path_base, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
file_cell_line_metadata = os.path.join(path_base, "Repurposing_Public_24Q2_Cell_Line_Meta_Data (2).csv")
file_dgidb_processed = os.path.join(path_base, "DGIdb_Interactions_CID_Filtered.csv")
file_gene_gene_processed = os.path.join(path_base, "GeneGene_Interactions_EntrezMapped_Filtered.csv")

# Mapping for printing loaded filenames
filename_map = {
    'df_interactions_cd': file_cell_drug_interactions,
    'df_drugs_meta': file_to_load_drug_metadata, # Uses the chosen drug metadata file
    'df_cell_meta': file_cell_line_metadata,
    'df_expression_raw': file_depmap_gene_expression,
    'df_dgidb': file_dgidb_processed,
    'df_gg': file_gene_gene_processed
}

# HGB Type IDs and Names
CELL_TYPE_ID = 0
DRUG_TYPE_ID = 1
GENE_TYPE_ID = 2
NODE_TYPES = {'cell': CELL_TYPE_ID, 'drug': DRUG_TYPE_ID, 'gene': GENE_TYPE_ID}
NODE_TYPE_ID_TO_NAME = {v: k for k, v in NODE_TYPES.items()}

# HGB Relation Type IDs and Names (7 types)
REL_CELL_DRUG_INTERACT = 0      # cell -> drug
REL_DRUG_CELL_INTERACT = 1      # drug -> cell
REL_CELL_GENE_EXPRESS = 2       # cell -> gene
REL_GENE_CELL_EXPRESSED_IN = 3  # gene -> cell
REL_DRUG_GENE_TARGET = 4        # drug -> gene
REL_GENE_DRUG_TARGETED_BY = 5   # gene -> drug
REL_GENE_GENE_INTERACT = 6      # gene <-> gene

RELATION_TYPES_INFO = { # For info.json
    str(REL_CELL_DRUG_INTERACT): ('cell', 'drug', 'cell-drug_interaction'),
    str(REL_DRUG_CELL_INTERACT): ('drug', 'cell', 'drug-cell_interaction_rev'),
    str(REL_CELL_GENE_EXPRESS): ('cell', 'gene', 'cell-gene_expression'),
    str(REL_GENE_CELL_EXPRESSED_IN): ('gene', 'cell', 'gene-cell_expressed_in_rev'),
    str(REL_DRUG_GENE_TARGET): ('drug', 'gene', 'drug-gene_target'),
    str(REL_GENE_DRUG_TARGETED_BY): ('gene', 'drug', 'gene-targeted_by_drug_rev'),
    str(REL_GENE_GENE_INTERACT): ('gene', 'gene', 'gene-interacts_gene')
}

# Column Names from your data files (as per your summary)
# Cell-Drug Interactions
CD_DRUG_ID_COL = 'PubChem_CID'
CD_CELL_ID_COL = 'row_id' # This will be mapped to DepMap_ID
CD_WEIGHT_COL = 'LMFI.normalized'

# Cell-Gene Interactions (Expression Matrix)
CG_CELL_ID_COL_TYPE = 'index' # Indicates DepMap_IDs are in the index
# Gene IDs are parsed from column headers "Symbol (EntrezID)"

# Drug-Gene Interactions
DG_DRUG_ID_COL = 'drug_pubchem_cid' # This was your "Cell identifier" typo, corrected to Drug
DG_GENE_ID_COL = 'gene_entrez_id'
DG_WEIGHT_COL = 'interaction_score'
DG_DRUG_NAME_COL = 'drug_name' # For augmenting drug nodes
DG_GENE_NAME_COL = 'gene_name'   # For augmenting gene nodes

# Gene-Gene Interactions
GG_GENE1_ID_COL = 'Gene1_EntrezID'
GG_GENE2_ID_COL = 'Gene2_EntrezID'
# Weight for GG is default 1.0

# Cell Line Metadata (for mapping row_id to DepMap_ID)
ROW_ID_COL_IN_INTERACTIONS = 'row_id' # From cell-drug interactions file
ROW_ID_COL_IN_METADATA = 'row_id'     # From cell line metadata file
DEPMAP_ID_COL_IN_METADATA = 'depmap_id' # From cell line metadata file
CCLE_NAME_COL_IN_METADATA = 'ccle_name' # From cell line metadata file
# --- End Configuration ---

def parse_gene_identifier_from_expr(gene_id_str_from_expr):
    match = re.match(r'^(.*)\s\((\d+)\)$', str(gene_id_str_from_expr))
    if match:
        return match.group(1), str(match.group(2)) # HUGO, Entrez
    # Fallback if parsing fails, though less ideal
    # print(f"Warning: Could not parse Entrez ID from gene column: {gene_id_str_from_expr}")
    return str(gene_id_str_from_expr), None 

if __name__ == "__main__":
    os.makedirs(output_hgb_dataset_dir, exist_ok=True)
    print(f"--- Preparing HGB Dataset in: {output_hgb_dataset_dir} ---")

    print("\n--- Loading Input Data Files ---")
    loaded_data = {}
    try:
        loaded_data['df_interactions_cd'] = pd.read_csv(file_cell_drug_interactions)
        loaded_data['df_drugs_meta'] = pd.read_csv(file_to_load_drug_metadata)
        loaded_data['df_drugs_meta']['PubChem_CID'] = loaded_data['df_drugs_meta']['PubChem_CID'].astype(str)
        
        loaded_data['df_cell_meta'] = pd.read_csv(file_cell_line_metadata)
        if DEPMAP_ID_COL_IN_METADATA in loaded_data['df_cell_meta'].columns:
             loaded_data['df_cell_meta'][DEPMAP_ID_COL_IN_METADATA] = loaded_data['df_cell_meta'][DEPMAP_ID_COL_IN_METADATA].astype(str)
        if ROW_ID_COL_IN_METADATA in loaded_data['df_cell_meta'].columns: # e.g. 'row_id'
            loaded_data['df_cell_meta'][ROW_ID_COL_IN_METADATA] = loaded_data['df_cell_meta'][ROW_ID_COL_IN_METADATA].astype(str)

        loaded_data['df_expression_raw'] = pd.read_csv(file_depmap_gene_expression, index_col=0) # index_col=0 for DepMap_IDs
        loaded_data['df_expression_raw'].index = loaded_data['df_expression_raw'].index.astype(str)
        loaded_data['df_expression_raw'].columns = [str(col) for col in loaded_data['df_expression_raw'].columns]
        
        loaded_data['df_dgidb'] = pd.read_csv(file_dgidb_processed)
        expected_dgidb_cols = [DG_DRUG_ID_COL, DG_DRUG_NAME_COL, DG_GENE_ID_COL, DG_GENE_NAME_COL, DG_WEIGHT_COL]
        missing_cols = [col for col in expected_dgidb_cols if col not in loaded_data['df_dgidb'].columns]
        if missing_cols:
            print(f"ERROR: Missing expected columns in {file_dgidb_processed}: {missing_cols}")
            print(f"       Required for DGIdb processing: {DG_DRUG_ID_COL}, {DG_DRUG_NAME_COL} (for node name), {DG_GENE_ID_COL}, {DG_GENE_NAME_COL} (for node name), {DG_WEIGHT_COL}")
            print(f"       Available columns: {loaded_data['df_dgidb'].columns.tolist()}")
            exit()
        loaded_data['df_dgidb'][DG_DRUG_ID_COL] = loaded_data['df_dgidb'][DG_DRUG_ID_COL].astype(str)
        loaded_data['df_dgidb'][DG_GENE_ID_COL] = loaded_data['df_dgidb'][DG_GENE_ID_COL].astype(str)
        
        loaded_data['df_gg'] = pd.read_csv(file_gene_gene_processed)
        loaded_data['df_gg'][GG_GENE1_ID_COL] = loaded_data['df_gg'][GG_GENE1_ID_COL].astype(str)
        loaded_data['df_gg'][GG_GENE2_ID_COL] = loaded_data['df_gg'][GG_GENE2_ID_COL].astype(str)
        
        for name_key, df_item in loaded_data.items():
            original_filename = filename_map.get(name_key)
            if original_filename: print(f"Loaded {os.path.basename(original_filename)} with shape {df_item.shape}")
            else: print(f"Loaded {name_key} with shape {df_item.shape} (Original filename not in map)")

    except FileNotFoundError as e: print(f"ERROR: Could not load input file: {e}. Please check paths."); exit()
    except KeyError as e: print(f"ERROR: A required column was not found during data loading: {e}. Check CSV headers and script config."); exit()
    except Exception as e: print(f"An unexpected error occurred during file loading: {e}"); exit()

    print("\n--- Consolidating and Defining Nodes ---")
    hgb_nodes_dict = {} 

    print("  Processing Cell Nodes (from interactions, cell metadata, and expression data)...")
    if not (ROW_ID_COL_IN_METADATA in loaded_data['df_cell_meta'].columns and DEPMAP_ID_COL_IN_METADATA in loaded_data['df_cell_meta'].columns):
        print(f"ERROR: Crucial mapping columns missing in cell metadata for DepMap_ID. Expected: '{ROW_ID_COL_IN_METADATA}', '{DEPMAP_ID_COL_IN_METADATA}'")
        exit()
    row_id_to_depmap_id_map = loaded_data['df_cell_meta'].dropna(subset=[ROW_ID_COL_IN_METADATA, DEPMAP_ID_COL_IN_METADATA]) \
                               .drop_duplicates(subset=[ROW_ID_COL_IN_METADATA], keep='first') \
                               .set_index(ROW_ID_COL_IN_METADATA)[DEPMAP_ID_COL_IN_METADATA]
    
    interaction_row_ids = loaded_data['df_interactions_cd'][CD_CELL_ID_COL].astype(str).unique()
    depmap_ids_from_interactions = set()
    for r_id in interaction_row_ids:
        mapped_id = row_id_to_depmap_id_map.get(r_id)
        if mapped_id: depmap_ids_from_interactions.add(str(mapped_id)) # Ensure string
    
    expression_depmap_ids = set(loaded_data['df_expression_raw'].index.map(str))
    common_depmap_ids_for_graph = list(depmap_ids_from_interactions.intersection(expression_depmap_ids))
    print(f"    Identified {len(common_depmap_ids_for_graph)} common cell lines (DepMap_IDs) for HGB graph.")

    depmap_id_to_display_name_map = {}
    if CCLE_NAME_COL_IN_METADATA in loaded_data['df_cell_meta'].columns:
        temp_map_ccle = loaded_data['df_cell_meta'].dropna(subset=[DEPMAP_ID_COL_IN_METADATA, CCLE_NAME_COL_IN_METADATA]) \
                                    .drop_duplicates(subset=[DEPMAP_ID_COL_IN_METADATA], keep='first') \
                                    .set_index(DEPMAP_ID_COL_IN_METADATA)[CCLE_NAME_COL_IN_METADATA]
        for depmap_id in common_depmap_ids_for_graph: # These are already strings
            depmap_id_to_display_name_map[depmap_id] = temp_map_ccle.get(depmap_id, depmap_id) # depmap_id itself as fallback name
    else:
        for depmap_id in common_depmap_ids_for_graph:
            depmap_id_to_display_name_map[depmap_id] = depmap_id 
            
    for depmap_id_str in common_depmap_ids_for_graph: # These are already strings
        node_key = (depmap_id_str, 'cell')
        if node_key not in hgb_nodes_dict:
            hgb_nodes_dict[node_key] = {'original_id': depmap_id_str, 
                                        'name': depmap_id_to_display_name_map.get(depmap_id_str, depmap_id_str), 
                                        'type_id': CELL_TYPE_ID, 'type_name': 'cell'}

    print("  Processing Drug Nodes (from primary drug metadata file)...")
    for _, row in loaded_data['df_drugs_meta'].iterrows():
        cid_str = str(row['PubChem_CID']) # Standardized ID
        node_key = (cid_str, 'drug')
        if node_key not in hgb_nodes_dict:
            hgb_nodes_dict[node_key] = {
                'original_id': cid_str, 'name': str(row['name']),
                'type_id': DRUG_TYPE_ID, 'type_name': 'drug',
                'smiles': row.get('SMILES'), 'short_broad_id': row.get('short_broad_id')
            }

    print("  Processing Gene Nodes (from gene expression file columns)...")
    df_expression_common = loaded_data['df_expression_raw'].loc[loaded_data['df_expression_raw'].index.isin(common_depmap_ids_for_graph)]
    gene_cols_from_expr = df_expression_common.columns.tolist()
    for gene_col_name_str in gene_cols_from_expr:
        hugo_symbol, entrez_id_str = parse_gene_identifier_from_expr(gene_col_name_str) # Standardized ID
        if entrez_id_str: # Will be None if parsing fails
            node_key = (entrez_id_str, 'gene')
            if node_key not in hgb_nodes_dict:
                hgb_nodes_dict[node_key] = {'original_id': entrez_id_str, 'name': hugo_symbol, 
                                            'type_id': GENE_TYPE_ID, 'type_name': 'gene'}
    
    print("  Augmenting with Drug/Gene Nodes from DGIdb interactions (if missing)...")
    new_drugs_from_dgidb = 0
    new_genes_from_dgidb = 0
    for _, row in tqdm(loaded_data['df_dgidb'].iterrows(), total=len(loaded_data['df_dgidb']), desc="Checking DGIdb nodes"):
        drug_cid_dgidb = str(row[DG_DRUG_ID_COL]) # Standardized ID
        drug_name_dgidb = str(row[DG_DRUG_NAME_COL]) if pd.notna(row[DG_DRUG_NAME_COL]) else drug_cid_dgidb
        drug_node_key = (drug_cid_dgidb, 'drug')
        if drug_node_key not in hgb_nodes_dict:
            hgb_nodes_dict[drug_node_key] = {
                'original_id': drug_cid_dgidb, 'name': drug_name_dgidb,
                'type_id': DRUG_TYPE_ID, 'type_name': 'drug'
            }
            new_drugs_from_dgidb += 1

        gene_entrez_dgidb = str(row[DG_GENE_ID_COL]) # Standardized ID
        gene_name_dgidb = str(row[DG_GENE_NAME_COL]) if pd.notna(row[DG_GENE_NAME_COL]) else gene_entrez_dgidb
        gene_node_key = (gene_entrez_dgidb, 'gene')
        if gene_node_key not in hgb_nodes_dict:
            hgb_nodes_dict[gene_node_key] = {
                'original_id': gene_entrez_dgidb, 'name': gene_name_dgidb,
                'type_id': GENE_TYPE_ID, 'type_name': 'gene'
            }
            new_genes_from_dgidb += 1
            
    if new_drugs_from_dgidb > 0: print(f"    Added {new_drugs_from_dgidb} new drug nodes from DGIdb.")
    if new_genes_from_dgidb > 0: print(f"    Added {new_genes_from_dgidb} new gene nodes from DGIdb.")

    hgb_nodes_list = list(hgb_nodes_dict.values())
    node_counts_summary = Counter(n['type_name'] for n in hgb_nodes_list)
    print(f"    Cell nodes defined: {node_counts_summary.get('cell', 0)}")
    print(f"    Drug nodes defined: {node_counts_summary.get('drug', 0)}")
    print(f"    Gene nodes defined: {node_counts_summary.get('gene', 0)}")
    print(f"  Total unique nodes identified for graph (after augmentation): {len(hgb_nodes_list)}")

    print("\n--- Creating node.dat and Global ID Map ---")
    hgb_nodes_list.sort(key=lambda x: (x['type_id'], str(x['original_id'])))
    original_id_to_global_hgb_id_map = {}
    node_dat_path = os.path.join(output_hgb_dataset_dir, "node.dat")
    with open(node_dat_path, 'w') as f_node:
        for global_id, node_info in enumerate(hgb_nodes_list):
            node_name_cleaned = str(node_info.get('name', node_info['original_id'])).replace('\t', ' ').replace('\n', ' ')
            f_node.write(f"{global_id}\t{node_name_cleaned}\t{node_info['type_id']}\n")
            original_id_to_global_hgb_id_map[(str(node_info['original_id']), node_info['type_name'])] = global_id
    print(f"  Written {len(hgb_nodes_list)} nodes to {node_dat_path}.")

    print("\n--- Preparing Links ---")
    all_hgb_links_tuples = []
    test_links_for_file = [] # Remains empty as per no-split requirement

    print("  Processing Cell-Drug links...")
    # Using CD_CELL_ID_COL which is 'row_id', then mapping to DepMap_ID
    loaded_data['df_interactions_cd']['mapped_depmap_id'] = loaded_data['df_interactions_cd'][CD_CELL_ID_COL].astype(str).map(row_id_to_depmap_id_map)
    # Filter interactions to only include cells that are part of the graph
    df_interactions_cd_filtered = loaded_data['df_interactions_cd'].dropna(subset=['mapped_depmap_id'])
    df_interactions_cd_filtered = df_interactions_cd_filtered[df_interactions_cd_filtered['mapped_depmap_id'].isin(common_depmap_ids_for_graph)].copy()
    df_interactions_cd_filtered[CD_DRUG_ID_COL] = df_interactions_cd_filtered[CD_DRUG_ID_COL].astype(str)
    
    cd_links_added_count = 0
    for _, row in tqdm(df_interactions_cd_filtered.iterrows(), total=len(df_interactions_cd_filtered), desc="Cell-Drug links"):
        cell_original_id = str(row['mapped_depmap_id']) # This is the DepMap_ID
        drug_original_id = str(row[CD_DRUG_ID_COL])   # This is PubChem_CID
        
        source_hgb_id = original_id_to_global_hgb_id_map.get((cell_original_id, 'cell'))
        target_hgb_id = original_id_to_global_hgb_id_map.get((drug_original_id, 'drug'))
        
        if source_hgb_id is not None and target_hgb_id is not None:
            weight = row[CD_WEIGHT_COL]
            if pd.isna(weight): continue # Skip if weight is NaN
            all_hgb_links_tuples.append((source_hgb_id, target_hgb_id, REL_CELL_DRUG_INTERACT, weight))
            all_hgb_links_tuples.append((target_hgb_id, source_hgb_id, REL_DRUG_CELL_INTERACT, weight))
            cd_links_added_count +=1
    print(f"    Added {cd_links_added_count} cell-drug interactions (and their reverses).")

    print("  Processing Cell-Gene Expression links...")
    cg_links_added_count = 0
    # df_expression_common uses DepMap_IDs as index and Entrez_IDs (parsed) as columns
    for cell_depmap_id_str, row_series in tqdm(df_expression_common.iterrows(), total=df_expression_common.shape[0], desc="Cell-Gene Links"):
        source_hgb_id = original_id_to_global_hgb_id_map.get((str(cell_depmap_id_str), 'cell'))
        if source_hgb_id is None: continue
        
        for gene_col_name_str, expression_value in row_series.items():
            _, gene_entrez_id_str = parse_gene_identifier_from_expr(gene_col_name_str)
            if gene_entrez_id_str:
                target_hgb_id = original_id_to_global_hgb_id_map.get((gene_entrez_id_str, 'gene'))
                if target_hgb_id is not None:
                    if pd.isna(expression_value): continue # Skip if weight is NaN
                    all_hgb_links_tuples.append((source_hgb_id, target_hgb_id, REL_CELL_GENE_EXPRESS, expression_value))
                    all_hgb_links_tuples.append((target_hgb_id, source_hgb_id, REL_GENE_CELL_EXPRESSED_IN, expression_value))
                    cg_links_added_count +=1 
    # Note: cg_links_added_count here counts pairs of forward/reverse links that are added.
    # If you want to count unique cell-gene connections, it would be cg_links_added_count / 2 (approx)
    print(f"    Added {cg_links_added_count} cell-gene expression links (raw count, includes forward/reverse pairs being added).")


    print("  Processing Drug-Gene Target links (from DGIdb)...")
    dgidb_links_added_count = 0
    for _, row in tqdm(loaded_data['df_dgidb'].iterrows(), total=len(loaded_data['df_dgidb']), desc="Drug-Gene Links"):
        drug_original_id = str(row[DG_DRUG_ID_COL]) # PubChem_CID
        gene_original_id = str(row[DG_GENE_ID_COL]) # Entrez_ID
        
        source_hgb_id = original_id_to_global_hgb_id_map.get((drug_original_id, 'drug'))
        target_hgb_id = original_id_to_global_hgb_id_map.get((gene_original_id, 'gene'))
        
        if source_hgb_id is not None and target_hgb_id is not None:
            weight = row.get(DG_WEIGHT_COL, 1.0) 
            if pd.isna(weight) or not isinstance(weight, (int, float)): weight = 1.0
            all_hgb_links_tuples.append((source_hgb_id, target_hgb_id, REL_DRUG_GENE_TARGET, weight))
            all_hgb_links_tuples.append((target_hgb_id, source_hgb_id, REL_GENE_DRUG_TARGETED_BY, weight))
            dgidb_links_added_count +=1
    print(f"    Added {dgidb_links_added_count} drug-gene interactions from DGIdb (and their reverses).")

    print("  Processing Gene-Gene links...")
    gg_links_added_count = 0
    for _, row in tqdm(loaded_data['df_gg'].iterrows(), total=len(loaded_data['df_gg']), desc="Gene-Gene Links"):
        gene1_original_id = str(row[GG_GENE1_ID_COL]) # Entrez_ID
        gene2_original_id = str(row[GG_GENE2_ID_COL]) # Entrez_ID
        
        source_hgb_id = original_id_to_global_hgb_id_map.get((gene1_original_id, 'gene'))
        target_hgb_id = original_id_to_global_hgb_id_map.get((gene2_original_id, 'gene'))
        
        if source_hgb_id is not None and target_hgb_id is not None and source_hgb_id != target_hgb_id: # Avoid self-loops
            weight = 1.0 # Default weight as per your summary
            all_hgb_links_tuples.append((source_hgb_id, target_hgb_id, REL_GENE_GENE_INTERACT, weight))
            all_hgb_links_tuples.append((target_hgb_id, source_hgb_id, REL_GENE_GENE_INTERACT, weight)) # Using same relation type for reverse
            gg_links_added_count += 1
    print(f"    Added {gg_links_added_count} gene-gene interactions (and their reverses, using same relation type).")


    print(f"  Total links before deduplication: {len(all_hgb_links_tuples)}")
    all_hgb_links_tuples = sorted(list(set(all_hgb_links_tuples))) # Remove duplicates and sort
    print(f"  Total links after deduplication and sorting: {len(all_hgb_links_tuples)}")


    print("\n--- Creating link.dat and link.dat.test ---")
    link_dat_path = os.path.join(output_hgb_dataset_dir, "link.dat")
    with open(link_dat_path, 'w') as f_link_train:
        for s, t, r, w in all_hgb_links_tuples:
            f_link_train.write(f"{s}\t{t}\t{r}\t{w:.6f}\n") # Format weight
    print(f"  Written {len(all_hgb_links_tuples)} links to {link_dat_path}.")

    link_dat_test_path = os.path.join(output_hgb_dataset_dir, "link.dat.test")
    with open(link_dat_test_path, 'w') as f_link_test: # Will create an empty file
        if test_links_for_file: # Should be empty now
             for s_test, t_test, r_test, w_test in test_links_for_file:
                 f_link_test.write(f"{s_test}\t{t_test}\t{r_test}\t{w_test:.6f}\n")
    print(f"  Written {len(test_links_for_file)} links to {link_dat_test_path} (expected to be empty).")

    print("\n--- Creating info.json ---")
    info_data = {"dataset": os.path.basename(output_hgb_dataset_dir), "node.dat": {}, "link.dat": {}, "link.dat.test": {}}
    node_counts_by_type_id_final = Counter(node_info['type_id'] for node_info in hgb_nodes_list)
    for type_id, count in sorted(node_counts_by_type_id_final.items()):
        info_data["node.dat"][str(type_id)] = [NODE_TYPE_ID_TO_NAME[type_id], count]

    link_counts_by_relation_id_train = Counter(r_link for _, _, r_link, _ in all_hgb_links_tuples) # Use different var name
    for rel_id_str, (s_type, t_type, rel_name) in RELATION_TYPES_INFO.items():
        count = link_counts_by_relation_id_train.get(int(rel_id_str), 0)
        if count > 0 : # Only add relation types present in link.dat
            info_data["link.dat"][rel_id_str] = [s_type, t_type, rel_name, count]
    
    if test_links_for_file: 
        link_counts_by_relation_id_test = Counter(r_link_test for _, _, r_link_test, _ in test_links_for_file) # Use different var name
        for rel_id_str_test, (s_type_test, t_type_test, rel_name_test) in RELATION_TYPES_INFO.items(): # Iterate RELATION_TYPES_INFO
            count_test = link_counts_by_relation_id_test.get(int(rel_id_str_test),0)
            if count_test > 0:
                 # Use the s_type_test, t_type_test, rel_name_test from RELATION_TYPES_INFO for consistency
                 info_data["link.dat.test"][rel_id_str_test] = [s_type_test, t_type_test, rel_name_test, count_test]
    else: # Ensure link.dat.test key exists even if empty
        info_data["link.dat.test"] = {}

    info_file_path = os.path.join(output_hgb_dataset_dir, "info.json") 
    with open(info_file_path, 'w') as f_info:
        json.dump(info_data, f_info, indent=4)
    print(f"  Written dataset info to {info_file_path}.")

    print("\n--- HGB Data File Generation Finished ---")
    print(f"Output directory: {output_hgb_dataset_dir}")
    print("Files created: node.dat, link.dat, link.dat.test, info.json")