# scripts/build_graph_files.py

"""
Constructs the final graph files (node.dat, link.dat, info.dat) and a node
mapping file, following the proven single-pass logic from the source notebook.
This version robustly handles raw link files with 3 or 4 columns.
"""
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import os
import argparse

def main(args):
    # --- Configuration ---
    link_paths = {
        "cell-drug": args.cell_drug_file, # This now points to the filtered file
        "gene-drug": args.gene_drug_file, # This also points to the filtered file
        "gene-cell": os.path.join(args.raw_dir, "link_cell_gene_mutation.txt"), # This is not filtered by drug, so it stays
        "gene-gene": os.path.join(args.raw_dir, "link_gene_gene.txt") # This also stays
    }

    type_map = {
        "cell-drug": (0, 1, "cell-drug_interaction"),
        "gene-cell": (0, 2, "cell-gene_mutation"),
        "gene-drug": (2, 1, "gene-drug_inhibition"),
        "gene-gene": (2, 2, "gene-interacts_gene")
    }
    
    # --- Initialization ---
    node2id = {}
    node_types = {}
    next_id = 0
    all_edges = []
    link_info = {}
    link_type_counter = 0

    # --- Process all link files in a single pass ---
    print("--- Building graph files from raw sources ---")
    for name, path in link_paths.items():
        if not os.path.exists(path):
            print(f"  > Warning: File not found for {name} at {path}. Skipping.")
            continue

        print(f"  > Processing {name} links from {path}...")
        
        # --- START OF FIX: Robustly handle 3 or 4 column files ---
        df = pd.read_csv(path, sep="\t", header=None)
        if df.shape[1] == 3:
            df.columns = ["src", "tgt", "weight"]
        elif df.shape[1] == 4:
            df.columns = ["src", "tgt", "type_from_file", "weight"]
        else:
            print(f"  > Warning: Skipping file {path} due to unexpected number of columns ({df.shape[1]}).")
            continue
        # --- END OF FIX ---

        src_type, tgt_type, relation_name = type_map[name]

        df["src"] = df["src"].astype(str).str.strip()
        df["tgt"] = df["tgt"].astype(str).str.strip()
        if src_type == 1: df["src"] = df["src"].str.upper()
        if tgt_type == 1: df["tgt"] = df["tgt"].str.upper()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  - {name}"):
            src, tgt = row["src"], row["tgt"]

            for node, ntype in [(src, src_type), (tgt, tgt_type)]:
                if node not in node2id:
                    node2id[node] = next_id
                    node_types[node] = ntype
                    next_id += 1

            src_id = node2id[src]
            tgt_id = node2id[tgt]
            weight = row["weight"]

            # Use our own consistent link type counter
            all_edges.append(f"{src_id}\t{tgt_id}\t{link_type_counter}\t{weight}")
        
        type_names = {0: "cell", 1: "drug", 2: "gene"}
        link_info[str(link_type_counter)] = [ 
            type_names[src_type],
            type_names[tgt_type],
            relation_name,
            len(df)
        ]
        link_type_counter += 1

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Save node.dat and mappings ---
    node_dat_path = os.path.join(args.output_dir, "node.dat")
    map_path = os.path.join(args.output_dir, "node_mappings.json")
    with open(node_dat_path, "w") as f:
        sorted_nodes = sorted(node2id.items(), key=lambda item: item[1])
        for node, nid in sorted_nodes:
            f.write(f"{nid}\t{node}\t{node_types[node]}\n")
    with open(map_path, 'w') as f:
        json.dump(node2id, f, indent=4)

    # --- Save link.dat ---
    link_dat_path = os.path.join(args.output_dir, "link.dat")
    with open(link_dat_path, "w") as f:
        f.write("\n".join(all_edges))

    # --- Save info.dat ---
    type_counts = defaultdict(int)
    for ntype in node_types.values():
        type_counts[ntype] += 1
    info = {
        "dataset": "PRELUDE_CellDrugGene_Network",
        "node.dat": { "0": ["cell", type_counts[0]], "1": ["drug", type_counts[1]], "2": ["gene", type_counts[2]] },
        "link.dat": link_info
    }
    info_dat_path = os.path.join(args.output_dir, "info.dat")
    with open(info_dat_path, "w") as f:
        json.dump(info, f, indent=4)

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  > Wrote {len(node2id)} nodes to {node_dat_path}")
    print(f"  > Wrote mapping for {len(node2id)} nodes to {map_path}")
    print(f"  > Wrote {len(all_edges)} links to {link_dat_path}")
    print(f"  > Wrote metadata to {info_dat_path}")
    print("\nGraph construction complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build graph .dat files from filtered raw link files.")
    parser.add_argument('--cell-drug-file', default='data/raw/link_cell_drug_filtered.txt',
                        help='Path to the FINAL filtered cell-drug link file.')
    parser.add_argument('--gene-drug-file', default='data/raw/link_gene_drug_filtered.txt',
                        help='Path to the FINAL filtered gene-drug link file.')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory containing other raw link files (gene-gene, etc.).')
    parser.add_argument('--output-dir', default='data/processed',
                        help='Directory to save the output .dat files.')
    
    args = parser.parse_args()
    main(args)