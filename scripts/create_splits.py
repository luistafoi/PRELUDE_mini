# scripts/create_splits.py

import os
import random
import sys
import json
from collections import defaultdict

# --- Configuration ---
PROCESSED_DIR = "data/processed"

# Ratios for CELL Splitting (New Patients)
# 80% Training (Old Patients)
# 10% Inductive Validation (Tuning on New Patients)
# 10% Inductive Test (Final Exam on New Patients)
VALID_RATIO = 0.10
TEST_RATIO = 0.10

def load_node_info():
    """Loads node types and IDs."""
    node_file = os.path.join(PROCESSED_DIR, "node.dat")
    if not os.path.exists(node_file):
        print(f"FATAL: {node_file} not found.")
        sys.exit(1)

    nodes = {}
    try:
        with open(node_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                nodes[int(parts[0])] = int(parts[2]) # NID -> Type
    except ValueError:
        print("Error parsing node.dat. Ensure it is formatted as: ID Name TypeID")
        sys.exit(1)
    return nodes

def main():
    print("--- Creating INDUCTIVE Splits (Cell-Only Cold Start) ---")

    # 1. Load Nodes
    nodes = load_node_info()

    # Define Types (Adjust if your dataset differs)
    # We assume 0=Cell, 1=Drug based on your previous logs
    CELL_TYPE = 0
    DRUG_TYPE = 1

    # 2. Split Cells (The Patients)
    all_cells = [nid for nid, ntype in nodes.items() if ntype == CELL_TYPE]
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(all_cells)

    n_total = len(all_cells)
    n_valid = int(n_total * VALID_RATIO)
    n_test = int(n_total * TEST_RATIO)

    # Sets of Cell IDs
    valid_cells = set(all_cells[:n_valid])
    test_cells = set(all_cells[n_valid : n_valid + n_test])
    train_cells = set(all_cells[n_valid + n_test:])

    inductive_pool = valid_cells.union(test_cells)

    print(f"Total Cells: {n_total}")
    print(f"  > Training Cells (Old): {len(train_cells)}")
    print(f"  > Valid Cells (New):    {len(valid_cells)}")
    print(f"  > Test Cells (New):     {len(test_cells)}")

    # 3. Process Links
    link_file = os.path.join(PROCESSED_DIR, "link.dat")
    if not os.path.exists(link_file):
        print("FATAL: link.dat not found.")
        sys.exit(1)

    train_graph_links = []    # goes to train.dat (Structure)
    train_lp_links = []       # goes to train_lp_links.dat (Supervision)

    valid_inductive_links = []
    test_inductive_links = []

    # We also keep transductive splits for the Old Patients (sanity check)
    valid_trans_links = []
    test_trans_links = []

    print("Processing links...")
    with open(link_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            src, tgt = int(parts[0]), int(parts[1])
            edge_type = int(parts[2])
            # Read label/weight from 4th column (soft labels for Cell-Drug, weights for others)
            label = float(parts[3]) if len(parts) > 3 else 1.0

            src_type = nodes.get(src)
            tgt_type = nodes.get(tgt)

            # Check if Drug-Cell Interaction
            is_drug_cell = (src_type == CELL_TYPE and tgt_type == DRUG_TYPE) or \
                           (src_type == DRUG_TYPE and tgt_type == CELL_TYPE)

            # --- CASE A: Biological Links (Gene-Gene, Cell-Gene, Drug-Gene, Cell-Cell) ---
            if not is_drug_cell:
                # CRITICAL: We KEEP these for EVERYONE (including Inductive cells).
                # This preserves the "Gene Bridge" and Cell-Cell similarity edges.
                train_graph_links.append(line)
                continue

            # --- CASE B: Drug-Cell Links (The Answers) ---

            # Is this involved in an Inductive Cell?
            is_inductive = (src in inductive_pool) or (tgt in inductive_pool)

            if is_inductive:
                # 1. HIDE from Graph Structure (Do not append to train_graph_links)
                # 2. HIDE from Training Supervision (Do not append to train_lp_links)
                # 3. Assign to Inductive Splits (preserving label)
                if (src in valid_cells) or (tgt in valid_cells):
                    valid_inductive_links.append((src, tgt, label))
                else:
                    test_inductive_links.append((src, tgt, label))
            else:
                # It is an Old Patient (Transductive)
                # We split these edges 80/10/10 for standard training
                r = random.random()
                if r < 0.8:
                    train_graph_links.append(line)          # Visible in Graph
                    train_lp_links.append((src, tgt, label)) # Used for Loss
                elif r < 0.9:
                    valid_trans_links.append((src, tgt, label))
                else:
                    test_trans_links.append((src, tgt, label))

    # 4. Write Outputs
    print(f"Writing outputs to {PROCESSED_DIR}...")

    # A. The Structural Graph (train.dat)
    with open(os.path.join(PROCESSED_DIR, "train.dat"), 'w') as f:
        for line in train_graph_links:
            f.write(line)
    print(f"  > train.dat: {len(train_graph_links)} links (Includes Gene links for ALL cells)")

    # B. The Splits — preserving actual GMM labels
    def save_links(fname, links):
        path = os.path.join(PROCESSED_DIR, fname)
        with open(path, 'w') as f:
            for s, t, lbl in links:
                f.write(f"{s}\t{t}\t{lbl}\n")
        # Count label distribution (>0.5 = positive for soft labels)
        n_pos = sum(1 for _, _, lbl in links if lbl > 0.5)
        n_neg = sum(1 for _, _, lbl in links if lbl <= 0.5)
        n_soft = sum(1 for _, _, lbl in links if 0 < lbl < 1)
        extra = f", soft={n_soft}" if n_soft > 0 else ""
        print(f"  > {fname}: {len(links)} links (pos={n_pos}, neg={n_neg}{extra})")

    save_links("train_lp_links.dat", train_lp_links)
    save_links("valid_inductive_links.dat", valid_inductive_links)
    save_links("test_inductive_links.dat", test_inductive_links)
    save_links("valid_transductive_links.dat", valid_trans_links)
    save_links("test_transductive_links.dat", test_trans_links)

    # 5. Save cell split info for downstream use
    split_info = {
        'train_cells': sorted(train_cells),
        'valid_cells': sorted(valid_cells),
        'test_cells': sorted(test_cells),
    }
    split_path = os.path.join(PROCESSED_DIR, "cell_splits.json")
    with open(split_path, 'w') as f:
        json.dump(split_info, f)
    print(f"  > cell_splits.json: saved cell split assignments")

    print("Done.")

if __name__ == "__main__":
    main()
