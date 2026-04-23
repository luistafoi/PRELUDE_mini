"""Step 4: Create train/val/test splits with zero data leakage.

Split strategy:
  1. Split CELLS 70/15/15 into train/val/test (inductive cold start)
  2. For training cells, split their Cell-Drug edges 80/10/10 (transductive)
  3. Build structural graph (train.dat) — EXCLUDES all held-out Cell-Drug edges
  4. Create Sanger evaluation subsets (S1-S4) based on cell/drug overlap

Leakage prevention:
  - Inductive val/test cells: ALL Cell-Drug edges removed from train.dat
  - Transductive val/test edges: those specific edges removed from train.dat
  - Cell-Gene and Cell-Cell edges: kept for ALL cells (biological, not prediction target)
  - Sanger subsets: defined by overlap with training cells/drugs, not by data split

Usage:
    python scripts/pipeline_v2/step4_create_splits.py
"""

import os
import sys
import json
import random
import pickle
import pandas as pd
import numpy as np

PROC_V2 = 'data/processed_v2'
MISC = 'data/misc'
SEED = 42


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(PROC_V2, exist_ok=True)

    # Load graph data
    with open(f'{PROC_V2}/id_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    cell_to_gid = mappings['cell_to_gid']
    drug_to_gid = mappings['drug_to_gid']
    gene_to_gid = mappings['gene_to_gid']

    gid_to_name = {}
    gid_to_type = {}
    with open(f'{PROC_V2}/node.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            gid, name, ntype = int(parts[0]), parts[1], int(parts[2])
            gid_to_name[gid] = name
            gid_to_type[gid] = ntype

    # Load labeled data
    prism_labeled = pd.read_csv(f'{PROC_V2}/prism_labeled.csv')
    sanger_labeled = pd.read_csv(f'{PROC_V2}/sanger_labeled.csv')

    # Load all edges from link.dat
    all_edges = []
    with open(f'{PROC_V2}/link.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            src, tgt, etype = int(parts[0]), int(parts[1]), int(parts[2])
            weight = float(parts[3])
            all_edges.append((src, tgt, etype, weight))

    # ==========================================
    # STEP 1: Split cells
    # ==========================================
    print("=" * 60)
    print("STEP 4a: Cell Split (Inductive)")
    print("=" * 60)

    # Only split cells that are in PRISM (trainable)
    master_cells = pd.read_csv(f'{PROC_V2}/master_cells.csv')
    trainable_cells = list(master_cells[master_cells['in_graph'] & master_cells['in_prism']]['ach_id'])
    eval_only_cells = list(master_cells[master_cells['in_graph'] & ~master_cells['in_prism'] & master_cells['in_sanger']]['ach_id'])

    random.shuffle(trainable_cells)
    n = len(trainable_cells)
    n_val = int(n * 0.15)
    n_test = int(n * 0.15)

    val_cells = set(trainable_cells[:n_val])
    test_cells = set(trainable_cells[n_val:n_val + n_test])
    train_cells = set(trainable_cells[n_val + n_test:])
    inductive_pool = val_cells | test_cells

    print(f"  Trainable cells (PRISM): {n}")
    print(f"    Training:    {len(train_cells)}")
    print(f"    Val (ind):   {len(val_cells)}")
    print(f"    Test (ind):  {len(test_cells)}")
    print(f"  Eval-only cells (Sanger only, no PRISM): {len(eval_only_cells)}")

    # ==========================================
    # STEP 2: Split edges
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 4b: Edge Split")
    print("=" * 60)

    # Separate edges by type
    cell_drug_edges = []  # type 0
    structural_edges = []  # types 1-4

    for src, tgt, etype, weight in all_edges:
        if etype == 0:
            cell_drug_edges.append((src, tgt, etype, weight))
        else:
            structural_edges.append((src, tgt, etype, weight))

    print(f"  Cell-Drug edges: {len(cell_drug_edges):,}")
    print(f"  Structural edges: {len(structural_edges):,}")

    # Split Cell-Drug edges
    train_lp = []          # Training supervision
    train_structural_cd = []  # Cell-Drug edges in structural graph (for GNN message passing)
    val_inductive = []     # Inductive val (held-out cells)
    test_inductive = []    # Inductive test (held-out cells)
    val_transductive = []  # Transductive val (held-out edges, known cells)
    test_transductive = [] # Transductive test (held-out edges, known cells)

    for src, tgt, etype, weight in cell_drug_edges:
        src_name = gid_to_name[src]
        tgt_name = gid_to_name[tgt]

        # Determine which is cell, which is drug
        if gid_to_type[src] == 0:  # cell
            cell_name = src_name
        else:
            cell_name = tgt_name

        is_inductive = cell_name in inductive_pool

        # Skip ambiguous labels (0.5) from supervision files
        # They stay in structural graph (train.dat) for GNN message passing
        is_ambiguous = (weight == 0.5)

        if is_inductive:
            # ALL Cell-Drug edges for inductive cells go to val or test
            # NONE go to structural graph or training
            if not is_ambiguous:
                if cell_name in val_cells:
                    val_inductive.append((src, tgt, weight))
                else:
                    test_inductive.append((src, tgt, weight))
        else:
            # Training cell: split edges 80/10/10
            r = random.random()
            if r < 0.80:
                if not is_ambiguous:
                    train_lp.append((src, tgt, weight))
                train_structural_cd.append((src, tgt, etype, weight))  # Always in GNN graph
            elif r < 0.90:
                if not is_ambiguous:
                    val_transductive.append((src, tgt, weight))
            else:
                if not is_ambiguous:
                    test_transductive.append((src, tgt, weight))

    print(f"\n  Training LP:           {len(train_lp):,}")
    print(f"  Training structural CD: {len(train_structural_cd):,} (in GNN message passing)")
    print(f"  Val inductive:         {len(val_inductive):,}")
    print(f"  Test inductive:        {len(test_inductive):,}")
    print(f"  Val transductive:      {len(val_transductive):,}")
    print(f"  Test transductive:     {len(test_transductive):,}")

    # Verify no leakage
    train_cell_set = train_cells
    val_ind_cells = set()
    for src, tgt, _ in val_inductive:
        c = gid_to_name[src] if gid_to_type[src] == 0 else gid_to_name[tgt]
        val_ind_cells.add(c)
    test_ind_cells = set()
    for src, tgt, _ in test_inductive:
        c = gid_to_name[src] if gid_to_type[src] == 0 else gid_to_name[tgt]
        test_ind_cells.add(c)

    train_lp_cells = set()
    for src, tgt, _ in train_lp:
        c = gid_to_name[src] if gid_to_type[src] == 0 else gid_to_name[tgt]
        train_lp_cells.add(c)

    leak_val = train_lp_cells & val_ind_cells
    leak_test = train_lp_cells & test_ind_cells
    print(f"\n  Leakage check:")
    print(f"    Train cells in val_ind: {len(leak_val)} {'CLEAN' if len(leak_val) == 0 else 'LEAK!'}")
    print(f"    Train cells in test_ind: {len(leak_test)} {'CLEAN' if len(leak_test) == 0 else 'LEAK!'}")

    # ==========================================
    # STEP 3: Write train.dat (structural graph)
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 4c: Write Structural Graph (train.dat)")
    print("=" * 60)

    # train.dat = ALL non-Cell-Drug edges + training Cell-Drug edges only
    train_dat_path = f'{PROC_V2}/train.dat'
    with open(train_dat_path, 'w') as f:
        # Structural edges (Cell-Gene, Drug-Gene, Gene-Gene, Cell-Cell) — ALL cells
        for src, tgt, etype, weight in structural_edges:
            f.write(f"{src}\t{tgt}\t{etype}\t{weight}\n")
        # Training Cell-Drug edges only (80% of training cells' edges)
        for src, tgt, etype, weight in train_structural_cd:
            f.write(f"{src}\t{tgt}\t{etype}\t{weight}\n")

    total_structural = len(structural_edges) + len(train_structural_cd)
    print(f"  Saved: {train_dat_path} ({total_structural:,} edges)")
    print(f"    Non-CD structural: {len(structural_edges):,}")
    print(f"    Training CD:       {len(train_structural_cd):,}")

    # ==========================================
    # STEP 4: Write LP split files
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 4d: Write LP Split Files")
    print("=" * 60)

    def write_split(path, edges):
        with open(path, 'w') as f:
            for src, tgt, weight in edges:
                f.write(f"{src}\t{tgt}\t{weight}\n")
        n_pos = sum(1 for _, _, w in edges if w > 0.5)
        n_neg = sum(1 for _, _, w in edges if w < 0.5)
        n_soft = sum(1 for _, _, w in edges if w == 0.5)
        print(f"  {os.path.basename(path)}: {len(edges):,} (pos={n_pos:,}, neg={n_neg:,}, soft={n_soft:,})")

    write_split(f'{PROC_V2}/train_lp_links.dat', train_lp)
    write_split(f'{PROC_V2}/valid_inductive_links.dat', val_inductive)
    write_split(f'{PROC_V2}/test_inductive_links.dat', test_inductive)
    write_split(f'{PROC_V2}/valid_transductive_links.dat', val_transductive)
    write_split(f'{PROC_V2}/test_transductive_links.dat', test_transductive)

    # ==========================================
    # STEP 5: Sanger Evaluation Subsets
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 4e: Sanger Evaluation Subsets (S1-S4)")
    print("=" * 60)

    # S1: Known cells + Known drugs (both in PRISM training)
    # S2: Known cells + New drugs (cell in PRISM, drug NOT in PRISM)
    # S3: New cells + Known drugs (cell NOT in PRISM train, drug in PRISM)
    # S4: New cells + New drugs (neither in PRISM training)

    # "Known cells" = training cells (not val/test inductive)
    # "Known drugs" = drugs that appear in training LP edges
    train_drugs = set()
    for src, tgt, _ in train_lp:
        if gid_to_type[src] == 1:  # drug
            train_drugs.add(gid_to_name[src])
        elif gid_to_type[tgt] == 1:
            train_drugs.add(gid_to_name[tgt])

    print(f"  Training cells: {len(train_cells)}")
    print(f"  Training drugs: {len(train_drugs)}")

    # Process Sanger pairs
    sanger_name_map = pd.read_csv(f'{PROC_V2}/sanger_cell_name_to_ach.csv')
    sanger_name_to_ach = dict(zip(sanger_name_map['sanger_name'], sanger_name_map['ach_id']))

    # Only use Sanger pairs that are NOT excluded (have clear labels)
    sanger_valid = sanger_labeled[~sanger_labeled['excluded']].copy()

    s1, s2, s3, s4 = [], [], [], []

    for _, row in sanger_valid.iterrows():
        cell_ach = row['cell_ach']
        drug_name = row['drug_name']

        # Must be in our graph
        cell_gid = cell_to_gid.get(cell_ach)
        drug_gid = drug_to_gid.get(drug_name)
        if cell_gid is None or drug_gid is None:
            continue

        cell_known = cell_ach in train_cells
        drug_known = drug_name in train_drugs

        pair = (cell_gid, drug_gid, float(row['label']))

        if cell_known and drug_known:
            s1.append(pair)
        elif cell_known and not drug_known:
            s2.append(pair)
        elif not cell_known and drug_known:
            s3.append(pair)
        else:
            s4.append(pair)

    # Write Sanger subsets
    for name, pairs in [('S1', s1), ('S2', s2), ('S3', s3), ('S4', s4)]:
        path = f'{PROC_V2}/sanger_{name}_links.dat'
        with open(path, 'w') as f:
            for src, tgt, label in pairs:
                f.write(f"{src}\t{tgt}\t{label}\n")
        n_pos = sum(1 for _, _, l in pairs if l > 0.5)
        n_neg = sum(1 for _, _, l in pairs if l < 0.5)

        # Count unique cells and drugs
        s_cells = set()
        s_drugs = set()
        for src, tgt, _ in pairs:
            if gid_to_type[src] == 0:
                s_cells.add(gid_to_name[src])
                s_drugs.add(gid_to_name[tgt])
            else:
                s_drugs.add(gid_to_name[src])
                s_cells.add(gid_to_name[tgt])

        desc = {
            'S1': 'Known cells + Known drugs',
            'S2': 'Known cells + New drugs',
            'S3': 'New cells + Known drugs',
            'S4': 'New cells + New drugs',
        }
        print(f"  {name} ({desc[name]}): {len(pairs):,} pairs (pos={n_pos:,}, neg={n_neg:,})")
        print(f"       Cells: {len(s_cells)}, Drugs: {len(s_drugs)}")

    # ==========================================
    # SAVE SPLIT CONFIG
    # ==========================================
    split_config = {
        'seed': SEED,
        'n_train_cells': len(train_cells),
        'n_val_cells': len(val_cells),
        'n_test_cells': len(test_cells),
        'n_eval_only_cells': len(eval_only_cells),
        'train_cells': sorted(train_cells),
        'val_cells': sorted(val_cells),
        'test_cells': sorted(test_cells),
        'n_train_drugs': len(train_drugs),
    }
    with open(f'{PROC_V2}/split_config.json', 'w') as f:
        json.dump(split_config, f, indent=2)
    print(f"\n  Saved: {PROC_V2}/split_config.json")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print("SPLITS COMPLETE — NO LEAKAGE")
    print(f"{'='*60}")
    print(f"\n  PRISM Splits:")
    print(f"    Train LP:         {len(train_lp):,}")
    print(f"    Val inductive:    {len(val_inductive):,}")
    print(f"    Test inductive:   {len(test_inductive):,}")
    print(f"    Val transductive: {len(val_transductive):,}")
    print(f"    Test transductive:{len(test_transductive):,}")
    print(f"\n  Sanger Evaluation:")
    print(f"    S1 (known/known): {len(s1):,}")
    print(f"    S2 (known/new):   {len(s2):,}")
    print(f"    S3 (new/known):   {len(s3):,}")
    print(f"    S4 (new/new):     {len(s4):,}")
    print(f"\n  Structural graph (train.dat): {total_structural:,} edges")


if __name__ == '__main__':
    main()
