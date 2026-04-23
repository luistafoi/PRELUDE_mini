#!/usr/bin/env python
"""
Graph Integrity Test Suite
--------------------------
Validates data consistency across all pipeline stages:
  1. Raw data files (data/raw/)
  2. Processed graph files (node.dat, link.dat, info.dat, train.dat)
  3. Split files (train_lp, valid, test)
  4. Precomputed neighbors (train_neighbors_preprocessed.pkl)
  5. Feature alignment (embeddings match node counts)
  6. Inductive isolation (no leakage of test cell drug edges)

Usage:
  python scripts/test_graph_integrity.py [--data_dir data/processed]
"""

import sys
import os
import json
import pickle
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"

failures = []
warnings = []


def check(condition, msg, warn_only=False):
    if condition:
        print(f"  [{PASS}] {msg}")
    elif warn_only:
        print(f"  [{WARN}] {msg}")
        warnings.append(msg)
    else:
        print(f"  [{FAIL}] {msg}")
        failures.append(msg)
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    # =========================================================================
    # STAGE 1: Core Graph Files
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Core Graph Files")
    print("=" * 60)

    # --- info.dat ---
    info_path = os.path.join(DATA_DIR, 'info.dat')
    check(os.path.exists(info_path), f"info.dat exists at {info_path}")
    with open(info_path) as f:
        info = json.load(f)

    node_type_counts = {}
    for tid_str, tinfo in info['node.dat'].items():
        node_type_counts[int(tid_str)] = tinfo[1]  # [name, count]
    edge_type_counts = {}
    for tid_str, tinfo in info['link.dat'].items():
        edge_type_counts[int(tid_str)] = tinfo[3]  # [src, tgt, name, count]

    expected_total_nodes = sum(node_type_counts.values())
    expected_total_edges = sum(edge_type_counts.values())
    print(f"  [{INFO}] info.dat: {len(node_type_counts)} node types, {len(edge_type_counts)} edge types")
    print(f"  [{INFO}] Expected: {expected_total_nodes} nodes, {expected_total_edges} edges")

    # --- node.dat ---
    node_path = os.path.join(DATA_DIR, 'node.dat')
    check(os.path.exists(node_path), "node.dat exists")

    nodes = {}  # gid -> (name, type)
    node_types = {}  # gid -> type
    node_names = {}  # gid -> name
    name_to_gid = {}
    with open(node_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                gid, name, ntype = int(parts[0]), parts[1], int(parts[2])
                nodes[gid] = (name, ntype)
                node_types[gid] = ntype
                node_names[gid] = name
                name_to_gid[name] = gid

    check(len(nodes) == expected_total_nodes,
          f"node.dat has {len(nodes)} nodes (expected {expected_total_nodes})")

    # Check sequential IDs
    max_id = max(nodes.keys())
    min_id = min(nodes.keys())
    check(min_id == 0, f"Node IDs start at 0 (got {min_id})")
    check(max_id == len(nodes) - 1, f"Node IDs are sequential 0..{len(nodes)-1} (max={max_id})")

    # Check type distribution
    type_dist = Counter(node_types.values())
    for tid, expected in node_type_counts.items():
        actual = type_dist.get(tid, 0)
        tname = info['node.dat'][str(tid)][0]  # [name, count]
        check(actual == expected, f"Type '{tname}' (id={tid}): {actual} nodes (expected {expected})")

    # Check no duplicate names WITHIN the same type (cross-type sharing like SAG drug/gene is OK)
    name_type_pairs = set()
    within_type_dups = 0
    for gid, name in node_names.items():
        key = (name, node_types[gid])
        if key in name_type_pairs:
            within_type_dups += 1
            print(f"    Duplicate within type: {name} (type {node_types[gid]})")
        name_type_pairs.add(key)
    n_cross_type = len(nodes) - len(name_to_gid)
    if n_cross_type > 0:
        print(f"  [INFO] {n_cross_type} name(s) shared across types (cross-type collisions, OK)")
    check(within_type_dups == 0, f"No duplicate node names within same type ({within_type_dups} found)")

    # --- link.dat ---
    link_path = os.path.join(DATA_DIR, 'link.dat')
    check(os.path.exists(link_path), "link.dat exists")

    edge_counts = Counter()
    orphan_src = set()
    orphan_tgt = set()
    link_lines = 0
    with open(link_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src, tgt, etype = int(parts[0]), int(parts[1]), int(parts[2])
                edge_counts[etype] += 1
                link_lines += 1
                if src not in nodes:
                    orphan_src.add(src)
                if tgt not in nodes:
                    orphan_tgt.add(tgt)

    check(link_lines == expected_total_edges,
          f"link.dat has {link_lines} edges (expected {expected_total_edges})")
    check(len(orphan_src) == 0, f"No orphan source nodes in link.dat ({len(orphan_src)} found)")
    check(len(orphan_tgt) == 0, f"No orphan target nodes in link.dat ({len(orphan_tgt)} found)")

    for eid, expected in edge_type_counts.items():
        actual = edge_counts.get(eid, 0)
        ename = info['link.dat'][str(eid)][2]
        check(actual == expected, f"Edge type '{ename}' (id={eid}): {actual} edges (expected {expected})")

    # =========================================================================
    # STAGE 2: Train.dat (Structural Graph for GNN)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Train.dat (GNN Structural Graph)")
    print("=" * 60)

    train_path = os.path.join(DATA_DIR, 'train.dat')
    check(os.path.exists(train_path), "train.dat exists")

    train_edges = Counter()  # etype -> count
    train_cd_edges = []  # Cell-Drug edges in train.dat
    train_nodes_seen = set()
    train_line_count = 0

    CELL_TYPE = 0
    DRUG_TYPE = 1

    with open(train_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src, tgt, etype = int(parts[0]), int(parts[1]), int(parts[2])
                train_edges[etype] += 1
                train_nodes_seen.add(src)
                train_nodes_seen.add(tgt)
                train_line_count += 1
                st = node_types.get(src)
                tt = node_types.get(tgt)
                if (st == CELL_TYPE and tt == DRUG_TYPE) or (st == DRUG_TYPE and tt == CELL_TYPE):
                    train_cd_edges.append((src, tgt))

    print(f"  [{INFO}] train.dat: {train_line_count} edges across {len(train_edges)} types")
    for eid in sorted(train_edges.keys()):
        ename = info['link.dat'][str(eid)][2] if str(eid) in info['link.dat'] else f"type_{eid}"
        print(f"  [{INFO}]   {ename}: {train_edges[eid]}")

    check(train_line_count < link_lines,
          f"train.dat ({train_line_count}) < link.dat ({link_lines}) — test edges held out")

    # Check all structural edge types are present (should be identical to link.dat)
    for eid in [1, 2, 3, 4]:  # GG, CG, DG, CC
        ename = info['link.dat'][str(eid)][2]
        expected = edge_type_counts[eid]
        actual = train_edges.get(eid, 0)
        check(actual == expected,
              f"All {ename} edges in train.dat ({actual}/{expected})",
              warn_only=(abs(actual - expected) <= 5))

    # =========================================================================
    # STAGE 3: Cell Splits & Inductive Isolation
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Cell Splits & Inductive Isolation")
    print("=" * 60)

    splits_path = os.path.join(DATA_DIR, 'cell_splits.json')
    check(os.path.exists(splits_path), "cell_splits.json exists")

    with open(splits_path) as f:
        cell_splits = json.load(f)

    train_cells = set(cell_splits['train_cells'])
    valid_cells = set(cell_splits['valid_cells'])
    test_cells = set(cell_splits['test_cells'])
    inductive_cells = valid_cells | test_cells

    total_split = len(train_cells) + len(valid_cells) + len(test_cells)
    total_cells = node_type_counts[CELL_TYPE]

    print(f"  [{INFO}] Train: {len(train_cells)}, Valid: {len(valid_cells)}, Test: {len(test_cells)}")
    check(total_split == total_cells,
          f"Splits cover all cells: {total_split} = {total_cells}")
    check(len(train_cells & valid_cells) == 0, "No overlap: train ∩ valid")
    check(len(train_cells & test_cells) == 0, "No overlap: train ∩ test")
    check(len(valid_cells & test_cells) == 0, "No overlap: valid ∩ test")

    # CRITICAL: No Cell-Drug edges for inductive cells in train.dat
    leaked_cells = set()
    for src, tgt in train_cd_edges:
        cell_gid = src if node_types[src] == CELL_TYPE else tgt
        if cell_gid in inductive_cells:
            leaked_cells.add(cell_gid)

    check(len(leaked_cells) == 0,
          f"No inductive Cell-Drug edges in train.dat ({len(leaked_cells)} leaked cells)")

    # =========================================================================
    # STAGE 4: Link Prediction Split Files
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 4: LP Split Files")
    print("=" * 60)

    split_files = {
        'train_lp': 'train_lp_links.dat',
        'valid_inductive': 'valid_inductive_links.dat',
        'valid_transductive': 'valid_transductive_links.dat',
        'test_inductive': 'test_inductive_links.dat',
        'test_transductive': 'test_transductive_links.dat',
    }

    split_data = {}
    for key, fname in split_files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            check(False, f"{fname} exists")
            continue
        check(True, f"{fname} exists")

        links = []
        bad_format = 0
        bad_nodes = 0
        bad_labels = 0
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    bad_format += 1
                    continue
                src, tgt, label = int(parts[0]), int(parts[1]), float(parts[2])
                if src not in nodes or tgt not in nodes:
                    bad_nodes += 1
                if not (0.0 <= label <= 1.0):
                    bad_labels += 1
                links.append((src, tgt, label))

        n_pos = sum(1 for _, _, l in links if l > 0.5)
        n_neg = sum(1 for _, _, l in links if l <= 0.5)
        n_soft = sum(1 for _, _, l in links if 0 < l < 1)
        pos_rate = n_pos / len(links) * 100 if links else 0

        split_data[key] = links
        soft_str = f", soft={n_soft}" if n_soft > 0 else ""
        print(f"  [{INFO}] {key}: {len(links)} links (pos={n_pos} [{pos_rate:.1f}%], neg={n_neg}{soft_str})")
        check(bad_format == 0, f"  {key}: no malformed lines ({bad_format} bad)")
        check(bad_nodes == 0, f"  {key}: all node IDs valid ({bad_nodes} bad)")
        check(bad_labels == 0, f"  {key}: all labels in [0,1] ({bad_labels} bad)")

    # Check inductive splits only contain inductive cells
    for key in ['test_inductive', 'valid_inductive']:
        if key not in split_data:
            continue
        wrong_cells = 0
        for src, tgt, _ in split_data[key]:
            cell = src if node_types.get(src) == CELL_TYPE else tgt
            if cell not in inductive_cells:
                wrong_cells += 1
        check(wrong_cells == 0,
              f"{key}: all links involve inductive cells ({wrong_cells} violations)")

    # Check transductive splits only contain train cells
    for key in ['test_transductive', 'valid_transductive']:
        if key not in split_data:
            continue
        wrong_cells = 0
        for src, tgt, _ in split_data[key]:
            cell = src if node_types.get(src) == CELL_TYPE else tgt
            if cell in inductive_cells:
                wrong_cells += 1
        check(wrong_cells == 0,
              f"{key}: no inductive cells in transductive split ({wrong_cells} violations)")

    # Check no duplicate (src, tgt) pairs within each split
    for key, links in split_data.items():
        pair_set = set()
        dups = 0
        for s, t, _ in links:
            if (s, t) in pair_set:
                dups += 1
            pair_set.add((s, t))
        check(dups == 0, f"{key}: no duplicate pairs ({dups} duplicates)")

    # Check no train LP links leak into test/valid (exact pair match)
    train_lp_set = set((s, t) for s, t, _ in split_data.get('train_lp', []))
    for key in ['test_inductive', 'test_transductive', 'valid_inductive', 'valid_transductive']:
        if key not in split_data:
            continue
        overlap = sum(1 for s, t, _ in split_data[key] if (s, t) in train_lp_set or (t, s) in train_lp_set)
        check(overlap == 0, f"No pair overlap between train_lp and {key} ({overlap} leaked)")

    # =========================================================================
    # STAGE 5: Precomputed Neighbors
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 5: Precomputed Neighbors")
    print("=" * 60)

    pkl_path = os.path.join(DATA_DIR, 'train_neighbors_preprocessed.pkl')
    check(os.path.exists(pkl_path), "train_neighbors_preprocessed.pkl exists")

    with open(pkl_path, 'rb') as f:
        precomputed = pickle.load(f)

    # Auto-detect MAX_NEIGHBORS from first entry
    first_entry = next(iter(precomputed.values()))
    first_type_data = next(iter(first_entry.values()))
    MAX_NEIGHBORS = len(first_type_data['ids'])
    print(f"  [INFO] Detected MAX_NEIGHBORS={MAX_NEIGHBORS} from precomputed data")
    PAD_VALUE = -1

    check(len(precomputed) == len(nodes),
          f"Precomputed covers all nodes: {len(precomputed)}/{len(nodes)}")

    # Validate structure
    bad_keys = 0
    bad_structure = 0
    bad_lengths = 0
    pad_weight_errors = 0

    cell_nb_counts = {0: [], 1: [], 2: []}  # type -> list of counts per cell
    inductive_cell_stats = {0: [], 1: [], 2: []}

    for (ntype, lid), type_dict in precomputed.items():
        if not isinstance(type_dict, dict):
            bad_structure += 1
            continue
        for nt_id, data in type_dict.items():
            if not isinstance(data, dict) or 'ids' not in data or 'weights' not in data:
                bad_structure += 1
                continue
            if len(data['ids']) != MAX_NEIGHBORS or len(data['weights']) != MAX_NEIGHBORS:
                bad_lengths += 1
                continue
            # Check pad positions have weight=0
            for i in range(MAX_NEIGHBORS):
                if data['ids'][i] == PAD_VALUE and data['weights'][i] != 0.0:
                    pad_weight_errors += 1

        # Track cell neighbor counts
        if ntype == CELL_TYPE:
            # Find GID for this cell
            gid = None
            for g, (t, l) in [(g, info) for g, info in
                               ((gid, nodes[gid]) for gid in nodes)
                               if info[1] == CELL_TYPE]:
                pass  # too slow, skip inline

    check(bad_structure == 0, f"All entries have correct structure ({bad_structure} bad)")
    check(bad_lengths == 0, f"All neighbor lists have length {MAX_NEIGHBORS} ({bad_lengths} bad)")
    check(pad_weight_errors == 0, f"All PAD positions have weight=0 ({pad_weight_errors} violations)")

    # Check inductive cells specifically
    inductive_with_nb = 0
    inductive_without = 0
    inductive_drug_nb = 0

    from dataloaders.data_loader import PRELUDEDataset
    ds = PRELUDEDataset(DATA_DIR)

    for gid in inductive_cells:
        tmap = ds.nodes['type_map'].get(gid)
        if tmap is None:
            continue
        ntype, lid = tmap
        key = (ntype, lid)
        if key in precomputed:
            inductive_with_nb += 1
            # Check drug neighbors are empty
            drug_data = precomputed[key].get(DRUG_TYPE, {})
            drug_non_pad = sum(1 for x in drug_data.get('ids', []) if x != PAD_VALUE)
            if drug_non_pad > 0:
                inductive_drug_nb += 1

            # Count cell and gene neighbors
            for nt_id in [0, 2]:
                data = precomputed[key].get(nt_id, {})
                non_pad = sum(1 for x in data.get('ids', []) if x != PAD_VALUE)
                inductive_cell_stats[nt_id].append(non_pad)
        else:
            inductive_without += 1

    n_inductive = len(inductive_cells)
    print(f"  [{INFO}] Inductive cells: {inductive_with_nb}/{n_inductive} have neighbors")
    check(inductive_without == 0,
          f"All inductive cells have precomputed neighbors ({inductive_without} missing)")
    check(inductive_drug_nb == 0,
          f"No inductive cells have drug neighbors ({inductive_drug_nb} have drugs)")

    if inductive_cell_stats[0]:
        avg_cell_nb = sum(inductive_cell_stats[0]) / len(inductive_cell_stats[0])
        zero_cell = sum(1 for c in inductive_cell_stats[0] if c == 0)
        print(f"  [{INFO}] Inductive cells: avg {avg_cell_nb:.1f} cell neighbors, {zero_cell} with zero")

    if inductive_cell_stats[2]:
        avg_gene_nb = sum(inductive_cell_stats[2]) / len(inductive_cell_stats[2])
        zero_gene = sum(1 for c in inductive_cell_stats[2] if c == 0)
        print(f"  [{INFO}] Inductive cells: avg {avg_gene_nb:.1f} gene neighbors, {zero_gene} with zero")

    # =========================================================================
    # STAGE 6: Feature Alignment
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 6: Feature Alignment")
    print("=" * 60)

    import torch
    from dataloaders.feature_loader import FeatureLoader
    device = torch.device('cpu')
    fl = FeatureLoader(ds, device)

    check(fl.cell_features.shape[0] == node_type_counts[CELL_TYPE],
          f"Cell features: {fl.cell_features.shape[0]} = {node_type_counts[CELL_TYPE]} cells")
    check(fl.drug_features.shape[0] == node_type_counts[DRUG_TYPE],
          f"Drug features: {fl.drug_features.shape[0]} = {node_type_counts[DRUG_TYPE]} drugs")
    check(fl.gene_features.shape[0] == node_type_counts[2],
          f"Gene features: {fl.gene_features.shape[0]} = {node_type_counts[2]} genes")

    # Check for all-zero rows (missing features)
    zero_cells = (fl.cell_features.sum(dim=1) == 0).sum().item()
    zero_drugs = (fl.drug_features.sum(dim=1) == 0).sum().item()
    zero_genes = (fl.gene_features.sum(dim=1) == 0).sum().item()

    check(zero_cells == 0, f"No all-zero cell feature vectors ({zero_cells} found)", warn_only=True)
    check(zero_drugs == 0, f"No all-zero drug feature vectors ({zero_drugs} found)", warn_only=True)
    check(zero_genes == 0, f"No all-zero gene feature vectors ({zero_genes} found)", warn_only=True)

    # =========================================================================
    # STAGE 7: DataGenerator Consistency
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 7: DataGenerator Consistency")
    print("=" * 60)

    from dataloaders.data_generator import DataGenerator
    gen = DataGenerator(DATA_DIR)

    # Check it loaded from train.dat
    total_gen_links = sum(len(v) for v in gen.links.values())
    check(total_gen_links == train_line_count,
          f"DataGenerator loaded {total_gen_links} links = train.dat ({train_line_count})")

    # Check bidirectional neighbors
    # Pick a random structural edge and verify both directions exist
    import random
    sample_edges = []
    for rtype, pairs in gen.links.items():
        for src, tgt in random.sample(pairs, min(5, len(pairs))):
            sample_edges.append((rtype, src, tgt))

    bidi_ok = 0
    bidi_fail = 0
    for rtype, src, tgt in sample_edges:
        st = node_types.get(src)
        tt = node_types.get(tgt)
        # Skip Cell-Drug (excluded from neighbors)
        if (st == CELL_TYPE and tt == DRUG_TYPE) or (st == DRUG_TYPE and tt == CELL_TYPE):
            continue
        fwd = tgt in gen.neighbors.get(src, {}).get(rtype, [])
        rev = src in gen.neighbors.get(tgt, {}).get(rtype, [])
        if fwd and rev:
            bidi_ok += 1
        else:
            bidi_fail += 1

    check(bidi_fail == 0,
          f"Bidirectional neighbors verified ({bidi_ok} OK, {bidi_fail} failed)")

    # Check Cell-Drug edges excluded from neighbor dicts
    cd_in_neighbors = 0
    for gid in random.sample(list(train_cells), min(20, len(train_cells))):
        for rtype, nbs in gen.neighbors.get(gid, {}).items():
            for nb in nbs:
                if node_types.get(nb) == DRUG_TYPE:
                    cd_in_neighbors += 1

    check(cd_in_neighbors == 0,
          f"Cell-Drug edges excluded from neighbor dicts ({cd_in_neighbors} found)")

    # Check no isolated transductive cells (should have some neighbors)
    isolated_train_cells = 0
    for gid in random.sample(list(train_cells), min(50, len(train_cells))):
        total_nb = sum(len(nbs) for nbs in gen.neighbors.get(gid, {}).values())
        if total_nb == 0:
            isolated_train_cells += 1

    check(isolated_train_cells == 0,
          f"No isolated training cells (sampled 50, {isolated_train_cells} isolated)", warn_only=True)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_checks = len(failures) + len(warnings)
    passed = total_checks - len(failures) - len(warnings)

    if failures:
        print(f"\n  [{FAIL}] {len(failures)} FAILURES:")
        for f in failures:
            print(f"    - {f}")

    if warnings:
        print(f"\n  [{WARN}] {len(warnings)} WARNINGS:")
        for w in warnings:
            print(f"    - {w}")

    if not failures and not warnings:
        print(f"\n  All checks passed!")
    elif not failures:
        print(f"\n  All critical checks passed ({len(warnings)} warnings).")
    else:
        print(f"\n  {len(failures)} critical failures found!")

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
