"""Build continuous-label versions of LP split files for regression training.

Reads existing binary split files (train_lp_links.dat, valid_*.dat, test_*.dat),
maps (cell_gid, drug_gid) -> (cell_name, drug_name), looks up mean LMFI.normalized
from the raw PRISM data, and writes parallel *_continuous.dat files.

LMFI.normalized semantics:
  - Lower values = more sensitive (drug kills cells)
  - Higher values = more resistant (drug has no effect)
  - We negate so that higher = more sensitive (consistent with binary label=1 = sensitive)

Usage:
    python scripts/build_continuous_labels.py
    python scripts/build_continuous_labels.py --data_dir data/processed --lmfi_path data/misc/Repurposing_Public_24Q2_LMFI_NORMALIZED_with_DrugNames.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_node_names(data_dir):
    """Load global_id -> node_name mapping from node.dat."""
    id2name = {}
    with open(os.path.join(data_dir, "node.dat")) as f:
        for line in f:
            parts = line.strip().split('\t')
            gid, name = int(parts[0]), parts[1]
            id2name[gid] = name
    return id2name


def load_lmfi_lookup(lmfi_path):
    """Build (cell_name, drug_name) -> mean LMFI.normalized lookup.

    Averages across replicates for the same cell-drug pair.
    """
    print(f"Loading LMFI data from {lmfi_path}...")
    df = pd.read_csv(lmfi_path)

    # Drop rows without drug names
    df = df.dropna(subset=['name'])

    # Extract cell name from row_id (ACH-XXXXXX::rest)
    df['cell'] = df['row_id'].str.split('::').str[0]

    # Uppercase drug names for case-insensitive matching
    df['drug_upper'] = df['name'].str.upper()

    # Average LMFI.normalized per (cell, drug) pair
    grouped = df.groupby(['cell', 'drug_upper'])['LMFI.normalized'].mean()

    lookup = {}
    for (cell, drug), val in grouped.items():
        lookup[(cell, drug)] = val

    print(f"  Built lookup: {len(lookup):,} unique (cell, drug) pairs")
    print(f"  LMFI range: [{grouped.min():.2f}, {grouped.max():.2f}], mean={grouped.mean():.2f}")

    return lookup


def convert_split_file(input_path, output_path, id2name, lmfi_lookup):
    """Convert a binary split file to continuous labels.

    For pairs not found in LMFI lookup (negatives from random sampling),
    we keep them but assign NaN — these will be filtered during training.

    Returns: (n_total, n_matched, n_missing)
    """
    if not os.path.exists(input_path):
        return 0, 0, 0

    lines_out = []
    n_total = 0
    n_matched = 0
    n_missing = 0

    with open(input_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            gid_a, gid_b = int(parts[0]), int(parts[1])
            old_label = float(parts[2]) if len(parts) > 2 else 1.0

            name_a = id2name.get(gid_a, "")
            name_b = id2name.get(gid_b, "")

            # Figure out which is cell, which is drug
            # Cells are ACH-*, drugs are everything else
            if name_a.startswith("ACH-"):
                cell_name, drug_name = name_a, name_b
            elif name_b.startswith("ACH-"):
                cell_name, drug_name = name_b, name_a
            else:
                # Neither is a cell — shouldn't happen in LP files
                n_missing += 1
                n_total += 1
                continue

            # Lookup LMFI
            lmfi_val = lmfi_lookup.get((cell_name, drug_name.upper()))

            if lmfi_val is not None:
                # Negate: lower LMFI = more sensitive -> higher score
                continuous_label = -lmfi_val
                lines_out.append(f"{gid_a}\t{gid_b}\t{continuous_label:.6f}\n")
                n_matched += 1
            else:
                # No LMFI data — this is a sampled negative (no actual measurement)
                # Keep with a sentinel value that training can filter
                # Use the old binary label as fallback indicator
                lines_out.append(f"{gid_a}\t{gid_b}\tNaN\n")
                n_missing += 1

            n_total += 1

    with open(output_path, 'w') as f:
        f.writelines(lines_out)

    return n_total, n_matched, n_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--lmfi_path', type=str,
                        default='data/misc/Repurposing_Public_24Q2_LMFI_NORMALIZED_with_DrugNames.csv')
    args = parser.parse_args()

    id2name = load_node_names(args.data_dir)
    print(f"Loaded {len(id2name)} node names")

    lmfi_lookup = load_lmfi_lookup(args.lmfi_path)

    # Convert each split file
    split_files = [
        'train_lp_links.dat',
        'valid_inductive_links.dat',
        'valid_transductive_links.dat',
        'test_inductive_links.dat',
        'test_transductive_links.dat',
    ]

    print("\nConverting split files:")
    total_matched = 0
    total_missing = 0

    for fname in split_files:
        input_path = os.path.join(args.data_dir, fname)
        # Insert _continuous before .dat
        out_fname = fname.replace('.dat', '_continuous.dat')
        output_path = os.path.join(args.data_dir, out_fname)

        n_total, n_matched, n_missing = convert_split_file(
            input_path, output_path, id2name, lmfi_lookup
        )

        if n_total > 0:
            print(f"  {fname} -> {out_fname}: "
                  f"{n_total:,} links, {n_matched:,} matched ({100*n_matched/n_total:.1f}%), "
                  f"{n_missing:,} missing LMFI")
            total_matched += n_matched
            total_missing += n_missing

    # Print summary stats for the training set
    train_path = os.path.join(args.data_dir, 'train_lp_links_continuous.dat')
    if os.path.exists(train_path):
        vals = []
        with open(train_path) as f:
            for line in f:
                v = line.strip().split('\t')[2]
                if v != 'NaN':
                    vals.append(float(v))
        vals = np.array(vals)
        print(f"\n--- Training Set Continuous Label Stats ---")
        print(f"  N valid: {len(vals):,}")
        print(f"  Range: [{vals.min():.3f}, {vals.max():.3f}]")
        print(f"  Mean: {vals.mean():.3f}, Std: {vals.std():.3f}")
        print(f"  Median: {np.median(vals):.3f}")
        print(f"  Quartiles: Q25={np.percentile(vals, 25):.3f}, Q75={np.percentile(vals, 75):.3f}")

    print(f"\nTotal: {total_matched:,} matched, {total_missing:,} missing across all splits")
    print("Done.")


if __name__ == '__main__':
    main()
