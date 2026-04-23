# scripts/create_masked_dataset.py
import os
import shutil
import random
import argparse
from tqdm import tqdm

def create_masked_dataset(source_dir, target_dir, mask_ratio=0.5):
    """
    Creates a new dataset folder where 'train.dat' has mask_ratio of 
    training cells STRIPPED of their drug edges.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    if os.path.exists(target_dir):
        print(f"Warning: Target directory {target_dir} already exists. Cleaning up...")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    print(f"--- Creating Masked Dataset ({mask_ratio*100}%) ---")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    # 1. Copy all files EXCEPT train.dat
    # We keep train_lp_links.dat intact because we want to calculate LOSS on everyone.
    files_to_copy = [f for f in os.listdir(source_dir) if f != 'train.dat' and f != 'train_neighbors_preprocessed.pkl']
    
    print("Copying base files...")
    for f in files_to_copy:
        src_file = os.path.join(source_dir, f)
        if os.path.isfile(src_file):
            shutil.copy(src_file, os.path.join(target_dir, f))

    # 2. Process train.dat (The Graph Structure)
    print("Processing train.dat...")
    
    # First pass: Find all cells that have drug interactions
    cell_drug_counts = {}
    
    with open(os.path.join(source_dir, 'train.dat'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Handle variable columns (h, t, r) OR (h, t, r, w)
            if len(parts) >= 3:
                h, t, r = parts[0], parts[1], parts[2]
                
                # Link type 0 is usually cell-drug in your dataset
                if r == '0': # Cell-Drug Interaction
                    if h not in cell_drug_counts: cell_drug_counts[h] = 0
                    cell_drug_counts[h] += 1

    all_train_cells = list(cell_drug_counts.keys())
    if not all_train_cells:
        print("Warning: No Cell-Drug links (type 0) found! Check your link types.")
    
    num_masked = int(len(all_train_cells) * mask_ratio)
    masked_cells = set(random.sample(all_train_cells, num_masked))
    
    print(f"Total Training Cells with Drug Edges: {len(all_train_cells)}")
    print(f"Cells selected for Masking: {len(masked_cells)}")

    # Second pass: Write new train.dat
    kept_links = 0
    dropped_links = 0
    
    with open(os.path.join(source_dir, 'train.dat'), 'r') as fin, \
         open(os.path.join(target_dir, 'train.dat'), 'w') as fout:
        
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 3: continue 
            
            h, t, r = parts[0], parts[1], parts[2]
            
            # If it's a Cell-Drug link (Type 0) involving a Masked Cell -> DROP IT
            if r == '0' and (h in masked_cells or t in masked_cells):
                dropped_links += 1
                continue 
            
            fout.write(line)
            kept_links += 1

    print(f"Done. Kept {kept_links} links. Dropped {dropped_links} links (masked).")
    print(f"Masked dataset ready at: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/processed')
    parser.add_argument('--target', type=str, default='data/processed_masked')
    parser.add_argument('--ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    create_masked_dataset(args.source, args.target, args.ratio)