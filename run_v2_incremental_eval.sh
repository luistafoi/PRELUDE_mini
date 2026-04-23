#!/bin/bash
# Run incremental inference on all evaluation sets for a given model
# Usage: bash run_v2_incremental_eval.sh [model_name]
# Default: v2_M03_drug_sim

set -e

MODEL_NAME="${1:-v2_M03_drug_sim}"
DATA_DIR="data/processed_v2"
OUT_DIR="results/${MODEL_NAME}"

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "INCREMENTAL INFERENCE: $MODEL_NAME"
echo "=========================================="
echo ""

# Run all sets and collect results
conda run -n hgb_env --no-capture-output python -c "
import sys, os
sys.path.insert(0, '.')

from scripts.inference_incremental import *

args = argparse.Namespace(
    model_name='$MODEL_NAME',
    data_dir='$DATA_DIR',
    emb_dir='$DATA_DIR/embeddings',
    gpu=0,
    confidence_threshold=0.8,
    cell_knn=15,
    output_dir='$OUT_DIR',
    split=None,
    sanger_scenario=None,
)

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
dataset = PRELUDEDataset(args.data_dir)
feature_loader = FeatureLoader(dataset, device, embedding_dir=args.emb_dir)
generator = DataGenerator(args.data_dir, include_cell_drug=True)

all_results = []

# DepMap test sets
for split_name, use_ind in [('test_inductive', True), ('test_transductive', False)]:
    links = dataset.links.get(split_name, [])
    if not links:
        continue
    print(f'\n{\"=\"*60}')
    print(f'Evaluating: DepMap {split_name} ({len(links):,} pairs)')
    print(f'{\"=\"*60}')

    print(f'\n--- Mode 1: Batch ---')
    model = load_model(args, dataset, feature_loader, device)
    import time
    t0 = time.time()
    m1, _ = batch_inference(model, dataset, links, device, generator, use_inductive_head=use_ind)
    t1 = time.time()
    print(f'  AUC={m1.get(\"ROC-AUC\",0):.4f} | F1={m1.get(\"F1-Score\",0):.4f} | Time={t1-t0:.1f}s')

    print(f'\n--- Mode 2: Independent ---')
    model = load_model(args, dataset, feature_loader, device)
    t0 = time.time()
    m2, _ = incremental_inference(model, dataset, links, device, generator, feature_loader, rolling=False)
    t1 = time.time()
    print(f'  AUC={m2.get(\"ROC-AUC\",0):.4f} | F1={m2.get(\"F1-Score\",0):.4f} | Time={t1-t0:.1f}s')

    print(f'\n--- Mode 3: Rolling ---')
    model = load_model(args, dataset, feature_loader, device)
    t0 = time.time()
    m3, _ = incremental_inference(model, dataset, links, device, generator, feature_loader, rolling=True, confidence_threshold=0.8)
    t1 = time.time()
    print(f'  AUC={m3.get(\"ROC-AUC\",0):.4f} | F1={m3.get(\"F1-Score\",0):.4f} | Edges={m3.get(\"edges_added\",0)} | Time={t1-t0:.1f}s')

    all_results.append({
        'eval_set': f'DepMap_{split_name}',
        'batch_auc': m1.get('ROC-AUC', 0), 'batch_f1': m1.get('F1-Score', 0),
        'independent_auc': m2.get('ROC-AUC', 0), 'independent_f1': m2.get('F1-Score', 0),
        'rolling_auc': m3.get('ROC-AUC', 0), 'rolling_f1': m3.get('F1-Score', 0),
        'rolling_edges': m3.get('edges_added', 0),
    })

# Sanger scenarios
for scenario in ['S1', 'S2', 'S3', 'S4']:
    links = load_sanger_links(args.data_dir, scenario)
    if not links:
        continue
    is_trans = (scenario == 'S1')

    print(f'\n{\"=\"*60}')
    print(f'Evaluating: Sanger {scenario} ({len(links):,} pairs)')
    print(f'{\"=\"*60}')

    print(f'\n--- Mode 1: Batch ---')
    model = load_model(args, dataset, feature_loader, device)
    t0 = time.time()
    m1, _ = batch_inference(model, dataset, links, device, generator, use_inductive_head=not is_trans)
    t1 = time.time()
    print(f'  AUC={m1.get(\"ROC-AUC\",0):.4f} | F1={m1.get(\"F1-Score\",0):.4f} | Time={t1-t0:.1f}s')

    print(f'\n--- Mode 2: Independent ---')
    model = load_model(args, dataset, feature_loader, device)
    t0 = time.time()
    m2, _ = incremental_inference(model, dataset, links, device, generator, feature_loader, rolling=False)
    t1 = time.time()
    print(f'  AUC={m2.get(\"ROC-AUC\",0):.4f} | F1={m2.get(\"F1-Score\",0):.4f} | Time={t1-t0:.1f}s')

    print(f'\n--- Mode 3: Rolling ---')
    model = load_model(args, dataset, feature_loader, device)
    t0 = time.time()
    m3, _ = incremental_inference(model, dataset, links, device, generator, feature_loader, rolling=True, confidence_threshold=0.8)
    t1 = time.time()
    print(f'  AUC={m3.get(\"ROC-AUC\",0):.4f} | F1={m3.get(\"F1-Score\",0):.4f} | Edges={m3.get(\"edges_added\",0)} | Time={t1-t0:.1f}s')

    all_results.append({
        'eval_set': f'Sanger_{scenario}',
        'batch_auc': m1.get('ROC-AUC', 0), 'batch_f1': m1.get('F1-Score', 0),
        'independent_auc': m2.get('ROC-AUC', 0), 'independent_f1': m2.get('F1-Score', 0),
        'rolling_auc': m3.get('ROC-AUC', 0), 'rolling_f1': m3.get('F1-Score', 0),
        'rolling_edges': m3.get('edges_added', 0),
    })

# Save combined results
import pandas as pd
results_df = pd.DataFrame(all_results)
results_df.to_csv('$OUT_DIR/incremental_inference_results.csv', index=False)

print(f'\n{\"=\"*60}')
print('ALL RESULTS')
print(f'{\"=\"*60}\n')
print(results_df.to_string(index=False))

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'$MODEL_NAME — Inference Mode Comparison', fontsize=14, fontweight='bold')
x = np.arange(len(all_results))
width = 0.25
for ax, metric, title in [(axes[0], 'auc', 'ROC-AUC'), (axes[1], 'f1', 'F1 Score')]:
    ax.bar(x - width, [r[f'batch_{metric}'] for r in all_results], width, label='Batch', color='tab:blue', alpha=0.8)
    ax.bar(x, [r[f'independent_{metric}'] for r in all_results], width, label='Independent', color='tab:orange', alpha=0.8)
    ax.bar(x + width, [r[f'rolling_{metric}'] for r in all_results], width, label='Rolling', color='tab:green', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['eval_set'] for r in all_results], fontsize=8, rotation=20)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('$OUT_DIR/incremental_inference_comparison.png')
plt.close()
print(f'\nPlot: $OUT_DIR/incremental_inference_comparison.png')
print(f'CSV:  $OUT_DIR/incremental_inference_results.csv')
"

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
