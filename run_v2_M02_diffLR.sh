#!/bin/bash
# V2 M02: Backbone LR 0.1x for better Sanger transfer
# Same as M01 but with backbone_lr_scale=0.1
# In v1, slower backbone consistently improved cross-dataset transfer

set -e

MODEL_NAME="v2_M02_diffLR"
DATA_DIR="data/processed_v2"
EMB_DIR="data/processed_v2/embeddings"

echo "=========================================="
echo "TRAINING: $MODEL_NAME"
echo "=========================================="

conda run -n hgb_env --no-capture-output python scripts/train.py \
    --data_dir $DATA_DIR \
    --save_dir checkpoints \
    --model_name $MODEL_NAME \
    --cell_feature_source vae \
    --include_cell_drug \
    --dual_head \
    --backbone_lr_scale 0.1 \
    --embed_d 256 \
    --n_layers 2 \
    --max_neighbors 10 \
    --use_node_gate \
    --use_skip_connection \
    --use_triplet_loss \
    --use_soft_margin_triplet \
    --use_dynamic_loss \
    --loss_target_lp 0.85 \
    --loss_target_triplet 0.15 \
    --isolation_ratio 0.5 \
    --lr 0.0003 \
    --l1_lambda 1e-5 \
    --weight_decay 1e-4 \
    --epochs 50 \
    --patience 20 \
    --dropout 0.2 \
    --mini_batch_s 10240 \
    --train_fraction 1.0 \
    --compute_mad \
    --use_minibatch_gnn \
    --neighbor_sample_size 5 \
    --align_lambda 1.0 \
    --align_types cell drug gene \
    --gpu 0 \
    --num_workers 4

echo ""
echo "=========================================="
echo "DEPMAP TEST EVALUATION"
echo "=========================================="

conda run -n hgb_env --no-capture-output python -c "
import sys, torch, argparse
sys.path.insert(0, '.')
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg
from utils.evaluation import evaluate_model
from scripts.train import LinkPredictionDataset
from torch.utils.data import DataLoader

device = torch.device('cuda:0')
dataset = PRELUDEDataset('$DATA_DIR')
feature_loader = FeatureLoader(dataset, device, embedding_dir='$EMB_DIR')
generator = DataGenerator('$DATA_DIR', include_cell_drug=True)

args = argparse.Namespace(
    data_dir='$DATA_DIR', embed_d=256, n_layers=2, max_neighbors=10,
    dropout=0.2, use_skip_connection=True, use_node_gate=True,
    cell_feature_source='vae', gene_encoder_dim=0, use_cross_attention=False,
    cross_attn_dim=32, freeze_cell_gnn=False, regression=False,
    use_residual_gnn=False, residual_scale=0.0, ema_lambda=0.0, ema_momentum=0.999,
    include_cell_drug=True, dual_head=True, inductive_loss_weight=1.0,
    backbone_lr_scale=0.1,
)

model = HetAgg(args, dataset, feature_loader, device).to(device)
model.setup_link_prediction('drug', 'cell')
model.load_state_dict(torch.load('checkpoints/${MODEL_NAME}.pth', map_location=device))
model.eval()

with torch.no_grad():
    eval_tables = model.compute_all_embeddings()

test_ind = dataset.links.get('test_inductive', [])
if test_ind:
    ds = LinkPredictionDataset(test_ind, dataset)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    metrics, _ = evaluate_model(model, loader, generator, device, dataset,
                                embedding_tables=eval_tables, use_inductive_head=True)
    print(f'  Test Inductive:    AUC={metrics[\"ROC-AUC\"]:.4f} | F1={metrics[\"F1-Score\"]:.4f}')

test_trans = dataset.links.get('test_transductive', [])
if test_trans:
    ds = LinkPredictionDataset(test_trans, dataset)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    metrics, _ = evaluate_model(model, loader, generator, device, dataset,
                                embedding_tables=eval_tables, use_inductive_head=False)
    print(f'  Test Transductive: AUC={metrics[\"ROC-AUC\"]:.4f} | F1={metrics[\"F1-Score\"]:.4f}')
"

echo ""
echo "=========================================="
echo "SANGER EVALUATION (S1-S4)"
echo "=========================================="

conda run -n hgb_env --no-capture-output python -c "
import sys, torch, argparse, os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg

device = torch.device('cuda:0')
DATA_DIR = '$DATA_DIR'
EMB_DIR = '$EMB_DIR'

dataset = PRELUDEDataset(DATA_DIR)
feature_loader = FeatureLoader(dataset, device, embedding_dir=EMB_DIR)
generator = DataGenerator(DATA_DIR, include_cell_drug=True)

args = argparse.Namespace(
    data_dir=DATA_DIR, embed_d=256, n_layers=2, max_neighbors=10,
    dropout=0.2, use_skip_connection=True, use_node_gate=True,
    cell_feature_source='vae', gene_encoder_dim=0, use_cross_attention=False,
    cross_attn_dim=32, freeze_cell_gnn=False, regression=False,
    use_residual_gnn=False, residual_scale=0.0, ema_lambda=0.0, ema_momentum=0.999,
    include_cell_drug=True, dual_head=True, inductive_loss_weight=1.0,
    backbone_lr_scale=0.1,
)

model = HetAgg(args, dataset, feature_loader, device).to(device)
model.setup_link_prediction('drug', 'cell')
model.load_state_dict(torch.load('checkpoints/${MODEL_NAME}.pth', map_location=device))
model.eval()

with torch.no_grad():
    eval_tables = model.compute_all_embeddings()

class SangerLPDataset(Dataset):
    def __init__(self, links_path, dataset):
        self.pairs = []
        with open(links_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                src, tgt = int(parts[0]), int(parts[1])
                label = float(parts[2])
                src_type, src_lid = dataset.nodes['type_map'][src]
                tgt_type, tgt_lid = dataset.nodes['type_map'][tgt]
                self.pairs.append((src_lid, tgt_lid, label, src_type))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        s, t, l, st = self.pairs[i]
        return (torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long),
                torch.tensor(l, dtype=torch.float), torch.tensor(st, dtype=torch.long))

drug_type = dataset.node_name2type['drug']
descs = {'S1': 'Known cells + Known drugs', 'S2': 'Known cells + New drugs',
         'S3': 'New cells + Known drugs', 'S4': 'New cells + New drugs'}

for scenario in ['S1', 'S2', 'S3', 'S4']:
    path = f'{DATA_DIR}/sanger_{scenario}_links.dat'
    if not os.path.exists(path): continue
    ds = SangerLPDataset(path, dataset)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for s_lids, t_lids, labels, s_types in loader:
            s_lids, t_lids = s_lids.to(device), t_lids.to(device)
            if s_types[0].item() == drug_type:
                drug_lids, cell_lids = s_lids, t_lids
            else:
                drug_lids, cell_lids = t_lids, s_lids
            use_ind = scenario != 'S1'
            preds = model.link_prediction_forward(drug_lids, cell_lids, generator,
                embedding_tables=eval_tables, use_inductive_head=use_ind)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    binary = (np.array(all_labels) > 0.5).astype(int)
    if len(np.unique(binary)) == 2:
        auc = roc_auc_score(binary, all_preds)
        f1 = f1_score(binary, all_preds > 0.5)
        print(f'  {scenario} ({descs[scenario]}): AUC={auc:.4f} | F1={f1:.4f} | Pairs={len(all_labels):,}')

print()
print('Done.')
"

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
