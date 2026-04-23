"""Clinical Decision Support Panel.

Single-page tool designed for clinicians, not data scientists.
Shows drug recommendations with biological evidence for each patient.

Usage:
    python scripts/build_clinical_panel.py --model_name v2_M03_drug_sim --data_dir data/processed_v2 --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_all_data(model_name, data_dir, device):
    """Load model, predictions, graph structure, everything."""
    from dataloaders.data_loader import PRELUDEDataset
    from dataloaders.feature_loader import FeatureLoader
    from dataloaders.data_generator import DataGenerator
    from models.tools import HetAgg
    from scripts.train import LinkPredictionDataset
    from torch.utils.data import DataLoader

    emb_dir = os.path.join(data_dir, 'embeddings')
    dataset = PRELUDEDataset(data_dir)
    feature_loader = FeatureLoader(dataset, device, embedding_dir=emb_dir)
    generator = DataGenerator(data_dir, include_cell_drug=True)

    model_args = argparse.Namespace(
        data_dir=data_dir, embed_d=256, n_layers=2, max_neighbors=10,
        dropout=0.2, use_skip_connection=True, use_node_gate=True,
        cell_feature_source='vae', gene_encoder_dim=0, use_cross_attention=False,
        cross_attn_dim=32, freeze_cell_gnn=False, regression=False,
        use_residual_gnn=False, residual_scale=0.0, ema_lambda=0.0, ema_momentum=0.999,
        include_cell_drug=True, dual_head=True, inductive_loss_weight=1.0,
        backbone_lr_scale=1.0,
    )
    model = HetAgg(model_args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction('drug', 'cell')
    model.load_state_dict(torch.load(f'checkpoints/{model_name}.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        eval_tables = model.compute_all_embeddings()

    gid_to_name = dict(dataset.id2node)
    name_to_gid = {v: k for k, v in gid_to_name.items()}
    cell_type = dataset.node_name2type['cell']
    drug_type = dataset.node_name2type['drug']

    # Metadata
    try:
        model_csv = pd.read_csv('data/misc/Model.csv')
        cell_tissues = {row['ModelID']: row.get('OncotreeLineage', 'Unknown')
                       for _, row in model_csv.iterrows()}
    except:
        cell_tissues = {}

    with open(os.path.join(data_dir, 'split_config.json')) as f:
        sc = json.load(f)
    train_cells = set(sc['train_cells'])
    test_cells = set(sc['test_cells'])

    # Embeddings
    cell_embeds = eval_tables[cell_type].cpu().numpy()

    # Cell lid mapping
    cell_lid_map = {}
    for gid, (t, lid) in dataset.nodes['type_map'].items():
        if t == cell_type:
            cell_lid_map[gid_to_name.get(gid, '')] = lid

    # Graph neighbors
    neighbors = defaultdict(lambda: defaultdict(list))
    etype_names = {0: 'cell-drug', 1: 'gene-gene', 2: 'cell-gene',
                  3: 'drug-gene', 4: 'cell-cell', 5: 'drug-drug'}
    with open(os.path.join(data_dir, 'link.dat')) as f:
        for line in f:
            parts = line.strip().split('\t')
            src, tgt, etype = int(parts[0]), int(parts[1]), int(parts[2])
            w = float(parts[3])
            sn = gid_to_name.get(src, '')
            tn = gid_to_name.get(tgt, '')
            en = etype_names.get(etype, '')
            neighbors[sn][en].append((tn, w))
            neighbors[tn][en].append((sn, w))

    # Predictions for test cells
    print("  Getting predictions...")
    all_data = []
    links = dataset.links.get('test_inductive', [])
    if links:
        ds = LinkPredictionDataset(links, dataset)
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
        with torch.no_grad():
            for u_lids, v_lids, labels, u_types, u_gids, v_gids in loader:
                u_lids, v_lids = u_lids.to(device), v_lids.to(device)
                if u_types[0].item() == drug_type:
                    d_lids, c_lids = u_lids, v_lids
                    d_gids, c_gids = u_gids, v_gids
                else:
                    d_lids, c_lids = v_lids, u_lids
                    d_gids, c_gids = v_gids, u_gids
                preds = model.link_prediction_forward(
                    d_lids, c_lids, generator,
                    embedding_tables=eval_tables, use_inductive_head=True)
                for i in range(len(preds)):
                    all_data.append({
                        'cell': gid_to_name.get(c_gids[i].item(), ''),
                        'drug': gid_to_name.get(d_gids[i].item(), ''),
                        'true_label': labels[i].item(),
                        'pred_score': preds[i].item(),
                    })

    predictions = pd.DataFrame(all_data)
    predictions['correct'] = (predictions['pred_score'] > 0.5) == (predictions['true_label'] > 0.5)

    # Calibration data: binned confidence vs accuracy
    print("  Computing calibration...")
    calibration = []
    for lo in np.arange(0, 1.0, 0.1):
        hi = lo + 0.1
        dist = np.abs(predictions['pred_score'] - 0.5)
        mask = (dist >= lo / 2) & (dist < hi / 2)
        subset = predictions[mask]
        if len(subset) >= 10:
            calibration.append({
                'bin': f'{50 + lo*50:.0f}-{50 + hi*50:.0f}%',
                'confidence': float(50 + (lo + hi) / 2 * 50),
                'accuracy': float(subset['correct'].mean()),
                'count': int(len(subset)),
            })

    return {
        'predictions': predictions,
        'cell_embeds': cell_embeds,
        'cell_tissues': cell_tissues,
        'train_cells': train_cells,
        'test_cells': test_cells,
        'neighbors': neighbors,
        'cell_lid_map': cell_lid_map,
        'calibration': calibration,
    }


def build_patient_data(data):
    """Precompute per-patient data for the panel."""
    predictions = data['predictions']
    top_cells = list(predictions.groupby('cell').size().sort_values(ascending=False).head(30).index)

    patients = {}
    for cell in top_cells:
        cp = predictions[predictions['cell'] == cell].copy()
        tissue = data['cell_tissues'].get(cell, 'Unknown')
        accuracy = float(cp['correct'].mean())

        # Three-tier drug classification
        recommended = cp[cp['pred_score'] >= 0.8].sort_values('pred_score', ascending=False)
        consider = cp[(cp['pred_score'] >= 0.5) & (cp['pred_score'] < 0.8)].sort_values('pred_score', ascending=False)
        not_recommended = cp[cp['pred_score'] < 0.5].sort_values('pred_score', ascending=True)

        # Mutations
        mutations = []
        for gene, w in data['neighbors'].get(cell, {}).get('cell-gene', []):
            mutations.append({'gene': gene, 'pathogenicity': round(w, 3)})
        mutations.sort(key=lambda x: x['pathogenicity'], reverse=True)

        # Similar training patients
        lid = data['cell_lid_map'].get(cell)
        similar_patients = []
        if lid is not None:
            sims = sklearn_cosine(data['cell_embeds'][lid:lid+1], data['cell_embeds'])[0]
            scored = []
            for other_cell, other_lid in data['cell_lid_map'].items():
                if other_cell in data['train_cells'] and other_lid != lid:
                    scored.append((other_cell, float(sims[other_lid])))
            scored.sort(key=lambda x: x[1], reverse=True)

            for sim_cell, sim_score in scored[:6]:
                sim_tissue = data['cell_tissues'].get(sim_cell, '?')
                # What drugs worked for this similar patient?
                sim_preds = predictions[predictions['cell'] == sim_cell] if sim_cell in predictions['cell'].values else pd.DataFrame()
                worked_drugs = []
                if len(sim_preds) > 0:
                    sim_sens = sim_preds[sim_preds['pred_score'] > 0.8].head(3)
                    worked_drugs = [str(d) for d in sim_sens['drug']]
                similar_patients.append({
                    'name': sim_cell.replace('ACH-', ''),
                    'full_id': sim_cell,
                    'tissue': sim_tissue,
                    'similarity': round(sim_score, 3),
                    'drugs_that_worked': worked_drugs[:3],
                })

        # Build drug evidence (for each recommended drug, show why)
        drug_evidence = {}
        for _, row in pd.concat([recommended.head(8), consider.head(4)]).iterrows():
            drug = row['drug']
            # Drug targets (dedup)
            targets = list(dict.fromkeys(g for g, _ in data['neighbors'].get(drug, {}).get('drug-gene', [])))
            # Patient mutations that match drug targets
            patient_genes = {m['gene'] for m in mutations}
            matching_mutations = [g for g in targets if g in patient_genes]
            # Similar drugs (dedup)
            seen_drugs = set()
            similar_drugs = []
            for d, w in data['neighbors'].get(drug, {}).get('drug-drug', []):
                if d not in seen_drugs:
                    seen_drugs.add(d)
                    similar_drugs.append((d, round(w, 2)))
            similar_drugs = similar_drugs[:3]

            if targets:
                mechanism = 'Targets ' + ', '.join(targets[:3])
            elif similar_drugs:
                mechanism = 'Similar to ' + ', '.join(d for d, _ in similar_drugs[:2])
            else:
                mechanism = 'No known gene targets'

            drug_evidence[drug] = {
                'targets': targets[:5],
                'matching_mutations': matching_mutations,
                'similar_drugs': similar_drugs,
                'mechanism': mechanism,
            }

        def drug_list(df, max_n=10):
            result = []
            for _, row in df.head(max_n).iterrows():
                d = str(row['drug'])
                result.append({
                    'name': d,
                    'short_name': d[:30],
                    'score': round(float(row['pred_score']), 3),
                    'true_label': int(row['true_label'] > 0.5),
                    'correct': bool(row['correct']),
                    'evidence': drug_evidence.get(d, {}),
                })
            return result

        patients[cell] = {
            'id': cell,
            'short_id': cell.replace('ACH-', ''),
            'tissue': tissue,
            'accuracy': round(accuracy, 3),
            'n_drugs': int(len(cp)),
            'n_mutations': len(mutations),
            'recommended': drug_list(recommended, 10),
            'consider': drug_list(consider, 8),
            'not_recommended': drug_list(not_recommended, 8),
            'mutations': mutations[:12],
            'similar_patients': similar_patients,
        }

    return patients, top_cells


def build_html(patients, top_cells, calibration, model_name, output_path):
    """Build the complete HTML page."""

    patients_json = json.dumps(patients, ensure_ascii=False)
    calibration_json = json.dumps(calibration, ensure_ascii=False)

    # JavaScript for the drug evidence detail
    js = """
function selectPatient(cellId, el) {
    document.querySelectorAll('.patient-item').forEach(e => e.classList.remove('active'));
    el.classList.add('active');
    const p = patients[cellId];
    if (!p) return;

    // Patient header
    document.getElementById('patient-header').innerHTML = `
        <div class="patient-card">
            <div class="patient-title">
                <span class="patient-id">${p.short_id}</span>
                <span class="patient-tissue">${p.tissue}</span>
            </div>
            <div class="patient-stats">
                <div class="stat">
                    <div class="stat-value">${(p.accuracy * 100).toFixed(0)}%</div>
                    <div class="stat-label">Model Accuracy</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${p.recommended.length}</div>
                    <div class="stat-label">Recommended</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${p.n_mutations}</div>
                    <div class="stat-label">Key Mutations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${p.n_drugs}</div>
                    <div class="stat-label">Drugs Screened</div>
                </div>
            </div>
        </div>`;

    // Drug recommendations
    let recHtml = '';
    if (p.recommended.length > 0) {
        recHtml += '<div class="tier-section"><div class="tier-header tier-green"><span class="tier-dot green"></span> Recommended (High Confidence)</div>';
        p.recommended.forEach(d => { recHtml += drugCard(d, 'green'); });
        recHtml += '</div>';
    }
    if (p.consider.length > 0) {
        recHtml += '<div class="tier-section"><div class="tier-header tier-amber"><span class="tier-dot amber"></span> Consider (Moderate Confidence)</div>';
        p.consider.forEach(d => { recHtml += drugCard(d, 'amber'); });
        recHtml += '</div>';
    }
    if (p.not_recommended.length > 0) {
        recHtml += '<div class="tier-section"><div class="tier-header tier-gray"><span class="tier-dot gray"></span> Not Recommended</div>';
        p.not_recommended.forEach(d => { recHtml += drugCard(d, 'gray'); });
        recHtml += '</div>';
    }
    document.getElementById('drug-recommendations').innerHTML = recHtml;

    // Mutations
    let mutHtml = '<div class="evidence-title">Pathogenic Mutations</div>';
    if (p.mutations.length > 0) {
        p.mutations.forEach(m => {
            const barW = (m.pathogenicity * 100).toFixed(0);
            mutHtml += `<div class="mutation-row">
                <span class="mutation-gene">${m.gene}</span>
                <div class="mutation-bar-bg"><div class="mutation-bar" style="width:${barW}%"></div></div>
                <span class="mutation-score">${m.pathogenicity.toFixed(2)}</span>
            </div>`;
        });
    } else {
        mutHtml += '<div class="empty-state">No high-confidence pathogenic mutations detected</div>';
    }
    document.getElementById('mutations-panel').innerHTML = mutHtml;

    // Similar patients
    let simHtml = '<div class="evidence-title">Similar Patients in Database</div>';
    p.similar_patients.forEach(s => {
        let drugTags = '';
        s.drugs_that_worked.forEach(d => {
            drugTags += `<span class="drug-mini-tag">${d.substring(0, 18)}</span>`;
        });
        simHtml += `<div class="similar-row">
            <div class="similar-info">
                <span class="similar-name">${s.name}</span>
                <span class="similar-tissue">${s.tissue}</span>
            </div>
            <div class="similar-sim">Similarity: ${(s.similarity * 100).toFixed(0)}%</div>
            <div class="similar-drugs">${drugTags || '<span class="empty-hint">No data</span>'}</div>
        </div>`;
    });
    document.getElementById('similar-panel').innerHTML = simHtml;
}

function drugCard(d, tier) {
    const scorePercent = (d.score * 100).toFixed(0);
    const correctIcon = d.correct ? '<span class="correct-icon">&#10003;</span>' : '<span class="wrong-icon">&#10007;</span>';
    const trueLabel = d.true_label ? 'Actually Sensitive' : 'Actually Resistant';

    let evidenceHtml = '';
    if (d.evidence) {
        // Gene targets (only if they exist)
        if (d.evidence.targets && d.evidence.targets.length > 0) {
            evidenceHtml += `<div class="drug-mechanism">Targets: ${d.evidence.targets.join(', ')}</div>`;
        }
        // Matching patient mutations (strongest evidence)
        if (d.evidence.matching_mutations && d.evidence.matching_mutations.length > 0) {
            evidenceHtml += `<div class="drug-match">Patient mutations match: <strong>${d.evidence.matching_mutations.join(', ')}</strong></div>`;
        }
        // Similar drugs (only if no gene targets — avoid redundancy)
        if ((!d.evidence.targets || d.evidence.targets.length === 0) && d.evidence.similar_drugs && d.evidence.similar_drugs.length > 0) {
            const simDrugs = d.evidence.similar_drugs.map(s => s[0].substring(0, 20) + ' (' + (s[1]*100).toFixed(0) + '%)').join(', ');
            evidenceHtml += `<div class="drug-similar">Structurally similar to: ${simDrugs}</div>`;
        }
    }

    return `<div class="drug-card ${tier}">
        <div class="drug-header">
            <span class="drug-name">${d.short_name}</span>
            <span class="drug-score score-${tier}">${scorePercent}%</span>
            ${correctIcon}
        </div>
        <div class="drug-bar-track"><div class="drug-bar-fill ${tier}" style="width:${scorePercent}%"></div></div>
        ${evidenceHtml ? '<div class="drug-evidence">' + evidenceHtml + '</div>' : ''}
        <div class="drug-ground-truth">${trueLabel}</div>
    </div>`;
}
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clinical Decision Support</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f7fa; color: #2d3748; font-size: 14px; }}

/* Header */
.top-bar {{ background: white; padding: 16px 32px; border-bottom: 1px solid #e2e8f0;
           display: flex; align-items: center; justify-content: space-between; }}
.top-bar h1 {{ font-size: 18px; font-weight: 600; color: #1a202c; }}
.top-bar .tag {{ background: #ebf4ff; color: #3182ce; padding: 4px 10px; border-radius: 4px;
                font-size: 12px; font-weight: 500; }}

/* Layout */
.layout {{ display: flex; height: calc(100vh - 57px); }}
.sidebar {{ width: 240px; background: white; border-right: 1px solid #e2e8f0;
           overflow-y: auto; flex-shrink: 0; }}
.sidebar-label {{ padding: 12px 16px 8px; font-size: 11px; font-weight: 600; color: #a0aec0;
                 text-transform: uppercase; letter-spacing: 0.05em; }}
.patient-item {{ padding: 10px 16px; cursor: pointer; border-left: 3px solid transparent;
                transition: background 0.15s; }}
.patient-item:hover {{ background: #f7fafc; }}
.patient-item.active {{ background: #ebf8ff; border-left-color: #3182ce; }}
.patient-name {{ font-weight: 600; font-size: 13px; color: #2d3748; }}
.patient-meta {{ font-size: 11px; color: #a0aec0; margin-top: 1px; }}
.patient-badge {{ float: right; font-size: 11px; font-weight: 600; padding: 1px 6px;
                 border-radius: 3px; }}
.badge-good {{ background: #c6f6d5; color: #276749; }}
.badge-ok {{ background: #fefcbf; color: #975a16; }}
.badge-bad {{ background: #fed7d7; color: #9b2c2c; }}

/* Main content */
.main {{ flex: 1; overflow-y: auto; padding: 24px 32px; }}
.patient-card {{ background: white; border-radius: 8px; padding: 20px 24px;
                border: 1px solid #e2e8f0; margin-bottom: 20px; }}
.patient-title {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; }}
.patient-id {{ font-size: 22px; font-weight: 700; color: #1a202c; }}
.patient-tissue {{ font-size: 14px; color: #718096; background: #edf2f7; padding: 2px 8px;
                  border-radius: 4px; }}
.patient-stats {{ display: flex; gap: 32px; }}
.stat-value {{ font-size: 24px; font-weight: 700; color: #2b6cb0; }}
.stat-label {{ font-size: 11px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.03em; }}

/* Two-column body */
.body-grid {{ display: grid; grid-template-columns: 1fr 340px; gap: 20px; }}

/* Drug tiers */
.tier-section {{ margin-bottom: 16px; }}
.tier-header {{ font-size: 13px; font-weight: 600; padding: 8px 0; display: flex; align-items: center; gap: 8px; }}
.tier-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.tier-dot.green {{ background: #48bb78; }}
.tier-dot.amber {{ background: #ed8936; }}
.tier-dot.gray {{ background: #a0aec0; }}
.tier-green {{ color: #276749; }}
.tier-amber {{ color: #975a16; }}
.tier-gray {{ color: #718096; }}

.drug-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 6px;
             padding: 12px 16px; margin-bottom: 8px; transition: box-shadow 0.15s; }}
.drug-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.drug-card.green {{ border-left: 3px solid #48bb78; }}
.drug-card.amber {{ border-left: 3px solid #ed8936; }}
.drug-card.gray {{ border-left: 3px solid #cbd5e0; }}
.drug-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.drug-name {{ font-weight: 600; font-size: 13px; flex: 1; color: #2d3748; }}
.drug-score {{ font-weight: 700; font-size: 13px; }}
.score-green {{ color: #276749; }}
.score-amber {{ color: #975a16; }}
.score-gray {{ color: #718096; }}
.correct-icon {{ color: #48bb78; font-weight: bold; }}
.wrong-icon {{ color: #fc8181; font-weight: bold; }}
.drug-bar-track {{ height: 4px; background: #edf2f7; border-radius: 2px; margin-bottom: 6px; }}
.drug-bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}
.drug-bar-fill.green {{ background: #48bb78; }}
.drug-bar-fill.amber {{ background: #ed8936; }}
.drug-bar-fill.gray {{ background: #cbd5e0; }}
.drug-evidence {{ font-size: 12px; color: #718096; margin-top: 4px; }}
.drug-mechanism {{ margin-bottom: 2px; }}
.drug-match {{ color: #2b6cb0; font-size: 12px; }}
.drug-similar {{ color: #a0aec0; font-size: 11px; }}
.drug-ground-truth {{ font-size: 11px; color: #a0aec0; margin-top: 4px; font-style: italic; }}

/* Right sidebar panels */
.side-panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px;
              padding: 16px; margin-bottom: 16px; }}
.evidence-title {{ font-size: 13px; font-weight: 600; color: #1a202c; margin-bottom: 12px;
                  padding-bottom: 8px; border-bottom: 1px solid #edf2f7; }}

/* Mutations */
.mutation-row {{ display: flex; align-items: center; gap: 8px; padding: 4px 0; }}
.mutation-gene {{ font-weight: 600; font-size: 12px; width: 80px; color: #553c9a; }}
.mutation-bar-bg {{ flex: 1; height: 6px; background: #edf2f7; border-radius: 3px; }}
.mutation-bar {{ height: 100%; background: #805ad5; border-radius: 3px; }}
.mutation-score {{ font-size: 11px; color: #a0aec0; width: 35px; text-align: right; }}

/* Similar patients */
.similar-row {{ padding: 8px 0; border-bottom: 1px solid #f7fafc; }}
.similar-info {{ display: flex; gap: 8px; align-items: center; }}
.similar-name {{ font-weight: 600; font-size: 12px; color: #2b6cb0; }}
.similar-tissue {{ font-size: 11px; color: #a0aec0; }}
.similar-sim {{ font-size: 11px; color: #718096; margin: 2px 0; }}
.similar-drugs {{ display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }}
.drug-mini-tag {{ background: #ebf8ff; color: #2b6cb0; padding: 1px 6px; border-radius: 3px;
                 font-size: 10px; }}
.empty-state {{ color: #cbd5e0; font-size: 12px; font-style: italic; padding: 8px 0; }}
.empty-hint {{ color: #cbd5e0; font-size: 10px; }}

/* Calibration */
.calibration-row {{ display: flex; align-items: center; gap: 8px; padding: 3px 0; }}
.cal-label {{ font-size: 11px; width: 60px; color: #718096; }}
.cal-bar-bg {{ flex: 1; height: 8px; background: #edf2f7; border-radius: 4px; }}
.cal-bar {{ height: 100%; background: #3182ce; border-radius: 4px; }}
.cal-value {{ font-size: 11px; font-weight: 600; width: 40px; text-align: right; color: #2d3748; }}
</style>
</head>
<body>

<div class="top-bar">
    <h1>Drug Sensitivity - Clinical Decision Support</h1>
    <span class="tag">{model_name}</span>
</div>

<div class="layout">
    <div class="sidebar">
        <div class="sidebar-label">Patients</div>
"""

    # Sidebar patient list
    for i, cell in enumerate(top_cells):
        p = patients[cell]
        acc = p['accuracy']
        badge_class = 'badge-good' if acc >= 0.8 else 'badge-ok' if acc >= 0.65 else 'badge-bad'
        active = ' active' if i == 0 else ''
        html += f"""
        <div class="patient-item{active}" onclick="selectPatient('{cell}', this)">
            <span class="patient-badge {badge_class}">{acc:.0%}</span>
            <div class="patient-name">{p['short_id']}</div>
            <div class="patient-meta">{p['tissue']}</div>
        </div>"""

    html += """
    </div>
    <div class="main">
        <div id="patient-header"></div>
        <div class="body-grid">
            <div id="drug-recommendations"></div>
            <div>
                <div class="side-panel" id="mutations-panel"></div>
                <div class="side-panel" id="similar-panel"></div>
                <div class="side-panel" id="calibration-panel">
                    <div class="evidence-title">Model Calibration</div>
"""

    # Static calibration bars
    for cal in calibration:
        bar_w = int(cal['accuracy'] * 100)
        html += f"""
                    <div class="calibration-row">
                        <span class="cal-label">{cal['bin']}</span>
                        <div class="cal-bar-bg"><div class="cal-bar" style="width:{bar_w}%"></div></div>
                        <span class="cal-value">{cal['accuracy']:.0%}</span>
                    </div>"""

    html += """
                </div>
            </div>
        </div>
    </div>
</div>

<script>
"""
    html += f"const patients = {patients_json};\n"
    html += js
    html += f"\nwindow.addEventListener('load', function() {{ selectPatient('{top_cells[0]}', document.querySelector('.patient-item')); }});\n"
    html += "</script>\n</body>\n</html>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.0f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed_v2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'results/{args.model_name}/interactive_dashboard'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"Clinical Decision Panel: {args.model_name}")
    print("=" * 60)

    print("\n--- Loading ---")
    data = load_all_data(args.model_name, args.data_dir, device)

    print("\n--- Building ---")
    patients, top_cells = build_patient_data(data)
    output_path = os.path.join(args.output_dir, 'clinical_panel.html')
    build_html(patients, top_cells, data['calibration'], args.model_name, output_path)

    print(f"\n  Open: {output_path}")


if __name__ == '__main__':
    main()
