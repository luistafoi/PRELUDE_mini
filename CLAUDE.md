# PRELUDE — HetGNN for Drug Response Prediction

Inductive heterogeneous GNN predicting drug-cell interactions. Active work: **v2 pipeline** with PRISM dose-response AUC labels.
Branch: `feature/inductive-triplet-loss` · Env: `hgb_env` (Python 3.12) · Active data: `data/processed_v2_dedup/` (symlinked mirror of `data/processed_v2/` with dedup'd neighbor pickle)

## Running things
- Always use `conda run -n hgb_env --no-capture-output python <script> ...` — no bare `python`.
- Long runs go in **tmux**; tee to `results/<model>/<task>.log`. Single GPU, `--gpu 0`.
- **Never** run destructive git ops (`reset --hard`, `push --force`, branch delete) without explicit ask.
- LFS data files are `.gitignore`d — never `git add -A` blindly.

## Graph
6 edge types: `cell-drug(0)`, `gene-gene(1)`, `cell-gene mutation(2)`, `gene-drug inhibition(3)`, `cell-cell similarity(4)`, `drug-drug Tanimoto(5)`.
Counts (v2): 947 cells, 1,720 drugs, 18,202 genes.

## Entry points
- **Train**: `scripts/train.py` → `checkpoints/<model_name>.pth` + `<model_name>_log.csv`
- **Tune**: `scripts/tune_optuna.py` (embed_d, n_layers, isolation_ratio, dropout, lr, align_lambda, neighbor_sample_size)
- **Inference**: `scripts/inference_sanger.py`, `scripts/inference_tcga.py`, `scripts/inference_incremental.py`
- **Predictions + paths**: `scripts/export_predictions.py` → `scripts/analyze_paths.py`

## V2 pipeline (current, use this)
`pipeline_v2/step1_resolve_ids.py → step2_label_gmm.py → step3_build_graph.py → step3b_drug_similarity.py → step4_create_splits.py → step5_build_features.py → step6_tcga_eval.py` then `scripts/generate_neighbors.py --include_cell_drug`.

V1 (`data/processed/`, LMFI labels) is **deprecated** — don't touch unless asked.

## Architecture flags (all additive, default-off)
- `--include_cell_drug` — Cell-Drug in GNN (v2 requires this)
- `--dual_head` — separate inductive + transductive heads; **requires `--include_cell_drug`**
- `--isolation_ratio` — Cell↔Drug edge masking per batch; **requires `--include_cell_drug`**
- `--backbone_lr_scale 0.1` — slow backbone, fast heads; best cross-dataset transfer
- `--inductive_loss_weight 3.0` — isolated cells contribute 3× to loss
- `--cell_feature_source {vae|scgpt|multiomic|hybrid}` — determines cell feature dim

## Data pipeline flags
- `generate_neighbors.py --dedup_symmetric` — skip redundant symmetric edges before pickle (used for `processed_v2_dedup/`)
- `DataGenerator(..., dedup_symmetric=True)` — same behavior at runtime; must match how pickle was generated

Implementation: `models/tools.py` (HetAgg: `create_isolation_masks`, `setup_link_prediction`, `link_prediction_forward`), `scripts/train.py` (param-group LR).

## Critical invariants (these bite)
- **`--max_neighbors` must match** the value used in `generate_neighbors.py`. Changing it requires **regenerating `train_neighbors_preprocessed.pkl`**.
- **Feature alignment is by uppercase name**. Missing names → **silent zero embeddings**. Check loader logs.
- **Label threshold**: `>0.5` = positive, `≤0.5` = negative in classification mode.
- Validation priority: `valid_transductive.dat` first, fallback to `valid_inductive.dat`.
- Graph edges are made bidirectional in `DataGenerator._build_neighbor_dicts` — don't double-add.
- Cell multiomic dim = 3858 (964 genes × 4 channels + 2 flags). Hybrid = 4370 (+512 VAE). scGPT = 512. VAE = 512. Gene ESM = 1536.

## Current state (v2)
- **Best model**: `v2_M04_dedup` (same arch as M03, trained on dedup'd neighbor graph — 341k duplicate symmetric edges removed, mostly gene-gene)
- DepMap test ind = **0.956**, trans = 0.957 (≈ M03)
- Sanger S1 (known cells/drugs) = **0.848**, S3 (new cells) = **0.842** (slight wins over M03)
- Sanger S2/S4 (new drugs) = 0.574 / 0.561 (-2.5 pts vs M03; within seed noise; still unsolved)
- TCGA: evaluated on M03 baseline (overall = 0.50, biological gap); per-drug: Tamoxifen 0.667, Gemcitabine 0.661, BRCA 0.688 — re-evaluate on M04 when relevant
- **Legacy M03 kept at `checkpoints/v2_M03_drug_sim.pth`** for comparison

## Known design decisions (from experiments)
- **Dedup kept** despite small S2/S4 drop — graph cleanliness + reviewer defensibility + tied on main metrics
- **M05 tested**: `--weighted_agg_pairs "drug:drug"` (proper weighted mean for drug-drug). ≈ M04 on every metric — weighted mean at forward time doesn't replicate the sampling-frequency effect that duplicates produced in M03. Flag retained for future use.
- **Bug 7 (triplet/isolation subgraph consistency) is NOT a real bug.** train.py:504-576 merges triplet seeds into the subgraph before one shared GNN forward, so LP and triplet loss use identical embeddings under identical isolation masks. Audit flag cleared.
- **Next direction**: mechanism-aware drug embeddings (swap MoleculeSTM → drug-target profile features). Needs detailed planning before implementation.

## Conventions
- **Additive, non-destructive**: flags default to old behavior; never break existing `run_*.sh`.
- **Script-first for experiments**: write `run_<name>.sh`, then execute via tmux.
- Results tree: `results/<model>/{predictions,interpretability,...}/`.
- Memory at `~/.claude/projects/-data-luis-PRELUDE-mini-inductive/memory/` has experiment history, TCGA results, architecture notes. Point-in-time — verify before citing.

## Token-efficient collaboration
- Don't re-read files unchanged in this session.
- Skip preambles ("I'll now..."); brief status on blockers/results only.
- Grep/Glob before Read when exploring. Read only the lines needed.
- Spawn Explore agents for open-ended searches (>3 queries); protects main context.
- Parallel tool calls for independent ops.
- Final summaries: 1–2 sentences.
- No code comments unless *why* is non-obvious.

## User preferences
- Discuss before implementing non-trivial changes; present 2–3 options with tradeoffs.
- Prioritize **clinical actionability** (no retraining per dataset, interpretable outputs).
- Focus shifting: architecture tuning → data alignment, benchmarking vs. DIPK/GraphTCDR/DiSyn, publication.
- Target venues: Nature Communications → Briefings in Bioinformatics → Bioinformatics.
