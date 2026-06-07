# Physics-Informed Spatiotemporal Water-Quality Forecasting

PyTorch framework for joint **water temperature** and **dissolved oxygen** forecasting on USGS monitoring networks, with graph attention (GAT), PatchTST (or TCN ablation), Benson–Krause DO saturation physics loss, and hypoxia explainability (attention + Captum IG / optional SHAP).

## Quick start (single site, existing RDB files)

Repository includes 15-minute USGS IV data for site **02334500** (`water_temperature.rdb`, `disolved_oxygen.rdb`).

```bash
cd /path/to/research
source myenv/bin/activate
pip install -r requirements.txt
python3 train.py --config configs/default.yaml
```

### Long unattended training (recommended)

Uses [`configs/long_run.yaml`](configs/long_run.yaml): **300 epochs**, patience **40**, checkpoints every **10** epochs, full resume support.

```bash
python3 train.py --config configs/long_run.yaml
```

| Artifact | Purpose |
|----------|---------|
| `checkpoints/best.pt` | Best validation model |
| `checkpoints/last.pt` | Latest epoch (optimizer + scheduler) for `--resume` |
| `checkpoints/epoch_*.pt` | Periodic snapshots (keeps last 5) |
| `logs/train_log.csv` | Per-epoch metrics (append-only) |
| `logs/train_log.jsonl` | Same log, JSON lines backup |

**Resume after Ctrl+C or crash:**

```bash
python3 train.py --config configs/long_run.yaml --resume checkpoints/last.pt
```

**Train the full epoch budget without early stopping:**

```bash
python3 train.py --config configs/long_run.yaml --no-early-stop
```

## High-accuracy physics-informed training (recommended)

Uses horizon cross-attention decoder, physics projection head, composite loss in physical units, and early stopping on `val_phys_mean_rmse`.

```bash
source myenv/bin/activate
python -m data.nwis_fetch --config configs/chattahoochee_graph.yaml
python3 train.py --config configs/high_accuracy.yaml
python3 scripts/eval_horizon.py --checkpoint checkpoints/best.pt
python3 scripts/plot_forecast.py
```

Config: [`configs/high_accuracy.yaml`](configs/high_accuracy.yaml). Works with local RDB files if NWIS fetch has not been run (single-site fallback).

## NWIS multi-site fetch (Chattahoochee)

```bash
python -m data.nwis_fetch --config configs/chattahoochee_graph.yaml
python3 train.py --config configs/high_accuracy.yaml
```

Sites with concurrent parameter codes **00010** (temperature) and **00300** (DO) are filtered by `min_paired_fraction` in the graph config. Outputs:

- `data/raw/{site_no}/iv.parquet`
- `data/raw/stations.json`
- `data/raw/edges.csv`

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `forecast.seq_len` | 96 | 24 h context @ 15 min |
| `forecast.pred_len` | 96 | 24 h horizon |
| `model.backbone` | `patchtst` | `patchtst` or `tcn` |
| `physics.lambda_physics` | 0.1 | Supersaturation penalty weight |
| `training.device` | `auto` | `auto` → CUDA, else Apple **MPS** (Metal GPU), else CPU |
| `metrics.hypoxia_threshold_mg_l` | 2.0 | Hypoxia event threshold |

### GPU acceleration

On **Apple Silicon** (M1/M2/M3), training uses **MPS** (Metal) automatically when `training.device: auto`. You should see:

`Training on mps (Apple GPU / Metal)...`

On **NVIDIA** Linux/Windows, `auto` selects CUDA. Force a backend:

```bash
python3 train.py --device mps    # Apple GPU
python3 train.py --device cuda   # NVIDIA GPU
python3 train.py --device cpu    # fallback
```

## Architecture

```
Input [B, N, T, F] → GAT (per timestep) → PatchTST/TCN → [B, N, H, 2]
```

Loss: `MSE + λ · mean(relu(DO_pred − DO_sat(T_pred))²)` with DO_sat from Benson–Krause (see `physics/do_saturation.py`).

**N=1 mode:** degenerate self-loop graph; works with bundled RDB files without NWIS fetch.

## Explainability

```bash
python train.py --config configs/default.yaml --explain_hypoxia
```

Generates under `figures/`:

- `attention_hypoxia_*.png` — encoder attention on hypoxia windows
- `ig_hypoxia.png` — Integrated Gradients (Captum)
- `shap_hypoxia.png` — optional (`explain.use_shap: true`)

## Ablations

```bash
python train.py --backbone tcn
# Edit configs/default.yaml: physics.lambda_physics: 0 for no-physics ablation
```

## Deprecated baseline

`model.py` is a thin wrapper; the original TensorFlow LSTM + polynomial DO heuristic is replaced by end-to-end multivariate learning.

## Project layout

```
configs/          YAML configs
data/             RDB parser, NWIS fetch, dataset, features
models/           GAT, PatchTST, TCN, spatiotemporal model
physics/          DO saturation
losses/           Physics-informed loss
training/         Trainer, metrics
explain/          Attention + IG/SHAP
viz/              Publication matplotlib style + forecast plots
train.py          CLI entry
```

## Citation

When using the physics term, cite Benson & Krause (1984) for DO saturation (documented in `physics/do_saturation.py`).
