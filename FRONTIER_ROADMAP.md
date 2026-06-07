# FRONTIER RESEARCH ROADMAP

## Physics-Informed Water Quality ML — Chattahoochee River (USGS 02334500)

**Current state:** Temp R² 0.502 / DO R² 0.509 | **Target:** R² 0.85+ | **Best checkpoint:** Epoch 31

See also: [`report.md`](report.md) for full codebase documentation.

---

## DIAGNOSIS: WHY YOU'RE STUCK AT R² 0.50

Your model has no meteorological forcing. Water temperature is *causally* driven by solar radiation, not by its own history. Every top-performing water quality model in literature (R² 0.92–0.98) includes shortwave radiation and air temperature. This is almost certainly the primary reason the model smooths over diurnal peaks and troughs. Fix this first — everything else builds on top of it.

---

## INITIATIVE 01 — ADD METEOROLOGICAL FORCING

**Impact:** Fastest path to R² 0.75+ | **Code effort:** Low | **Status:** In progress

Water temperature is a heat balance problem. Solar radiation input, air temperature, wind speed, and humidity are the *causal drivers* of the diurnal cycle your model currently can't track. You're asking the model to predict an effect without giving it the cause.

**What to do:**

- Add `data/meteo_fetch.py` — pull Open-Meteo historical + forecast API for lat/lon 34.18, -84.09 at 15-min intervals (free, reanalysis-quality, no API key needed)
- New feature group in `data/features.py`: `shortwave_rad`, `air_temp`, `wind_speed`, `cloud_cover`, `rh` — adds ~5 features (total ~28)
- For the decoder: pass the 24h NWS/Open-Meteo forecast as *known future covariates* — the model sees tomorrow's weather when forecasting tomorrow's water temperature
- Run `--no-early-stop` through at least epoch 135 so the physics curriculum fully engages

**Expected lift:** Temp R² +0.20–0.30 | DO R² +0.10–0.15  
**Est. resulting R²:** Temp ~0.72, DO ~0.62

**Implementation in repo:**

```bash
python -m data.meteo_fetch --config configs/high_accuracy.yaml
python3 train.py --config configs/high_accuracy.yaml --no-early-stop
```

---

## INITIATIVE 02 — ACTIVATE THE MULTI-SITE GRAPH (DAM CAUSALITY)

**Impact:** Biggest structural blind spot | **Code effort:** Medium | **Status:** Not started

Site 02334500 is immediately downstream of Buford Dam (Lake Lanier). Dam operations — cold hypolimnetic releases, power generation pulses — directly control temperature and DO at your station with a predictable travel-time lag of roughly 15–90 minutes. Your model has zero knowledge of this. The GAT layer is already written in your codebase — you just need to populate the data.

**What to do:**

- Add upstream/downstream USGS sites to `stations.json`: 02334430 (above dam), 02335000, 02335350, 02335757
- Add travel-time delay as an edge feature in `data/graph_builder.py` — each upstream→downstream edge carries a learned lag parameter
- Engineer a **dam release signal**: rate-of-change of gage height at 02334430 as a feature in `data/features.py` — when the dam releases, this spikes before temperature/DO respond
- Upgrade `gat_layer.py` to use edge attributes with lag-aware message passing: node j sends message to node i with time shift τ_ij
- Ablation: train N=1, N=3, N=6 — quantify R² gain per additional node

**Expected lift:** Temp R² +0.10–0.15 | DO R² +0.15–0.20  
**Novel scientific angle:** First model to explicitly encode hydropower dam release causality in a water quality graph network

---

## INITIATIVE 03 — KOOPMAN OPERATOR EMBEDDING

**Impact:** Highest novelty | Publishable in Nature Water, WRR | **Code effort:** Medium-High | **Status:** Not started

This is the most scientifically novel initiative. The Benson–Krause equation and reaeration ODE are approximations. The true governing dynamics of river DO and temperature are a nonlinear system whose phase-space structure is unknown. Koopman operator theory says any nonlinear system has an exact infinite-dimensional *linear* representation in a lifted observable space — and we can learn that lifting with a neural network.

Train a deep autoencoder to map (T, DO, discharge, time-of-day) into a ~32-dimensional Koopman latent space where dynamics are linear and predictable. The autoencoder enforces K·z_t ≈ z_{t+1} via a learned linear transition matrix K. Forecast in latent space; decode back to physical units.

The payoff beyond accuracy: the eigenvalues of K directly reveal the dominant oscillation frequencies of the river — diurnal cycles, storm events, dam-forced modes — without any manual feature engineering. This is **new physics discovery**, not just better prediction.

**What to do:**

- New module `models/koopman_encoder.py` — encoder E, decoder D, linear operator K (32×32 learnable matrix)
- Triple loss: reconstruction ||D(E(x))−x|| + prediction ||E(x_{t+1})−K·E(x_t)|| + multi-step consistency ||K^n·E(x_t)−E(x_{t+n})|| for n∈{1,4,24}
- Use Koopman latent z as an additional input channel to the existing PatchTST horizon decoder
- After training: SVD of K → plot eigenvalue spectrum → identify dominant river modes → compare to known diurnal/tidal frequencies
- Ablation: Koopman+PatchTST vs. PatchTST alone to prove the latent space adds accuracy

**Expected lift:** Temp R² +0.15–0.20 | DO R² +0.12–0.18  
**Discovery potential:** Quantitative characterization of how Buford Dam regulation distorts the natural Koopman spectrum of the river — a publishable finding in its own right

---

## INITIATIVE 04 — CONDITIONAL DIFFUSION PROBABILISTIC FORECASTING

**Impact:** First in field for DO | Publishable at NeurIPS/ICLR | **Code effort:** High | **Status:** Not started

Your model outputs a single deterministic forecast. Real operators need: "What is the probability that DO drops below 2 mg/L in the next 6 hours?" That requires a *distribution* over futures. No existing water quality ML system produces this.

Implement a conditional score-based diffusion model: the context window is the conditioning signal, and the model learns to sample from p(y|x) by iteratively denoising random noise. Each sample is a physically plausible 24-hour trajectory. With 100 samples you get prediction intervals, risk probabilities, and tail-event detection.

**What to do:**

- New module `models/diffusion_forecaster.py` — 1D U-Net denoiser conditioned on existing PatchTST embeddings (reuse your encoder)
- Training: add Gaussian noise at timestep t to target y; train denoiser ε_θ(y_t, t, context) to predict the noise
- Inference: DDIM sampling (50 steps); generate 100 trajectory samples per input window
- **Novel contribution**: physics-constrained sampling — reject or reweight samples that violate the DO_sat ceiling from PhysicsProjectionHead
- New metrics: hypoxia exceedance probability curve (P(DO < 2 mg/L) vs lead time); calibration plots; CRPS score

**Expected lift:** Hypoxia F1 +0.25–0.35 | CRPS benchmark state-of-art  
**Discovery potential:** First calibrated early warning system for river hypoxia — identifies which upstream signals are 6–12h precursors to low-DO events

---

## INITIATIVE 05 — NEURAL ODE BACKBONE

**Impact:** Best physics fidelity | Handles data gaps | **Code effort:** High | **Status:** Not started

The current model treats time as discrete 96-step patches. But the underlying physics is continuous: dDO/dt = k·(DO_sat−DO) − BOD − ... PatchTST cannot represent this correctly, especially across data gaps (sensor outages) which are common in NWIS data.

Replace the backbone with a latent Neural ODE: encode the context window to an initial latent state z₀, then integrate a learned ODE dz/dt = f_θ(z, t) forward 24 hours using an adaptive RK45 solver (torchdiffeq). Decode z(t) → (T, DO). This automatically handles irregular observation times and is guaranteed smooth. The learned f_θ is the discovered governing equation.

**What to do:**

- Add `torchdiffeq` to requirements.txt; new `models/neural_ode_backbone.py`
- ODE function: small 3-layer MLP with physics soft-constraints (reaeration term, Benson–Krause coupling) baked into f_θ architecture — the network learns corrections to the analytical model
- Encoder: compress 96-step context → z₀ (reuse existing PatchTST encoder, remove horizon decoder)
- Support `irregular_timestamps: true` in dataset.py — pass actual timestamps to ODE integrator instead of assuming uniform 15-min grid
- Sensitivity analysis of learned f_θ: which terms dominate? Compare learned k₂(Q) curve to O'Connor–Dobbins formula in literature

**Expected lift:** Temp R² +0.08–0.12 | Transforms gap-handling from heuristic to principled  
**Discovery potential:** Data-driven reaeration rate as a function of discharge — corrects 60-year-old empirical formulas with actual Chattahoochee observations

---

## INITIATIVE 06 — MAMBA STATE-SPACE BACKBONE

**Impact:** 4× longer context at same compute | Architecture efficiency | **Code effort:** Medium | **Status:** Not started

PatchTST's Transformer is O(L²) in sequence length. Extending context from 96 to 384 steps (4 days) or 2016 steps (3 weeks) — needed to capture multi-day thermal regimes and slow hypoxia buildup — is computationally prohibitive with attention. Mamba (S4/S6) achieves O(L) with selective state-space modeling that adaptively chooses what to remember.

**What to do:**

- Add `mamba-ssm` to requirements.txt; implement `models/mamba_backbone.py` as drop-in for `patchtst.py`
- Extend `seq_len` from 96 → 384 in configs; measure val R² vs context length (96/192/384)
- Novel architectural contribution: condition Mamba's selection mechanism (Δ parameter) on DO-deficit and time-of-day features — physics-selective memory
- Benchmark: MambaPhysics vs PatchTST at identical compute budget across context lengths

**Expected lift:** Temp R² +0.10–0.15 from longer context | 4× faster training per epoch

---

## INITIATIVE 07 — PRE-TRAIN A WATER QUALITY FOUNDATION MODEL

**Impact:** National-scale generalization | Publishable in Nature/Science | **Code effort:** Very High | Timeline: 6–12 months | **Status:** Not started

USGS operates 8,700+ active monitoring sites. Hundreds have decades of temperature and DO records. A self-supervised pre-training phase on this corpus would give the model general hydrological representations before fine-tuning on Chattahoochee — similar to how HydroGEM (2025) does it for discharge QC, but applied to water quality for the first time.

The resulting model could do zero-shot forecasting at any new USGS site with minimal data. No water quality foundation model exists anywhere.

**What to do:**

- Extend `nwis_fetch.py` to bulk-download 500+ sites with temp+DO; write to sharded parquet dataset
- Pre-training objective: masked patch prediction (mask 25% of patches; predict them from context) using existing PatchTST or Mamba backbone
- Station embedding: add NLCD land-use, StreamStats basin area, slope as static covariates encoded as learnable site_id vectors
- Fine-tune on 02334500 from pre-trained weights; measure few-shot efficiency vs training from scratch
- Zero-shot eval: apply to 3 held-out USGS sites in different river systems with no fine-tuning

**Expected lift:** Fine-tune R² +0.15–0.25 | Zero-shot R² ~0.60+ at unseen sites  
**Discovery potential:** What hydrological representations are universal vs site-specific? The pre-trained site embeddings cluster rivers by geomorphic type — a new taxonomy of US river dynamics

---

## RECOMMENDED EXECUTION ORDER

| Phase | Initiative | Focus |
|-------|------------|--------|
| Week 1–2 | **01** Meteo forcing | Lowest effort, highest immediate R² gain; `--no-early-stop` through epoch 135 |
| Week 2–4 | **02** Multi-site graph + dam | Data + config; GAT already implemented |
| Month 2 | **03** Koopman encoder | Parallel branch → horizon decoder; eigenspectrum analysis |
| Month 2–3 | **04** Diffusion head | Hypoxia probability curves on trained encoder |
| Month 3–4 | **05 + 06** Neural ODE vs Mamba | Ablations; pick winner |
| Month 6+ | **07** Foundation model | 500+ USGS sites; long-game publication |

---

## PROJECTED R² TRAJECTORY

| After initiative | Temp R² | DO R² |
|------------------|---------|-------|
| Baseline (now) | 0.50 | 0.51 |
| + Meteo forcing (01) | ~0.72 | ~0.62 |
| + Multi-site graph (02) | ~0.78 | ~0.75 |
| + Koopman (03) | ~0.82 | ~0.82 |
| + Diffusion + Neural ODE (04+05) | ~0.85 | ~0.85 |
| + Foundation model fine-tune (07) | ~0.88 | ~0.88 |

---

## CODEBASE READINESS BY INITIATIVE

| Initiative | Existing infrastructure | Gap |
|------------|-------------------------|-----|
| 01 Meteo | `features.py`, `dataset.py`, horizon decoder | `meteo_fetch.py`, future covariate path in decoder |
| 02 Multi-site | `nwis_fetch.py`, `gat_layer.py`, `graph_builder.py` | Sites in `stations.json`, lag-aware GAT |
| 03 Koopman | PatchTST encoder, physics loss | `models/koopman_encoder.py`, triple loss |
| 04 Diffusion | Physics head, PatchTST embeddings | `models/diffusion_forecaster.py`, CRPS metrics |
| 05 Neural ODE | `physics/do_saturation.py`, reaeration helper | `torchdiffeq`, irregular timestamps |
| 06 Mamba | `patchtst.py` plug-in pattern | `mamba-ssm`, extended `seq_len` configs |
| 07 Foundation | `nwis_fetch.py`, PatchTST | Bulk download, masked pretrain loop |
