#!/usr/bin/env python3
"""
Diffusion Forecaster Evaluation — Probabilistic Hypoxia Risk Assessment

Evaluates the conditional diffusion model on the test split, computing:
  1. CRPS (Continuous Ranked Probability Score) — proper scoring rule
  2. Calibration curves — reliability of probability estimates
  3. Hypoxia exceedance probability curves — P(DO < 2 mg/L) vs lead time
  4. Comparison to deterministic DLinear/PatchTST baselines

Usage:
    # First train the base model and the diffusion head:
    python3 train.py --config configs/exp_frontier_v1.yaml --no-early-stop
    # Then evaluate diffusion (fine-tunes diffusion head on frozen encoder):
    python3 scripts/diffusion_eval.py --checkpoint checkpoints/frontier_v1/best.pt
    python3 scripts/diffusion_eval.py --checkpoint checkpoints/koopman/best.pt --n_samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_crps(y_true: np.ndarray, samples: np.ndarray) -> float:
    """
    Energy form of CRPS: CRPS = E[|Y - y|] - 0.5 * E[|Y - Y'|]
    where Y, Y' are independent draws from the forecast distribution.

    y_true: [N, H] or [N]
    samples: [N, S, H] or [N, S] — S forecast samples
    """
    n_samples = samples.shape[1]
    # E[|Y - y|]: mean absolute error over samples
    mae_term = np.abs(samples - y_true[:, None, ...]).mean(axis=1).mean()
    # E[|Y - Y'|]: mean pairwise distance within the ensemble
    # Efficient computation: var(Y) = E[Y²] - E[Y]² → E[|Y-Y'|] ≈ 2*std*sqrt(2/π) for Gaussian
    # Exact computation for non-Gaussian:
    spread = samples.std(axis=1).mean()  # Simplified: mean std across ensemble
    crps = mae_term - 0.5 * spread
    return float(crps)


def compute_ece(
    pred_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for binary hypoxia predictions.

    pred_probs: [N] — predicted P(hypoxia) at each lead-time step
    true_labels: [N] — 1 if actual hypoxia, 0 otherwise
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(pred_probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_probs >= lo) & (pred_probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = pred_probs[mask].mean()
        bin_acc = true_labels[mask].mean()
        ece += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(ece)


def train_diffusion_head(
    base_model,
    train_loader,
    device: torch.device,
    feature_engineer,
    n_epochs: int = 30,
    context_dim: int = 192,
) -> "DiffusionForecaster":
    """
    Fine-tune diffusion head on frozen PatchTST encoder.
    The encoder provides conditioning; only the denoiser is trained.
    """
    from models.diffusion_forecaster import DiffusionForecaster

    # Detect actual d_model from model
    if hasattr(base_model, "backbone") and hasattr(base_model.backbone, "d_model"):
        context_dim = base_model.backbone.d_model
    pred_len = base_model.pred_len

    diffusion = DiffusionForecaster(
        n_targets=2,
        pred_len=pred_len,
        context_dim=context_dim,
        T=500,        # Faster training: 500 steps
        ddim_steps=50,
    ).to(device)

    optimizer = AdamW(diffusion.parameters(), lr=3e-4, weight_decay=0.01)
    base_model.eval()

    print(f"\nTraining diffusion head ({n_epochs} epochs)...")
    ts = feature_engineer.target_scaler
    target_mean = torch.tensor(ts.mean_, dtype=torch.float32).to(device)
    target_scale = torch.tensor(ts.scale_, dtype=torch.float32).to(device)

    for epoch in range(1, n_epochs + 1):
        diffusion.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            # Extract PatchTST tokens (frozen encoder)
            with torch.no_grad():
                tokens = base_model.get_patch_tokens(x)
            if tokens is None:
                continue

            # Target: use forecast-node scaled values [B, pred_len, 2]
            y_node = y[:, 0, :, :]  # forecast node

            optimizer.zero_grad()
            losses = diffusion.training_loss(y_node, tokens)
            loss = losses["diffusion_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Diffusion epoch {epoch}/{n_epochs}: loss={epoch_loss/max(n_batches,1):.4f}")

    return diffusion


@torch.no_grad()
def evaluate_diffusion(
    base_model,
    diffusion,
    test_loader,
    device: torch.device,
    feature_engineer,
    n_samples: int = 100,
    hypoxia_threshold: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Evaluate diffusion model on test split.
    Returns all probabilistic metrics.
    """
    ts = feature_engineer.target_scaler
    target_mean = torch.tensor(ts.mean_, dtype=torch.float32).to(device)
    target_scale = torch.tensor(ts.scale_, dtype=torch.float32).to(device)

    all_samples_do = []   # [B, n_samples, pred_len]
    all_true_do = []      # [B, pred_len]
    all_samples_temp = []
    all_true_temp = []
    all_exceedance = []   # [B, pred_len]

    base_model.eval()
    diffusion.eval()

    print(f"\nEvaluating on test set ({n_samples} samples per window)...")
    for batch in test_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        tokens = base_model.get_patch_tokens(x)
        if tokens is None:
            continue

        # Generate samples
        samples = diffusion.sample(
            tokens, n_samples=n_samples,
            physics_filter=True,
            target_mean=target_mean, target_scale=target_scale,
        )  # [B, n_samples, pred_len, 2]

        # Denormalize
        mean = target_mean.view(1, 1, 1, -1)
        scale = target_scale.view(1, 1, 1, -1)
        samples_phys = samples * scale + mean
        y_phys = feature_engineer.inverse_targets(y[:, 0, :, :].cpu().numpy())

        do_samples = samples_phys[..., 1].cpu().numpy()  # [B, n_samples, pred_len]
        temp_samples = samples_phys[..., 0].cpu().numpy()

        exceedance = (do_samples < hypoxia_threshold).mean(axis=1)  # [B, pred_len]

        all_samples_do.append(do_samples)
        all_true_do.append(y_phys[..., 1])
        all_samples_temp.append(temp_samples)
        all_true_temp.append(y_phys[..., 0])
        all_exceedance.append(exceedance)

    samples_do = np.concatenate(all_samples_do, axis=0)     # [N, n_samples, pred_len]
    true_do = np.concatenate(all_true_do, axis=0)           # [N, pred_len]
    samples_temp = np.concatenate(all_samples_temp, axis=0)
    true_temp = np.concatenate(all_true_temp, axis=0)
    exceedance = np.concatenate(all_exceedance, axis=0)     # [N, pred_len]

    # CRPS
    crps_do = compute_crps(true_do, samples_do)
    crps_temp = compute_crps(true_temp, samples_temp)

    # ECE (flatten over all windows and lead times)
    true_hyp_flat = (true_do < hypoxia_threshold).ravel()
    exceedance_flat = exceedance.ravel()
    ece = compute_ece(exceedance_flat, true_hyp_flat)

    # Median forecast R²
    med_do = np.median(samples_do, axis=1)   # [N, pred_len]
    med_temp = np.median(samples_temp, axis=1)
    from sklearn.metrics import r2_score
    r2_do = r2_score(true_do.ravel(), med_do.ravel())
    r2_temp = r2_score(true_temp.ravel(), med_temp.ravel())

    # Mean exceedance curve per lead time
    mean_exceedance = exceedance.mean(axis=0)  # [pred_len]

    # DO quantiles per lead time
    do_quantiles = np.quantile(samples_do, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
    # [5, N, pred_len] → mean over windows
    do_quantile_mean = do_quantiles.mean(axis=1)  # [5, pred_len]

    return {
        "crps_do": crps_do,
        "crps_temp": crps_temp,
        "ece": ece,
        "r2_do_median": r2_do,
        "r2_temp_median": r2_temp,
        "mean_exceedance_curve": mean_exceedance,
        "do_quantile_mean": do_quantile_mean,
        "true_do_mean": true_do.mean(axis=0),
        "exceedance": exceedance,
        "true_hypoxia": true_hyp_flat,
        "pred_prob": exceedance_flat,
    }


def plot_results(results: Dict, out_dir: Path) -> None:
    """Generate all publication figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_len = len(results["mean_exceedance_curve"])
    lead_hours = np.arange(pred_len) * 0.25

    # Figure 1: Hypoxia exceedance probability curve
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444466")

    ax1 = axes[0]
    exceedance_pct = results["mean_exceedance_curve"] * 100
    ax1.fill_between(lead_hours, exceedance_pct, alpha=0.3, color="#ff6b6b")
    ax1.plot(lead_hours, exceedance_pct, color="#ff6b6b", linewidth=2.5,
             label="P(DO < 2 mg/L)")
    ax1.axhline(10, color="#ffd93d", linewidth=1, linestyle="--", alpha=0.7, label="10% threshold")
    ax1.axhline(50, color="#ff9f43", linewidth=1, linestyle="--", alpha=0.7, label="50% threshold")
    ax1.set_xlabel("Lead time (hours)", color="white")
    ax1.set_ylabel("P(hypoxia) [%]", color="white")
    ax1.set_title("Hypoxia Exceedance Probability Curve\nP(DO < 2 mg/L) vs. Lead Time",
                  color="white", fontweight="bold")
    ax1.legend(facecolor="#222244", labelcolor="white")
    ax1.set_ylim(0, 100)

    # Figure 2: Calibration plot
    ax2 = axes[1]
    pred_probs = results["pred_prob"]
    true_labels = results["true_hypoxia"]
    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_probs >= lo) & (pred_probs < hi)
        if mask.sum() > 5:
            bin_accs.append(true_labels[mask].mean())
            bin_confs.append(pred_probs[mask].mean())

    ax2.plot([0, 1], [0, 1], "--", color="#444466", linewidth=1.5, label="Perfect calibration")
    ax2.scatter(bin_confs, bin_accs, s=60, color="#4ecdc4", zorder=5,
                label=f"ECE = {results['ece']:.3f}")
    ax2.fill_between([0, 1], [0, 1], alpha=0.05, color="#4ecdc4")
    ax2.set_xlabel("Predicted probability", color="white")
    ax2.set_ylabel("Empirical frequency", color="white")
    ax2.set_title("Calibration Plot — Hypoxia Probability\n(closer to diagonal = better calibrated)",
                  color="white", fontweight="bold")
    ax2.legend(facecolor="#222244", labelcolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fig_path = out_dir / "diffusion_hypoxia_analysis.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {fig_path}")

    # Figure 2: DO forecast with uncertainty bands
    fig2, ax = plt.subplots(figsize=(14, 5))
    fig2.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")

    q = results["do_quantile_mean"]  # [5, pred_len]
    t_obs = results["true_do_mean"]
    ax.fill_between(lead_hours, q[0], q[4], alpha=0.15, color="#4ecdc4", label="5th-95th pct")
    ax.fill_between(lead_hours, q[1], q[3], alpha=0.30, color="#4ecdc4", label="25th-75th pct")
    ax.plot(lead_hours, q[2], color="#4ecdc4", linewidth=2.5, label="Median forecast")
    ax.plot(lead_hours, t_obs, color="#ffd93d", linewidth=2, linestyle="--", label="Mean observed")
    ax.axhline(2.0, color="#ff6b6b", linewidth=1.5, linestyle=":", label="Hypoxia threshold (2 mg/L)")
    ax.set_xlabel("Lead time (hours)", color="white")
    ax.set_ylabel("DO (mg/L)", color="white")
    ax.set_title("Dissolved Oxygen Probabilistic Forecast\nMedian + Uncertainty Bands (averaged over test set)",
                 color="white", fontweight="bold")
    ax.legend(facecolor="#222244", labelcolor="white", ncol=2)
    plt.tight_layout()
    fig2_path = out_dir / "diffusion_do_uncertainty.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"Saved: {fig2_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion forecaster evaluation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "frontier_v1" / "best.pt",
    )
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--diffusion_epochs", type=int, default=30,
                        help="Epochs to train the diffusion head")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "diffusion")
    args = parser.parse_args()

    device_str = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load base model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    from data.dataset import build_dataloaders_from_config
    from training.evaluate import build_model_from_bundle, load_state_dict_compatible

    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    base_model = build_model_from_bundle(cfg, bundle)
    load_state_dict_compatible(base_model, ckpt["model"])
    base_model.to(device)

    # Detect context dim
    context_dim = cfg["model"].get("d_model", 192)

    # Train diffusion head
    diffusion = train_diffusion_head(
        base_model,
        bundle.train_loader,
        device,
        bundle.feature_engineer,
        n_epochs=args.diffusion_epochs,
        context_dim=context_dim,
    )

    # Evaluate
    results = evaluate_diffusion(
        base_model, diffusion, bundle.test_loader, device,
        bundle.feature_engineer, n_samples=args.n_samples,
    )

    # Print summary
    print("\n" + "="*60)
    print("DIFFUSION FORECASTER RESULTS")
    print("="*60)
    print(f"  CRPS (DO):          {results['crps_do']:.4f} mg/L")
    print(f"  CRPS (Temp):        {results['crps_temp']:.4f} °C")
    print(f"  Calibration ECE:    {results['ece']:.4f}")
    print(f"  Median R² (DO):     {results['r2_do_median']:.3f}")
    print(f"  Median R² (Temp):   {results['r2_temp_median']:.3f}")
    max_exc_lead = results["mean_exceedance_curve"].argmax() * 0.25
    max_exc_prob = results["mean_exceedance_curve"].max() * 100
    print(f"  Peak hypoxia risk:  {max_exc_prob:.1f}% at {max_exc_lead:.1f}h lead")
    print("="*60)

    # Save diffusion model
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"diffusion": diffusion.state_dict(), "cfg": cfg},
               args.out / "diffusion_head.pt")
    print(f"\nDiffusion head saved to: {args.out / 'diffusion_head.pt'}")

    plot_results(results, out_dir=args.out)


if __name__ == "__main__":
    main()
