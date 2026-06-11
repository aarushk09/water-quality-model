#!/usr/bin/env python3
"""
Paper Figures Generator — Publication-quality figures for all sections.

Generates the complete figure set for the paper:
  Fig 1 — Prediction vs. observed (test set, 72h window)
  Fig 2 — Koopman eigenspectrum (requires koopman checkpoint)
  Fig 3 — Ablation table heatmap (R² by model × variable)
  Fig 4 — Hypoxia exceedance probability curve
  Fig 5 — Diurnal amplitude tracking scatter (model vs. observed)
  Fig 6 — Lag recovery: learned τ vs physical travel time
  Fig S1 — Training curves (loss + R² vs epoch)

Usage:
    python3 scripts/paper_figures.py --frontier checkpoints/frontier_v1/best.pt
    python3 scripts/paper_figures.py --koopman checkpoints/koopman/best.pt --ablation logs/ablation_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Matplotlib dark theme ────────────────────────────────────────────────────
def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor":  "#0f0f1a",
        "axes.facecolor":    "#1a1a2e",
        "axes.edgecolor":    "#444466",
        "axes.labelcolor":   "white",
        "axes.titlecolor":   "white",
        "xtick.color":       "white",
        "ytick.color":       "white",
        "text.color":        "white",
        "grid.color":        "#2a2a44",
        "grid.alpha":        0.5,
        "legend.facecolor":  "#22223a",
        "legend.edgecolor":  "#444466",
        "legend.labelcolor": "white",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "figure.dpi":        150,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "#0f0f1a",
    })
    return plt


PALETTE = {
    "temp":     "#ff9f43",   # warm orange — temperature
    "do":       "#4ecdc4",   # teal — dissolved oxygen
    "observed": "#ffd93d",   # yellow — observed
    "model":    "#a29bfe",   # purple — model
    "physics":  "#ff6b6b",   # red — physics threshold / violation
    "koopman":  "#6c5ce7",   # deep purple — Koopman mode
    "diurnal":  "#fd79a8",   # pink — diurnal mode
    "synoptic": "#00b894",   # green — synoptic mode
}


# ─── Fig 1: Forecast vs Observed ──────────────────────────────────────────────
def fig_forecast_vs_observed(
    plt, checkpoint_path: Path, out_dir: Path, window_hours: float = 72.0
):
    """72-hour test set forecast overlaid on observations."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    from data.dataset import build_dataloaders_from_config
    from training.evaluate import build_model_from_bundle, load_state_dict_compatible

    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    model  = build_model_from_bundle(cfg, bundle)
    load_state_dict_compatible(model, ckpt["model"])
    model.eval()

    pred_len   = cfg["forecast"]["pred_len"]
    n_steps    = int(window_hours * 4)   # 15-min steps
    n_windows  = max(1, n_steps // pred_len)

    all_true, all_pred = [], []
    with torch.no_grad():
        for i, batch in enumerate(bundle.test_loader):
            if i >= n_windows:
                break
            x   = batch["x"]
            y   = batch["y"]
            out, _ = model(x)
            all_true.append(y[:1, 0].numpy())   # first sample, forecast node
            all_pred.append(out[:1, 0].detach().numpy())

    true_np = np.concatenate(all_true, axis=1)[0]   # [H*W, 2]
    pred_np = np.concatenate(all_pred, axis=1)[0]

    fe = bundle.feature_engineer
    true_phys = fe.inverse_targets(true_np)
    pred_phys = fe.inverse_targets(pred_np)
    t_hours   = np.arange(len(true_phys)) * 0.25

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, var_idx, name, unit, color in [
        (axes[0], 0, "Water Temperature", "°C",    PALETTE["temp"]),
        (axes[1], 1, "Dissolved Oxygen",  "mg/L",  PALETTE["do"]),
    ]:
        ax.plot(t_hours, true_phys[:, var_idx], color=PALETTE["observed"],
                linewidth=1.8, label="Observed", alpha=0.9)
        ax.plot(t_hours, pred_phys[:, var_idx], color=color,
                linewidth=2.0, label="Forecast", linestyle="--")

        if var_idx == 1:
            ax.axhline(2.0, color=PALETTE["physics"], linewidth=1.2,
                       linestyle=":", label="Hypoxia threshold (2 mg/L)", alpha=0.8)

        # Metrics annotation
        valid = np.isfinite(true_phys[:, var_idx]) & np.isfinite(pred_phys[:, var_idx])
        from sklearn.metrics import r2_score, mean_squared_error
        r2   = r2_score(true_phys[valid, var_idx], pred_phys[valid, var_idx])
        rmse = np.sqrt(mean_squared_error(true_phys[valid, var_idx], pred_phys[valid, var_idx]))
        ax.text(0.02, 0.97, f"R² = {r2:.3f}   RMSE = {rmse:.3f} {unit}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#22223a", alpha=0.8))
        ax.set_ylabel(f"{name} ({unit})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True)
        ax.set_title(name)

    axes[1].set_xlabel("Lead time (hours)")
    fig.suptitle(
        "PatchTST+Koopman Forecast — Chattahoochee River @ USGS 02334500\n"
        f"{window_hours:.0f}-hour test window",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_path = out_dir / "fig1_forecast_vs_observed.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Fig 3: Ablation Heatmap ──────────────────────────────────────────────────
def fig_ablation_heatmap(plt, ablation_csv: Path, out_dir: Path):
    """R² heatmap: rows = models, cols = [Temp, DO]."""
    import pandas as pd
    df = pd.read_csv(ablation_csv)
    df = df[df["status"] == "done"].copy()
    if df.empty:
        print("  Skipping ablation heatmap — no completed runs yet")
        return

    models  = df["name"].tolist()
    temp_r2 = df["temp_r2"].tolist()
    do_r2   = df["do_r2"].tolist()
    data    = np.array([temp_r2, do_r2]).T   # [N_models, 2]

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.5 + 1)))
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="R²").ax.yaxis.label.set_color("white")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Temperature R²", "Dissolved O₂ R²"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j, val in enumerate([temp_r2[i], do_r2[i]]):
            if np.isfinite(val):
                color = "white" if val < 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    ax.set_title("Ablation Study — R² by Model", fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "fig3_ablation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Fig 5: Diurnal Amplitude Tracking ───────────────────────────────────────
def fig_diurnal_tracking(plt, checkpoint_path: Path, out_dir: Path):
    """Scatter: predicted diurnal range vs observed diurnal range, per window."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    from data.dataset import build_dataloaders_from_config
    from training.evaluate import build_model_from_bundle, load_state_dict_compatible

    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    model  = build_model_from_bundle(cfg, bundle)
    load_state_dict_compatible(model, ckpt["model"])
    model.eval()

    true_ranges, pred_ranges = [], []
    with torch.no_grad():
        for batch in bundle.test_loader:
            x   = batch["x"]
            y   = batch["y"]
            out, _ = model(x)
            t_np = bundle.feature_engineer.inverse_targets(
                y[:, 0].numpy().reshape(-1, y.shape[-1])
            ).reshape(y.shape[0], y.shape[2], 2)
            p_np = bundle.feature_engineer.inverse_targets(
                out[:, 0].detach().numpy().reshape(-1, out.shape[-1])
            ).reshape(out.shape[0], out.shape[2], 2)

            true_ranges.append(t_np.max(axis=1) - t_np.min(axis=1))  # [B, 2]
            pred_ranges.append(p_np.max(axis=1) - p_np.min(axis=1))

    tr = np.concatenate(true_ranges, axis=0)  # [N, 2]
    pr = np.concatenate(pred_ranges, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, var_idx, name, unit, color in [
        (axes[0], 0, "Temperature", "°C",   PALETTE["temp"]),
        (axes[1], 1, "DO",          "mg/L", PALETTE["do"]),
    ]:
        valid = np.isfinite(tr[:, var_idx]) & np.isfinite(pr[:, var_idx])
        x_v, y_v = tr[valid, var_idx], pr[valid, var_idx]
        corr = np.corrcoef(x_v, y_v)[0, 1]
        ratio = y_v.mean() / (x_v.mean() + 1e-8)

        ax.scatter(x_v, y_v, s=15, alpha=0.4, color=color)
        lim = max(x_v.max(), y_v.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="white", linewidth=1.2,
                alpha=0.6, label="Perfect tracking")
        ax.text(0.05, 0.93,
                f"r = {corr:.3f}\nAmplitude ratio = {ratio:.2f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#22223a", alpha=0.8))
        ax.set_xlabel(f"Observed diurnal range ({unit})")
        ax.set_ylabel(f"Predicted diurnal range ({unit})")
        ax.set_title(f"{name} Diurnal Amplitude Tracking")
        ax.legend(fontsize=9)
        ax.grid(True)

    fig.suptitle("Diurnal Amplitude Tracking\n(model vs. observed, test set)",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = out_dir / "fig5_diurnal_tracking.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Fig S1: Training Curves ──────────────────────────────────────────────────
def fig_training_curves(plt, log_csv: Path, out_dir: Path, label: str = ""):
    """Loss + R² vs epoch for a single training run."""
    import csv
    rows = list(csv.DictReader(open(log_csv)))
    if not rows:
        print(f"  Skipping training curves — {log_csv} is empty")
        return

    epochs  = [int(r["epoch"]) for r in rows]
    t_loss  = [float(r.get("train_loss", "nan") or "nan") for r in rows]
    v_mse   = [float(r.get("val_mse", "nan") or "nan") for r in rows]
    t_r2    = [float(r.get("val_phys_temperature_r2", "nan") or "nan") for r in rows]
    do_r2   = [float(r.get("val_phys_dissolved_oxygen_r2", "nan") or "nan") for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, t_loss, color=PALETTE["model"],  linewidth=2, label="Train loss")
    ax.plot(epochs, v_mse,  color=PALETTE["do"],     linewidth=2, label="Val MSE (scaled)", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves{' — ' + label if label else ''}")
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log")

    # R² curves
    ax2 = axes[1]
    ax2.plot(epochs, t_r2,  color=PALETTE["temp"], linewidth=2, label="Temperature R²")
    ax2.plot(epochs, do_r2, color=PALETTE["do"],   linewidth=2, label="DO R²")
    ax2.axhline(0.75, color="white", linewidth=1, linestyle=":", alpha=0.5, label="R²=0.75 target")
    ax2.axhline(0.85, color=PALETTE["physics"], linewidth=1, linestyle=":", alpha=0.5, label="R²=0.85 target")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R²")
    ax2.set_title("Validation R² over Training")
    ax2.legend(fontsize=9)
    ax2.grid(True)
    ax2.set_ylim(-0.6, 1.0)

    fig.suptitle(f"Training Curves{' — ' + label if label else ''}", fontweight="bold")
    plt.tight_layout()
    tag = label.lower().replace(" ", "_").replace("/", "_") if label else "model"
    out_path = out_dir / f"figS1_training_curves_{tag}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--frontier", type=Path,
                        default=ROOT / "checkpoints" / "frontier_v1_ft4" / "last.pt")
    parser.add_argument("--koopman",  type=Path,
                        default=ROOT / "checkpoints" / "koopman_posthoc" / "best.pt")
    parser.add_argument("--ablation", type=Path,
                        default=ROOT / "logs" / "ablation_table.csv")
    parser.add_argument("--out",      type=Path, default=ROOT / "figures" / "paper")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plt = setup_style()

    print(f"\nGenerating paper figures → {args.out}\n")

    # Fig 1: Forecast vs Observed (uses best available checkpoint)
    ckpt = args.frontier
    if ckpt.exists():
        print("Fig 1 — Forecast vs. Observed...")
        try:
            fig_forecast_vs_observed(plt, ckpt, args.out)
        except Exception as e:
            print(f"  SKIPPED: {e}")
    else:
        print("Fig 1 — SKIPPED: no checkpoint available yet")

    # Fig 2: Koopman eigenspectrum
    # Handled separately via scripts/plot_koopman_spectrum.py

    # Fig 3: Ablation heatmap
    if args.ablation.exists():
        print("Fig 3 — Ablation heatmap...")
        try:
            fig_ablation_heatmap(plt, args.ablation, args.out)
        except Exception as e:
            print(f"  SKIPPED: {e}")
    else:
        print("Fig 3 — SKIPPED: run scripts/ablation_runner.py first")

    # Fig 5: Diurnal amplitude tracking
    if ckpt.exists():
        print("Fig 5 — Diurnal amplitude tracking...")
        try:
            fig_diurnal_tracking(plt, ckpt, args.out)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # Fig S1: Training curves for each available log
    for name, log_path in [
        ("Frontier V1 FT4",ROOT / "logs" / "frontier_v1_ft4" / "train_log.csv"),
        ("Koopman",        ROOT / "logs" / "koopman_posthoc" / "train_log.csv"),
        ("Multi-site Dam", ROOT / "logs" / "multisite_dam" / "train_log.csv"),
    ]:
        if log_path.exists() and log_path.stat().st_size > 100:
            print(f"Fig S1 — Training curves ({name})...")
            try:
                fig_training_curves(plt, log_path, args.out, label=name)
            except Exception as e:
                print(f"  SKIPPED: {e}")
        else:
            print(f"Fig S1 ({name}) — SKIPPED: log not available yet")

    print(f"\nAll done. Figures in: {args.out}")


if __name__ == "__main__":
    main()
