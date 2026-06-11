#!/usr/bin/env python3
"""
Koopman Eigenspectrum Analysis — Physics Discovery Tool

After training with configs/exp_koopman.yaml, run this script to:
  1. Extract the learned Koopman operator K (32×32 matrix)
  2. Compute eigenvalues → reveal dominant river dynamics modes
  3. Identify diurnal cycles, dam release signatures, weather system modes
  4. Generate publication-quality figures for the paper

Usage:
    python3 scripts/koopman_analysis.py --checkpoint checkpoints/koopman/best.pt
    python3 scripts/koopman_analysis.py --checkpoint checkpoints/koopman/best.pt --out figures/koopman/

Physical interpretation of Koopman eigenvalues:
  - Magnitude ≈ 1.0: persistent mode (slowly decaying)
  - Magnitude < 1.0: transient mode (decaying perturbation)
  - Period T = 1/(|angle|/2π) timesteps = T×0.25h hours
  - Real eigenvalue: monotone decay mode (e.g., thermal equilibration)
  - Complex conjugate pair: oscillatory mode at period T
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_koopman_K(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load the Koopman transition matrix K from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    from data.dataset import build_dataloaders_from_config
    from training.evaluate import build_model_from_bundle

    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    model = build_model_from_bundle(cfg, bundle)

    state = ckpt["model"]
    # Remap legacy keys
    remapped = {}
    for k, v in state.items():
        if k.startswith("gat.lin."):
            k = k.replace("gat.lin.", "gat.node_proj.", 1)
        remapped[k] = v
    model.load_state_dict(remapped, strict=False)
    model.eval()

    if not hasattr(model, "koopman") or model.koopman is None:
        raise ValueError(
            "Checkpoint does not contain a Koopman encoder. "
            "Train with backbone: koopman_patchtst (configs/exp_koopman.yaml)"
        )

    spectrum = model.koopman.get_eigenspectrum()
    K_matrix = model.koopman.K.detach().cpu().numpy()
    return K_matrix, spectrum, bundle, model


def identify_physical_modes(period_hours: np.ndarray) -> list:
    """Map eigenvalue periods to known physical processes."""
    modes = []
    for i, period in enumerate(period_hours):
        if period == float("inf") or period > 1000:
            label = "DC mode (mean)"
        elif 20 <= period <= 28:
            label = f"Diurnal cycle ({period:.1f}h ≈ 24h)"
        elif 11 <= period <= 14:
            label = f"Semi-diurnal ({period:.1f}h ≈ 12h)"
        elif 5 <= period <= 8:
            label = f"Dam pulse? ({period:.1f}h)"
        elif 48 <= period <= 96:
            label = f"2-4 day weather ({period:.1f}h)"
        elif 96 <= period <= 200:
            label = f"Synoptic system ({period:.1f}h ≈ {period/24:.1f}d)"
        elif period < 3:
            label = f"Sub-diurnal fast ({period:.1f}h)"
        else:
            label = f"Unknown ({period:.1f}h)"
        modes.append(label)
    return modes


def print_spectrum_table(spectrum: dict, top_k: int = 20) -> None:
    """Print the top-k eigenvalue modes in a readable table."""
    mags = spectrum["magnitudes"].numpy()
    freqs = spectrum["freq_per_hour"].numpy()
    periods = spectrum["period_hours"].numpy()
    reals = spectrum["eigenvalues_real"].numpy()
    imags = spectrum["eigenvalues_imag"].numpy()

    print("\n" + "="*80)
    print("KOOPMAN EIGENSPECTRUM — TOP DYNAMICAL MODES")
    print("="*80)
    print(f"{'Rank':>4}  {'Magnitude':>10}  {'Period (h)':>12}  {'Freq (cyc/h)':>14}  {'Mode label'}")
    print("-"*80)

    modes = identify_physical_modes(periods)
    for i in range(min(top_k, len(mags))):
        period_str = f"{periods[i]:.1f}" if periods[i] < 500 else "∞ (DC)"
        print(
            f"{i+1:>4}  {mags[i]:>10.4f}  {period_str:>12}  {freqs[i]:>14.4f}  {modes[i]}"
        )
    print("="*80)

    # Highlight key modes
    diurnal_idx = np.where((periods >= 20) & (periods <= 28))[0]
    if len(diurnal_idx):
        print(f"\n✓ Diurnal mode detected at period {periods[diurnal_idx[0]]:.2f}h "
              f"(magnitude {mags[diurnal_idx[0]]:.4f})")

    dam_idx = np.where((periods >= 4) & (periods <= 10))[0]
    if len(dam_idx):
        print(f"✓ Possible dam pulse mode at period {periods[dam_idx[0]]:.2f}h "
              f"(magnitude {mags[dam_idx[0]]:.4f})")


def plot_spectrum(spectrum: dict, out_dir: Path) -> None:
    """Generate publication-quality eigenspectrum figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    mags = spectrum["magnitudes"].numpy()
    reals = spectrum["eigenvalues_real"].numpy()
    imags = spectrum["eigenvalues_imag"].numpy()
    periods = spectrum["period_hours"].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444466")

    # Left: complex plane (eigenvalue portrait)
    ax1 = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), "--", color="#445566", linewidth=1, alpha=0.6,
             label="Unit circle")
    sc = ax1.scatter(reals, imags, c=mags, cmap="plasma", s=80, alpha=0.9,
                     vmin=0.5, vmax=1.0, zorder=5)
    plt.colorbar(sc, ax=ax1, label="Magnitude").ax.yaxis.label.set_color("white")

    # Annotate dominant modes
    for i, (r, im, p) in enumerate(zip(reals[:5], imags[:5], periods[:5])):
        label = f"{p:.0f}h" if p < 200 else "DC"
        ax1.annotate(label, (r, im), fontsize=8, color="white",
                     xytext=(r + 0.03, im + 0.03))

    ax1.set_xlabel("Re(λ)", color="white")
    ax1.set_ylabel("Im(λ)", color="white")
    ax1.set_title("Koopman Eigenvalue Portrait\n(Complex Plane)", color="white", fontweight="bold")
    ax1.legend(facecolor="#222244", labelcolor="white", fontsize=8)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect("equal")
    ax1.axhline(0, color="#444466", linewidth=0.5)
    ax1.axvline(0, color="#444466", linewidth=0.5)

    # Right: Period spectrum (bar chart)
    ax2 = axes[1]
    valid = periods < 500
    period_plot = periods[valid]
    mag_plot = mags[valid]
    colors = []
    for p in period_plot:
        if 20 <= p <= 28:
            colors.append("#ff6b6b")   # red for diurnal
        elif 48 <= p <= 100:
            colors.append("#4ecdc4")   # teal for synoptic
        elif 4 <= p <= 10:
            colors.append("#ffd93d")   # yellow for dam pulse range
        else:
            colors.append("#6c757d")   # grey for other
    bars = ax2.bar(range(len(period_plot)), mag_plot, color=colors, edgecolor="none", alpha=0.85)

    # Reference lines for known physics
    known_periods = {"24h\n(Diurnal)": 24, "12h\n(Semi-diurnal)": 12, "6h\n(Dam?)": 6}
    # Add period labels on top 8 bars
    top_idx = np.argsort(mag_plot)[-8:]
    for idx in top_idx:
        p = period_plot[idx]
        ax2.text(idx, mag_plot[idx] + 0.005, f"{p:.0f}h",
                 ha="center", va="bottom", fontsize=7, color="white", rotation=45)

    ax2.set_xlabel("Mode rank (sorted by magnitude)", color="white")
    ax2.set_ylabel("Magnitude |λ|", color="white")
    ax2.set_title("Koopman Mode Spectrum\n(Dominant River Dynamics)", color="white", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="#445566", linewidth=0.8, linestyle="--", alpha=0.5)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color="#ff6b6b", label="Diurnal (24h)"),
        mpatches.Patch(color="#ffd93d", label="Dam pulse (4-10h)"),
        mpatches.Patch(color="#4ecdc4", label="Synoptic (2-4 day)"),
        mpatches.Patch(color="#6c757d", label="Other modes"),
    ]
    ax2.legend(handles=legend_patches, facecolor="#222244", labelcolor="white", fontsize=8)

    plt.suptitle(
        "Koopman Operator Spectral Analysis\nChattahoochee River @ USGS 02334500",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "koopman_eigenspectrum.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nEigenspectrum figure saved to: {out_path}")

    # Also save K matrix as CSV for supplementary material
    import pandas as pd
    csv_out = out_dir / "koopman_eigenvalues.csv"
    df = pd.DataFrame({
        "rank": range(1, len(mags) + 1),
        "magnitude": mags,
        "period_hours": periods,
        "freq_per_hour": spectrum["freq_per_hour"].numpy(),
        "eigenvalue_real": reals,
        "eigenvalue_imag": imags,
    })
    df.to_csv(csv_out, index=False)
    print(f"Eigenvalue table saved to: {csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Koopman eigenspectrum analysis for river dynamics discovery"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "koopman" / "best.pt",
        help="Path to trained Koopman checkpoint",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "koopman",
        help="Output directory for figures",
    )
    parser.add_argument("--top_k", type=int, default=20, help="Top-k modes to display")
    args = parser.parse_args()

    device = torch.device("cpu")  # Use CPU for stable eigendecomposition
    print(f"Loading checkpoint: {args.checkpoint}")
    K_matrix, spectrum, bundle, model = load_koopman_K(args.checkpoint, device)

    print(f"\nKoopman operator K: {K_matrix.shape[0]}×{K_matrix.shape[1]} matrix")
    print(f"Frobenius norm: {np.linalg.norm(K_matrix, 'fro'):.4f}")

    print_spectrum_table(spectrum, top_k=args.top_k)
    plot_spectrum(spectrum, out_dir=args.out)

    # Report dominant diurnal mode for paper
    periods = spectrum["period_hours"].numpy()
    mags = spectrum["magnitudes"].numpy()
    diurnal = np.where((periods >= 18) & (periods <= 30))[0]
    if len(diurnal):
        best = diurnal[0]
        print(f"\nPAPER FINDING: Dominant diurnal mode at {periods[best]:.2f}h "
              f"(magnitude {mags[best]:.4f}, expected 24.0h)")
    else:
        print("\nNote: No clear diurnal mode found — model may need more training.")


if __name__ == "__main__":
    main()
