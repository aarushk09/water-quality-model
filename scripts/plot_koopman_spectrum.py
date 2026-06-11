#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def plot_spectrum_from_json(json_path: Path, out_dir: Path):
    with open(json_path) as f:
        spectrum = json.load(f)

    mags = np.array(spectrum["magnitudes"])
    reals = np.array(spectrum["eigenvalues_real"])
    imags = np.array(spectrum["eigenvalues_imag"])
    periods = np.array(spectrum["period_hours"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

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
        label = f"{p:.0f}h" if p < 20000 else "DC"
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
    valid = periods < 50000
    period_plot = periods[valid]
    mag_plot = mags[valid]
    colors = []
    for p in period_plot:
        if 20 <= p <= 28:
            colors.append("#ff6b6b")   # red for diurnal
        elif 80 <= p <= 100:
            colors.append("#4ecdc4")   # teal for seasonal/quarterly
        elif 4 <= p <= 10:
            colors.append("#ffd93d")   # yellow for dam pulse range
        elif p > 1000:
            colors.append("#4ecdc4")   # teal for seasonal
        else:
            colors.append("#6c757d")   # grey for other
    bars = ax2.bar(range(len(period_plot)), mag_plot, color=colors, edgecolor="none", alpha=0.85)

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
        mpatches.Patch(color="#4ecdc4", label="Seasonal/Quarterly"),
        mpatches.Patch(color="#6c757d", label="Other modes"),
    ]
    ax2.legend(handles=legend_patches, facecolor="#222244", labelcolor="white", fontsize=8)

    plt.suptitle(
        "Koopman Operator Spectral Analysis\nChattahoochee River @ USGS 02334500",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig2_koopman_eigenspectrum.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    json_path = ROOT / "logs" / "koopman_posthoc" / "spectrum.json"
    out_dir = ROOT / "figures" / "paper"
    plot_spectrum_from_json(json_path, out_dir)
