"""Captum Integrated Gradients and optional SHAP for hypoxia explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from viz.style import COLORS, apply_publication_style

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _do_output(model, x: torch.Tensor) -> torch.Tensor:
    """Scalar DO forecast: mean over horizon and nodes."""
    y_hat, _ = model(x)
    return y_hat[..., 1].mean(dim=(1, 2))


def integrated_gradients_attribution(
    model,
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> np.ndarray:
    """
    IG attributions for DO output w.r.t. input x [B, N, T, F].

    Returns attributions shaped like x.
    """
    if not HAS_CAPTUM:
        raise ImportError("captum required for Integrated Gradients")

    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)

    ig = IntegratedGradients(lambda inp: _do_output(model, inp))
    attr = ig.attribute(x, baselines=baseline, n_steps=n_steps)
    return attr.detach().cpu().numpy()


def shap_gradient_attribution(
    model,
    x: torch.Tensor,
    background: torch.Tensor,
) -> np.ndarray:
    """Optional SHAP GradientExplainer (smaller background batch)."""
    if not HAS_SHAP:
        raise ImportError("shap required for SHAP attribution")

    model.eval()

    def forward_flat(inp):
        return _do_output(model, inp)

    explainer = shap.GradientExplainer(forward_flat, background)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values)


def plot_attribution_surface(
    attr: np.ndarray,
    feature_names: List[str],
    out_path: Optional[Path] = None,
    title: str = "Integrated Gradients (DO output)",
) -> plt.Figure:
    """Aggregate |attr| over batch/nodes -> [T, F] heatmap."""
    apply_publication_style()
    # attr: [B, N, T, F]
    agg = np.abs(attr).mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(agg.T, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Time step (within window)")
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax, label="|attribution|")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


def run_hypoxia_explainability(
    model,
    loader,
    feature_names: List[str],
    figures_dir: Path,
    hypoxia_threshold: float = 2.0,
    background_samples: int = 50,
    use_shap: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Path]:
    """Run IG (and optional SHAP) on hypoxia vs non-hypoxia windows."""
    device = device or torch.device("cpu")
    model.to(device)
    figures_dir = Path(figures_dir)
    outputs: Dict[str, Path] = {}

    # Collect background batch
    backgrounds = []
    hypoxia_batches = []
    for batch in loader:
        y = batch["y"]
        if (y[..., 1] < hypoxia_threshold).any():
            hypoxia_batches.append(batch)
        if len(backgrounds) < background_samples:
            backgrounds.append(batch["x"][:1])
        if len(hypoxia_batches) >= 3:
            break

    if not hypoxia_batches:
        print("No hypoxia windows found in loader subset.")
        return outputs

    bg = torch.cat(backgrounds, dim=0).to(device)[:background_samples]
    batch = hypoxia_batches[0]
    x = batch["x"].to(device)

    if HAS_CAPTUM:
        attr = integrated_gradients_attribution(model, x, baseline=bg[:1])
        p = figures_dir / "ig_hypoxia.png"
        plot_attribution_surface(attr, feature_names, out_path=p)
        plt.close()
        outputs["ig"] = p

    if use_shap and HAS_SHAP:
        sv = shap_gradient_attribution(model, x[:1], bg)
        p = figures_dir / "shap_hypoxia.png"
        plot_attribution_surface(
            sv[np.newaxis, ...] if sv.ndim == 3 else sv,
            feature_names,
            out_path=p,
            title="SHAP (DO output)",
        )
        plt.close()
        outputs["shap"] = p

    return outputs
