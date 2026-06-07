"""PatchTST / GAT attention heatmaps for hypoxia windows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from viz.style import COLORS, apply_publication_style


def plot_attention_heatmap(
    attention: np.ndarray,
    feature_names: list,
    do_trace: Optional[np.ndarray] = None,
    hypoxia_mask: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    title: str = "Encoder attention (patch × patch)",
) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(2 if do_trace is not None else 1, 1, figsize=(10, 6))
    if do_trace is None:
        ax_attn = axes
    else:
        ax_do, ax_attn = axes

    if do_trace is not None:
        ax_do.plot(do_trace, color=COLORS[1], linewidth=0.8, label="Observed DO")
        if hypoxia_mask is not None:
            idx = np.where(hypoxia_mask)[0]
            if len(idx):
                ax_do.axvspan(idx[0], idx[-1], alpha=0.2, color=COLORS[3], label="Hypoxia window")
        ax_do.set_ylabel("DO (mg/L)")
        ax_do.legend()
        ax_do.set_title("(a) Dissolved oxygen trace", fontweight="bold")

    im = ax_attn.imshow(attention, aspect="auto", cmap="viridis")
    ax_attn.set_xlabel("Patch index")
    ax_attn.set_ylabel("Patch index")
    ax_attn.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax_attn, fraction=0.046)

    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


def extract_and_save_hypoxia_attention(
    model,
    batch: dict,
    edge_index: torch.Tensor,
    hypoxia_threshold: float,
    figures_dir: Path,
    event_id: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[Path]:
    """Run forward with attention; save figure if batch contains hypoxia."""
    device = device or torch.device("cpu")
    model.eval()
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    with torch.no_grad():
        y_hat, attn = model(x, return_attention=True)
    do_true = y[0, 0, :, 1].cpu().numpy()
    # Note: y is scaled; threshold comparison is approximate unless caller passes physical y
    hyp_mask = do_true < hypoxia_threshold
    if not hyp_mask.any():
        return None
    if attn is None:
        return None
    att_np = attn[0].cpu().numpy()
    out = figures_dir / f"attention_hypoxia_{event_id}.png"
    plot_attention_heatmap(
        att_np,
        feature_names=[f"f{i}" for i in range(x.shape[-1])],
        do_trace=do_true,
        hypoxia_mask=hyp_mask,
        out_path=out,
    )
    plt.close()
    return out
