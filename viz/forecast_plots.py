"""Publication forecast plots for temperature and dissolved oxygen."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.style import COLORS, apply_publication_style


def resolve_forecast_datetimes(batch, bundle, n: int) -> pd.DatetimeIndex:
    """Datetime index for the first sample in a loader batch."""
    if "forecast_times" in batch:
        ft = batch["forecast_times"]
        if ft.dim() == 1:
            times = ft.detach().cpu().numpy()
        elif ft.dim() == 2:
            times = ft[0].detach().cpu().numpy()
        else:
            times = ft.reshape(-1, ft.shape[-1])[0].detach().cpu().numpy()
        times = np.atleast_1d(times)
        return pd.to_datetime(times[:n].astype("datetime64[ns]"))

    if len(getattr(bundle, "datetimes", [])) > 0:
        t0 = int(batch["window_start"][0].item()) + bundle.seq_len
        return pd.to_datetime(bundle.datetimes[t0 : t0 + n])

    return pd.date_range("2024-01-01", periods=n, freq="15min")


def create_multivariate_forecast_plot(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_masks: Optional[Dict[str, np.ndarray]] = None,
    physics_violation: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Dual-panel temperature + DO forecast figure.

    y_true, y_pred: [T, 2] physical units [temp, DO].
    """
    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, idx, label, unit in zip(
        axes,
        [0, 1],
        ["Temperature", "Dissolved Oxygen"],
        ["°C", "mg/L"],
    ):
        ax.plot(dates, y_true[:, idx], color=COLORS[idx], linewidth=0.8, alpha=0.7, label="Observed")
        ax.plot(dates, y_pred[:, idx], color=COLORS[idx + 2], linewidth=1.0, linestyle="--", label="Forecast")
        if split_masks:
            for name, mask, color in [
                ("train", split_masks.get("train"), "#6A994E"),
                ("val", split_masks.get("val"), "#F18F01"),
                ("test", split_masks.get("test"), "#C73E1D"),
            ]:
                if mask is not None and mask.any():
                    ax.axvspan(dates[mask][0], dates[mask][-1], alpha=0.05, color=color, label=name)
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        if idx == 1:
            ax.axhline(2.0, color=COLORS[3], linestyle=":", label="Hypoxia 2 mg/L")

    if physics_violation is not None:
        inset = axes[1].inset_axes([0.65, 0.55, 0.32, 0.35])
        inset.plot(dates, physics_violation, color=COLORS[3], linewidth=0.8)
        inset.set_title("Physics violation", fontsize=9)
        inset.tick_params(labelsize=7)

    axes[0].set_title("(a) Water temperature", fontweight="bold")
    axes[1].set_title("(b) Dissolved oxygen", fontweight="bold")
    axes[1].set_xlabel("Date")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig
