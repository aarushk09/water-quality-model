"""Evaluation metrics including hypoxia-event detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score


TARGET_NAMES = ["temperature", "dissolved_oxygen"]


@dataclass
class MetricsConfig:
    hypoxia_threshold_mg_l: float = 2.0
    hypoxia_sensitivity_mg_l: float = 3.0
    sudden_drop_mg_l: float = 1.0
    sudden_drop_steps: int = 4


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    y_true, y_pred: [..., 2] last dim = [temp, DO].
    """
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        yt = y_true[..., i].ravel()
        yp = y_pred[..., i].ravel()
        valid = np.isfinite(yt) & np.isfinite(yp)
        if not valid.any():
            metrics[f"{name}_rmse"] = float("nan")
            metrics[f"{name}_mae"] = float("nan")
            metrics[f"{name}_r2"] = float("nan")
            continue
        yt, yp = yt[valid], yp[valid]
        metrics[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
        metrics[f"{name}_mae"] = float(mean_absolute_error(yt, yp))
        metrics[f"{name}_r2"] = float(r2_score(yt, yp))
    return metrics


def hypoxia_event_metrics(
    do_true: np.ndarray,
    do_pred: np.ndarray,
    cfg: Optional[MetricsConfig] = None,
) -> Dict[str, float]:
    """Binary hypoxia classification metrics (DO < threshold)."""
    cfg = cfg or MetricsConfig()
    thresh = cfg.hypoxia_threshold_mg_l
    yt = (do_true.ravel() < thresh).astype(int)
    yp = (do_pred.ravel() < thresh).astype(int)
    if len(np.unique(yt)) < 2:
        f1 = float("nan")
    else:
        f1 = float(f1_score(yt, yp, zero_division=0))
    return {
        "hypoxia_f1": f1,
        "hypoxia_prevalence": float(yt.mean()),
    }


def sudden_drop_events(
    do: np.ndarray,
    drop_mg_l: float = 1.0,
    steps: int = 4,
) -> np.ndarray:
    """Boolean mask where DO drops more than drop_mg_l within `steps` intervals."""
    do = do.ravel()
    events = np.zeros(len(do), dtype=bool)
    for i in range(steps, len(do)):
        delta = do[i] - do[i - steps]
        if delta < -drop_mg_l:
            events[i] = True
    return events


def evaluate_batch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    cfg: Optional[MetricsConfig] = None,
) -> Dict[str, float]:
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    out = compute_regression_metrics(yt, yp)
    out.update(hypoxia_event_metrics(yt[..., 1], yp[..., 1], cfg))
    return out


def evaluate_batch_physical(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    feature_engineer,
    cfg: Optional[MetricsConfig] = None,
    forecast_node: Optional[int] = None,
) -> Dict[str, float]:
    """Regression metrics in °C and mg/L."""
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    if forecast_node is not None and yt.ndim == 4:
        yt = yt[:, forecast_node, :, :]
        yp = yp[:, forecast_node, :, :]
    yt = feature_engineer.inverse_targets(yt)
    yp = feature_engineer.inverse_targets(yp)
    out = compute_regression_metrics(yt, yp)
    out.update(hypoxia_event_metrics(yt[..., 1], yp[..., 1], cfg))
    out["mean_rmse"] = (out["temperature_rmse"] + out["dissolved_oxygen_rmse"]) / 2.0
    return out


def mean_physical_rmse(metrics: Dict[str, float]) -> float:
    return (metrics["temperature_rmse"] + metrics["dissolved_oxygen_rmse"]) / 2.0
