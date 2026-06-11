"""Evaluation metrics including hypoxia-event detection, CRPS, and diurnal tracking."""

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


def persistence_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_obs: np.ndarray,
) -> Dict[str, float]:
    """
    Skill score relative to persistence baseline: SS = 1 - MSE_model / MSE_persist
    SS > 0 means model beats persistence; SS = 1 is perfect.

    y_true, y_pred, last_obs: [..., 2] — last dim = [temp, DO]
    last_obs: the last observed value before the forecast horizon (the persistence forecast)
    """
    out = {}
    for i, name in enumerate(TARGET_NAMES):
        yt = y_true[..., i].ravel()
        yp = y_pred[..., i].ravel()
        # last_obs: [N, 2] → expand to match y_true shape [N, H, 2] → ravel
        lo_i = last_obs[..., i]  # [N] or [N, 1, ...]
        # Broadcast last_obs to full horizon shape
        while lo_i.ndim < y_true[..., i].ndim:
            lo_i = np.expand_dims(lo_i, axis=-1)
        persist = np.broadcast_to(lo_i, y_true[..., i].shape).ravel()
        valid = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(persist)
        if not valid.any():
            out[f"{name}_skill"] = float("nan")
            continue
        mse_model = mean_squared_error(yt[valid], yp[valid])
        mse_persist = mean_squared_error(yt[valid], persist[valid])
        out[f"{name}_skill"] = float(1.0 - mse_model / (mse_persist + 1e-8))
    return out



def diurnal_amplitude_tracking(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 96,  # full 24h
) -> Dict[str, float]:
    """
    Measures how well the model tracks the diurnal amplitude.

    Computes correlation between predicted and observed diurnal ranges
    across all windows. Low correlation = model is smoothing the peaks.

    y_true, y_pred: [N, H, 2] or [N, H] — N windows, H timesteps
    """
    out = {}
    if y_true.ndim == 2:
        y_true = y_true[:, :, np.newaxis]
        y_pred = y_pred[:, :, np.newaxis]
    n_vars = min(y_true.shape[-1], 2)
    for i in range(n_vars):
        name = TARGET_NAMES[i]
        yt = y_true[:, :, i]  # [N, H]
        yp = y_pred[:, :, i]
        # Diurnal range per window
        true_range = yt.max(axis=1) - yt.min(axis=1)  # [N]
        pred_range = yp.max(axis=1) - yp.min(axis=1)
        valid = np.isfinite(true_range) & np.isfinite(pred_range)
        if valid.sum() < 5:
            out[f"{name}_diurnal_corr"] = float("nan")
            out[f"{name}_diurnal_ratio"] = float("nan")
        else:
            corr = np.corrcoef(true_range[valid], pred_range[valid])[0, 1]
            ratio = pred_range[valid].mean() / (true_range[valid].mean() + 1e-8)
            out[f"{name}_diurnal_corr"] = float(corr)
            out[f"{name}_diurnal_ratio"] = float(ratio)
    return out


def crps_gaussian(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Dict[str, float]:
    """
    Closed-form CRPS for Gaussian predictive distribution.
    CRPS(N(μ,σ), y) = σ[2φ(z) + zΦ(z) - 1/√π] where z = (y-μ)/σ

    y_true, y_mean, y_std: [..., 2]
    """
    from scipy import stats as scipy_stats
    out = {}
    for i, name in enumerate(TARGET_NAMES):
        if i >= y_true.shape[-1]:
            break
        yt = y_true[..., i].ravel()
        mu = y_mean[..., i].ravel()
        sigma = np.abs(y_std[..., i].ravel()) + 1e-6
        z = (yt - mu) / sigma
        crps = sigma * (2 * scipy_stats.norm.pdf(z) + z * (2 * scipy_stats.norm.cdf(z) - 1)
                        - 1.0 / np.sqrt(np.pi))
        out[f"{name}_crps"] = float(np.nanmean(crps))
    return out
