"""Evaluate checkpoint on test split in physical units (°C, mg/L)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from data.dataset import DatasetBundle
from models.spatiotemporal_model import SpatioTemporalWaterModel
from training.metrics import MetricsConfig, compute_regression_metrics, hypoxia_event_metrics


def remap_checkpoint_state_dict(state_dict: dict) -> dict:
    """
    Map legacy GAT keys to the current module layout.

    Older single-site runs saved ``gat.lin.*`` (no PyG); current N=1 uses ``gat.node_proj.*``.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("gat.lin."):
            key = key.replace("gat.lin.", "gat.node_proj.", 1)
        remapped[key] = value
    return remapped


def load_state_dict_compatible(model: torch.nn.Module, state_dict: dict) -> None:
    remapped = remap_checkpoint_state_dict(state_dict)

    # Graph topology buffers (edge_index, edge_attr) are set from the current
    # dataset config and must NOT be restored from checkpoint — the graph
    # structure may differ between single-site and multi-site runs.
    TOPOLOGY_KEYS = {"edge_index", "edge_attr"}
    remapped = {k: v for k, v in remapped.items()
                if k.split(".")[-1] not in TOPOLOGY_KEYS}

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if unexpected:
        print(f"Warning: ignored unexpected keys: {unexpected[:5]}...")
    if missing:
        # Only warn about keys that are not unused optional GAT branches or topology
        critical = [k for k in missing
                    if not k.startswith("gat.conv.")
                    and k.split(".")[-1] not in TOPOLOGY_KEYS]
        if critical:
            print(f"Warning: missing keys when loading checkpoint: {critical[:5]}...")



@torch.no_grad()
def evaluate_loader(
    model: SpatioTemporalWaterModel,
    loader,
    feature_engineer,
    device: torch.device,
    metrics_cfg: Optional[MetricsConfig] = None,
    forecast_node: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    metrics_cfg = metrics_cfg or MetricsConfig()
    preds, trues = [], []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        fut_cov = batch.get("fut_cov")
        if fut_cov is not None:
            fut_cov = fut_cov.to(device)
        y_hat, _ = model(x, fut_cov=fut_cov)
        preds.append(y_hat.cpu().numpy())
        trues.append(y.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    if forecast_node is not None and y_pred.ndim == 4:
        y_pred = y_pred[:, forecast_node, :, :]
        y_true = y_true[:, forecast_node, :, :]
    y_pred_phys = feature_engineer.inverse_targets(y_pred)
    y_true_phys = feature_engineer.inverse_targets(y_true)

    out = compute_regression_metrics(y_true_phys, y_pred_phys)
    out.update(
        hypoxia_event_metrics(
            y_true_phys[..., 1],
            y_pred_phys[..., 1],
            metrics_cfg,
        )
    )
    return out


def build_model_from_bundle(cfg: dict, bundle: DatasetBundle) -> SpatioTemporalWaterModel:
    mcfg = cfg["model"]
    ts = bundle.feature_engineer.target_scaler
    target_mean = torch.tensor(ts.mean_, dtype=torch.float32)
    target_scale = torch.tensor(ts.scale_, dtype=torch.float32)
    use_physics = mcfg.get("use_physics_head", True)
    return SpatioTemporalWaterModel(
        n_features=bundle.n_features,
        seq_len=bundle.seq_len,
        pred_len=bundle.pred_len,
        edge_index=bundle.edge_index,
        backbone=mcfg.get("backbone", "patchtst"),
        gat_hidden=mcfg.get("gat_hidden", 64),
        gat_heads=mcfg.get("gat_heads", 4),
        gat_dropout=mcfg.get("gat_dropout", 0.1),
        patch_len=mcfg.get("patch_len", 16),
        patch_stride=mcfg.get("patch_stride", 8),
        d_model=mcfg.get("d_model", 128),
        n_heads=mcfg.get("n_heads", 4),
        n_layers=mcfg.get("n_layers", 3),
        dropout=mcfg.get("dropout", 0.1),
        tcn_channels=mcfg.get("tcn_channels"),
        tcn_kernel_size=mcfg.get("tcn_kernel_size", 3),
        use_revin=mcfg.get("use_revin", True),
        use_horizon_decoder=mcfg.get("use_horizon_decoder", True),
        use_physics_head=use_physics,
        target_mean=target_mean if use_physics else None,
        target_scale=target_scale if use_physics else None,
        use_ar_refinement=mcfg.get("use_ar_refinement", True),
        use_residual_baseline=mcfg.get("use_residual_baseline", True),
        use_local_conv=mcfg.get("use_local_conv", True),
        n_fut_cov=len(bundle.feature_engineer.meteo_col_indices),
        edge_attr=bundle.edge_attr,
        # Koopman parameters
        koopman_latent_dim=mcfg.get("koopman_latent_dim", 32),
        koopman_lambda_recon=mcfg.get("koopman_lambda_recon", 1.0),
        koopman_lambda_pred=mcfg.get("koopman_lambda_pred", 1.0),
        koopman_lambda_multi=mcfg.get("koopman_lambda_multi", 0.5),
        koopman_lambda_spectral=mcfg.get("koopman_lambda_spectral", 0.01),
        koopman_loss_weight=mcfg.get("koopman_loss_weight", 0.1),
    )


def load_model_from_checkpoint(
    ckpt_path: Path,
    bundle: DatasetBundle,
    device: torch.device,
) -> SpatioTemporalWaterModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model_from_bundle(cfg, bundle)
    load_state_dict_compatible(model, ckpt["model"])
    model.to(device)
    return model


def print_metrics(split: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {split} metrics (physical units) ===")
    print(
        f"  Temperature — RMSE: {metrics['temperature_rmse']:.3f} °C, "
        f"MAE: {metrics['temperature_mae']:.3f} °C, R²: {metrics['temperature_r2']:.3f}"
    )
    print(
        f"  Dissolved O₂ — RMSE: {metrics['dissolved_oxygen_rmse']:.3f} mg/L, "
        f"MAE: {metrics['dissolved_oxygen_mae']:.3f} mg/L, R²: {metrics['dissolved_oxygen_r2']:.3f}"
    )
    if not np.isnan(metrics.get("hypoxia_f1", float("nan"))):
        print(
            f"  Hypoxia F1 (DO < 2 mg/L): {metrics['hypoxia_f1']:.3f} "
            f"(prevalence {metrics['hypoxia_prevalence']:.1%})"
        )
