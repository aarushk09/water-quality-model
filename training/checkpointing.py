"""Save and load full training checkpoints for resume and long runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from training.evaluate import load_state_dict_compatible


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: dict,
    epoch: int,
    best_val: float,
    wait: int,
    val_mse: float,
    feature_scaler: Any,
    target_scaler: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "best_val": best_val,
        "wait": wait,
        "val_mse": val_mse,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    if model is not None and "model" in ckpt:
        load_state_dict_compatible(model, ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
