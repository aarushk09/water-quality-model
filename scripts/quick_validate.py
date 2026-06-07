#!/usr/bin/env python3
"""Short training run to verify val metrics improve vs a persistence baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders_from_config
from training.device import resolve_device
from training.evaluate import build_model_from_bundle, evaluate_loader
from training.metrics import MetricsConfig
from training.seed import set_seed
from training.trainer import Trainer


def persistence_rmse(loader, feature_engineer) -> float:
    errs = []
    for batch in loader:
        x = batch["x"].numpy()
        y = batch["y"].numpy()
        last = x[:, :, -1, :2]
        pred = np.tile(last[:, :, None, :], (1, 1, y.shape[2], 1))
        yp = feature_engineer.inverse_targets(pred)
        yt = feature_engineer.inverse_targets(y)
        errs.append(np.sqrt(np.mean((yp - yt) ** 2)))
    return float(np.mean(errs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "high_accuracy.yaml")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["project_root"] = str(ROOT)
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["enable_early_stopping"] = False
    cfg["training"]["min_epochs"] = 1
    cfg["training"]["checkpoint_dir"] = "checkpoints/quick_validate"
    cfg["training"]["log_dir"] = "logs/quick_validate"

    set_seed(cfg["training"].get("seed", 42))
    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    base = persistence_rmse(bundle.val_loader, bundle.feature_engineer)
    print(f"Persistence val mean RMSE (scaled space proxy): {base:.4f}")

    model = build_model_from_bundle(cfg, bundle)
    trainer = Trainer(model, bundle, cfg, device=resolve_device("auto"))
    result = trainer.fit()
    val_rmse = result["test_metrics"]["temperature_rmse"] + result["test_metrics"][
        "dissolved_oxygen_rmse"
    ]
    print(f"Test combined RMSE proxy: {val_rmse:.4f}")
    print(f"Best tracked score: {result['best_val_mse']:.4f}")


if __name__ == "__main__":
    main()
