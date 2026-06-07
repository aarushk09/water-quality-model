#!/usr/bin/env python3
"""Generate forecast plot from checkpoints/best.pt using real timestamps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders_from_config
from training.device import resolve_device
from training.evaluate import load_model_from_checkpoint
from viz.forecast_plots import create_multivariate_forecast_plot, resolve_forecast_datetimes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    args = parser.parse_args()

    cfg_path = args.config or ROOT / "configs" / "high_accuracy.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "configs" / "long_run.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cfg["project_root"] = str(ROOT)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {args.checkpoint}")

    bundle = build_dataloaders_from_config(cfg, ROOT)
    dev = resolve_device("auto")
    model = load_model_from_checkpoint(args.checkpoint, bundle, dev)

    batch = next(iter(bundle.test_loader))
    with torch.no_grad():
        y_hat, _ = model(batch["x"].to(dev))

    yt = bundle.feature_engineer.inverse_targets(batch["y"][0, 0].numpy())
    yp = bundle.feature_engineer.inverse_targets(y_hat[0, 0].cpu().numpy())
    n = min(500, len(yt))
    dates = resolve_forecast_datetimes(batch, bundle, n)

    out = ROOT / "figures" / "test_forecast_sample.png"
    create_multivariate_forecast_plot(dates, yt[:n], yp[:n], out_path=out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
