#!/usr/bin/env python3
"""Per-lead RMSE and R² for each forecast step (1..H)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders_from_config
from training.device import resolve_device
from training.evaluate import load_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "high_accuracy.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["project_root"] = str(ROOT)
    bundle = build_dataloaders_from_config(cfg, ROOT)
    dev = resolve_device("auto")
    model = load_model_from_checkpoint(args.checkpoint, bundle, dev)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch in bundle.test_loader:
            y_hat, _ = model(batch["x"].to(dev))
            preds.append(y_hat.cpu().numpy())
            trues.append(batch["y"].numpy())

    yp = bundle.feature_engineer.inverse_targets(np.concatenate(preds, axis=0))
    yt = bundle.feature_engineer.inverse_targets(np.concatenate(trues, axis=0))
    h = yp.shape[2]
    names = ["temperature", "dissolved_oxygen"]

    print(f"\nPer-horizon metrics (test, H={h})")
    print(f"{'step':>5} {'hours':>6}  " + "  ".join(f"{n}_rmse" for n in names))
    for k in range(h):
        hrs = (k + 1) * 0.25
        parts = [f"{k+1:5d} {hrs:6.2f}"]
        for i, name in enumerate(names):
            rmse = float(np.sqrt(mean_squared_error(yt[:, :, k, i].ravel(), yp[:, :, k, i].ravel())))
            r2 = float(r2_score(yt[:, :, k, i].ravel(), yp[:, :, k, i].ravel()))
            parts.append(f"{rmse:.3f}")
        print("  ".join(parts))

    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    leads = np.arange(1, h + 1)
    for i, name in enumerate(names):
        rmse_by_lead = [
            float(np.sqrt(mean_squared_error(yt[:, :, k, i].ravel(), yp[:, :, k, i].ravel())))
            for k in range(h)
        ]
        np.savez(
            out_dir / f"horizon_metrics_{name}.npz",
            lead=leads,
            rmse=np.array(rmse_by_lead),
        )
    print(f"\nSaved horizon metric arrays to {out_dir}")


if __name__ == "__main__":
    main()
