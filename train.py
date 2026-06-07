#!/usr/bin/env python3
"""
CLI entry for physics-informed spatiotemporal water-quality forecasting.

Usage:
    # Recommended for long unattended runs (300 epochs, checkpoints every 10):
    python3 train.py --config configs/long_run.yaml

    # Resume after interrupt or crash:
    python3 train.py --config configs/long_run.yaml --resume checkpoints/last.pt

    # Run all epochs without early stopping:
    python3 train.py --config configs/long_run.yaml --no-early-stop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders_from_config
from training.device import device_label
from training.seed import set_seed
from training.evaluate import build_model_from_bundle
from training.trainer import Trainer


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["project_root"] = str(ROOT)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spatiotemporal water-quality model")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_run.yaml",
        help="Config file (default: long_run.yaml for extended training)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--explain_hypoxia", action="store_true")
    parser.add_argument("--backbone", choices=["patchtst", "tcn"], default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint (e.g. checkpoints/last.pt)",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from checkpoints/last.pt if it exists",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Train for full epoch budget without early stopping",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.explain_hypoxia:
        cfg["explain"]["explain_hypoxia"] = True
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.device:
        cfg["training"]["device"] = args.device
    if args.no_early_stop:
        cfg["training"]["enable_early_stopping"] = False
    if args.auto_resume:
        cfg["training"]["auto_resume_last"] = True

    set_seed(cfg["training"].get("seed", 42))

    print("Building dataset...")
    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    print(
        f"Nodes={bundle.n_nodes}, features={bundle.n_features}, "
        f"seq={bundle.seq_len}, pred={bundle.pred_len}, "
        f"epochs={cfg['training']['epochs']}"
    )

    model = build_model_from_bundle(cfg, bundle)
    resume = args.resume
    if resume is None and cfg["training"].get("auto_resume_last"):
        last = ROOT / cfg["training"]["checkpoint_dir"] / "last.pt"
        if last.exists():
            resume = last

    trainer = Trainer(model, bundle, cfg, resume_path=resume)
    print(f"Training on {device_label(trainer.device)}...")
    result = trainer.fit()

    print(f"Best val MSE (scaled): {result['best_val_mse']:.4f}")
    print(f"Last completed epoch: {result.get('last_epoch', '?')}")
    if result.get("interrupted"):
        print("Run was interrupted — resume with: python3 train.py --resume checkpoints/last.pt")

    tm = result["test_metrics"]
    print(
        f"Test — temp RMSE {tm['temperature_rmse']:.3f} °C, "
        f"DO RMSE {tm['dissolved_oxygen_rmse']:.3f} mg/L"
    )

    figures_dir = ROOT / cfg.get("explain", {}).get("figures_dir", "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        from viz.forecast_plots import create_multivariate_forecast_plot

        batch = next(iter(bundle.test_loader))
        x = batch["x"].to(trainer.device)
        y = batch["y"]
        with torch.no_grad():
            y_hat, _ = trainer.model(x)
        yt = bundle.feature_engineer.inverse_targets(y[0, 0].numpy())
        yp = bundle.feature_engineer.inverse_targets(y_hat[0, 0].cpu().numpy())
        n = min(len(yt), 500)
        from viz.forecast_plots import resolve_forecast_datetimes

        dates = resolve_forecast_datetimes(batch, bundle, n)
        create_multivariate_forecast_plot(
            dates, yt[:n], yp[:n], out_path=figures_dir / "test_forecast_sample.png"
        )
        print(f"Saved sample forecast plot to {figures_dir / 'test_forecast_sample.png'}")
    except Exception as e:
        print(f"Forecast plot skipped: {e}")

    if cfg["explain"].get("explain_hypoxia") or args.explain_hypoxia:
        from explain.attribution import run_hypoxia_explainability
        from explain.attention_viz import extract_and_save_hypoxia_attention

        figures_dir = ROOT / cfg["explain"]["figures_dir"]
        figures_dir.mkdir(parents=True, exist_ok=True)
        batch = next(iter(bundle.test_loader))
        extract_and_save_hypoxia_attention(
            trainer.model,
            batch,
            bundle.edge_index,
            cfg["metrics"]["hypoxia_threshold_mg_l"],
            figures_dir,
            device=trainer.device,
        )
        run_hypoxia_explainability(
            trainer.model,
            bundle.test_loader,
            bundle.feature_cols,
            figures_dir,
            hypoxia_threshold=cfg["metrics"]["hypoxia_threshold_mg_l"],
            background_samples=cfg["explain"].get("background_samples", 50),
            use_shap=cfg["explain"].get("use_shap", False),
            device=trainer.device,
        )
        print(f"Explainability figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
