#!/usr/bin/env python3
"""Benchmark forward-pass throughput (it/s) on MPS/CUDA/CPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders_from_config
from training.device import resolve_device
from training.evaluate import build_model_from_bundle


def benchmark(cfg_path: Path, n_iters: int = 100, warmup: int = 10) -> float:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["project_root"] = str(ROOT)
    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    device = resolve_device(cfg["training"].get("device", "auto"))
    model = build_model_from_bundle(cfg, bundle).to(device)
    model.train()
    batch = next(iter(bundle.train_loader))
    x = batch["x"].to(device)
    fut_cov = batch.get("fut_cov")
    if fut_cov is not None:
        fut_cov = fut_cov.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x, fut_cov=fut_cov)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(x, fut_cov=fut_cov)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    its = n_iters / elapsed
    print(f"Device: {device}")
    print(f"Forward passes: {n_iters} in {elapsed:.3f}s → {its:.2f} it/s")
    return its


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model forward speed")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "high_accuracy.yaml",
    )
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    its = benchmark(args.config, n_iters=args.iters)
    ok = its >= 15.0
    print(f"PASS (≥15 it/s): {ok}")


if __name__ == "__main__":
    main()
