#!/usr/bin/env python3
"""
Ablation Runner — Automated experiment suite for paper Table 1.

Runs all ablation configurations and records metrics in a CSV table.
Handles: context length, architecture, physics loss, multi-site graph.

Usage:
    python3 scripts/ablation_runner.py --quick     # 30-epoch smoke test
    python3 scripts/ablation_runner.py             # Full 200-epoch runs
    python3 scripts/ablation_runner.py --dry-run   # Just print the plan
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

ABLATION_CONFIGS = [
    # Baseline comparisons
    {
        "name": "DLinear_baseline",
        "config": "configs/exp_02_dlinear.yaml",
        "description": "DLinear — trend+seasonal decomposition",
        "group": "Architecture",
    },
    {
        "name": "PatchTST_96ctx",
        "config": "configs/high_accuracy.yaml",
        "description": "PatchTST, 96-step context (24h)",
        "group": "Architecture",
    },
    {
        "name": "PatchTST_192ctx",
        "config": "configs/exp_frontier_v1.yaml",
        "description": "PatchTST, 192-step context (48h) + solar features + fixed physics",
        "group": "Architecture",
    },
    {
        "name": "Koopman_PatchTST",
        "config": "configs/exp_koopman.yaml",
        "description": "Koopman + PatchTST parallel branch",
        "group": "Architecture",
    },
    # Physics loss ablation
    {
        "name": "No_physics_loss",
        "config": "configs/exp_frontier_v1.yaml",
        "description": "PatchTST frontier, physics_mode=off",
        "group": "Physics",
        "overrides": {"physics.physics_mode": "off"},
    },
    {
        "name": "Physics_soft",
        "config": "configs/exp_frontier_v1.yaml",
        "description": "PatchTST frontier, physics_mode=soft",
        "group": "Physics",
    },
    # Feature ablation
    {
        "name": "No_solar_features",
        "config": "configs/exp_frontier_v1.yaml",
        "description": "PatchTST frontier, no solar forcing features",
        "group": "Features",
        "overrides": {"model.use_solar": False},  # handled by feature_engineer
    },
    # Context length ablation
    {
        "name": "ctx_96",
        "config": "configs/exp_frontier_v1.yaml",
        "description": "96-step context (24h)",
        "group": "ContextLength",
        "overrides": {"forecast.seq_len": 96},
    },
    {
        "name": "ctx_192",
        "config": "configs/exp_frontier_v1.yaml",
        "description": "192-step context (48h) — default frontier",
        "group": "ContextLength",
    },
]


@dataclass
class AblationResult:
    name: str
    description: str
    group: str
    config: str
    temp_rmse: float = float("nan")
    temp_r2: float = float("nan")
    do_rmse: float = float("nan")
    do_r2: float = float("nan")
    hypoxia_f1: float = float("nan")
    train_time_min: float = float("nan")
    status: str = "pending"
    error: str = ""


def run_experiment(
    ablation: dict,
    epochs: int,
    dry_run: bool = False,
) -> AblationResult:
    """Run a single ablation experiment and return its metrics."""
    name = ablation["name"]
    config = ablation["config"]
    description = ablation["description"]
    group = ablation.get("group", "Other")
    checkpoint_dir = ROOT / "checkpoints" / "ablations" / name

    result = AblationResult(name=name, description=description, group=group, config=config)

    if dry_run:
        print(f"  [DRY RUN] {name}: python3 train.py --config {config} "
              f"--epochs {epochs} --no-early-stop")
        result.status = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        sys.executable, "train.py",
        "--config", config,
        "--epochs", str(epochs),
        "--no-early-stop",
    ]

    # Write override config if needed
    overrides = ablation.get("overrides", {})
    if overrides:
        import yaml
        base_cfg_path = ROOT / config
        with open(base_cfg_path) as f:
            cfg = yaml.safe_load(f)
        # Apply overrides (supports dotted keys like "physics.physics_mode")
        for key, val in overrides.items():
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
        # Override checkpoint dir
        cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)
        cfg["training"]["log_dir"] = str(ROOT / "logs" / "ablations" / name)
        override_path = ROOT / "configs" / f"_ablation_{name}.yaml"
        with open(override_path, "w") as f:
            yaml.dump(cfg, f)
        cmd[3] = str(override_path)

    start_time = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=7200)
        elapsed = (time.time() - start_time) / 60

        if proc.returncode != 0:
            result.status = "failed"
            result.error = proc.stderr[-500:] if proc.stderr else proc.stdout[-500:]
            print(f"FAILED: {result.error}")
            return result

        # Parse metrics from stdout
        output = proc.stdout
        for line in output.split("\n"):
            if "temp RMSE" in line and "DO RMSE" in line:
                try:
                    import re
                    match = re.search(
                        r"temp RMSE (\d+\.\d+).*?DO RMSE (\d+\.\d+)", line
                    )
                    if match:
                        result.temp_rmse = float(match.group(1))
                        result.do_rmse = float(match.group(2))
                except Exception:
                    pass

        # Try to load checkpoint metrics directly
        ckpt_path = checkpoint_dir / "best.pt"
        if ckpt_path.exists():
            import torch
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                # Metrics are logged in logs/ablations/{name}/train_log.csv
                log_path = ROOT / "logs" / "ablations" / name / "train_log.csv"
                if log_path.exists():
                    import pandas as pd
                    df = pd.read_csv(log_path)
                    best_row = df.loc[df["val_phys_mean_rmse"].dropna().idxmin()]
                    result.temp_rmse = best_row.get("val_phys_temperature_rmse", float("nan"))
                    result.temp_r2 = best_row.get("val_phys_temperature_r2", float("nan"))
                    result.do_rmse = best_row.get("val_phys_dissolved_oxygen_rmse", float("nan"))
                    result.do_r2 = best_row.get("val_phys_dissolved_oxygen_r2", float("nan"))
                    result.hypoxia_f1 = best_row.get("val_phys_hypoxia_f1", float("nan"))
            except Exception as e:
                print(f"  (Could not load detailed metrics: {e})")

        result.train_time_min = elapsed
        result.status = "done"
        print(f"\nDone in {elapsed:.1f} min: temp_rmse={result.temp_rmse:.3f}, "
              f"do_rmse={result.do_rmse:.3f}")

    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error = "Exceeded 2h timeout"
    except Exception as e:
        result.status = "error"
        result.error = str(e)

    # Clean up override config
    override_path = ROOT / "configs" / f"_ablation_{name}.yaml"
    if override_path.exists():
        override_path.unlink()

    return result


def save_results(results: List[AblationResult], out_path: Path) -> None:
    """Save ablation table as CSV and print Markdown table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name", "group", "description", "temp_rmse", "temp_r2",
        "do_rmse", "do_r2", "hypoxia_f1", "train_time_min", "status"
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fieldnames})
    print(f"\nAblation table saved to: {out_path}")

    # Print Markdown table for paper
    print("\n" + "="*100)
    print("ABLATION TABLE (Markdown format)")
    print("="*100)
    print(f"| {'Model':<35} | {'Group':<15} | {'Temp R²':>8} | {'DO R²':>8} | {'DO RMSE':>8} | {'Status'} |")
    print(f"|{'-'*37}|{'-'*17}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
    for r in results:
        r2t = f"{r.temp_r2:.3f}" if r.temp_r2 == r.temp_r2 else "—"
        r2d = f"{r.do_r2:.3f}" if r.do_r2 == r.do_r2 else "—"
        rmsd = f"{r.do_rmse:.3f}" if r.do_rmse == r.do_rmse else "—"
        print(f"| {r.name:<35} | {r.group:<15} | {r2t:>8} | {r2d:>8} | {rmsd:>8} | {r.status} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation suite for paper Table 1")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per experiment")
    parser.add_argument("--quick", action="store_true", help="30-epoch quick test")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Run only specific configs by name (e.g. --configs DLinear_baseline Koopman_PatchTST)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "ablation_table.csv")
    args = parser.parse_args()

    epochs = 30 if args.quick else args.epochs
    configs_to_run = ABLATION_CONFIGS
    if args.configs:
        configs_to_run = [c for c in ABLATION_CONFIGS if c["name"] in args.configs]
        if not configs_to_run:
            print(f"No matching configs found. Available: {[c['name'] for c in ABLATION_CONFIGS]}")
            sys.exit(1)

    print(f"Ablation suite: {len(configs_to_run)} experiments, {epochs} epochs each")
    if args.dry_run:
        print("(DRY RUN — no actual training)")

    results = []
    for ablation in configs_to_run:
        r = run_experiment(ablation, epochs=epochs, dry_run=args.dry_run)
        results.append(r)

    if not args.dry_run:
        save_results(results, args.out)


if __name__ == "__main__":
    main()
