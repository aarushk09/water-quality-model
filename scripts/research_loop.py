#!/usr/bin/env python3
"""Automated research loop: train experiments, check pass criteria, diagnose failures."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_speed import benchmark
from scripts.data_audit import audit_iv
from scripts.log_experiment import log_experiment, COLUMNS


def _run(cmd: List[str], dry_run: bool = False) -> int:
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def _parse_test_metrics(log_dir: Path) -> Dict[str, float]:
    """Read final test metrics from trainer output or log."""
    csv_path = log_dir / "train_log.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {
        "temp_r2": float(last.get("val_phys_temperature_r2", last.get("val_temperature_r2", 0))),
        "do_r2": float(last.get("val_phys_dissolved_oxygen_r2", last.get("val_dissolved_oxygen_r2", 0))),
        "temp_rmse": float(last.get("val_phys_temperature_rmse", last.get("val_temperature_rmse", 0))),
        "do_rmse": float(last.get("val_phys_dissolved_oxygen_rmse", last.get("val_dissolved_oxygen_rmse", 0))),
        "epochs": int(last.get("epoch", 0)),
    }


def diagnose_training(exp_id: str, log_dir: Path) -> str:
    lines = [f"=== Diagnosis for {exp_id} ==="]
    csv_path = log_dir / "train_log.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        lines.append("Last 5 log rows:")
        lines.append(df.tail(5).to_string())
        for col in df.columns:
            if df[col].dtype in ("float64", "float32", "int64"):
                if df[col].isna().any():
                    lines.append(f"NaN detected in {col}")
    else:
        lines.append(f"No train_log.csv at {log_dir}")
    text = "\n".join(lines)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "diagnosis.txt").write_text(text)
    print(text)
    return text


EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "id": "00_data",
        "kind": "audit",
        "pass_fn": lambda r: r.get("paired_fraction", 0) >= 0.85 and r.get("rows", 0) >= 200_000,
    },
    {
        "id": "01_speed",
        "kind": "benchmark",
        "config": "configs/high_accuracy.yaml",
        "pass_fn": lambda r: r.get("its", 0) >= 15,
    },
    {
        "id": "02_dlinear",
        "kind": "train",
        "config": "configs/exp_02_dlinear.yaml",
        "pass_fn": lambda r: r.get("temp_r2", 0) >= 0.40 and r.get("do_r2", 0) >= 0.35,
    },
    {
        "id": "03_itransformer",
        "kind": "train",
        "config": "configs/exp_03_itransformer.yaml",
        "pass_fn": lambda r: r.get("temp_r2", 0) >= 0.60 and r.get("do_r2", 0) >= 0.50,
    },
    {
        "id": "04_physics",
        "kind": "train",
        "config": "configs/exp_04_physics.yaml",
        "pass_fn": lambda r: r.get("physics_ok", False),
    },
    {
        "id": "05_daymet",
        "kind": "train",
        "config": "configs/exp_05_daymet.yaml",
        "pass_fn": lambda r: r.get("daymet_gain_ok", False),
    },
]


def run_experiment(exp: Dict[str, Any], dry_run: bool = False, max_retries: int = 3) -> Dict[str, Any]:
    exp_id = exp["id"]
    result: Dict[str, Any] = {"exp_id": exp_id, "passed": False}

    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}\nExperiment {exp_id} attempt {attempt}/{max_retries}\n{'='*60}")
        kind = exp["kind"]

        if kind == "audit":
            stats = audit_iv(ROOT / "data/raw/02334500/iv.parquet")
            result.update(stats)
            result["passed"] = exp["pass_fn"](result)
            break

        elif kind == "benchmark":
            if dry_run:
                result["its"] = 0
                result["passed"] = True
                break
            its = benchmark(ROOT / exp["config"])
            result["its"] = its
            result["passed"] = exp["pass_fn"](result)
            break

        elif kind == "train":
            cfg_path = ROOT / exp["config"]
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            log_dir = ROOT / cfg["training"]["log_dir"]
            rc = _run(
                [sys.executable, "train.py", "--config", str(cfg_path.relative_to(ROOT))],
                dry_run=dry_run,
            )
            if dry_run:
                result["passed"] = True
                break
            if rc != 0:
                diagnose_training(exp_id, log_dir)
                continue
            metrics = _parse_test_metrics(log_dir)
            result.update(metrics)
            if exp_id == "04_physics":
                df = pd.read_csv(log_dir / "train_log.csv")
                by_ep10 = df[df["epoch"] <= 10]
                pv = by_ep10["val_physics_violation"].max() if not by_ep10.empty else 999
                result["physics_violation_ep10"] = float(pv)
                result["physics_ok"] = pv < 1e-3
            if exp_id == "05_daymet":
                table = ROOT / "logs/experiment_table.csv"
                if table.exists():
                    tdf = pd.read_csv(table)
                    p4 = tdf[tdf["exp_id"] == "04_physics"]
                    if not p4.empty:
                        gain = result.get("temp_r2", 0) - float(p4.iloc[-1]["temp_r2"])
                        result["temp_r2_gain"] = gain
                        result["daymet_gain_ok"] = gain >= 0.02
            result["passed"] = exp["pass_fn"](result)
            if result["passed"]:
                break
            diagnose_training(exp_id, log_dir)

    log_experiment({
        "exp_id": exp_id,
        "temp_r2": result.get("temp_r2", ""),
        "do_r2": result.get("do_r2", ""),
        "temp_rmse": result.get("temp_rmse", ""),
        "do_rmse": result.get("do_rmse", ""),
        "its": result.get("its", ""),
        "epochs": result.get("epochs", ""),
        "notes": "PASS" if result.get("passed") else f"FAIL {result}",
    })
    return result


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n## Experiment Summary\n")
    print("| exp_id | passed | temp_r2 | do_r2 | it/s | notes |")
    print("|--------|--------|---------|-------|------|-------|")
    for r in results:
        print(
            f"| {r['exp_id']} | {r.get('passed', False)} | "
            f"{r.get('temp_r2', '-')} | {r.get('do_r2', '-')} | "
            f"{r.get('its', '-')} | {r.get('rows', r.get('notes', ''))} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research loop driver")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    results = []
    for exp in EXPERIMENTS:
        results.append(run_experiment(exp, dry_run=args.dry_run, max_retries=args.max_retries))
    print_summary(results)


if __name__ == "__main__":
    main()
