#!/usr/bin/env python3
"""Audit NWIS IV parquet completeness for the forecast target site."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def audit_iv(path: Path) -> dict:
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        raise ValueError(f"{path} missing datetime column")
    df["datetime"] = pd.to_datetime(df["datetime"])
    n = len(df)
    dmin, dmax = df["datetime"].min(), df["datetime"].max()
    if "temperature" in df.columns and "dissolved_oxygen" in df.columns:
        paired = (df["temperature"].notna() & df["dissolved_oxygen"].notna()).mean()
    else:
        paired = 0.0
    return {
        "rows": n,
        "date_min": str(dmin),
        "date_max": str(dmax),
        "paired_fraction": float(paired),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NWIS IV parquet")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/raw/02334500/iv.parquet"),
    )
    args = parser.parse_args()
    stats = audit_iv(args.path)
    print(f"File: {args.path}")
    print(f"Rows: {stats['rows']:,}")
    print(f"Date range: {stats['date_min']} → {stats['date_max']}")
    print(f"Paired temp/DO fraction: {stats['paired_fraction']:.4f}")
    ok = stats["paired_fraction"] >= 0.85 and stats["rows"] >= 200_000
    print(f"PASS: {ok}")


if __name__ == "__main__":
    main()
