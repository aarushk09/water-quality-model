#!/usr/bin/env python3
"""Append a row to logs/experiment_table.csv."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

COLUMNS = [
    "exp_id",
    "temp_r2",
    "do_r2",
    "temp_rmse",
    "do_rmse",
    "its",
    "epochs",
    "notes",
]


def log_experiment(
    row: Dict[str, Any],
    path: Path = Path("logs/experiment_table.csv"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
