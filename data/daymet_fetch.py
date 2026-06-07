"""
Fetch DAYMET daily climate for the forecast site via single-pixel API.

Usage:
    python -m data.daymet_fetch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import urllib.request
except ImportError:
    urllib = None  # type: ignore

DAYMET_URL = (
    "https://daymet.ornl.gov/single-pixel/api/data"
    "?lat=34.1261&lon=-84.0934"
    "&vars=srad,tmax,tmin,prcp,vp"
    "&start=2010-01-01&end=2025-06-30&format=json"
)

COL_MAP = {
    "srad (W/m^2)": "srad",
    "tmax (deg c)": "tmax",
    "tmin (deg c)": "tmin",
    "prcp (mm/day)": "prcp",
    "vp (Pa)": "vp",
}


def fetch_daymet(url: str = DAYMET_URL) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=180) as resp:
        payload = json.loads(resp.read().decode())
    block = payload.get("data", payload)
    if not isinstance(block, dict):
        raise ValueError(f"Unexpected DAYMET data block: {type(block)}")
    df = pd.DataFrame(block)
    if "year" in df.columns and "yday" in df.columns:
        years = df["year"].astype(int)
        ydays = df["yday"].astype(int)
        df["date"] = pd.to_datetime(
            years.astype(str) + ydays.astype(str).str.zfill(3),
            format="%Y%j",
        )
    df = df.rename(columns=COL_MAP)
    keep = ["date", "srad", "tmax", "tmin", "prcp", "vp"]
    cols = [c for c in keep if c in df.columns]
    return df[cols].sort_values("date").drop_duplicates("date")


def run_fetch(out_path: Path, url: str = DAYMET_URL) -> Path:
    print(f"Fetching DAYMET from {url[:80]}...")
    df = fetch_daymet(url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} daily rows)")
    return out_path


def load_daymet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            path = csv_path
        else:
            return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch DAYMET daily climate")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/daymet.parquet"),
    )
    args = parser.parse_args()
    run_fetch(args.out)


if __name__ == "__main__":
    main()
