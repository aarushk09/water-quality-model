#!/usr/bin/env python3
"""
Fetch Chattahoochee River multi-site USGS data for the dam-causality graph.

Sites:
  02334430 — Chattahoochee R. above Buford Dam (directly captures dam releases)
  02334500 — Chattahoochee R. near Buford (primary forecast site)
  02334885 — Chattahoochee R. at Lanier Bridge (downstream)
  02335000 — Chattahoochee R. at Norcross (further downstream)
  02335757 — Suwanee Creek near Suwanee (tributary, storm signal)

Graph structure (directed, with travel-time lags):
  02334430 → 02334500 (~45 min travel time)
  02334500 → 02334885 (~60 min)
  02334885 → 02335000 (~90 min)
  02335757 → 02335000 (tributary, no lag)

Usage:
    python3 scripts/fetch_chattahoochee_network.py
    python3 scripts/fetch_chattahoochee_network.py --start 2023-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Site definitions with physical properties for the graph
CHATTAHOOCHEE_NETWORK = {
    "02334430": {
        "name": "Chattahoochee R. above Buford Dam NR Buford GA",
        "lat": 34.1743,
        "lon": -84.0977,
        "dam_controlled": True,       # Above Buford Dam — captures dam releases
        "order": 0,                   # Upstream order
        "downstream": "02334500",
        "travel_time_min": 45,        # Estimated travel time to next site
        "params": ["00010", "00300", "00060", "00065"],  # T, DO, Q, gage
    },
    "02334500": {
        "name": "Chattahoochee R. near Buford GA",
        "lat": 34.1801,
        "lon": -84.0927,
        "dam_controlled": False,
        "order": 1,
        "downstream": "02334885",
        "travel_time_min": 60,
        "params": ["00010", "00300", "00060", "00065"],
    },
    "02334885": {
        "name": "Chattahoochee R. at Lanier Bridge nr Gainesville GA",
        "lat": 34.2618,
        "lon": -83.9352,
        "dam_controlled": False,
        "order": 2,
        "downstream": "02335000",
        "travel_time_min": 90,
        "params": ["00010", "00300", "00060", "00065"],
    },
    "02335000": {
        "name": "Chattahoochee R. at Norcross GA",
        "lat": 33.9735,
        "lon": -84.2155,
        "dam_controlled": False,
        "order": 3,
        "downstream": None,
        "travel_time_min": None,
        "params": ["00010", "00300", "00060", "00065"],
    },
    "02335757": {
        "name": "Suwanee Cr. near Suwanee GA",
        "lat": 34.0568,
        "lon": -84.0697,
        "dam_controlled": False,
        "order": 2,
        "downstream": "02335000",
        "travel_time_min": 0,         # Tributary junction, no delay
        "params": ["00010", "00300", "00060"],  # No gage at this site
    },
}

PARAM_NAMES = {
    "00010": "temperature",
    "00300": "dissolved_oxygen",
    "00060": "discharge",
    "00065": "gage_height",
}


def fetch_nwis_iv(
    site_no: str,
    params: list,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch NWIS instantaneous values for a single site."""
    try:
        import requests
    except ImportError:
        raise ImportError("requests package required: pip install requests")

    param_str = ",".join(params)
    url = (
        f"https://waterservices.usgs.gov/nwis/iv/"
        f"?sites={site_no}&parameterCd={param_str}"
        f"&startDT={start_date}&endDT={end_date}"
        f"&format=rdb&siteStatus=all"
    )

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Failed after {max_retries} attempts: {e}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)

    lines = resp.text.split("\n")
    data_lines = [l for l in lines if l and not l.startswith("#")]
    if len(data_lines) < 3:
        print(f"  No data returned for site {site_no}")
        return pd.DataFrame()

    # Parse RDB format
    header = data_lines[0].split("\t")
    # Skip data-type row (line 1)
    rows = []
    for line in data_lines[2:]:
        parts = line.split("\t")
        if len(parts) >= len(header):
            rows.append(parts[:len(header)])

    df = pd.DataFrame(rows, columns=header)
    if "datetime" not in df.columns:
        print(f"  No datetime column in response for site {site_no}")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Rename value columns based on parameter codes
    result = {"datetime": df["datetime"]}
    for col in df.columns:
        for param_cd, name in PARAM_NAMES.items():
            if param_cd in col and "_00000" in col:  # Value column (not approval flag)
                result[name] = pd.to_numeric(df[col], errors="coerce")
                break

    return pd.DataFrame(result)


def update_stations_json(sites: dict, stations_path: Path) -> None:
    """Update data/raw/stations.json with all network sites."""
    if stations_path.exists():
        with open(stations_path) as f:
            raw = json.load(f)
        # Handle both flat list and {"sites": [...]} formats
        if isinstance(raw, list):
            stations_list = raw
        else:
            stations_list = raw.get("sites", [])
    else:
        stations_list = []

    # Index existing sites by site_no
    existing = {s["site_no"]: i for i, s in enumerate(stations_list)}

    for site_no, meta in sites.items():
        entry = {
            "site_no": site_no,
            "name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "dam_controlled": meta.get("dam_controlled", False),
            "order": meta.get("order", 0),
            "role": "upstream_dam" if meta.get("dam_controlled") else "auxiliary",
        }
        if site_no in existing:
            stations_list[existing[site_no]].update(entry)
        else:
            stations_list.append(entry)

    # Write back as flat list (matching existing format)
    with open(stations_path, "w") as f:
        json.dump(stations_list, f, indent=2)
    print(f"Updated {stations_path} with {len(sites)} network sites ({len(stations_list)} total)")


def update_edges_csv(sites: dict, edges_path: Path) -> None:
    """Update data/raw/edges.csv with directed travel-time edges."""
    rows = []
    for src_site, meta in sites.items():
        dst_site = meta.get("downstream")
        if dst_site and dst_site in sites:
            rows.append({
                "upstream": src_site,
                "downstream": dst_site,
                "travel_minutes": meta.get("travel_time_min", 0),
                "dam_release": meta.get("dam_controlled", False),
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["upstream", "downstream", "travel_minutes", "dam_release"]
    )
    df.to_csv(edges_path, index=False)
    print(f"Updated {edges_path} with {len(df)} directional edges")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Chattahoochee multi-site network for dam-causality graph"
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--sites", nargs="+", default=list(CHATTAHOOCHEE_NETWORK.keys()),
                        help="Sites to fetch (default: all 5 network sites)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to requested sites
    sites = {k: v for k, v in CHATTAHOOCHEE_NETWORK.items() if k in args.sites}

    print(f"Fetching {len(sites)} sites: {args.start} → {args.end}")
    print()

    for site_no, meta in sites.items():
        site_dir = args.out_dir / site_no
        site_dir.mkdir(exist_ok=True)
        out_path = site_dir / "iv.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            print(f"  {site_no}: existing data ({len(existing)} rows), appending...")
        else:
            existing = None
            print(f"  {site_no}: fetching from scratch...")

        print(f"    {meta['name']}")
        df = fetch_nwis_iv(site_no, meta["params"], args.start, args.end)

        if df.empty:
            print(f"    ⚠ No data fetched for {site_no}")
            continue

        # Resample to 15-min intervals
        df = df.set_index("datetime").sort_index()
        df = df.resample("15min").mean()
        df = df.reset_index()
        df.columns = ["datetime"] + [c for c in df.columns if c != "datetime"]

        if existing is not None:
            df = pd.concat([existing, df]).drop_duplicates("datetime").sort_values("datetime")

        df.to_parquet(out_path, index=False)
        print(f"    ✓ {len(df)} rows → {out_path}")

        # Quick data quality check
        if "temperature" in df.columns:
            pct_t = df["temperature"].notna().mean() * 100
            print(f"      Temperature: {pct_t:.1f}% coverage")
        if "dissolved_oxygen" in df.columns:
            pct_do = df["dissolved_oxygen"].notna().mean() * 100
            print(f"      DO:          {pct_do:.1f}% coverage")
        print()

    # Update graph definition files
    print("Updating graph definition files...")
    update_stations_json(sites, args.out_dir / "stations.json")
    update_edges_csv(sites, args.out_dir / "edges.csv")

    print("\nDone! Run multi-site training with:")
    print("  python3 train.py --config configs/exp_multisite_dam.yaml --no-early-stop")


if __name__ == "__main__":
    main()
