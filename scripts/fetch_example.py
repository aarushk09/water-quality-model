"""Minimal NWIS + Open-Meteo fetch — runnable in ~30 seconds.

Usage:
    source myenv/bin/activate
    python scripts/fetch_example.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

SITE = "02334500"
START, END = "2025-05-15", "2025-05-22"
PARAMS = "00010,00300,00060,00065"
LAT, LON = 34.126, -84.093


def fetch_nwis() -> pd.DataFrame:
    import dataretrieval.nwis as nwis

    iv, _ = nwis.get_iv(sites=SITE, start=START, end=END, parameterCd=PARAMS)
    iv = iv.reset_index()
    rename = {}
    for col in iv.columns:
        cs = str(col)
        if "00010" in cs:
            rename[col] = "temperature"
        elif "00300" in cs:
            rename[col] = "dissolved_oxygen"
        elif "00060" in cs:
            rename[col] = "discharge"
        elif "00065" in cs:
            rename[col] = "gage_height"
        elif cs.lower() in ("datetime", "datetime_1") or "dateTime" in cs:
            rename[col] = "datetime"
    iv = iv.rename(columns=rename)
    if "datetime" not in iv.columns:
        iv = iv.rename(columns={iv.columns[0]: "datetime"})
    keep = ["datetime"] + [c for c in ("temperature", "dissolved_oxygen", "discharge", "gage_height") if c in iv.columns]
    iv = iv.loc[:, ~iv.columns.duplicated()].filter(items=keep, axis=1)
    iv["datetime"] = pd.to_datetime(iv["datetime"]).dt.tz_convert("UTC").dt.tz_localize(None)
    return iv


def fetch_meteo() -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={START}&end_date={END}"
        "&hourly=shortwave_radiation,temperature_2m,wind_speed_10m"
        "&timezone=America%2FNew_York"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        meteo = pd.DataFrame(json.loads(resp.read())["hourly"])
    meteo = meteo.rename(
        columns={
            "time": "datetime",
            "temperature_2m": "air_temp",
            "shortwave_radiation": "shortwave_rad",
        }
    )
    meteo["datetime"] = (
        pd.to_datetime(meteo["datetime"])
        .dt.tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    return meteo


def main() -> None:
    iv = fetch_nwis()
    meteo = fetch_meteo()

    print("NWIS rows:", len(iv))
    cols = [c for c in ("datetime", "temperature", "dissolved_oxygen") if c in iv.columns]
    print(iv[cols].dropna().head(3))

    print("\nMeteo rows:", len(meteo))
    print(meteo.head(3))

    merged = iv.merge(meteo, on="datetime", how="left")
    merged["air_temp"] = merged["air_temp"].interpolate().bfill()
    print("\nMerged sample:")
    print(
        merged[["datetime", "temperature", "air_temp", "shortwave_rad"]]
        .dropna()
        .head(5)
    )

    out = Path(__file__).resolve().parent.parent / "fetch_example_out"
    out.mkdir(exist_ok=True)
    iv.to_parquet(out / f"{SITE}_iv.parquet", index=False)
    meteo.to_parquet(out / "meteo.parquet", index=False)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
