# Fetching water-quality and weather data

This project pulls two free data sources — no API keys, no accounts.

| Source | What you get | Library |
|--------|----------------|---------|
| **USGS NWIS** | 15-min water temperature, dissolved oxygen, discharge, gage height | `dataretrieval` |
| **Open-Meteo** | Hourly solar radiation, air temp, wind, humidity (ERA5 reanalysis) | `urllib` (stdlib) |

Target forecast site: **USGS 02334500** (Chattahoochee River near Buford, GA).  
Upstream dam site: **02334430** (Buford Dam release signal).

---

## Setup (one time)

```bash
cd /path/to/research
python3 -m venv myenv && source myenv/bin/activate
pip install dataretrieval pandas pyarrow
```

---

## Option A — project CLI (full year, multi-site graph)

```bash
source myenv/bin/activate

# 1) USGS IV for dam + target + downstream aux sites → data/raw/<site>/iv.parquet
python -m data.nwis_fetch --config configs/chattahoochee_graph.yaml

# 2) Open-Meteo at station lat/lon → data/raw/meteo.parquet
python -m data.meteo_fetch --config configs/high_accuracy.yaml
```

Outputs:

```
data/raw/stations.json     # site metadata (lat/lon, paired fraction)
data/raw/edges.csv         # upstream→downstream edges + travel time
data/raw/02334500/iv.parquet
data/raw/meteo.parquet
```

---

## Option B — copy-paste example (7 days, single site)

Save as `fetch_example.py` anywhere, or run from the project root after `pip install dataretrieval pandas`.

```python
"""Minimal NWIS + Open-Meteo fetch — runnable in ~30 seconds."""
from pathlib import Path

import pandas as pd

# --- 1) USGS instantaneous values (15-min) ---
import dataretrieval.nwis as nwis

SITE = "02334500"
START, END = "2025-05-15", "2025-05-22"
PARAMS = "00010,00300,00060,00065"  # temp, DO, discharge, gage height

iv, _ = nwis.get_iv(sites=SITE, start=START, end=END, parameterCd=PARAMS)
iv = iv.reset_index()
# Rename NWIS parameter codes to readable columns
rename = {}
for col in iv.columns:
    if "00010" in str(col):
        rename[col] = "temperature"
    elif "00300" in str(col):
        rename[col] = "dissolved_oxygen"
    elif "00060" in str(col):
        rename[col] = "discharge"
    elif "00065" in str(col):
        rename[col] = "gage_height"
    elif col.lower() in ("datetime", "datetime_1") or "dateTime" in str(col):
        rename[col] = "datetime"
iv = iv.rename(columns=rename)
if "datetime" not in iv.columns:
    iv = iv.rename(columns={iv.columns[0]: "datetime"})
keep = ["datetime"] + [c for c in ("temperature", "dissolved_oxygen", "discharge", "gage_height") if c in iv.columns]
iv = iv.loc[:, ~iv.columns.duplicated()].filter(items=keep, axis=1)
iv["datetime"] = pd.to_datetime(iv["datetime"]).dt.tz_convert("UTC").dt.tz_localize(None)

print("NWIS rows:", len(iv))
print(iv[["datetime", "temperature", "dissolved_oxygen"]].dropna().head(3))

# --- 2) Open-Meteo archive (hourly, no API key) ---
import json
import urllib.request

LAT, LON = 34.126, -84.093  # USGS 02334500
url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={START}&end_date={END}"
    "&hourly=shortwave_radiation,temperature_2m,wind_speed_10m"
    "&timezone=America%2FNew_York"
)
with urllib.request.urlopen(url, timeout=60) as resp:
    meteo = pd.DataFrame(json.loads(resp.read())["hourly"])
meteo = meteo.rename(columns={"time": "datetime", "temperature_2m": "air_temp",
                              "shortwave_radiation": "shortwave_rad"})
meteo["datetime"] = (
    pd.to_datetime(meteo["datetime"])
    .dt.tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward")
    .dt.tz_convert("UTC")
    .dt.tz_localize(None)
)

print("\nMeteo rows:", len(meteo))
print(meteo.head(3))

# --- 3) Quick join on date (hourly meteo → water timestamps) ---
merged = iv.merge(meteo, on="datetime", how="left")
merged["air_temp"] = merged["air_temp"].interpolate().bfill()
print("\nMerged sample (15-min water + hourly meteo):")
print(merged[["datetime", "temperature", "air_temp", "shortwave_rad"]].dropna().head(5))

# Optional: save
out = Path("fetch_example_out")
out.mkdir(exist_ok=True)
iv.to_parquet(out / f"{SITE}_iv.parquet", index=False)
meteo.to_parquet(out / "meteo.parquet", index=False)
print(f"\nSaved to {out.resolve()}/")
```

Run it (from project root):

```bash
python scripts/fetch_example.py
```

Expected output (truncated):

```
NWIS rows: 672
                 datetime  temperature  dissolved_oxygen
0 2025-05-15 04:00:00+00:00         11.2               8.4
...
Meteo rows: 192
             datetime  air_temp  shortwave_rad
0 2025-05-15 00:00:00      14.1            0.0
...
Saved to .../fetch_example_out/
```

---

## Parameter codes (NWIS)

| Code | Column | Units |
|------|--------|-------|
| `00010` | temperature | °C |
| `00300` | dissolved_oxygen | mg/L |
| `00060` | discharge | ft³/s |
| `00065` | gage_height | ft |

Find more sites: [USGS Water Data for the Nation](https://waterdata.usgs.gov/nwis) → search by state/HUC → copy the 8-digit site ID.

---

## Why two sources?

Water temperature and DO follow **weather** (solar heating, air temp) and **dam operations** (cold hypolimnetic releases from Buford Dam). NWIS gives the river response; Open-Meteo gives the atmospheric forcing the model needs for diurnal peaks.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ImportError: dataretrieval` | `pip install dataretrieval` |
| `ImportError: pyarrow` | `pip install pyarrow` (or CSV fallback is automatic in project fetchers) |
| Empty NWIS result | Check site has IV (not daily) data for your date range |
| Meteo timezone mismatch | Project fetcher converts Open-Meteo (Eastern) → UTC to match NWIS parquet |

For the full training pipeline after fetch, see `configs/high_accuracy.yaml` and `report.md`.
