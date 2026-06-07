"""
Fetch historical meteorology from Open-Meteo (free, no API key).

Aligns to site datetime grid for feature engineering.

Usage:
    python -m data.meteo_fetch --config configs/high_accuracy.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

try:
    import urllib.request
except ImportError:
    urllib = None  # type: ignore

METEO_COLUMNS = [
    "shortwave_rad",  # W/m²
    "air_temp",       # °C
    "wind_speed",     # m/s
    "cloud_cover",    # %
    "rh",             # %
]

OPEN_METEO_HOURLY_MAP = {
    "shortwave_radiation": "shortwave_rad",
    "temperature_2m": "air_temp",
    "wind_speed_10m": "wind_speed",
    "cloud_cover": "cloud_cover",
    "relative_humidity_2m": "rh",
}


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _infer_date_range_from_site(root: Path, data_cfg: dict) -> tuple[str, str]:
    """Use NWIS parquet or RDB span when meteo config omits dates."""
    site = data_cfg.get("site_no", "02334500")
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    parquet = root / raw_dir / site / "iv.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet, columns=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return (
            df["datetime"].min().strftime("%Y-%m-%d"),
            df["datetime"].max().strftime("%Y-%m-%d"),
        )
    return "2024-07-15", "2025-07-15"


def fetch_open_meteo_archive(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Download hourly ERA5-based archive from Open-Meteo."""
    hourly_vars = ",".join(OPEN_METEO_HOURLY_MAP.keys())
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={hourly_vars}&timezone={timezone.replace('/', '%2F')}"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode())

    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(f"Open-Meteo returned no hourly data: {payload}")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns=OPEN_METEO_HOURLY_MAP)
    df = df[["datetime"] + METEO_COLUMNS].sort_values("datetime")
    df["datetime"] = _to_utc_naive(df["datetime"], timezone)
    return df


def _to_utc_naive(series: pd.Series, assume_tz: str = "America/New_York") -> pd.Series:
    """Normalize datetimes to UTC-naive for alignment with NWIS parquet."""
    dt = pd.to_datetime(series)
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(
            assume_tz, ambiguous=False, nonexistent="shift_forward"
        )
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)


def resample_meteo_to_grid(
    meteo: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    freq_minutes: int = 15,
    meteo_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Interpolate hourly meteo onto the water-quality 15-min grid."""
    meteo = meteo.copy()
    meteo["datetime"] = _to_utc_naive(meteo["datetime"], meteo_tz)
    meteo = meteo.set_index("datetime").sort_index()
    meteo = meteo[~meteo.index.duplicated(keep="first")]
    if target_index.tz is not None:
        target_index = target_index.tz_convert("UTC").tz_localize(None)
    start = min(meteo.index.min(), target_index.min())
    end = max(meteo.index.max(), target_index.max())
    full_idx = pd.date_range(start, end, freq=f"{freq_minutes}min")
    aligned = meteo.reindex(full_idx)
    aligned[METEO_COLUMNS] = aligned[METEO_COLUMNS].interpolate(
        method="time", limit_direction="both"
    )
    out = aligned.reindex(target_index)
    return out.reset_index().rename(columns={"index": "datetime"})


def merge_meteo_into_site_df(
    site_df: pd.DataFrame,
    meteo_df: pd.DataFrame,
    resample_minutes: int = 15,
    meteo_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Left-join meteo columns onto a site dataframe by datetime."""
    site_df = site_df.copy()
    site_df["datetime"] = _to_utc_naive(site_df["datetime"])
    idx = pd.DatetimeIndex(site_df["datetime"])
    meteo_aligned = resample_meteo_to_grid(
        meteo_df, idx, resample_minutes, meteo_tz=meteo_tz
    )
    merged = site_df.merge(meteo_aligned, on="datetime", how="left")
    for col in METEO_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].interpolate(limit_direction="both").bfill().ffill()
    return merged


def _site_lat_lon_from_config(root: Path, data_cfg: dict) -> tuple[float, float]:
    stations_path = root / data_cfg.get("stations_json", "data/raw/stations.json")
    site_no = str(data_cfg.get("site_no", "02334500"))
    if stations_path.exists():
        with open(stations_path) as f:
            stations = json.load(f)
        for st in stations:
            if str(st.get("site_no")) == site_no:
                return float(st.get("lat", 34.18)), float(st.get("lon", -84.09))
    return 34.18, -84.09


def run_fetch(cfg: dict, project_root: Optional[Path] = None) -> Path:
    root = Path(project_root or cfg.get("project_root", "."))
    meteo_cfg = cfg.get("meteo", {})
    data_cfg = cfg.get("data", {})

    lat_cfg = meteo_cfg.get("latitude")
    lon_cfg = meteo_cfg.get("longitude")
    if lat_cfg is None or lon_cfg is None:
        lat, lon = _site_lat_lon_from_config(root, data_cfg)
    else:
        lat, lon = float(lat_cfg), float(lon_cfg)
    tz = meteo_cfg.get("timezone", "America/New_York")
    start = meteo_cfg.get("start_date")
    end = meteo_cfg.get("end_date")
    if not start or not end:
        start, end = _infer_date_range_from_site(root, data_cfg)

    print(f"Fetching Open-Meteo archive {start} → {end} @ ({lat}, {lon})")
    df = fetch_open_meteo_archive(lat, lon, start, end, tz)

    out_dir = root / data_cfg.get("raw_dir", "data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / meteo_cfg.get("filename", "meteo.parquet")
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} hourly rows)")
    return out_path


def load_meteo_for_config(cfg: dict, project_root: Path) -> Optional[pd.DataFrame]:
    """Load cached meteo if enabled in config."""
    meteo_cfg = cfg.get("meteo", {})
    if not meteo_cfg.get("enabled", False):
        return None
    data_cfg = cfg.get("data", {})
    raw_dir = project_root / data_cfg.get("raw_dir", "data/raw")
    fname = meteo_cfg.get("filename", "meteo.parquet")
    path = raw_dir / fname
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            path = csv_path
        else:
            print(f"Meteo enabled but {path} missing — run: python -m data.meteo_fetch")
            return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo historical meteo")
    parser.add_argument("--config", type=Path, default=Path("configs/high_accuracy.yaml"))
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = args.project_root or Path(cfg.get("project_root", "."))
    run_fetch(cfg, root)


if __name__ == "__main__":
    main()
