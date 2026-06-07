"""
Download multi-site USGS NWIS IV data for Chattahoochee graph stations.

Usage:
    python -m data.nwis_fetch --config configs/chattahoochee_graph.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

try:
    import dataretrieval.nwis as nwis
except ImportError:
    nwis = None  # type: ignore

# NWIS parameter code -> canonical column name
PARAM_CODE_MAP = {
    "00010": "temperature",
    "00300": "dissolved_oxygen",
    "00060": "discharge",
    "00065": "gage_height",
}


def load_graph_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _unwrap_nwis_df(result) -> pd.DataFrame:
    """dataretrieval >= 1.0 often returns (DataFrame, metadata)."""
    if isinstance(result, tuple):
        result = result[0]
    if result is None:
        return pd.DataFrame()
    return result


def parameter_codes_string(cfg: Dict[str, Any]) -> str:
    if cfg.get("parameter_codes"):
        return str(cfg["parameter_codes"])
    params = cfg.get("parameters", {})
    codes = [params[k] for k in ("temperature", "dissolved_oxygen", "discharge", "gage_height") if k in params]
    return ",".join(codes) if codes else "00010,00300,00060,00065"


def normalize_iv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map NWIS IV columns to canonical names; keep datetime index as column."""
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex) or out.index.name:
        out = out.reset_index()

    dt_col = None
    for c in out.columns:
        cl = str(c).lower()
        if cl in ("datetime", "dateTime", "date_time") or "datetime" in cl:
            dt_col = c
            break
    if dt_col is None:
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                dt_col = c
                break
    if dt_col is not None:
        out = out.rename(columns={dt_col: "datetime"})
        out["datetime"] = pd.to_datetime(out["datetime"])

    rename: Dict[str, str] = {}
    for c in out.columns:
        if c == "datetime":
            continue
        for code, name in PARAM_CODE_MAP.items():
            if code in str(c) and name not in rename.values():
                rename[c] = name
                break
    out = out.rename(columns=rename)

    keep = ["datetime"] + [c for c in PARAM_CODE_MAP.values() if c in out.columns]
    out = out[[c for c in keep if c in out.columns]]
    if "datetime" in out.columns:
        out = out.sort_values("datetime").drop_duplicates("datetime")
    return out


def _normalize_sites_df(sites_df: pd.DataFrame) -> pd.DataFrame:
    if sites_df is None or sites_df.empty:
        return pd.DataFrame(columns=["site_no"])
    site_col = "site_no" if "site_no" in sites_df.columns else sites_df.columns[0]
    out = sites_df.rename(columns={site_col: "site_no"})
    out["site_no"] = out["site_no"].astype(str).str.zfill(8)
    return out.drop_duplicates("site_no")


def _query_site_catalog(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Query NWIS site catalog with fallbacks.

    USGS often returns 404 for invalid/empty HUC filters; we try shorter HUCs,
    state filter, bounding box, then fall back to seed sites only.
    """
    seed_sites = [str(s) for s in cfg.get("seed_sites", [])]
    if cfg.get("discovery_mode") == "seeds_only":
        print("discovery_mode=seeds_only — skipping NWIS site catalog query.")
        return pd.DataFrame({"site_no": seed_sites})

    huc8 = str(cfg.get("huc8", "")).strip()
    state_cd = cfg.get("state_cd")
    bbox = cfg.get("bbox")
    param_cd = "00010,00300"
    max_catalog = int(cfg.get("max_catalog_sites", 40))

    attempts: List[tuple] = []
    if huc8:
        for n in (8, 6, 4):
            if len(huc8) >= n:
                attempts.append(("huc", {"huc": huc8[:n]}))
    if bbox:
        attempts.append(("bBox", {"bBox": bbox}))
    if state_cd:
        attempts.append(("stateCd", {"stateCd": str(state_cd)}))

    for label, kwargs in attempts:
        try:
            sites_df = _unwrap_nwis_df(
                nwis.what_sites(
                    parameterCd=param_cd,
                    siteType="ST",
                    **kwargs,
                )
            )
            sites_df = _normalize_sites_df(sites_df)
            if not sites_df.empty:
                if len(sites_df) > max_catalog:
                    print(
                        f"Site catalog from NWIS ({label}): {len(sites_df)} sites "
                        f"(capped to {max_catalog} for probing)"
                    )
                    sites_df = sites_df.iloc[:max_catalog].reset_index(drop=True)
                else:
                    print(f"Site catalog from NWIS ({label}): {len(sites_df)} sites")
                return sites_df
        except Exception as exc:
            print(f"NWIS site catalog ({label}) failed: {exc}")

    print("Site catalog query failed — using seed_sites only.")
    return pd.DataFrame({"site_no": seed_sites})


def _site_has_iv_signal(df: pd.DataFrame, require_do: bool) -> bool:
    if "temperature" not in df.columns:
        return False
    if require_do and "dissolved_oxygen" not in df.columns:
        return False
    paired = df["temperature"].notna()
    if require_do:
        paired = paired & df["dissolved_oxygen"].notna()
    return float(paired.mean()) > 0.0


def discover_sites(
    cfg: Dict[str, Any],
    start: str,
    end: str,
    min_paired_fraction: float,
    parameter_codes: str,
) -> List[Dict[str, Any]]:
    if nwis is None:
        raise ImportError("Install dataretrieval: pip install dataretrieval")

    seed_sites = [str(s) for s in cfg.get("seed_sites", [])]
    aux_sites = [str(s) for s in cfg.get("auxiliary_sites", [])]
    sites_df = _query_site_catalog(cfg)

    for s in seed_sites + aux_sites:
        if s not in sites_df["site_no"].astype(str).values:
            sites_df = pd.concat(
                [sites_df, pd.DataFrame({"site_no": [s]})], ignore_index=True
            )

    selected: List[Dict[str, Any]] = []
    seen: set = set()
    for site_no in sites_df["site_no"].astype(str).unique():
        try:
            df = _unwrap_nwis_df(
                nwis.get_iv(
                    sites=site_no,
                    start=start,
                    end=end,
                    parameterCd=parameter_codes,
                )
            )
        except Exception as exc:
            print(f"Skip {site_no}: {exc}")
            continue
        if df is None or df.empty:
            print(f"Skip {site_no}: no IV data")
            continue
        df = normalize_iv_columns(df)
        is_aux = site_no in aux_sites
        require_do = site_no not in aux_sites
        if not _site_has_iv_signal(df, require_do=require_do):
            print(f"Skip {site_no}: missing required IV columns")
            continue
        if "dissolved_oxygen" in df.columns:
            paired = df["temperature"].notna() & df["dissolved_oxygen"].notna()
            frac = float(paired.mean())
        else:
            paired = df["temperature"].notna()
            frac = float(paired.mean())
        min_frac = cfg.get("auxiliary_min_paired_fraction", 0.3) if is_aux else min_paired_fraction
        if frac < min_frac:
            print(f"Skip {site_no}: paired fraction {frac:.2f} < {min_frac}")
            continue
        meta: Dict[str, Any] = {
            "site_no": site_no,
            "paired_fraction": frac,
            "role": "auxiliary" if is_aux else "forecast",
        }
        try:
            info = _unwrap_nwis_df(nwis.get_info(sites=site_no))
            if info is not None and not info.empty:
                row = info.iloc[0]
                meta["lat"] = float(row.get("dec_lat_va", row.get("lat", 0.0)) or 0.0)
                meta["lon"] = float(row.get("dec_long_va", row.get("long", 0.0)) or 0.0)
                meta["name"] = str(row.get("station_nm", ""))
        except Exception:
            meta["lat"] = 0.0
            meta["lon"] = 0.0
        if site_no not in seen:
            selected.append(meta)
            seen.add(site_no)
            role = meta.get("role", "forecast")
            print(f"Selected {site_no} ({role}, paired={frac:.2%})")

    return selected


def download_site_iv(
    site_no: str,
    start: str,
    end: str,
    out_dir: Path,
    parameter_codes: str,
) -> Optional[Path]:
    if nwis is None:
        raise ImportError("Install dataretrieval: pip install dataretrieval")
    df = _unwrap_nwis_df(
        nwis.get_iv(
            sites=site_no,
            start=start,
            end=end,
            parameterCd=parameter_codes,
        )
    )
    if df is None or df.empty:
        return None
    df = normalize_iv_columns(df)
    site_dir = out_dir / site_no
    site_dir.mkdir(parents=True, exist_ok=True)
    path = site_dir / "iv.parquet"
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        path = site_dir / "iv.csv"
        df.to_csv(path, index=False)
        print(f"  (pyarrow missing — saved CSV instead of parquet)")
    return path


def build_edges(
    stations: List[Dict[str, Any]],
    manual_edges: List,
    k_nn: int,
    resample_minutes: int = 15,
) -> pd.DataFrame:
    rows = []
    for edge in manual_edges:
        if isinstance(edge, dict):
            up = edge["upstream"]
            down = edge["downstream"]
            travel_min = float(edge.get("travel_minutes", 60))
            travel_steps = max(1, int(round(travel_min / resample_minutes)))
            rows.append(
                {
                    "upstream": up,
                    "downstream": down,
                    "travel_minutes": travel_min,
                    "travel_steps": travel_steps,
                }
            )
        else:
            up, down = edge[0], edge[1]
            rows.append({"upstream": up, "downstream": down, "travel_minutes": 60, "travel_steps": 4})
    if not rows and len(stations) > 1:
        lats = [s.get("lat", 0.0) for s in stations]
        order = sorted(range(len(stations)), key=lambda i: lats[i], reverse=True)
        for i in range(len(order) - 1):
            rows.append(
                {
                    "upstream": stations[order[i]]["site_no"],
                    "downstream": stations[order[i + 1]]["site_no"],
                    "travel_minutes": 60,
                    "travel_steps": max(1, int(round(60 / resample_minutes))),
                }
            )
    elif len(stations) == 1:
        s = stations[0]["site_no"]
        rows.append(
            {
                "upstream": s,
                "downstream": s,
                "travel_minutes": 0,
                "travel_steps": 0,
            }
        )
    return pd.DataFrame(rows)


def run_fetch(config_path: Path, project_root: Optional[Path] = None) -> None:
    cfg = load_graph_config(config_path)
    root = Path(project_root or ".")
    out_dir = root / cfg["output"]["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pcodes = parameter_codes_string(cfg)

    stations = discover_sites(
        cfg,
        start=cfg["start_date"],
        end=cfg["end_date"],
        min_paired_fraction=cfg.get("min_paired_fraction", 0.5),
        parameter_codes=pcodes,
    )
    if not stations:
        print("No stations selected; using seed sites only.")
        stations = [{"site_no": s, "lat": 0.0, "lon": 0.0} for s in cfg.get("seed_sites", [])]

    for st in stations:
        download_site_iv(
            st["site_no"], cfg["start_date"], cfg["end_date"], out_dir, pcodes
        )

    stations_path = root / cfg["output"]["stations_json"]
    stations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stations_path, "w") as f:
        json.dump(stations, f, indent=2)

    resample = int(cfg.get("resample_minutes", 15))
    edges = build_edges(
        stations,
        cfg.get("graph", {}).get("manual_edges", []),
        cfg.get("graph", {}).get("k_nn", 3),
        resample_minutes=resample,
    )
    edges_path = root / cfg["output"]["edges_csv"]
    edges["upstream"] = edges["upstream"].astype(str).str.zfill(8)
    edges["downstream"] = edges["downstream"].astype(str).str.zfill(8)
    edges.to_csv(edges_path, index=False)
    print(f"Wrote {stations_path} ({len(stations)} sites) and {edges_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NWIS IV for graph stations")
    parser.add_argument("--config", type=Path, default=Path("configs/chattahoochee_graph.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    run_fetch(args.config, args.project_root)


if __name__ == "__main__":
    main()
