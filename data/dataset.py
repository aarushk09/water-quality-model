"""SpatioTemporalWaterDataset: sliding windows with masks for missing nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.features import FeatureEngineer, chronological_split_indices
from data.cross_site import add_cross_site_lag_features
from data.meteo_fetch import load_meteo_for_config, merge_meteo_into_site_df
from data.daymet_fetch import load_daymet
from data.graph_builder import load_graph_from_files, self_loop_edge_index
from data.usgs_rdb import load_single_site_from_config


@dataclass
class DatasetBundle:
    """Container for splits, graph, and scalers."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor]
    feature_engineer: FeatureEngineer
    site_ids: List[str]
    feature_cols: List[str]
    seq_len: int
    pred_len: int
    n_nodes: int
    n_features: int
    forecast_node: int = 0
    datetimes: np.ndarray = field(default_factory=lambda: np.array([]))
    test_window_starts: np.ndarray = field(default_factory=lambda: np.array([]))


class SpatioTemporalWaterDataset(Dataset):
    """Sliding-window dataset: x [N,T,F], y [N,H,2], mask [N]."""

    def __init__(
        self,
        panel: np.ndarray,
        masks: np.ndarray,
        seq_len: int,
        pred_len: int,
        indices: np.ndarray,
        window_datetimes: Optional[np.ndarray] = None,
        meteo_col_indices: Optional[List[int]] = None,
    ):
        self.panel = panel
        self.masks = masks
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indices = indices
        self.window_datetimes = window_datetimes
        self.meteo_col_indices = meteo_col_indices or []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = int(self.indices[idx])
        x = self.panel[t : t + self.seq_len]
        y = self.panel[
            t + self.seq_len : t + self.seq_len + self.pred_len, :, :2
        ]
        m = self.masks[t + self.seq_len : t + self.seq_len + self.pred_len].mean(axis=0)
        out = {
            "x": torch.from_numpy(x).permute(1, 0, 2).float(),
            "y": torch.from_numpy(y).permute(1, 0, 2).float(),
            "mask": torch.from_numpy(m).float(),
            "window_start": torch.tensor(t, dtype=torch.long),
        }
        if self.window_datetimes is not None:
            # Store as int64 nanoseconds for DataLoader collation
            ft = self.window_datetimes[idx].astype("datetime64[ns]").astype(np.int64)
            out["forecast_times"] = torch.from_numpy(ft)
        if self.meteo_col_indices:
            fut = self.panel[
                t + self.seq_len : t + self.seq_len + self.pred_len, :, self.meteo_col_indices
            ]
            out["fut_cov"] = torch.from_numpy(fut).permute(1, 0, 2).float()
        return out


def _build_panel_multi_site(
    site_dfs: List[pd.DataFrame],
    feat_eng: FeatureEngineer,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-site scaled features into [T, N, F] and observation masks [T, N]."""
    scaled = [feat_eng.transform(df) for df in site_dfs]
    n_time = len(scaled[0])
    n_nodes = len(scaled)
    f_dim = len(feat_eng.feature_cols)
    panel = np.zeros((n_time, n_nodes, f_dim), dtype=np.float32)
    mask = np.zeros((n_time, n_nodes), dtype=np.float32)

    for i, df in enumerate(scaled):
        panel[:, i, :] = df[feat_eng.feature_cols].values.astype(np.float32)
        raw = site_dfs[i]
        obs = raw["temperature"].notna() & raw["dissolved_oxygen"].notna()
        mask[:, i] = obs.values.astype(np.float32)

    return panel, mask


def _window_indices(n_time: int, seq_len: int, pred_len: int) -> np.ndarray:
    max_start = n_time - seq_len - pred_len
    if max_start < 0:
        raise ValueError(
            f"Series too short: need {seq_len + pred_len}, got {n_time}"
        )
    return np.arange(0, max_start + 1)


def _normalize_site_iv_df(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        df = df.reset_index()
    rename = {}
    for c in df.columns:
        cs = str(c)
        if "00010" in cs:
            rename[c] = "temperature"
        elif "00300" in cs:
            rename[c] = "dissolved_oxygen"
        elif "00060" in cs:
            rename[c] = "discharge"
        elif "00065" in cs:
            rename[c] = "gage_height"
    df = df.rename(columns=rename)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    cols = ["datetime", "temperature", "dissolved_oxygen"]
    for opt in ("discharge", "gage_height"):
        if opt in df.columns:
            cols.append(opt)
    if "dissolved_oxygen" not in df.columns:
        df["dissolved_oxygen"] = np.nan
    if "temperature" not in df.columns:
        df["temperature"] = np.nan
    return df[cols].sort_values("datetime")


def _load_site_parquet(path: Path) -> pd.DataFrame:
    return _normalize_site_iv_df(pd.read_parquet(path))


def _align_multi_site(
    panels: List[pd.DataFrame],
    site_ids: List[str],
    resample_minutes: int = 15,
) -> Tuple[List[pd.DataFrame], pd.DatetimeIndex]:
    """Align all sites to a common regular grid; forward-fill short gaps."""
    starts = [df["datetime"].min() for df in panels]
    ends = [df["datetime"].max() for df in panels]
    common_start = max(starts)
    common_end = min(ends)
    freq = f"{resample_minutes}min"
    idx = pd.date_range(common_start, common_end, freq=freq)

    aligned: List[pd.DataFrame] = []
    for df in panels:
        s = df.set_index("datetime").sort_index()
        s = s.reindex(idx)
        for col in ("temperature", "dissolved_oxygen", "discharge", "gage_height"):
            if col in s.columns:
                s[col] = s[col].interpolate(limit=8, limit_direction="both")
        s = s.reset_index().rename(columns={"index": "datetime"})
        if "datetime" not in s.columns:
            s.insert(0, "datetime", idx)
        aligned.append(s)
    return aligned, idx


def _merge_daymet_into_site_df(
    site_df: pd.DataFrame, daymet_df: pd.DataFrame
) -> pd.DataFrame:
    """Broadcast daily DAYMET values to all 15-min steps in each day."""
    out = site_df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["_date"] = out["datetime"].dt.normalize()
    dm = daymet_df.copy()
    dm["date"] = pd.to_datetime(dm["date"]).dt.normalize()
    dm = dm.rename(
        columns={
            "srad": "daymet_srad",
            "tmax": "daymet_tmax",
            "tmin": "daymet_tmin",
            "prcp": "daymet_prcp",
            "vp": "daymet_vp",
        }
    )
    keep = ["date"] + [c for c in dm.columns if c.startswith("daymet_")]
    dm = dm[keep]
    merged = out.merge(dm, left_on="_date", right_on="date", how="left")
    for col in [c for c in dm.columns if c.startswith("daymet_")]:
        merged[col] = merged[col].ffill().bfill().fillna(0.0)
    merged = merged.drop(columns=["_date", "date"], errors="ignore")
    return merged


def _load_site_data(
    root: Path,
    site: str,
    data_cfg: dict,
) -> pd.DataFrame:
    site_dir = root / data_cfg["raw_dir"] / site
    parquet = site_dir / "iv.parquet"
    if parquet.exists():
        return _load_site_parquet(parquet)
    csv_path = site_dir / "iv.csv"
    if csv_path.exists():
        return _normalize_site_iv_df(pd.read_csv(csv_path))
    return load_single_site_from_config(
        root,
        data_cfg["temperature_rdb"],
        data_cfg["dissolved_oxygen_rdb"],
        data_cfg["temp_value_column"],
        data_cfg["do_value_column"],
    )


def build_dataloaders_from_config(
    cfg: dict,
    project_root: Optional[Path] = None,
) -> DatasetBundle:
    root = Path(project_root or cfg.get("project_root", "."))
    data_cfg = cfg["data"]
    seq_len = cfg["forecast"]["seq_len"]
    pred_len = cfg["forecast"]["pred_len"]
    resample = int(data_cfg.get("resample_minutes", 15))

    stations_path = root / data_cfg.get("stations_json", "data/raw/stations.json")
    edges_path = root / data_cfg.get("edges_csv", "data/raw/edges.csv")

    if stations_path.exists():
        site_ids, edge_index_np, edge_attr_np = load_graph_from_files(
            stations_path, edges_path if edges_path.exists() else None
        )
        site_filter = data_cfg.get("sites")
        if site_filter:
            keep = [str(s) for s in site_filter]
            # Build new-index mapping from old site ordering
            old_site_ids = list(site_ids)   # full list before filter
            site_ids = [s for s in old_site_ids if s in keep]
            keep_set  = set(site_ids)
            old_to_new = {old_site_ids.index(s): i for i, s in enumerate(site_ids)}

            if len(site_ids) == 1:
                edge_index_np = self_loop_edge_index(1)
                edge_attr_np = None
            elif edge_index_np is not None:
                # Keep only edges whose both endpoints are in the filtered set,
                # then remap indices to new 0-based ordering.
                old_n = len(old_site_ids)
                valid_mask = np.array([
                    int(edge_index_np[0, e]) in old_to_new
                    and int(edge_index_np[1, e]) in old_to_new
                    for e in range(edge_index_np.shape[1])
                ])
                ei_kept = edge_index_np[:, valid_mask]
                # Vectorized remap
                src_new = np.array([old_to_new[int(v)] for v in ei_kept[0]])
                dst_new = np.array([old_to_new[int(v)] for v in ei_kept[1]])
                edge_index_np = np.stack([src_new, dst_new], axis=0)
                if edge_attr_np is not None:
                    edge_attr_np = edge_attr_np[valid_mask]

                # Ensure every node has a self-loop
                n_new = len(site_ids)
                has_self = set(zip(edge_index_np[0], edge_index_np[1]))
                extra_src, extra_dst = [], []
                for i in range(n_new):
                    if (i, i) not in has_self:
                        extra_src.append(i)
                        extra_dst.append(i)
                if extra_src:
                    sl = np.array([extra_src, extra_dst], dtype=np.int64)
                    edge_index_np = np.concatenate([edge_index_np, sl], axis=1)
                    if edge_attr_np is not None:
                        edge_attr_np = np.concatenate(
                            [edge_attr_np, np.zeros((len(extra_src), edge_attr_np.shape[1]))], axis=0
                        )
        raw_panels = [_load_site_data(root, site, data_cfg) for site in site_ids]
        if len(raw_panels) > 1:
            site_dfs, time_index = _align_multi_site(raw_panels, site_ids, resample)
        else:
            site_dfs = raw_panels
            time_index = pd.DatetimeIndex(site_dfs[0]["datetime"])
    else:
        site_ids = [data_cfg.get("site_no", "02334500")]
        site_dfs = [
            load_single_site_from_config(
                root,
                data_cfg["temperature_rdb"],
                data_cfg["dissolved_oxygen_rdb"],
                data_cfg["temp_value_column"],
                data_cfg["do_value_column"],
            )
        ]
        time_index = pd.DatetimeIndex(site_dfs[0]["datetime"])
        edge_index_np = self_loop_edge_index(len(site_ids))
        edge_attr_np = None

    meteo_df = load_meteo_for_config(cfg, root)
    if meteo_df is not None:
        meteo_tz = cfg.get("meteo", {}).get("timezone", "America/New_York")
        site_dfs = [
            merge_meteo_into_site_df(df, meteo_df, resample, meteo_tz=meteo_tz)
            for df in site_dfs
        ]

    daymet_cfg = cfg.get("daymet", {})
    if daymet_cfg.get("enabled", False):
        daymet_path = root / data_cfg.get("raw_dir", "data/raw") / daymet_cfg.get(
            "filename", "daymet.parquet"
        )
        daymet_df = load_daymet(daymet_path)
        if daymet_df is not None:
            site_dfs = [
                _merge_daymet_into_site_df(df, daymet_df) for df in site_dfs
            ]

    if edges_path.exists() and len(site_ids) > 1:
        edges_df = pd.read_csv(edges_path)
        edge_records = edges_df.to_dict("records")
        site_dfs = add_cross_site_lag_features(
            site_dfs, site_ids, edge_records, resample
        )

    n_time = len(time_index)
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    train_end, val_end = chronological_split_indices(n_time, train_ratio, val_ratio)

    feat_eng = FeatureEngineer()
    if len(site_dfs) > 1:
        feat_eng.fit_multi_site(site_dfs, train_end)
    else:
        feat_eng.fit(site_dfs[0].iloc[:train_end])

    panel, mask = _build_panel_multi_site(site_dfs, feat_eng)
    all_idx = _window_indices(n_time, seq_len, pred_len)

    datetimes = time_index.values
    forecast_times = np.stack(
        [
            datetimes[t + seq_len : t + seq_len + pred_len]
            for t in all_idx
        ],
        axis=0,
    )

    train_mask = all_idx + seq_len < train_end
    val_mask = (all_idx + seq_len >= train_end) & (all_idx + seq_len < val_end)
    test_mask = all_idx + seq_len >= val_end

    meteo_idx = list(feat_eng.meteo_col_indices)
    ds_kw = dict(meteo_col_indices=meteo_idx)
    train_ds = SpatioTemporalWaterDataset(
        panel, mask, seq_len, pred_len, all_idx[train_mask],
        forecast_times[train_mask], **ds_kw,
    )
    val_ds = SpatioTemporalWaterDataset(
        panel, mask, seq_len, pred_len, all_idx[val_mask],
        forecast_times[val_mask], **ds_kw,
    )
    test_ds = SpatioTemporalWaterDataset(
        panel, mask, seq_len, pred_len, all_idx[test_mask],
        forecast_times[test_mask], **ds_kw,
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)
    pin_memory = cfg["training"].get("pin_memory", True)

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and num_workers >= 0,
    )
    if num_workers > 0:
        loader_kw["persistent_workers"] = cfg["training"].get(
            "persistent_workers", True
        )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    edge_index = torch.from_numpy(edge_index_np).long()
    edge_attr = (
        torch.from_numpy(edge_attr_np).float() if edge_attr_np is not None else None
    )
    forecast_site = str(data_cfg.get("forecast_site", data_cfg.get("site_no", "")))
    forecast_node = (
        site_ids.index(forecast_site) if forecast_site in site_ids else 0
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_attr=edge_attr,
        feature_engineer=feat_eng,
        site_ids=site_ids,
        feature_cols=list(feat_eng.feature_cols),
        seq_len=seq_len,
        pred_len=pred_len,
        n_nodes=len(site_ids),
        n_features=len(feat_eng.feature_cols),
        forecast_node=forecast_node,
        datetimes=datetimes,
        test_window_starts=all_idx[test_mask],
    )
