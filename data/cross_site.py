"""Cross-site lag features for dam / upstream causality."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _gage_roc(series: pd.Series, steps: int = 4) -> pd.Series:
    """Rate of change over `steps` 15-min intervals (1 h at 15-min cadence)."""
    if series.isna().all():
        return pd.Series(0.0, index=series.index)
    return series.diff(steps).fillna(0.0)


def add_cross_site_lag_features(
    site_dfs: List[pd.DataFrame],
    site_ids: List[str],
    edges: List[Dict],
    resample_minutes: int = 15,
) -> List[pd.DataFrame]:
    """
    For each upstream→downstream edge, add lagged upstream signals to the
    downstream dataframe (travel-time shifted).

    Columns added to downstream site:
      up_gage_roc, up_temp_lag, up_do_lag
    """
    id_to_idx: Dict[str, int] = {s: i for i, s in enumerate(site_ids)}
    out = [df.copy() for df in site_dfs]

    for edge in edges:
        up = str(edge.get("upstream", ""))
        down = str(edge.get("downstream", ""))
        if up not in id_to_idx or down not in id_to_idx or up == down:
            continue
        travel_min = float(edge.get("travel_minutes", 60))
        lag_steps = max(1, int(round(travel_min / resample_minutes)))

        up_df = out[id_to_idx[up]]
        down_idx = id_to_idx[down]
        down_df = out[down_idx]

        if "gage_height" in up_df.columns:
            roc = _gage_roc(up_df["gage_height"], steps=4)
            down_df["up_gage_roc"] = roc.shift(lag_steps).fillna(0.0).values
        if "temperature" in up_df.columns:
            down_df["up_temp_lag"] = (
                up_df["temperature"].shift(lag_steps).bfill().ffill().values
            )
        if "dissolved_oxygen" in up_df.columns:
            down_df["up_do_lag"] = (
                up_df["dissolved_oxygen"].shift(lag_steps).bfill().ffill().values
            )
        out[down_idx] = down_df

    cross_cols = ["up_gage_roc", "up_temp_lag", "up_do_lag"]
    for i, df in enumerate(out):
        for col in cross_cols:
            if col not in df.columns:
                df[col] = 0.0
        out[i] = df

    return out
