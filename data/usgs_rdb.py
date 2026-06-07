"""Portable USGS RDB (tab-delimited) parser."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_usgs_rdb(
    filepath: Union[str, Path],
    value_column_name: str,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """
    Load and preprocess USGS RDB format data.

    Parameters
    ----------
    filepath : path to .rdb file
    value_column_name : e.g. '334112_00010' or '334113_00300'

    Returns
    -------
    DataFrame with columns ['datetime', <short_name>]
    where short_name is the parameter code (e.g. '00010').
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, sep="\t", comment="#", skip_blank_lines=True)
    df = df[df["agency_cd"] != "5s"]
    df = df.rename(columns=str.strip)
    df = df[df[datetime_col].notna()]

    df[datetime_col] = pd.to_datetime(df[datetime_col], format="%Y-%m-%d %H:%M")
    short_name = value_column_name.split("_")[-1]
    out = df[[datetime_col, value_column_name]].copy()
    out = out.rename(columns={value_column_name: short_name})
    out[short_name] = pd.to_numeric(out[short_name], errors="coerce")
    return out.dropna().reset_index(drop=True)


def merge_temperature_do(
    temp_df: pd.DataFrame,
    do_df: pd.DataFrame,
    temp_col: str = "00010",
    do_col: str = "00300",
) -> pd.DataFrame:
    """Merge temperature and DO on datetime via merge_asof (nearest)."""
    temp_df = temp_df.rename(columns={temp_col: "temperature"})
    do_df = do_df.rename(columns={do_col: "dissolved_oxygen"})
    merged = pd.merge_asof(
        temp_df.sort_values("datetime"),
        do_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",
    )
    return merged.dropna().reset_index(drop=True)


def load_single_site_from_config(
    project_root: Path,
    temp_rdb: str,
    do_rdb: str,
    temp_column: str,
    do_column: str,
) -> pd.DataFrame:
    """Load merged single-site panel from repo-relative RDB paths."""
    root = Path(project_root)
    temp_df = load_usgs_rdb(root / temp_rdb, temp_column)
    do_df = load_usgs_rdb(root / do_rdb, do_column)
    return merge_temperature_do(temp_df, do_df)
