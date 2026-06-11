"""Feature engineering for spatiotemporal water-quality forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLS = ["temperature", "dissolved_oxygen"]

# Targets are first two columns; remaining are inputs.
BASE_FEATURE_COLS = [
    "temperature",
    "dissolved_oxygen",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "temp_ma_6h",
    "temp_ma_24h",
    "do_ma_6h",
    "temp_diff_1h",
    "temp_diff_6h",
    "do_deficit",
    "do_sat_frac",
    "temp_accel_4",
    "temp_lag_4",
    "temp_lag_24",
    "temp_lag_96",
    "do_lag_4",
    "do_lag_24",
    "do_lag_96",
]

OPTIONAL_FEATURE_COLS = ["discharge_log", "gage_height"]

METEO_FEATURE_COLS = [
    "shortwave_rad",
    "air_temp",
    "wind_speed",
    "cloud_cover",
    "rh",
]

METEO_DERIVED_COLS = [
    "air_water_delta",
    "shortwave_ma_4",
]

DAM_FEATURE_COLS = [
    "gage_height_roc",
]

CROSS_SITE_COLS = [
    "up_gage_roc",
    "up_temp_lag",
    "up_do_lag",
]

DAYMET_FEATURE_COLS = ["daymet_srad", "daymet_tmax", "daymet_prcp", "daymet_vp"]

DAYMET_DERIVED_COLS = ["daymet_srad_ma3d"]

# Computed from lat/lon + timestamp — no API required
SOLAR_FEATURE_COLS = [
    "solar_zenith_cos",     # cos(zenith angle) — proxy for shortwave intensity
    "clearsky_rad",         # theoretical clear-sky irradiance (W/m²)
    "temp_range_24h",       # diurnal thermal amplitude over past 24h
    "temp_range_48h",       # 48h thermal range (multi-day regime)
    "do_diurnal_phase",     # phase of DO within its diurnal cycle (0-1)
]


def do_saturation_np(temperature_c: np.ndarray) -> np.ndarray:
    """Benson–Krause freshwater DO saturation (mg/L)."""
    t = np.asarray(temperature_c, dtype=np.float64)
    tk = t + 273.15
    ln_do = (
        -139.34411
        + (1.575701e5) / tk
        - (6.642308e7) / (tk**2)
        + (1.243800e10) / (tk**3)
        - (8.621949e11) / (tk**4)
    )
    return np.exp(ln_do)


# Chattahoochee River, near Buford GA — used for computed solar features
_DEFAULT_LAT = 34.18
_DEFAULT_LON = -84.09


def _solar_zenith_cos(dt: pd.Series, lat_deg: float = _DEFAULT_LAT) -> np.ndarray:
    """Approximate cos(solar zenith angle) from hour-of-day and day-of-year.
    Uses simplified Spencer equation — no API needed."""
    lat = math.radians(lat_deg)
    doy = dt.dt.dayofyear.values.astype(float)
    # Solar declination (Spencer 1971)
    B = (2 * math.pi / 365) * (doy - 1)
    decl = (
        0.006918
        - 0.399912 * np.cos(B)
        + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2 * B)
        + 0.000907 * np.sin(2 * B)
        - 0.002697 * np.cos(3 * B)
        + 0.00148 * np.sin(3 * B)
    )
    # Equation of time (minutes)
    eqt = 229.18 * (
        0.000075
        + 0.001868 * np.cos(B)
        - 0.032077 * np.sin(B)
        - 0.014615 * np.cos(2 * B)
        - 0.04089 * np.sin(2 * B)
    )
    # Hour angle in radians
    hour_frac = dt.dt.hour.values + dt.dt.minute.values / 60.0 + eqt / 60.0
    ha = math.radians(15) * (hour_frac - 12.0)
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.clip(cos_z, 0.0, 1.0)


def _clearsky_irradiance(cos_z: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Simplified Ineichen clear-sky model: I_cs = I₀ · cos(z) · transmittance.
    I₀ varies with Earth-Sun distance correction."""
    B = (2 * math.pi / 365) * (doy - 1)
    # Eccentricity correction (distance factor)
    E0 = 1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B)
    I0 = 1361.0 * E0  # W/m²
    # Simple Beer-Lambert atmosphere transmittance
    # Using typical midlatitude clear-sky optical depth ≈ 0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        airmass = np.where(cos_z > 0.087, 1.0 / cos_z, 11.5)  # cap at AM≈11.5
    tau = np.exp(-0.25 * airmass)
    return I0 * cos_z * tau


def build_feature_columns(
    has_discharge: bool = False,
    has_gage: bool = False,
    has_meteo: bool = False,
    has_dam: bool = False,
    has_cross_site: bool = False,
    has_daymet: bool = False,
    has_solar: bool = True,
) -> List[str]:
    cols = list(BASE_FEATURE_COLS)
    if has_solar:
        cols.extend(SOLAR_FEATURE_COLS)
    if has_meteo:
        cols.extend(METEO_FEATURE_COLS)
        cols.extend(METEO_DERIVED_COLS)
    if has_daymet:
        cols.extend(DAYMET_FEATURE_COLS)
        cols.extend(DAYMET_DERIVED_COLS)
    if has_dam:
        cols.extend(DAM_FEATURE_COLS)
    if has_cross_site:
        cols.extend(CROSS_SITE_COLS)
    if has_discharge:
        cols.append("discharge_log")
    if has_gage:
        cols.append("gage_height")
    return cols


@dataclass
class FeatureEngineer:
    """Fit scalers on train split only; transform full series."""

    feature_cols: List[str] = field(default_factory=lambda: list(BASE_FEATURE_COLS))
    target_cols: List[str] = field(default_factory=lambda: list(TARGET_COLS))
    feature_scaler: Optional[StandardScaler] = None
    target_scaler: Optional[StandardScaler] = None
    has_discharge: bool = False
    has_gage: bool = False
    has_meteo: bool = False
    has_dam: bool = False
    has_cross_site: bool = False
    has_daymet: bool = False
    has_solar: bool = True
    site_lat: float = _DEFAULT_LAT
    site_lon: float = _DEFAULT_LON
    meteo_col_indices: Optional[List[int]] = field(default_factory=list)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "datetime" not in out.columns:
            raise ValueError("DataFrame must contain 'datetime'")

        out["hour"] = out["datetime"].dt.hour
        out["day_of_year"] = out["datetime"].dt.dayofyear
        out["month"] = out["datetime"].dt.month

        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["day_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
        out["day_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

        out["temp_ma_6h"] = out["temperature"].rolling(6, min_periods=1).mean()
        out["temp_ma_24h"] = out["temperature"].rolling(24, min_periods=1).mean()
        # Aux sites may lack DO — use zero proxy for input features only; keep
        # dissolved_oxygen NaN for target masking.
        do_feat = out["dissolved_oxygen"].fillna(0.0)
        out["do_ma_6h"] = do_feat.rolling(6, min_periods=1).mean()
        out["temp_diff_1h"] = out["temperature"].diff(1).fillna(0)
        out["temp_diff_6h"] = out["temperature"].diff(6).fillna(0)

        do_sat = do_saturation_np(out["temperature"].values)
        out["do_deficit"] = do_sat - do_feat.values
        out["do_sat_frac"] = (do_feat.values / np.clip(do_sat, 0.5, None)).clip(
            0.0, 1.5
        )
        out["temp_accel_4"] = out["temperature"].diff(4).fillna(0) / 4.0

        for lag, name in [(4, "4"), (24, "24"), (96, "96")]:
            out[f"temp_lag_{name}"] = out["temperature"].shift(lag).bfill()
            out[f"do_lag_{name}"] = do_feat.shift(lag).bfill().fillna(0.0)

        if self.has_discharge or "discharge" in out.columns:
            if "discharge" in out.columns:
                out["discharge_log"] = np.log1p(
                    out["discharge"].clip(lower=0).fillna(0)
                )
            else:
                out["discharge_log"] = 0.0
        if self.has_gage or "gage_height" in out.columns:
            if "gage_height" in out.columns:
                out["gage_height"] = out["gage_height"].bfill().ffill()
            else:
                out["gage_height"] = 0.0
        if self.has_dam or "gage_height" in out.columns:
            if "gage_height" in out.columns:
                out["gage_height_roc"] = out["gage_height"].diff(4).fillna(0.0)
            else:
                out["gage_height_roc"] = 0.0

        if "air_temp" in out.columns:
            out["air_water_delta"] = out["air_temp"] - out["temperature"]
        if "shortwave_rad" in out.columns:
            out["shortwave_ma_4"] = (
                out["shortwave_rad"].rolling(4, min_periods=1).mean()
            )

        if "daymet_srad" in out.columns:
            out["daymet_srad_ma3d"] = (
                out["daymet_srad"].rolling(96 * 3, min_periods=1).mean()
            )

        # Computed solar features — purely from lat/lon + timestamp, no API
        if self.has_solar:
            cos_z = _solar_zenith_cos(out["datetime"], lat_deg=self.site_lat)
            doy = out["datetime"].dt.dayofyear.values.astype(float)
            cs_rad = _clearsky_irradiance(cos_z, doy)
            out["solar_zenith_cos"] = cos_z
            out["clearsky_rad"] = cs_rad
            # Thermal range features (multi-timescale inertia)
            out["temp_range_24h"] = (
                out["temperature"].rolling(96, min_periods=4).max()
                - out["temperature"].rolling(96, min_periods=4).min()
            ).fillna(0.0)
            out["temp_range_48h"] = (
                out["temperature"].rolling(192, min_periods=4).max()
                - out["temperature"].rolling(192, min_periods=4).min()
            ).fillna(0.0)
            # DO diurnal phase (normalized 0–1 within daily range)
            do_feat_solar = out["dissolved_oxygen"].fillna(0.0)
            do_daily_min = do_feat_solar.rolling(96, min_periods=4).min()
            do_daily_max = do_feat_solar.rolling(96, min_periods=4).max()
            do_range = (do_daily_max - do_daily_min).clip(lower=0.01)
            out["do_diurnal_phase"] = (
                (do_feat_solar - do_daily_min) / do_range
            ).clip(0.0, 1.0).fillna(0.5)

        for col in CROSS_SITE_COLS:
            if col not in out.columns:
                out[col] = 0.0

        skip = {"datetime", *self.target_cols}
        input_cols = [c for c in out.columns if c not in skip]
        out[input_cols] = out[input_cols].bfill().ffill().fillna(0.0)
        return out

    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        self.has_discharge = "discharge" in train_df.columns
        self.has_gage = "gage_height" in train_df.columns
        self.has_meteo = all(c in train_df.columns for c in METEO_FEATURE_COLS)
        self.has_dam = "gage_height" in train_df.columns
        self.has_cross_site = any(c in train_df.columns for c in CROSS_SITE_COLS)
        self.has_daymet = any(c in train_df.columns for c in DAYMET_FEATURE_COLS)
        # Solar features are always computed (lat/lon only)
        self.has_solar = True
        self.feature_cols = build_feature_columns(
            self.has_discharge,
            self.has_gage,
            self.has_meteo,
            self.has_dam,
            self.has_cross_site,
            self.has_daymet,
            has_solar=self.has_solar,
        )
        train_feat = self.build_features(train_df)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_scaler.fit(train_feat[self.feature_cols].values)
        valid_tgt = train_feat[self.target_cols].notna().all(axis=1)
        tgt_rows = train_feat.loc[valid_tgt, self.target_cols]
        if tgt_rows.empty:
            raise ValueError("No valid target rows in training split for scaler fit.")
        self.target_scaler.fit(tgt_rows.values)
        self.meteo_col_indices = [
            self.feature_cols.index(c) for c in METEO_FEATURE_COLS if c in self.feature_cols
        ]
        return self

    def fit_multi_site(
        self, site_dfs: List[pd.DataFrame], train_end: int
    ) -> "FeatureEngineer":
        """Fit scalers on train portion of all sites (stacked rows)."""
        parts = []
        for df in site_dfs:
            parts.append(df.iloc[:train_end])
        stacked = pd.concat(parts, ignore_index=True)
        return self.fit(stacked)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_scaler is None or self.target_scaler is None:
            raise RuntimeError("Call fit() before transform()")
        feat = self.build_features(df)
        scaled = feat.copy()
        scaled[self.feature_cols] = self.feature_scaler.transform(
            feat[self.feature_cols].values
        )
        tgt = feat[self.target_cols].copy()
        valid = tgt.notna().all(axis=1)
        scaled_tgt = tgt.values.copy()
        if valid.any():
            scaled_tgt[valid.values] = self.target_scaler.transform(
                tgt.loc[valid].values
            )
        scaled_tgt[~valid.values] = 0.0
        scaled[self.target_cols] = scaled_tgt
        return scaled

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            raise RuntimeError("Scaler not fitted")
        shape = y_scaled.shape
        flat = y_scaled.reshape(-1, len(self.target_cols))
        inv = self.target_scaler.inverse_transform(flat)
        return inv.reshape(shape)


def chronological_split_indices(
    n: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return train_end, val_end


def feature_dim(feature_cols: Sequence[str] | None = None) -> int:
    return len(feature_cols or BASE_FEATURE_COLS)
