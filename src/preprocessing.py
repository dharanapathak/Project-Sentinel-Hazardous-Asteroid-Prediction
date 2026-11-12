from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_ALIASES = ("hazardous", "is_hazardous", "potentially_hazardous")
EXCLUDE_COLUMNS = {
    "hazardous", "id", "neo_id", "name", "orbiting_body", "sentry_object",
    "designation", "full_name", "spkid", "pdes"
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    aliases = {
        "est_diameter_min": ("estimated_diameter_min", "estimated_diameter_min_km"),
        "est_diameter_max": ("estimated_diameter_max", "estimated_diameter_max_km"),
        "relative_velocity": ("relative_velocity_km_per_sec", "velocity"),
        "miss_distance": ("miss_distance_km", "miss_distance_au"),
        "absolute_magnitude": ("h", "magnitude"),
        "hazardous": ("is_hazardous", "potentially_hazardous"),
    }
    rename = {}
    for canonical, candidates in aliases.items():
        if canonical not in out.columns:
            for candidate in candidates:
                if candidate in out.columns:
                    rename[candidate] = canonical
                    break
    return out.rename(columns=rename)

def parse_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        y = series.astype("Int64")
    elif pd.api.types.is_numeric_dtype(series):
        y = pd.to_numeric(series, errors="coerce")
    else:
        y = series.astype(str).str.strip().str.lower().map({
            "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0,
            "hazardous": 1, "non-hazardous": 0, "non hazardous": 0
        })
    if y.isna().any():
        raise ValueError("Target column contains missing or unrecognized labels.")
    values = set(pd.Series(y).astype(int).unique())
    if not values.issubset({0, 1}) or len(values) != 2:
        raise ValueError("Target 'hazardous' must contain both binary classes 0 and 1.")
    return pd.Series(y.astype(int), index=series.index, name="hazardous")

def prepare_xy(raw: pd.DataFrame):
    if raw.empty:
        raise ValueError("The dataset is empty.")
    df = normalize_columns(raw).drop_duplicates().dropna(axis=1, how="all").copy()
    if "hazardous" not in df.columns:
        raise ValueError("Missing target column 'hazardous'.")
    y = parse_target(df["hazardous"])
    X = df.drop(columns=[c for c in df.columns if c in EXCLUDE_COLUMNS], errors="ignore").copy()
    for col in X.columns:
        if X[col].dtype == "object":
            numeric = pd.to_numeric(X[col], errors="coerce")
            if numeric.notna().mean() >= 0.8:
                X[col] = numeric
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after cleaning.")
    return X, y
