"""
Pipeline de ETL (Fase 3). Funções puras sobre DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aeroquant.sensor_data.domain.value_objects import SensorSchema


def clean(df: pd.DataFrame, schema: SensorSchema) -> pd.DataFrame:
    """Remove duplicatas e interpola nulos por unidade (nunca entre unidades)."""
    df = df.drop_duplicates(subset=["unit_id", "cycle"]).sort_values(["unit_id", "cycle"])
    sensor_cols = [s.name for s in schema.sensors if s.name in df.columns]
    df[sensor_cols] = df.groupby("unit_id")[sensor_cols].transform(lambda s: s.interpolate().ffill().bfill())
    return df


def normalize(df: pd.DataFrame, schema: SensorSchema) -> pd.DataFrame:
    """Z-score por sensor/condição no DataFrame inteiro.

    Preferir fit_normalize_stats + apply_normalize em pipelines de ML
    para evitar leakage train/test.
    """
    df = df.copy()
    sensor_cols = [s.name for s in schema.sensors if s.name in df.columns]
    for col in sensor_cols:
        grouped = df.groupby("operating_condition")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1.0)
        df[f"{col}_z"] = (df[col] - mean) / std
    return df


def engineer_features(df: pd.DataFrame, schema: SensorSchema, window: int = 5) -> pd.DataFrame:
    """Rolling mean/std por unidade + delta desde o primeiro ciclo (causal)."""
    df = df.copy()
    sensor_cols = [f"{s.name}_z" for s in schema.sensors if f"{s.name}_z" in df.columns]
    for col in sensor_cols:
        grouped = df.groupby("unit_id")[col]
        df[f"{col}_roll_mean{window}"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{col}_roll_std{window}"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
        first_value = grouped.transform("first")
        df[f"{col}_delta_first"] = df[col] - first_value
    return df


def add_rul_labels(df: pd.DataFrame, max_rul_cap: int = 125) -> pd.DataFrame:
    """RUL piecewise-linear com cap (padrão C-MAPSS)."""
    df = df.copy()
    true_rul = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    df["rul"] = true_rul.clip(upper=max_rul_cap)
    return df


def select_features(df: pd.DataFrame, feature_cols: list[str], min_abs_corr_with_rul: float = 0.05) -> list[str]:
    selected = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].std() < 1e-8:
            continue
        corr = df[col].corr(df["rul"])
        if pd.notna(corr) and abs(corr) >= min_abs_corr_with_rul:
            selected.append(col)
    return selected


def readings_to_dataframe(readings, schema: SensorSchema) -> pd.DataFrame:
    rows = []
    for r in readings:
        row = {"unit_id": r.unit_id, "cycle": r.cycle, "operating_condition": r.operating_condition}
        row.update(r.values)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_normalize_stats(df: pd.DataFrame, schema: SensorSchema) -> dict[str, dict[int, tuple[float, float]]]:
    """Estatísticas de z-score por (sensor, condition) — apenas TREINO."""
    stats: dict[str, dict[int, tuple[float, float]]] = {}
    sensor_cols = [s.name for s in schema.sensors if s.name in df.columns]
    for col in sensor_cols:
        stats[col] = {}
        for cond, g in df.groupby("operating_condition")[col]:
            mu = float(g.mean())
            sigma = float(g.std()) if len(g) > 1 else 1.0
            if sigma < 1e-8 or not np.isfinite(sigma):
                sigma = 1.0
            stats[col][int(cond)] = (mu, sigma)
    return stats


def apply_normalize(
    df: pd.DataFrame,
    schema: SensorSchema,
    stats: dict[str, dict[int, tuple[float, float]]],
) -> pd.DataFrame:
    """Aplica z-score com estatísticas fixas do treino."""
    df = df.copy()
    sensor_cols = [s.name for s in schema.sensors if s.name in df.columns]
    for col in sensor_cols:
        if col not in stats:
            continue
        means = df["operating_condition"].map(lambda c, _col=col: stats[_col].get(int(c), (0.0, 1.0))[0])
        stds = df["operating_condition"].map(lambda c, _col=col: stats[_col].get(int(c), (0.0, 1.0))[1])
        stds = stds.replace(0, 1.0)
        df[f"{col}_z"] = (df[col] - means) / stds
    return df
