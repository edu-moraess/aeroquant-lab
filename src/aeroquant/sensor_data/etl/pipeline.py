"""
Pipeline de ETL (Fase 3). Funções puras sobre DataFrame — deliberadamente
sem estado e sem I/O, para serem testáveis isoladamente e reaproveitáveis
tanto para dados sintéticos quanto para C-MAPSS real (mesmo SensorSchema).
"""
from __future__ import annotations

import pandas as pd

from aeroquant.sensor_data.domain.value_objects import SensorSchema


def clean(df: pd.DataFrame, schema: SensorSchema) -> pd.DataFrame:
    """Remove duplicatas e interpola nulos por unidade (nunca entre unidades)."""
    df = df.drop_duplicates(subset=["unit_id", "cycle"]).sort_values(["unit_id", "cycle"])
    sensor_cols = [s.name for s in schema.sensors if s.name in df.columns]
    df[sensor_cols] = df.groupby("unit_id")[sensor_cols].transform(lambda s: s.interpolate().ffill().bfill())
    return df


def normalize(df: pd.DataFrame, schema: SensorSchema) -> pd.DataFrame:
    """Z-score por sensor, calculado por condição operacional.

    Justificativa: em subconjuntos com múltiplos regimes operacionais
    (ex.: FD002/FD004 do C-MAPSS, 6 condições), normalizar globalmente
    mistura a variância natural entre regimes com a variância devida à
    degradação — o sinal de interesse. Normalizar por condição isola
    melhor o efeito de degradação.
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
    """Rolling mean/std por unidade + delta em relação ao primeiro ciclo.

    Estas duas famílias de atributos são as mais usadas na literatura de
    RUL sobre C-MAPSS para capturar tendência (rolling stats) e desvio
    acumulado desde o início de vida (delta), complementando o valor
    instantâneo do sensor.
    """
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
    """RUL piecewise-linear: constante em `max_rul_cap` no início de vida,
    depois decresce linearmente. Escolha padrão na literatura de C-MAPSS
    (usada, por exemplo, em modelos CAELSTM e DAB-LSTM revisados na Fase 2)
    porque a degradação real só se torna mensurável perto do fim de vida —
    antes disso, tratar RUL como decrescente linear desde o ciclo 1 introduz
    um sinal de treino que os sensores simplesmente não carregam ainda.
    """
    df = df.copy()
    true_rul = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    df["rul"] = true_rul.clip(upper=max_rul_cap)
    return df


def select_features(df: pd.DataFrame, feature_cols: list[str], min_abs_corr_with_rul: float = 0.05) -> list[str]:
    """Seleção univariada simples: descarta features quase sem correlação
    com RUL ou quase constantes (variância ~0). Um passo de seleção mais
    sofisticado (ex.: mutual information, VIF para colinearidade) fica
    para quando houver dados reais suficientes para justificar o custo.
    """
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
    """Converte list[SensorReading] (domínio) para DataFrame (fronteira ETL)."""
    rows = []
    for r in readings:
        row = {"unit_id": r.unit_id, "cycle": r.cycle, "operating_condition": r.operating_condition}
        row.update(r.values)
        rows.append(row)
    return pd.DataFrame(rows)
