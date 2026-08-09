"""
Métricas de avaliação para RUL.

NASA asymmetric scoring (PHM C-MAPSS):
    d = RUL_previsto - RUL_verdadeiro
    d < 0  (subestimou, conservador):  s = exp(-d/13) - 1
    d >= 0 (superestimou, otimista):  s = exp(d/10) - 1

Superestimar RUL atrasa manutenção — custo operacionalmente maior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.ml.domain.value_objects import RULMetrics


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Error = mean(pred - true). Positivo = superestima RUL."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    d_clipped = np.clip(d, -700, 700)
    scores = np.where(
        d_clipped < 0,
        np.exp(-d_clipped / 13.0) - 1.0,
        np.exp(d_clipped / 10.0) - 1.0,
    )
    return float(np.sum(scores))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def percentile_abs_error(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
    return float(np.percentile(err, q))


@dataclass
class ExtendedRULMetrics:
    rmse: float
    mae: float
    r2: float
    nasa_score: float
    bias: float
    p50_abs_error: float
    p90_abs_error: float
    p95_abs_error: float
    p99_abs_error: float
    n_samples: int
    underprediction_rate: float
    overprediction_rate: float
    interval_coverage_90: float | None = None

    def to_row(self, model: str) -> dict:
        return {
            "Model": model,
            "RMSE": round(self.rmse, 3),
            "MAE": round(self.mae, 3),
            "R²": round(self.r2, 4),
            "NASA Score": round(self.nasa_score, 2),
            "Bias": round(self.bias, 3),
            "P50 Abs Err": round(self.p50_abs_error, 3),
            "P90 Abs Err": round(self.p90_abs_error, 3),
            "P95 Abs Err": round(self.p95_abs_error, 3),
            "N": self.n_samples,
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> RULMetrics:
    coverage = (
        interval_coverage(y_true, lower, upper)
        if lower is not None and upper is not None
        else None
    )
    return RULMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        nasa_score=nasa_asymmetric_score(y_true, y_pred),
        n_samples=len(y_true),
        interval_coverage_90=coverage,
    )


def evaluate_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> ExtendedRULMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    coverage = (
        interval_coverage(y_true, lower, upper)
        if lower is not None and upper is not None
        else None
    )
    return ExtendedRULMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        nasa_score=nasa_asymmetric_score(y_true, y_pred),
        bias=bias(y_true, y_pred),
        p50_abs_error=percentile_abs_error(y_true, y_pred, 50),
        p90_abs_error=percentile_abs_error(y_true, y_pred, 90),
        p95_abs_error=percentile_abs_error(y_true, y_pred, 95),
        p99_abs_error=percentile_abs_error(y_true, y_pred, 99),
        n_samples=len(y_true),
        underprediction_rate=float(np.mean(d < 0)),
        overprediction_rate=float(np.mean(d > 0)),
        interval_coverage_90=coverage,
    )


RUL_BUCKETS = (
    ("RUL < 10", 0, 10),
    ("10 ≤ RUL < 30", 10, 30),
    ("30 ≤ RUL < 60", 30, 60),
    ("RUL ≥ 60", 60, float("inf")),
)


def evaluate_by_rul_bucket(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for label, lo, hi in RUL_BUCKETS:
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"Bucket": label, "N": 0, "RMSE": np.nan, "MAE": np.nan, "NASA Score": np.nan, "Bias": np.nan})
            continue
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "Bucket": label, "N": n,
            "RMSE": round(rmse(yt, yp), 3),
            "MAE": round(mae(yt, yp), 3),
            "NASA Score": round(nasa_asymmetric_score(yt, yp), 2),
            "Bias": round(bias(yt, yp), 3),
        })
    return pd.DataFrame(rows)
