"""Correção de bias positivo do RUL e Late Failure Risk.

Diagnóstico: mean error ≈ +3.7 ciclos → SUPERESTIMA RUL (false-safe).
ŷ_corr = ŷ - bias; Late Failure Risk = overestimation rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiasReport:
    mean_error: float
    median_error: float
    overestimation_rate: float
    underestimation_rate: float
    late_failure_risk: float
    bias_message: str


@dataclass(frozen=True)
class CorrectedPrediction:
    point: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    bias_applied: float
    method: str


def compute_bias_report(y_true: np.ndarray, y_pred: np.ndarray) -> BiasReport:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return BiasReport(0.0, 0.0, 0.0, 0.0, 0.0, "Sem resíduos válidos.")
    mean_e = float(np.mean(err))
    med_e = float(np.median(err))
    over = float(np.mean(err > 0))
    under = float(np.mean(err < 0))
    if mean_e > 1.0:
        msg = (
            f"POSITIVE BIAS: modelo SUPERESTIMA RUL em média {mean_e:.2f} ciclos "
            f"(overestimation rate={100*over:.0f}%). Late Failure Risk elevado."
        )
    elif mean_e < -1.0:
        msg = (
            f"NEGATIVE BIAS: modelo SUBESTIMA RUL em média {mean_e:.2f} ciclos "
            f"(safer side operacionalmente)."
        )
    else:
        msg = f"Bias residual moderado (mean error={mean_e:.2f} ciclos)."
    return BiasReport(
        mean_error=mean_e, median_error=med_e,
        overestimation_rate=over, underestimation_rate=under,
        late_failure_risk=over, bias_message=msg,
    )


def fit_bias_correction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.mean(err)) if len(err) else 0.0


def apply_bias_correction(
    y_pred: np.ndarray, bias: float, residual_std: float | None = None,
) -> CorrectedPrediction:
    point = np.clip(np.asarray(y_pred, dtype=float) - bias, 0.0, None)
    if residual_std is None or residual_std <= 0:
        return CorrectedPrediction(
            point=point, p10=point, p50=point, p90=point,
            bias_applied=bias, method="mean_bias_only",
        )
    z10, z90 = -1.2816, 1.2816
    p10 = np.clip(point + z10 * residual_std, 0.0, None)
    p90 = np.clip(point + z90 * residual_std, 0.0, None)
    return CorrectedPrediction(
        point=point, p10=p10, p50=point.copy(), p90=p90,
        bias_applied=bias, method="mean_bias + gaussian residual bands",
    )


def failure_region_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    bins: tuple[tuple[str, float, float], ...] = (
        ("Failure Region (RUL<10)", 0.0, 10.0),
        ("Critical (10≤RUL<30)", 10.0, 30.0),
        ("Degradation (30≤RUL<60)", 30.0, 60.0),
        ("Healthy (RUL≥60)", 60.0, 1e9),
    ),
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for name, lo, hi in bins:
        m = (y_true >= lo) & (y_true < hi)
        if not np.any(m):
            rows.append({"region": name, "n": 0, "mae": None, "bias": None, "over_rate": None})
            continue
        err = y_pred[m] - y_true[m]
        rows.append({
            "region": name, "n": int(m.sum()),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "over_rate": float(np.mean(err > 0)),
        })
    return rows
