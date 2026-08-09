"""Análise de resíduos para modelos de RUL."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import percentile_abs_error


@dataclass
class ResidualReport:
    residual: np.ndarray
    mean: float
    median: float
    std: float
    p90_abs: float
    p95_abs: float
    p99_abs: float
    bias_message: str
    has_positive_bias: bool
    has_negative_bias: bool

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Mean Residual": round(self.mean, 3),
                "Median Residual": round(self.median, 3),
                "Std Residual": round(self.std, 3),
                "P90 Abs Err": round(self.p90_abs, 3),
                "P95 Abs Err": round(self.p95_abs, 3),
                "P99 Abs Err": round(self.p99_abs, 3),
            }
        ])


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray, *, bias_threshold: float = 1.0) -> ResidualReport:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    mean_b = float(np.mean(residual))
    has_pos = mean_b > bias_threshold
    has_neg = mean_b < -bias_threshold
    if has_pos:
        msg = f"Positive bias detected: model tends to overestimate remaining useful life (mean error = {mean_b:.2f} cycles)."
    elif has_neg:
        msg = f"Negative bias detected: model tends to underestimate remaining useful life (mean error = {mean_b:.2f} cycles)."
    else:
        msg = f"No strong systematic bias detected (mean error = {mean_b:.2f} cycles)."
    return ResidualReport(
        residual=residual, mean=mean_b, median=float(np.median(residual)), std=float(np.std(residual)),
        p90_abs=percentile_abs_error(y_true, y_pred, 90),
        p95_abs=percentile_abs_error(y_true, y_pred, 95),
        p99_abs=percentile_abs_error(y_true, y_pred, 99),
        bias_message=msg, has_positive_bias=has_pos, has_negative_bias=has_neg,
    )
