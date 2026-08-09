"""Estimativa de incerteza para RUL (residual, ensemble, amostras)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionUncertainty:
    expected: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    method: str

    def summary(self) -> dict[str, float]:
        return {
            "Expected RUL": float(np.mean(self.expected)),
            "P10": float(np.mean(self.p10)),
            "P50": float(np.mean(self.p50)),
            "P90": float(np.mean(self.p90)),
        }


def residual_based_intervals(
    y_pred: np.ndarray, residual_std: float, *,
    z_p10: float = -1.2816, z_p90: float = 1.2816,
) -> PredictionUncertainty:
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = max(float(residual_std), 1e-6)
    return PredictionUncertainty(
        expected=y_pred,
        p10=np.clip(y_pred + z_p10 * sigma, 0, None),
        p50=y_pred.copy(),
        p90=np.clip(y_pred + z_p90 * sigma, 0, None),
        method=f"residual_gaussian(σ={sigma:.2f})",
    )


def quantiles_from_samples(samples: np.ndarray) -> PredictionUncertainty:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return PredictionUncertainty(
        expected=np.clip(np.mean(samples, axis=0), 0, None),
        p10=np.clip(np.percentile(samples, 10, axis=0), 0, None),
        p50=np.clip(np.percentile(samples, 50, axis=0), 0, None),
        p90=np.clip(np.percentile(samples, 90, axis=0), 0, None),
        method="empirical_samples",
    )


def rf_tree_quantiles(tree_preds: np.ndarray) -> PredictionUncertainty:
    return quantiles_from_samples(tree_preds)
