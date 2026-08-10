"""Calibration: cobertura empírica de intervalos P10/P90."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    p10_coverage: float
    p90_coverage: float
    interval_coverage: float
    mean_width: float
    n: int
    message: str


def evaluate_interval_calibration(
    y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray,
) -> CalibrationReport:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(p10, dtype=float)
    hi = np.asarray(p90, dtype=float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[m], lo[m], hi[m]
    n = int(len(y))
    if n == 0:
        return CalibrationReport(0.0, 0.0, 0.0, 0.0, 0, "Sem amostras.")
    cov90 = float(np.mean(y <= hi))
    cov10 = float(np.mean(y >= lo))
    interval = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    msg = (
        f"Interval coverage={100*interval:.1f}% (alvo ~80% para P10–P90). "
        f"P90 coverage (y≤P90)={100*cov90:.1f}% (alvo ~90%). "
        f"Mean width={width:.1f} cycles."
    )
    return CalibrationReport(cov10, cov90, interval, width, n, msg)
