"""Detector estatístico: z-score do incremento de Health Index (alinhado ao Digital Twin)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aeroquant.anomaly.domain.entities import AnomalyPoint, AnomalyReport


class ResidualZScoreDetector:
    def __init__(self, z_threshold: float = 3.0, min_history: int = 6) -> None:
        self._z_threshold = z_threshold
        self._min_history = min_history

    def detect(self, df: pd.DataFrame) -> AnomalyReport:
        """Espera colunas: unit_id, cycle, health_index."""
        points: list[AnomalyPoint] = []
        for unit_id, g in df.groupby("unit_id", sort=False):
            g = g.sort_values("cycle")
            his = g["health_index"].to_numpy(dtype=float)
            cycles = g["cycle"].to_numpy()
            for i in range(len(his)):
                if i < self._min_history:
                    points.append(
                        AnomalyPoint(
                            unit_id=str(unit_id),
                            cycle=int(cycles[i]),
                            score=0.0,
                            is_anomaly=False,
                            method="residual_zscore",
                        )
                    )
                    continue
                window = his[i - self._min_history : i]
                mu, sigma = float(window.mean()), float(window.std())
                sigma = max(sigma, 1e-8)
                z = abs((his[i] - mu) / sigma)
                is_anom = z > self._z_threshold
                points.append(
                    AnomalyPoint(
                        unit_id=str(unit_id),
                        cycle=int(cycles[i]),
                        score=float(z),
                        is_anomaly=is_anom,
                        method="residual_zscore",
                        reason=f"z={z:.2f}" if is_anom else None,
                    )
                )
        n_anom = sum(1 for p in points if p.is_anomaly)
        return AnomalyReport(
            method="residual_zscore",
            n_samples=len(points),
            n_anomalies=n_anom,
            threshold=self._z_threshold,
            points=points,
        )
