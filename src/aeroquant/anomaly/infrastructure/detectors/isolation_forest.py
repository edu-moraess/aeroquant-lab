"""Isolation Forest sobre features de sensores (treino em janela saudável inicial)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from aeroquant.anomaly.domain.entities import AnomalyPoint, AnomalyReport


class IsolationForestDetector:
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        healthy_cycles: int = 30,
        seed: int = 42,
    ) -> None:
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._healthy_cycles = healthy_cycles
        self._seed = seed

    def detect(self, df: pd.DataFrame, feature_cols: list[str]) -> AnomalyReport:
        cols = [c for c in feature_cols if c in df.columns]
        if not cols:
            raise ValueError("Nenhuma feature válida para Isolation Forest.")

        train_parts = []
        for _, g in df.groupby("unit_id", sort=False):
            g = g.sort_values("cycle")
            train_parts.append(g.head(self._healthy_cycles))
        train_df = pd.concat(train_parts, ignore_index=True)
        X_train = np.nan_to_num(train_df[cols].to_numpy(dtype=float), nan=0.0)

        model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._seed,
            n_jobs=-1,
        )
        model.fit(X_train)

        X_all = np.nan_to_num(df[cols].to_numpy(dtype=float), nan=0.0)
        raw = model.decision_function(X_all)
        pred = model.predict(X_all)
        scores = -raw
        threshold = float(np.percentile(scores, 100 * (1 - self._contamination)))

        points: list[AnomalyPoint] = []
        for i, row in enumerate(df.itertuples(index=False)):
            is_anom = bool(pred[i] == -1)
            points.append(
                AnomalyPoint(
                    unit_id=str(getattr(row, "unit_id")),
                    cycle=int(getattr(row, "cycle")),
                    score=float(scores[i]),
                    is_anomaly=is_anom,
                    method="isolation_forest",
                    reason=f"score={scores[i]:.3f}" if is_anom else None,
                )
            )

        n_anom = sum(1 for p in points if p.is_anomaly)
        return AnomalyReport(
            method="isolation_forest",
            n_samples=len(points),
            n_anomalies=n_anom,
            threshold=threshold,
            points=points,
        )
