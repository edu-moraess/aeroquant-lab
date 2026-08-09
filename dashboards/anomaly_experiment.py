"""
Experimento de detecção de anomalias para o dashboard.

Métodos:
- residual_zscore: z-score do HI (mesma família do Digital Twin)
- isolation_forest: Isolation Forest em sensores z-score (treino em início de vida)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.anomaly.infrastructure.detectors.isolation_forest import IsolationForestDetector
from aeroquant.anomaly.infrastructure.detectors.residual_zscore import ResidualZScoreDetector
from aeroquant.digital_twin.application.use_cases import UpdateDigitalTwin
from aeroquant.digital_twin.infrastructure.estimators.linear_extrapolation_rul import (
    LinearExtrapolationRULEstimator,
)
from aeroquant.digital_twin.infrastructure.estimators.threshold_calibration import (
    calibrate_failure_threshold,
)
from aeroquant.digital_twin.infrastructure.estimators.welford_fleet_baseline import (
    OnlineFleetBaseline,
)
from aeroquant.digital_twin.infrastructure.estimators.zscore_health_index import (
    ZScoreHealthIndexEstimator,
)
from aeroquant.digital_twin.infrastructure.repositories.in_memory_repository import (
    InMemoryDigitalTwinRepository,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)


@dataclass
class AnomalyExperimentResult:
    method: str
    n_units: int
    n_samples: int
    n_anomalies: int
    rate: float
    threshold: float
    timeline: pd.DataFrame
    by_unit: pd.DataFrame


def run_anomaly_experiment(
    n_units: int = 16,
    seed: int = 42,
    noise_std: float = 0.012,
    abrupt_rate: float = 0.005,
    method: str = "isolation_forest",
    z_threshold: float = 3.0,
    contamination: float = 0.05,
    coupling_threshold: float = 0.2,
) -> AnomalyExperimentResult:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling_threshold)
    failure_threshold = calibrate_failure_threshold(schema, OnlineFleetBaseline(), hi_estimator)

    rows = []
    fault_modes = {}
    for i in range(n_units):
        mode = FaultMode.ABRUPT if i % 3 == 0 else FaultMode.GRADUAL
        unit = Unit(
            unit_id=f"A{i:03d}",
            fleet_id="anomaly-exp",
            max_cycles=150,
            fault_mode=mode,
        )
        fault_modes[unit.unit_id] = mode.name
        params = DegradationParams(
            seed=seed + i,
            noise_std=noise_std,
            abrupt_fault_rate=abrupt_rate,
            abrupt_fault_magnitude=0.5,
        )
        readings = generator.generate_unit(unit, schema, params)
        dt = UpdateDigitalTwin(
            baseline_tracker=OnlineFleetBaseline(),
            hi_estimator=hi_estimator,
            rul_estimator=LinearExtrapolationRULEstimator(),
            repository=InMemoryDigitalTwinRepository(),
        )
        for r in readings:
            snap = dt.ingest(
                unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold
            )
            row = {
                "unit_id": unit.unit_id,
                "cycle": r.cycle,
                "health_index": snap.health_index,
                "dt_anomaly": snap.is_anomaly,
            }
            row.update({k: r.values[k] for k in r.values})
            rows.append(row)

    df = pd.DataFrame(rows)
    sensor_cols = [s.name for s in schema.sensors if s.name in df.columns]
    for col in sensor_cols:
        mu, sigma = df[col].mean(), df[col].std()
        sigma = sigma if sigma and sigma > 1e-8 else 1.0
        df[f"{col}_z"] = (df[col] - mu) / sigma
    z_cols = [f"{c}_z" for c in sensor_cols]

    if method == "residual_zscore":
        report = ResidualZScoreDetector(z_threshold=z_threshold).detect(df)
    else:
        report = IsolationForestDetector(contamination=contamination, seed=seed).detect(df, z_cols)

    points_df = pd.DataFrame(
        [
            {
                "unit_id": p.unit_id,
                "cycle": p.cycle,
                "score": p.score,
                "is_anomaly": p.is_anomaly,
            }
            for p in report.points
        ]
    )
    timeline = points_df.merge(
        df[["unit_id", "cycle", "health_index", "dt_anomaly"]],
        on=["unit_id", "cycle"],
        how="left",
    )

    by_unit = (
        timeline.groupby("unit_id")
        .agg(n_anomalies=("is_anomaly", "sum"), mean_score=("score", "mean"))
        .reset_index()
    )
    by_unit["fault_mode"] = by_unit["unit_id"].map(fault_modes)
    by_unit = by_unit.sort_values("n_anomalies", ascending=False)

    return AnomalyExperimentResult(
        method=report.method,
        n_units=n_units,
        n_samples=report.n_samples,
        n_anomalies=report.n_anomalies,
        rate=report.rate,
        threshold=report.threshold,
        timeline=timeline,
        by_unit=by_unit,
    )
