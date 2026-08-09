"""
Experimento leve de comparação ML (Fase 6) para o dashboard Streamlit.

Gera frota sintética pequena, treina os 3 modelos scikit-learn já existentes
e devolve métricas + predições — cacheável por seed. Não substitui
scripts/demo_ml_vs_baseline.py (comparação formal streaming); serve para
exploração interativa no dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.ml.application.use_cases import TrainAndCompareModels
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    GradientBoostingQuantileTrainer,
    LinearRegressionTrainer,
    RandomForestTrainer,
)
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.etl.pipeline import (
    add_rul_labels,
    clean,
    engineer_features,
    normalize,
    readings_to_dataframe,
)
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit


@dataclass
class MLExperimentResult:
    metrics_table: pd.DataFrame
    test_true: np.ndarray
    test_pred_best: np.ndarray
    best_model_name: str
    feature_importance: pd.DataFrame | None
    n_train_units: int
    n_test_units: int
    n_features: int


def run_ml_experiment(
    n_units: int = 40,
    seed: int = 2026,
    noise_std: float = 0.015,
    n_estimators: int = 80,
) -> MLExperimentResult:
    """Treina e avalia os 3 modelos da Fase 6 em frota sintética."""
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()

    readings = []
    rng = np.random.default_rng(seed)
    for i in range(n_units):
        lifetime = int(np.clip(rng.normal(160, 25), 80, 250))
        unit = Unit(
            unit_id=f"ml-{i:03d}",
            fleet_id="ml-exp",
            max_cycles=lifetime,
            fault_mode=FaultMode.ABRUPT if i % 4 == 0 else FaultMode.GRADUAL,
        )
        params = DegradationParams(
            seed=seed + i,
            noise_std=noise_std,
            abrupt_fault_rate=0.005,
            abrupt_fault_magnitude=0.5,
        )
        readings.extend(generator.generate_unit(unit, schema, params))

    df = readings_to_dataframe(readings, schema)
    df = clean(df, schema)
    df = normalize(df, schema)
    df = engineer_features(df, schema, window=5)
    df = add_rul_labels(df, max_rul_cap=125)

    feature_cols = [
        c for c in df.columns if c.endswith("_z") or "_roll_" in c or "_delta_" in c
    ]
    train_df, test_df = split_by_unit(df, test_fraction=0.3, seed=seed)

    trainers = {
        "linear_regression": LinearRegressionTrainer(),
        "random_forest": RandomForestTrainer(n_estimators=n_estimators, max_depth=8),
        "gradient_boosting_quantile": GradientBoostingQuantileTrainer(
            n_estimators=n_estimators, max_depth=3
        ),
    }
    result = TrainAndCompareModels(trainers).run(train_df, test_df, feature_cols)

    rows = []
    for name, m in result.results.items():
        rows.append(
            {
                "modelo": name,
                "RMSE": round(m.rmse, 2),
                "MAE": round(m.mae, 2),
                "NASA score": f"{m.nasa_score:.3e}",
                "n": m.n_samples,
                "cobertura 90%": (
                    f"{m.interval_coverage_90:.1%}"
                    if m.interval_coverage_90 is not None
                    else "—"
                ),
            }
        )
    metrics_table = pd.DataFrame(rows).sort_values("RMSE")

    best_name = metrics_table.iloc[0]["modelo"]
    best_trainer = trainers[best_name]
    X_train, y_train = train_df[feature_cols], train_df["rul"]
    X_test, y_test = test_df[feature_cols], test_df["rul"]
    model = best_trainer.train(X_train, y_train)
    y_pred = best_trainer.predict(model, X_test)

    importance = None
    if best_name == "random_forest":
        imp = model.predictor.feature_importances_
        importance = (
            pd.DataFrame({"feature": feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(15)
        )
    elif best_name == "gradient_boosting_quantile":
        imp = model.predictor[0.5].feature_importances_
        importance = (
            pd.DataFrame({"feature": feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(15)
        )

    return MLExperimentResult(
        metrics_table=metrics_table,
        test_true=y_test.to_numpy(),
        test_pred_best=y_pred,
        best_model_name=best_name,
        feature_importance=importance,
        n_train_units=int(train_df["unit_id"].nunique()),
        n_test_units=int(test_df["unit_id"].nunique()),
        n_features=len(feature_cols),
    )
