"""
Experimento ML tabular — protocolo leakage-free.

Split por unit_id; normalização fit só no treino; métricas estendidas + ranking NASA-first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import evaluate_extended
from aeroquant.ml.infrastructure.evaluation.ranking import RankingPolicy
from aeroquant.ml.infrastructure.evaluation.residuals import analyze_residuals
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    GradientBoostingQuantileTrainer,
    LinearRegressionTrainer,
    MLPTrainer,
    RandomForestTrainer,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.etl.pipeline import (
    add_rul_labels,
    apply_normalize,
    clean,
    engineer_features,
    fit_normalize_stats,
    readings_to_dataframe,
)
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)


@dataclass
class MLExperimentResult:
    metrics_table: pd.DataFrame
    ranked_table: pd.DataFrame
    test_true: np.ndarray
    test_pred_best: np.ndarray
    best_model_name: str
    feature_importance: pd.DataFrame | None
    n_train_units: int
    n_test_units: int
    n_features: int
    trained_model: object | None = None
    X_test: pd.DataFrame | None = None
    feature_cols: list | None = None
    residual_report: object | None = None
    protocol_note: str = ""


def run_ml_experiment(
    n_units: int = 40,
    seed: int = 2026,
    noise_std: float = 0.015,
    n_estimators: int = 80,
) -> MLExperimentResult:
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
            seed=seed + i, noise_std=noise_std, abrupt_fault_rate=0.005, abrupt_fault_magnitude=0.5,
        )
        readings.extend(generator.generate_unit(unit, schema, params))

    df = readings_to_dataframe(readings, schema)
    df = clean(df, schema)
    df = add_rul_labels(df, max_rul_cap=125)
    train_df, test_df = split_by_unit(df, test_fraction=0.3, seed=seed)

    stats = fit_normalize_stats(train_df, schema)
    train_df = apply_normalize(train_df, schema, stats)
    test_df = apply_normalize(test_df, schema, stats)
    train_df = engineer_features(train_df, schema, window=5)
    test_df = engineer_features(test_df, schema, window=5)

    feature_cols = [c for c in train_df.columns if c.endswith("_z") or "_roll_" in c or "_delta_" in c]
    feature_cols = [c for c in feature_cols if c in test_df.columns]

    X_train, y_train = train_df[feature_cols], train_df["rul"]
    X_test, y_test = test_df[feature_cols], test_df["rul"]
    y_true = y_test.to_numpy(dtype=float)

    trainers = {
        "Linear Regression": LinearRegressionTrainer(),
        "Random Forest": RandomForestTrainer(n_estimators=n_estimators, max_depth=8, seed=seed),
        "GBM Quantile": GradientBoostingQuantileTrainer(n_estimators=n_estimators, max_depth=3),
        "MLP": MLPTrainer(hidden_layer_sizes=(64, 32), max_iter=200, seed=seed),
    }

    rows, preds, models = [], {}, {}
    for name, trainer in trainers.items():
        model = trainer.train(X_train, y_train)
        pred = np.clip(trainer.predict(model, X_test), 0, None)
        preds[name] = pred
        models[name] = model
        rows.append(evaluate_extended(y_true, pred).to_row(name))

    metrics_table = pd.DataFrame(rows)
    ranked = RankingPolicy().rank(metrics_table)
    best_name = str(ranked.iloc[0]["Model"])
    y_pred = preds[best_name]
    model = models[best_name]

    importance = None
    if best_name == "Random Forest" and hasattr(model.predictor, "feature_importances_"):
        imp = model.predictor.feature_importances_
        importance = (
            pd.DataFrame({"feature": feature_cols, "importance": imp})
            .sort_values("importance", ascending=False).head(15)
        )
    elif best_name == "GBM Quantile":
        try:
            imp = model.predictor[0.5].feature_importances_
            importance = (
                pd.DataFrame({"feature": feature_cols, "importance": imp})
                .sort_values("importance", ascending=False).head(15)
            )
        except Exception:
            pass

    return MLExperimentResult(
        metrics_table=metrics_table,
        ranked_table=ranked,
        test_true=y_true,
        test_pred_best=y_pred,
        best_model_name=best_name,
        feature_importance=importance,
        n_train_units=int(train_df["unit_id"].nunique()),
        n_test_units=int(test_df["unit_id"].nunique()),
        n_features=len(feature_cols),
        trained_model=model,
        X_test=X_test.copy(),
        feature_cols=list(feature_cols),
        residual_report=analyze_residuals(y_true, y_pred),
        protocol_note="Split por unit_id (30% test). Normalização fit só no treino. Ranking: NASA + 0.5·RMSE + 0.25·|Bias|.",
    )
