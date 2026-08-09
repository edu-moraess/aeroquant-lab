"""
Experimento de rede neural (MLP) para RUL — dashboard.

Treina MLPRegressor (sklearn) com early stopping e devolve métricas,
curva de perda e predições. Compara opcionalmente com Linear/RF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import evaluate
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    LinearRegressionTrainer,
    MLPTrainer,
    RandomForestTrainer,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
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


@dataclass
class NNExperimentResult:
    metrics_table: pd.DataFrame
    test_true: np.ndarray
    test_pred_mlp: np.ndarray
    loss_curve: list[float] | None
    n_epochs: int
    n_train_units: int
    n_test_units: int
    n_features: int
    trained_model: object | None
    X_test: pd.DataFrame | None
    feature_cols: list[str]
    architecture: str


def run_nn_experiment(
    n_units: int = 28,
    seed: int = 2026,
    noise_std: float = 0.015,
    hidden: tuple[int, ...] = (64, 32),
    max_iter: int = 250,
    alpha: float = 1e-4,
    compare_baselines: bool = True,
) -> NNExperimentResult:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    readings = []
    rng = np.random.default_rng(seed)
    for i in range(n_units):
        lifetime = int(np.clip(rng.normal(160, 25), 80, 250))
        unit = Unit(
            unit_id=f"nn-{i:03d}",
            fleet_id="nn-exp",
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
    X_train, y_train = train_df[feature_cols], train_df["rul"]
    X_test, y_test = test_df[feature_cols], test_df["rul"]

    mlp_trainer = MLPTrainer(hidden_layer_sizes=hidden, max_iter=max_iter, alpha=alpha, seed=seed)
    mlp_model = mlp_trainer.train(X_train, y_train)
    y_pred = mlp_trainer.predict(mlp_model, X_test)
    mlp_metrics = evaluate(y_test.to_numpy(), y_pred)

    rows = [
        {
            "modelo": "mlp",
            "RMSE": round(mlp_metrics.rmse, 2),
            "MAE": round(mlp_metrics.mae, 2),
            "NASA score": f"{mlp_metrics.nasa_score:.3e}",
            "n": mlp_metrics.n_samples,
        }
    ]

    if compare_baselines:
        for name, trainer in [
            ("linear_regression", LinearRegressionTrainer()),
            ("random_forest", RandomForestTrainer(n_estimators=60, max_depth=8, seed=seed)),
        ]:
            m = trainer.train(X_train, y_train)
            pred = trainer.predict(m, X_test)
            met = evaluate(y_test.to_numpy(), pred)
            rows.append(
                {
                    "modelo": name,
                    "RMSE": round(met.rmse, 2),
                    "MAE": round(met.mae, 2),
                    "NASA score": f"{met.nasa_score:.3e}",
                    "n": met.n_samples,
                }
            )

    metrics_table = pd.DataFrame(rows).sort_values("RMSE")

    loss_curve = None
    n_epochs = 0
    try:
        mlp = mlp_model.predictor.named_steps["mlp"]
        loss_curve = list(mlp.loss_curve_) if hasattr(mlp, "loss_curve_") else None
        n_epochs = int(getattr(mlp, "n_iter_", 0) or 0)
    except Exception:
        pass

    n_train = train_df["unit_id"].nunique() if "unit_id" in train_df.columns else 0
    n_test = test_df["unit_id"].nunique() if "unit_id" in test_df.columns else 0

    return NNExperimentResult(
        metrics_table=metrics_table,
        test_true=y_test.to_numpy(),
        test_pred_mlp=y_pred,
        loss_curve=loss_curve,
        n_epochs=n_epochs,
        n_train_units=int(n_train),
        n_test_units=int(n_test),
        n_features=len(feature_cols),
        trained_model=mlp_model,
        X_test=X_test,
        feature_cols=feature_cols,
        architecture=f"MLP{list(hidden)}",
    )
