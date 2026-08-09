"""Incerteza residual/RF/GBM/MC + Risk assessment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import evaluate_extended
from aeroquant.ml.infrastructure.evaluation.uncertainty import (
    PredictionUncertainty, residual_based_intervals, rf_tree_quantiles,
)
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit_three_way
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    GradientBoostingQuantileTrainer, RandomForestTrainer,
)
from aeroquant.risk.assessment import RiskAssessment, assess_risk
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.etl.pipeline import (
    add_rul_labels, apply_normalize, clean, engineer_features,
    fit_normalize_stats, readings_to_dataframe,
)
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import StochasticSensorGenerator
from aeroquant.uncertainty.monte_carlo_rul import run_monte_carlo_rul


@dataclass
class UncertaintyRiskResult:
    method: str
    metrics_row: dict
    uncertainty_summary: dict
    risk: RiskAssessment
    y_true: np.ndarray
    y_pred: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    residual_std: float | None
    mc_result: object | None
    protocol_note: str


def run_uncertainty_risk_experiment(
    n_units: int = 28, seed: int = 2026, noise_std: float = 0.015,
    maintenance_threshold: float = 30.0, method: str = "random_forest",
) -> UncertaintyRiskResult:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    readings = []
    rng = np.random.default_rng(seed)
    for i in range(n_units):
        lifetime = int(np.clip(rng.normal(160, 25), 90, 250))
        unit = Unit(
            unit_id=f"ur-{i:03d}", fleet_id="unc-risk", max_cycles=lifetime,
            fault_mode=FaultMode.ABRUPT if i % 4 == 0 else FaultMode.GRADUAL,
        )
        params = DegradationParams(
            seed=seed + i, noise_std=noise_std, abrupt_fault_rate=0.005, abrupt_fault_magnitude=0.5,
        )
        readings.extend(generator.generate_unit(unit, schema, params))

    df = readings_to_dataframe(readings, schema)
    df = clean(df, schema)
    df = add_rul_labels(df, max_rul_cap=125)
    train_df, val_df, test_df = split_by_unit_three_way(df, seed=seed)
    stats = fit_normalize_stats(train_df, schema)
    train_df = engineer_features(apply_normalize(train_df, schema, stats), schema)
    val_df = engineer_features(apply_normalize(val_df, schema, stats), schema)
    test_df = engineer_features(apply_normalize(test_df, schema, stats), schema)

    feature_cols = [c for c in train_df.columns if c.endswith("_z") or "_roll_" in c or "_delta_" in c]
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    X_train, y_train = train_df[feature_cols], train_df["rul"]
    X_val, y_val = val_df[feature_cols], val_df["rul"]
    X_test, y_test = test_df[feature_cols], test_df["rul"]
    y_true = y_test.to_numpy(dtype=float)
    residual_std = None
    mc_result = None

    if method == "monte_carlo":
        mc_result = run_monte_carlo_rul(
            n_runs=24, base_seed=seed, max_cycles=160, noise_std=noise_std,
            reference_cycle_fraction=0.6,
        )
        samples = mc_result.rul_samples
        unc = PredictionUncertainty(
            expected=np.array([mc_result.mean]),
            p10=np.array([float(np.percentile(samples, 10))]),
            p50=np.array([mc_result.q50]),
            p90=np.array([float(np.percentile(samples, 90))]),
            method="monte_carlo_digital_twin",
        )
        y_pred = unc.expected
        y_true_pt = np.array([mc_result.true_rul_at_ref if mc_result.true_rul_at_ref else mc_result.mean])
        ext = evaluate_extended(y_true_pt, y_pred)
        risk = assess_risk(samples, maintenance_threshold=maintenance_threshold)
        return UncertaintyRiskResult(
            method=method, metrics_row=ext.to_row("Monte Carlo DT"),
            uncertainty_summary=unc.summary(), risk=risk,
            y_true=y_true_pt, y_pred=y_pred, p10=unc.p10, p50=unc.p50, p90=unc.p90,
            residual_std=None, mc_result=mc_result,
            protocol_note="Monte Carlo sobre Digital Twin; distribuição empírica de RUL.",
        )

    if method == "gbm_quantile":
        trainer = GradientBoostingQuantileTrainer(n_estimators=80, max_depth=3, seed=seed)
        model = trainer.train(X_train, y_train)
        y_pred = trainer.predict(model, X_test)
        lower, upper = trainer.predict_interval(model, X_test)
        p10 = lower + 0.25 * (y_pred - lower)
        p90 = y_pred + 0.75 * (upper - y_pred)
        unc = PredictionUncertainty(
            expected=y_pred, p10=np.clip(p10, 0, None), p50=y_pred, p90=np.clip(p90, 0, None),
            method="gbm_quantile_approx_p10_p90",
        )
    elif method == "random_forest":
        trainer = RandomForestTrainer(n_estimators=100, max_depth=10, seed=seed)
        model = trainer.train(X_train, y_train)
        y_pred = trainer.predict(model, X_test)
        Xf = X_test[model.features_used]
        tree_preds = np.stack([tree.predict(Xf.to_numpy()) for tree in model.predictor.estimators_], axis=0)
        unc = rf_tree_quantiles(tree_preds)
        unc.expected = y_pred
    else:
        trainer = RandomForestTrainer(n_estimators=80, max_depth=8, seed=seed)
        model = trainer.train(X_train, y_train)
        y_pred = trainer.predict(model, X_test)
        val_pred = trainer.predict(model, X_val)
        residual_std = float(np.std(val_pred - y_val.to_numpy(dtype=float)))
        unc = residual_based_intervals(y_pred, residual_std)

    ext = evaluate_extended(y_true, y_pred)
    risk = assess_risk(y_pred, maintenance_threshold=maintenance_threshold)
    return UncertaintyRiskResult(
        method=method, metrics_row=ext.to_row(method),
        uncertainty_summary=unc.summary(), risk=risk,
        y_true=y_true, y_pred=y_pred, p10=unc.p10, p50=unc.p50, p90=unc.p90,
        residual_std=residual_std, mc_result=mc_result,
        protocol_note=f"Split 3-way unit_id; normalize fit treino. Incerteza: {unc.method}.",
    )
