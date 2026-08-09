"""
Benchmark unificado — mesmo protocolo experimental para todos os modelos.

1. Frota sintética com seed fixo
2. Clean + RUL labels
3. Split 3-way por unit_id (sem overlap)
4. fit_normalize no TREINO; apply em val/test
5. Feature engineering causal
6. Avalia Linear, RF, MLP, SequenceMLP (+ LSTM opcional)
7. Ranking NASA-first
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import evaluate_by_rul_bucket, evaluate_extended
from aeroquant.ml.infrastructure.evaluation.ranking import RankingPolicy
from aeroquant.ml.infrastructure.evaluation.residuals import analyze_residuals
from aeroquant.ml.infrastructure.sequences.windowing import build_sequence_windows
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit_three_way
from aeroquant.ml.infrastructure.trainers.sequence_trainers import HAS_TORCH, LSTMTrainer, SequenceMLPTrainer
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    LinearRegressionTrainer, MLPTrainer, RandomForestTrainer,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.etl.pipeline import (
    add_rul_labels, apply_normalize, clean, engineer_features,
    fit_normalize_stats, readings_to_dataframe,
)
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import StochasticSensorGenerator


@dataclass
class BenchmarkResult:
    experiment_id: str
    timestamp: str
    seed: int
    n_units: int
    n_train_units: int
    n_val_units: int
    n_test_units: int
    n_features: int
    seq_len: int
    metrics_table: pd.DataFrame
    ranked_table: pd.DataFrame
    best_model: str
    predictions: dict
    residual_reports: dict
    bucket_tables: dict
    protocol_note: str


def _generate_fleet(n_units: int, seed: int, noise_std: float):
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    readings = []
    rng = np.random.default_rng(seed)
    for i in range(n_units):
        lifetime = int(np.clip(rng.normal(160, 25), 90, 250))
        unit = Unit(
            unit_id=f"bm-{i:03d}", fleet_id="benchmark", max_cycles=lifetime,
            fault_mode=FaultMode.ABRUPT if i % 4 == 0 else FaultMode.GRADUAL,
        )
        params = DegradationParams(
            seed=seed + i, noise_std=noise_std, abrupt_fault_rate=0.005, abrupt_fault_magnitude=0.5,
        )
        readings.extend(generator.generate_unit(unit, schema, params))
    df = readings_to_dataframe(readings, schema)
    df = clean(df, schema)
    df = add_rul_labels(df, max_rul_cap=125)
    return df, schema


def run_benchmark(
    n_units: int = 30, seed: int = 2026, noise_std: float = 0.015,
    seq_len: int = 30, include_lstm: bool = False, ranking: RankingPolicy | None = None,
) -> BenchmarkResult:
    ranking = ranking or RankingPolicy()
    df, schema = _generate_fleet(n_units, seed, noise_std)
    train_df, val_df, test_df = split_by_unit_three_way(
        df, val_fraction=0.15, test_fraction=0.20, seed=seed,
    )
    stats = fit_normalize_stats(train_df, schema)
    train_df = apply_normalize(train_df, schema, stats)
    val_df = apply_normalize(val_df, schema, stats)
    test_df = apply_normalize(test_df, schema, stats)
    train_df = engineer_features(train_df, schema, window=5)
    val_df = engineer_features(val_df, schema, window=5)
    test_df = engineer_features(test_df, schema, window=5)

    feature_cols = [c for c in train_df.columns if c.endswith("_z") or "_roll_" in c or "_delta_" in c]
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    X_train, y_train = train_df[feature_cols], train_df["rul"]
    X_test, y_test = test_df[feature_cols], test_df["rul"]
    y_true = y_test.to_numpy(dtype=float)

    predictions, rows, residual_reports, bucket_tables = {}, [], {}, {}
    tabular = [
        ("Linear Regression", LinearRegressionTrainer()),
        ("Random Forest", RandomForestTrainer(n_estimators=80, max_depth=10, seed=seed)),
        ("MLP", MLPTrainer(hidden_layer_sizes=(64, 32), max_iter=200, seed=seed)),
    ]
    for name, trainer in tabular:
        model = trainer.train(X_train, y_train)
        pred = np.clip(trainer.predict(model, X_test), 0, None)
        predictions[name] = (y_true, pred)
        rows.append(evaluate_extended(y_true, pred).to_row(name))
        residual_reports[name] = analyze_residuals(y_true, pred)
        bucket_tables[name] = evaluate_by_rul_bucket(y_true, pred)

    train_units = set(train_df["unit_id"].unique())
    test_units = set(test_df["unit_id"].unique())
    z_cols = [c for c in train_df.columns if c.endswith("_z") and "_roll_" not in c and "_delta_" not in c]
    if not z_cols:
        z_cols = feature_cols[: min(10, len(feature_cols))]
    full_norm = pd.concat([train_df, val_df, test_df], ignore_index=True)
    seq_all = build_sequence_windows(full_norm, z_cols, seq_len=seq_len, stride=2)
    tr_mask = np.array([u in train_units for u in seq_all.unit_ids])
    te_mask = np.array([u in test_units for u in seq_all.unit_ids])
    if tr_mask.sum() >= 10 and te_mask.sum() >= 5:
        seq_result = SequenceMLPTrainer(hidden_layer_sizes=(128, 64), max_iter=150, seed=seed).train_predict(
            seq_all.X[tr_mask], seq_all.y[tr_mask], seq_all.X[te_mask]
        )
        pred, yt = seq_result.y_pred, seq_all.y[te_mask]
        predictions["Sequence MLP"] = (yt, pred)
        rows.append(evaluate_extended(yt, pred).to_row("Sequence MLP"))
        residual_reports["Sequence MLP"] = analyze_residuals(yt, pred)
        bucket_tables["Sequence MLP"] = evaluate_by_rul_bucket(yt, pred)
        if include_lstm and HAS_TORCH:
            lstm_result = LSTMTrainer(hidden=64, epochs=20, seed=seed).train_predict(
                seq_all.X[tr_mask], seq_all.y[tr_mask], seq_all.X[te_mask]
            )
            pred = lstm_result.y_pred
            predictions["LSTM"] = (yt, pred)
            rows.append(evaluate_extended(yt, pred).to_row("LSTM"))
            residual_reports["LSTM"] = analyze_residuals(yt, pred)
            bucket_tables["LSTM"] = evaluate_by_rul_bucket(yt, pred)

    metrics_table = pd.DataFrame(rows)
    ranked = ranking.rank(metrics_table)
    best = str(ranked.iloc[0]["Model"]) if len(ranked) else "n/a"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    protocol = (
        "Split por unit_id (3-way train/val/test). Normalização fit só no treino. "
        "Sequence: target = RUL do último ciclo. Ranking: NASA + 0.5·RMSE + 0.25·|Bias|. "
        "N de amostras pode diferir entre tabular (ciclos) e sequencial (janelas)."
    )
    return BenchmarkResult(
        experiment_id=f"bm-{seed}-{ts}", timestamp=ts, seed=seed, n_units=n_units,
        n_train_units=int(train_df["unit_id"].nunique()),
        n_val_units=int(val_df["unit_id"].nunique()),
        n_test_units=int(test_df["unit_id"].nunique()),
        n_features=len(feature_cols), seq_len=seq_len,
        metrics_table=metrics_table, ranked_table=ranked, best_model=best,
        predictions=predictions, residual_reports=residual_reports,
        bucket_tables=bucket_tables, protocol_note=protocol,
    )
