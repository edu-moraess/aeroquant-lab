"""
Experimento sequencial (janelas temporais) para RUL.

Protocolo leakage-free:
- Split por unit_id ANTES do scaling
- fit_normalize_stats só no treino
- Target da janela = RUL do último ciclo (sem futuro)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import evaluate
from aeroquant.ml.infrastructure.sequences.windowing import SequenceDataset, build_sequence_windows
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit
from aeroquant.ml.infrastructure.trainers.sequence_trainers import (
    HAS_TORCH, LSTMTrainer, SequenceMLPTrainer, TransformerTrainer,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.etl.pipeline import (
    add_rul_labels, apply_normalize, clean, fit_normalize_stats, readings_to_dataframe,
)
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import StochasticSensorGenerator


@dataclass
class SeqExperimentResult:
    metrics_table: pd.DataFrame
    test_true: np.ndarray
    test_pred: np.ndarray
    loss_curve: list[float] | None
    n_epochs: int
    n_train_windows: int
    n_test_windows: int
    n_features: int
    seq_len: int
    algorithm: str
    model_name: str
    torch_available: bool


def run_seq_experiment(
    n_units: int = 24, seed: int = 2026, noise_std: float = 0.015,
    seq_len: int = 30, stride: int = 2, model: str = "sequence_mlp",
    hidden: tuple[int, ...] = (128, 64), max_iter: int = 200,
    lstm_hidden: int = 64, lstm_epochs: int = 30,
) -> SeqExperimentResult:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    readings = []
    rng = np.random.default_rng(seed)
    for i in range(n_units):
        lifetime = int(np.clip(rng.normal(160, 25), 90, 250))
        unit = Unit(
            unit_id=f"seq-{i:03d}", fleet_id="seq-exp", max_cycles=lifetime,
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
    df_scaled = pd.concat([train_df, test_df], ignore_index=True)

    feature_cols = [c for c in df_scaled.columns if c.endswith("_z")]
    if not feature_cols:
        feature_cols = [s.name for s in schema.sensors if s.name in df_scaled.columns]

    data = build_sequence_windows(df_scaled, feature_cols, seq_len=seq_len, stride=stride)
    train_units = set(train_df["unit_id"].astype(str))
    test_units = set(test_df["unit_id"].astype(str))
    tr_mask = np.array([u in train_units for u in data.unit_ids])
    te_mask = np.array([u in test_units for u in data.unit_ids])
    train = SequenceDataset(
        X=data.X[tr_mask], y=data.y[tr_mask], unit_ids=data.unit_ids[tr_mask],
        feature_names=data.feature_names, seq_len=data.seq_len,
    )
    test = SequenceDataset(
        X=data.X[te_mask], y=data.y[te_mask], unit_ids=data.unit_ids[te_mask],
        feature_names=data.feature_names, seq_len=data.seq_len,
    )

    if len(train.y) < 10 or len(test.y) < 5:
        raise RuntimeError("Janelas insuficientes — aumente unidades ou reduza seq_len.")

    if model == "lstm":
        result = LSTMTrainer(hidden=lstm_hidden, epochs=lstm_epochs, seed=seed).train_predict(
            train.X, train.y, test.X
        )
    elif model == "transformer":
        result = TransformerTrainer(epochs=lstm_epochs, seed=seed).train_predict(
            train.X, train.y, test.X
        )
    else:
        result = SequenceMLPTrainer(hidden_layer_sizes=hidden, max_iter=max_iter, seed=seed).train_predict(
            train.X, train.y, test.X
        )

    met = evaluate(test.y, result.y_pred)
    metrics_table = pd.DataFrame([{
        "modelo": result.name,
        "RMSE": round(met.rmse, 2),
        "MAE": round(met.mae, 2),
        "NASA score": f"{met.nasa_score:.3e}",
        "n": met.n_samples,
    }])

    return SeqExperimentResult(
        metrics_table=metrics_table, test_true=test.y, test_pred=result.y_pred,
        loss_curve=result.loss_curve, n_epochs=result.n_epochs,
        n_train_windows=len(train.y), n_test_windows=len(test.y),
        n_features=len(feature_cols), seq_len=seq_len,
        algorithm=result.algorithm, model_name=result.name, torch_available=HAS_TORCH,
    )
