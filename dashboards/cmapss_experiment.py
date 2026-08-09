"""Treino RUL com NASA C-MAPSS real (FD001–FD004)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import evaluate_by_rul_bucket, evaluate_extended
from aeroquant.ml.infrastructure.evaluation.ranking import RankingPolicy
from aeroquant.ml.infrastructure.evaluation.residuals import analyze_residuals
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    LinearRegressionTrainer, MLPTrainer, RandomForestTrainer,
)
from aeroquant.sensor_data.etl.pipeline import apply_normalize, engineer_features, fit_normalize_stats
from aeroquant.sensor_data.infrastructure.adapters.cmapss_adapter import CMAPSSAdapter
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema

_ROOT = Path(__file__).resolve().parent.parent
_EXTERNAL = _ROOT / "data" / "external"


@dataclass
class CMAPSSExperimentResult:
    subset: str
    metrics_table: pd.DataFrame
    ranked_table: pd.DataFrame
    best_model_name: str
    test_true: np.ndarray
    test_pred_best: np.ndarray
    residual_report: object
    bucket_table: pd.DataFrame
    n_train_units: int
    n_test_units: int
    n_train_rows: int
    n_test_rows: int
    n_features: int
    protocol_note: str
    data_source: str = "NASA C-MAPSS"
    feature_importance: object = None
    trained_model: object = None
    X_test: object = None


def cmapss_available(subset: str = "FD001") -> bool:
    return (
        (_EXTERNAL / f"train_{subset}.txt").exists()
        and (_EXTERNAL / f"test_{subset}.txt").exists()
        and (_EXTERNAL / f"RUL_{subset}.txt").exists()
    )


def list_available_subsets() -> list[str]:
    return [s for s in ("FD001", "FD002", "FD003", "FD004") if cmapss_available(s)]


def run_cmapss_experiment(
    subset: str = "FD001", max_rul_cap: int = 125, seed: int = 42, n_estimators: int = 100,
) -> CMAPSSExperimentResult:
    if not cmapss_available(subset):
        raise FileNotFoundError(
            f"Arquivos C-MAPSS {subset} não encontrados em {_EXTERNAL}. "
            "Execute: python scripts/download_cmapss.py"
        )

    schema = build_cmapss_like_schema()
    adapter = CMAPSSAdapter()
    train_df = adapter.to_dataframe(str(_EXTERNAL / f"train_{subset}.txt"), schema)
    test_df = adapter.to_dataframe(str(_EXTERNAL / f"test_{subset}.txt"), schema)
    train_df = adapter.attach_train_rul(train_df, max_rul_cap=max_rul_cap)
    test_df = adapter.attach_test_rul(test_df, str(_EXTERNAL / f"RUL_{subset}.txt"), max_rul_cap=max_rul_cap)

    stats = fit_normalize_stats(train_df, schema)
    train_df = engineer_features(apply_normalize(train_df, schema, stats), schema, window=5)
    test_df = engineer_features(apply_normalize(test_df, schema, stats), schema, window=5)

    feature_cols = [c for c in train_df.columns if c.endswith("_z") or "_roll_" in c or "_delta_" in c]
    feature_cols = [c for c in feature_cols if c in test_df.columns and train_df[c].std() > 1e-8]

    X_train, y_train = train_df[feature_cols], train_df["rul"]
    X_test, y_test = test_df[feature_cols], test_df["rul"]
    y_true = y_test.to_numpy(dtype=float)

    trainers = {
        "Linear Regression": LinearRegressionTrainer(),
        "Random Forest": RandomForestTrainer(n_estimators=n_estimators, max_depth=12, seed=seed),
        "MLP": MLPTrainer(hidden_layer_sizes=(64, 32), max_iter=200, seed=seed),
    }
    rows, preds = [], {}
    for name, trainer in trainers.items():
        model = trainer.train(X_train, y_train)
        pred = np.clip(trainer.predict(model, X_test), 0, None)
        preds[name] = pred
        rows.append(evaluate_extended(y_true, pred).to_row(name))

    metrics_table = pd.DataFrame(rows)
    ranked = RankingPolicy().rank(metrics_table)
    best = str(ranked.iloc[0]["Model"])
    y_pred = preds[best]

    return CMAPSSExperimentResult(
        subset=subset, metrics_table=metrics_table, ranked_table=ranked, best_model_name=best,
        test_true=y_true, test_pred_best=y_pred,
        residual_report=analyze_residuals(y_true, y_pred),
        bucket_table=evaluate_by_rul_bucket(y_true, y_pred),
        n_train_units=int(train_df["unit_id"].nunique()),
        n_test_units=int(test_df["unit_id"].nunique()),
        n_train_rows=len(train_df), n_test_rows=len(test_df), n_features=len(feature_cols),
        protocol_note=(
            f"NASA C-MAPSS {subset}. Train=run-to-failure; Test=parcial + RUL file. "
            "Normalize fit só no treino. Ranking NASA-first. "
            "C-MAPSS é simulação NASA (benchmark), não telemetria comercial."
        ),
    )
