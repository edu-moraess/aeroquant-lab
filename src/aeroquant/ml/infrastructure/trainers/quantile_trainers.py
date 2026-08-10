"""Quantile trainers: RF tree-distribution + GBM asymmetric quantiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from aeroquant.ml.domain.entities import TrainedModel


class RandomForestQuantileTrainer:
    def __init__(self, n_estimators: int = 200, max_depth: int | None = 12, seed: int = 42) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._seed = seed

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        model = RandomForestRegressor(
            n_estimators=self._n_estimators, max_depth=self._max_depth,
            random_state=self._seed, n_jobs=-1,
        )
        model.fit(X, y)
        return TrainedModel(
            name="random_forest_quantile", algorithm="RandomForestRegressor(tree-quantiles)",
            features_used=list(X.columns), predictor=model, supports_uncertainty=True,
        )

    def predict(self, model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
        return np.clip(model.predictor.predict(X[model.features_used]), 0, None)

    def predict_quantiles(self, model: TrainedModel, X: pd.DataFrame):
        Xf = X[model.features_used].to_numpy()
        tree_preds = np.stack([tree.predict(Xf) for tree in model.predictor.estimators_], axis=0)
        p10 = np.clip(np.percentile(tree_preds, 10, axis=0), 0, None)
        p50 = np.clip(np.percentile(tree_preds, 50, axis=0), 0, None)
        p90 = np.clip(np.percentile(tree_preds, 90, axis=0), 0, None)
        return p10, p50, p90

    def predict_interval(self, model: TrainedModel, X: pd.DataFrame):
        p10, _, p90 = self.predict_quantiles(model, X)
        return p10, p90


class AsymmetricQuantileGBMTrainer:
    """GBM 0.1/0.5/0.9 — quantis assimétricos para RUL."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 3, seed: int = 42) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._seed = seed

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        models = {}
        for alpha in (0.1, 0.5, 0.9):
            m = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, n_estimators=self._n_estimators,
                max_depth=self._max_depth, random_state=self._seed,
            )
            m.fit(X, y)
            models[alpha] = m
        return TrainedModel(
            name="asymmetric_gbm_quantile", algorithm="GBM quantile 0.1/0.5/0.9",
            features_used=list(X.columns), predictor=models, supports_uncertainty=True,
        )

    def predict(self, model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
        return np.clip(model.predictor[0.5].predict(X[model.features_used]), 0, None)

    def predict_quantiles(self, model: TrainedModel, X: pd.DataFrame):
        Xf = X[model.features_used]
        p10 = np.clip(model.predictor[0.1].predict(Xf), 0, None)
        p50 = np.clip(model.predictor[0.5].predict(Xf), 0, None)
        p90 = np.clip(model.predictor[0.9].predict(Xf), 0, None)
        return p10, p50, p90

    def predict_interval(self, model: TrainedModel, X: pd.DataFrame):
        p10, _, p90 = self.predict_quantiles(model, X)
        return np.minimum(p10, p90), np.maximum(p10, p90)
