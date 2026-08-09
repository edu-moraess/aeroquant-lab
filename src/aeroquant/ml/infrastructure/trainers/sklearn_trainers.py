"""Trainers sklearn: Linear, RF, GBM quantile, MLP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from aeroquant.ml.domain.entities import TrainedModel


class LinearRegressionTrainer:
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        model = LinearRegression()
        model.fit(X, y)
        return TrainedModel(name="linear_regression", algorithm="LinearRegression", features_used=list(X.columns), predictor=model)

    def predict(self, model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
        return np.clip(model.predictor.predict(X[model.features_used]), 0, None)

    def predict_interval(self, model: TrainedModel, X: pd.DataFrame) -> None:
        return None


class RandomForestTrainer:
    def __init__(self, n_estimators: int = 200, max_depth: int | None = 10, seed: int = 42) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._seed = seed

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        model = RandomForestRegressor(
            n_estimators=self._n_estimators, max_depth=self._max_depth, random_state=self._seed, n_jobs=-1
        )
        model.fit(X, y)
        return TrainedModel(
            name="random_forest", algorithm="RandomForestRegressor", features_used=list(X.columns),
            predictor=model, supports_uncertainty=True,
        )

    def predict(self, model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
        return np.clip(model.predictor.predict(X[model.features_used]), 0, None)

    def predict_interval(self, model: TrainedModel, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xf = X[model.features_used]
        tree_preds = np.stack([tree.predict(Xf.to_numpy()) for tree in model.predictor.estimators_], axis=0)
        lower = np.clip(np.percentile(tree_preds, 5, axis=0), 0, None)
        upper = np.clip(np.percentile(tree_preds, 95, axis=0), 0, None)
        return lower, upper


class GradientBoostingQuantileTrainer:
    def __init__(self, n_estimators: int = 200, max_depth: int = 3, seed: int = 42) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._seed = seed

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        models = {}
        for alpha in (0.05, 0.5, 0.95):
            m = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, n_estimators=self._n_estimators,
                max_depth=self._max_depth, random_state=self._seed,
            )
            m.fit(X, y)
            models[alpha] = m
        return TrainedModel(
            name="gradient_boosting_quantile", algorithm="GradientBoostingRegressor(quantile)",
            features_used=list(X.columns), predictor=models, supports_uncertainty=True,
        )

    def predict(self, model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
        return np.clip(model.predictor[0.5].predict(X[model.features_used]), 0, None)

    def predict_interval(self, model: TrainedModel, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xf = X[model.features_used]
        lower = np.clip(model.predictor[0.05].predict(Xf), 0, None)
        upper = np.clip(model.predictor[0.95].predict(Xf), 0, None)
        return np.minimum(lower, upper), np.maximum(lower, upper)


class MLPTrainer:
    def __init__(
        self, hidden_layer_sizes: tuple[int, ...] = (64, 32), max_iter: int = 250,
        alpha: float = 1e-4, seed: int = 42,
    ) -> None:
        self._hidden = hidden_layer_sizes
        self._max_iter = max_iter
        self._alpha = alpha
        self._seed = seed

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=self._hidden, max_iter=self._max_iter, alpha=self._alpha,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                random_state=self._seed, learning_rate_init=1e-3,
            )),
        ])
        pipe.fit(X, y)
        mlp = pipe.named_steps["mlp"]
        meta = {
            "loss_curve": list(getattr(mlp, "loss_curve_", []) or []),
            "validation_scores": list(getattr(mlp, "validation_scores_", []) or []),
            "n_iter": int(getattr(mlp, "n_iter_", 0) or 0),
            "best_loss": float(getattr(mlp, "best_loss_", float("nan"))),
            "n_layers": len(self._hidden),
            "n_params_approx": int(sum(
                a * b for a, b in zip([X.shape[1]] + list(self._hidden), list(self._hidden) + [1])
            )),
        }
        pipe._aeroquant_meta = meta  # type: ignore[attr-defined]
        return TrainedModel(
            name="mlp", algorithm=f"MLP{list(self._hidden)}", features_used=list(X.columns),
            predictor=pipe, supports_uncertainty=False,
        )

    def predict(self, model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
        return np.clip(model.predictor.predict(X[model.features_used]), 0, None)

    def predict_interval(self, model: TrainedModel, X: pd.DataFrame) -> None:
        return None
