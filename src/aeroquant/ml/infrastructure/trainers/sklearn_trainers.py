"""
Trainers concretos (infrastructure) implementando o port `ModelTrainer`.

Por que estes três algoritmos, nesta ordem de complexidade crescente
(exigência do master prompt: "sempre comparar diferentes algoritmos antes
da escolha final"):

1. **LinearRegression** — o baseline de ML mais simples possível. Se um
   modelo mais complexo não superar isto, a complexidade extra não se
   justifica. Serve de ponte entre o baseline estatístico da Fase 5
   (extrapolação de HI) e modelos de fato "aprendidos" de features.
2. **RandomForestRegressor** — captura não-linearidades e interações entre
   sensores sem exigir escalonamento cuidadoso nem grande volume de dados
   (importante aqui, onde só temos frota sintética + eventualmente uma
   frota pequena de C-MAPSS). A variância entre árvores também dá uma
   forma barata de incerteza epistêmica (usada em `predict_interval`).
3. **GradientBoostingRegressor com quantile loss** — treinado 3 vezes
   (alpha=0.05, 0.5, 0.95) para produzir um intervalo de predição real,
   não decorativo, comparável ao intervalo OLS do baseline da Fase 5.
   GBM com quantile loss foi escolhido em vez de um único modelo com
   dropout/ensemble bootstrap porque scikit-learn oferece isso nativamente
   e sem necessidade de tunar um ensemble manual.

Deep learning (LSTM/Transformer, mencionados na arquitetura da Fase 1)
FICOU DE FORA desta rodada — torch não está disponível neste container
sem rede. Ver `docs/roadmap.md` para isso ser retomado assim que houver
ambiente com acesso à internet.
"""
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
        # Incerteza epistêmica via dispersão entre as árvores do ensemble —
        # cada árvore prevê separadamente, e usamos os percentis 5/95 da
        # distribuição de previsões como intervalo (não é Bayesiano, mas é
        # uma incerteza real derivada do modelo, não inventada).
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
        Xf = X[model.features_used]
        return np.clip(model.predictor[0.5].predict(Xf), 0, None)

    def predict_interval(self, model: TrainedModel, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xf = X[model.features_used]
        lower = np.clip(model.predictor[0.05].predict(Xf), 0, None)
        upper = np.clip(model.predictor[0.95].predict(Xf), 0, None)
        # Quantile regression treinada independentemente por alpha pode,
        # raramente, produzir lower > upper numa amostra específica —
        # corrigido aqui em vez de deixar propagar como intervalo inválido.
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        return lower, upper