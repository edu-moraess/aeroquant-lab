"""
Métricas de avaliação para modelos de RUL. A métrica mais importante aqui
não é RMSE/MAE (simétricas) — é a NASA asymmetric scoring function, usada
no PHM Data Challenge original do C-MAPSS e replicada consistentemente na
literatura levantada na Fase 2: penaliza SUPERESTIMAR RUL mais que
subestimar, porque prever "sobra mais vida útil do que realmente sobra" é
o erro operacionalmente perigoso (atrasa manutenção além do seguro).

Fórmula (d = RUL_previsto - RUL_verdadeiro):
    d < 0 (previsão conservadora, subestimou):  s = exp(-d/13) - 1
    d >= 0 (previsão otimista, superestimou):   s = exp(d/10) - 1

O denominador menor no lado d>=0 (10 vs 13) é o que faz a penalidade
crescer mais rápido para superestimação — é essa assimetria que torna a
métrica cientificamente mais relevante que RMSE puro para este domínio.
"""
from __future__ import annotations

import numpy as np

from aeroquant.ml.domain.value_objects import RULMetrics


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    # Clipping de segurança: mesmo um erro genuinamente grande não deveria
    # estourar float64 em exp() e virar inf — um erro de 500 ciclos já é
    # catastrófico o suficiente para o score refletir isso sem precisar
    # calcular exp(50). Encontrado ao comparar contra o baseline da Fase 5
    # antes da correção do teto de extrapolação (ver linear_extrapolation_rul.py).
    d_clipped = np.clip(d, -700, 700)
    scores = np.where(d_clipped < 0, np.exp(-d_clipped / 13.0) - 1.0, np.exp(d_clipped / 10.0) - 1.0)
    return float(np.sum(scores))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> RULMetrics:
    coverage = interval_coverage(y_true, lower, upper) if lower is not None and upper is not None else None
    return RULMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        nasa_score=nasa_asymmetric_score(y_true, y_pred),
        n_samples=len(y_true),
        interval_coverage_90=coverage,
    )