"""
LinearExtrapolationRULEstimator — baseline estatístico clássico para RUL:
ajusta uma regressão linear HI(cycle) sobre a janela recente de histórico,
projeta até o limiar de falha, e retorna intervalo de predição OLS padrão
(que cresce corretamente com a distância de extrapolação — propriedade
estatística real, não decorativa).

Este é DELIBERADAMENTE o baseline exigido pela Fase 12 (Validação
Científica: "comparação com baseline") — qualquer modelo de ML da Fase 6
precisa superar isto para justificar a complexidade adicional.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from aeroquant.digital_twin.domain.entities import DigitalTwinSnapshot
from aeroquant.sensor_data.domain.value_objects import RULEstimate

_UNINFORMATIVE_RUL = RULEstimate(point=200.0, lower=0.0, upper=400.0, confidence=0.90)


class LinearExtrapolationRULEstimator:
    def __init__(self, min_points: int = 5, window: int = 30, confidence: float = 0.90) -> None:
        self._min_points = min_points
        self._window = window
        self._confidence = confidence

    def estimate(self, history: list[DigitalTwinSnapshot], failure_threshold: float = 1.0) -> RULEstimate:
        if len(history) < self._min_points:
            return _UNINFORMATIVE_RUL

        recent = history[-self._window :]
        x = np.array([s.cycle for s in recent], dtype=float)
        y = np.array([s.health_index for s in recent], dtype=float)
        n = len(x)
        current_cycle = x[-1]

        slope, intercept, _, _, _ = scipy_stats.linregress(x, y)

        if slope <= 1e-8:
            # HI não está piorando de forma detectável -> não extrapolar agressivamente
            return RULEstimate(point=_UNINFORMATIVE_RUL.point, lower=0.0, upper=_UNINFORMATIVE_RUL.upper, confidence=self._confidence)

        x_failure = (failure_threshold - intercept) / slope
        rul_point = max(0.0, x_failure - current_cycle)

        y_pred = slope * x + intercept
        residual_std = float(np.sqrt(np.sum((y - y_pred) ** 2) / max(n - 2, 1)))
        x_mean = x.mean()
        sxx = float(np.sum((x - x_mean) ** 2)) or 1e-8

        se_pred = residual_std * np.sqrt(1 + 1 / n + (x_failure - x_mean) ** 2 / sxx)
        t_val = scipy_stats.t.ppf(0.5 + self._confidence / 2, df=max(n - 2, 1))
        # erro na estimativa de x_failure, propagado para o eixo de RUL (divide por slope)
        rul_half_width = float(t_val * se_pred / slope) if slope > 1e-8 else _UNINFORMATIVE_RUL.upper

        lower = max(0.0, rul_point - abs(rul_half_width))
        upper = rul_point + abs(rul_half_width)
        return RULEstimate(point=rul_point, lower=lower, upper=upper, confidence=self._confidence)
