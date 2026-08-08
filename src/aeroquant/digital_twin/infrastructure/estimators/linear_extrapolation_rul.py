"""
LinearExtrapolationRULEstimator — baseline estatístico clássico para RUL:
ajusta uma regressão linear HI(cycle) sobre a janela recente de histórico,
projeta até o limiar de falha, e retorna intervalo de predição OLS padrão
(que cresce corretamente com a distância de extrapolação — propriedade
estatística real, não decorativa).

Este é DELIBERADAMENTE o baseline exigido pela Fase 12 (Validação
Científica: "comparação com baseline") — qualquer modelo de ML da Fase 6
precisa superar isto para justificar a complexidade adicional.

LIMITAÇÃO REAL ENCONTRADA E CORRIGIDA (Fase 6, ao comparar contra ML):
inversão de regressão (resolver x a partir de y) é numericamente instável
quando a inclinação estimada é pequena mas positiva — pequenos erros de
inclinação viram erros ENORMES no ciclo de falha projetado (dividir por
quase-zero). Sem limite, isso produzia RUL previsto de milhares de ciclos
para uma unidade com \~150 ciclos de vida, quebrando RMSE e até causando
overflow na métrica NASA (exp de um erro gigantesco). Corrigido com um
teto de extrapolação: nunca projetar além de `max_extrapolation_multiple`
vezes o espaço de ciclos observado na janela de ajuste — heurística padrão
para evitar extrapolação absurda em regressão, não um número arbitrário
novo (documentado e testado abaixo).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from aeroquant.digital_twin.domain.entities import DigitalTwinSnapshot
from aeroquant.sensor_data.domain.value_objects import RULEstimate

_UNINFORMATIVE_RUL = RULEstimate(point=200.0, lower=0.0, upper=400.0, confidence=0.90)


class LinearExtrapolationRULEstimator:
    def __init__(
        self,
        min_points: int = 5,
        window: int = 30,
        confidence: float = 0.90,
        max_extrapolation_multiple: float = 3.0,
    ) -> None:
        self._min_points = min_points
        self._window = window
        self._confidence = confidence
        self._max_extrapolation_multiple = max_extrapolation_multiple

    def estimate(self, history: list[DigitalTwinSnapshot], failure_threshold: float = 1.0) -> RULEstimate:
        if len(history) < self._min_points:
            return _UNINFORMATIVE_RUL

        recent = history[-self._window :]
        x = np.array([s.cycle for s in recent], dtype=float)
        y = np.array([s.health_index for s in recent], dtype=float)
        n = len(x)
        current_cycle = x[-1]
        # Teto de extrapolação baseado no total de ciclos JÁ OBSERVADOS da
        # unidade (current_cycle), não no span da janela local — assim o
        # teto cresce de forma monotônica conforme mais dados chegam,
        # coerente com a heurística "não prever vida remanescente muito
        # além do que já foi observado". Usar o span da janela local aqui
        # tinha um efeito colateral real: no início de vida a janela é
        # curta, o teto ficava mais apertado que no fim de vida, invertendo
        # o padrão esperado de incerteza encolhendo com mais dados (achado
        # ao rodar os testes de integração da Fase 5 após esta mudança).
        rul_cap = self._max_extrapolation_multiple * max(current_cycle, 1.0)

        slope, intercept, _, _, _ = scipy_stats.linregress(x, y)

        if slope <= 1e-8:
            # HI não está piorando de forma detectável -> não extrapolar agressivamente
            return RULEstimate(point=_UNINFORMATIVE_RUL.point, lower=0.0, upper=_UNINFORMATIVE_RUL.upper, confidence=self._confidence)

        x_failure = (failure_threshold - intercept) / slope
        rul_point = min(max(0.0, x_failure - current_cycle), rul_cap)

        y_pred = slope * x + intercept
        residual_std = float(np.sqrt(np.sum((y - y_pred) ** 2) / max(n - 2, 1)))
        x_mean = x.mean()
        sxx = float(np.sum((x - x_mean) ** 2)) or 1e-8

        se_pred = residual_std * np.sqrt(1 + 1 / n + (x_failure - x_mean) ** 2 / sxx)
        t_val = scipy_stats.t.ppf(0.5 + self._confidence / 2, df=max(n - 2, 1))
        # erro na estimativa de x_failure, propagado para o eixo de RUL (divide por slope),
        # também limitado ao teto de extrapolação — um intervalo não pode ser mais largo
        # que o próprio espaço de valores plausíveis de RUL.
        rul_half_width = min(float(t_val * se_pred / slope), rul_cap) if slope > 1e-8 else _UNINFORMATIVE_RUL.upper

        lower = max(0.0, rul_point - abs(rul_half_width))
        upper = min(rul_point + abs(rul_half_width), rul_cap)
        return RULEstimate(point=rul_point, lower=lower, upper=upper, confidence=self._confidence)