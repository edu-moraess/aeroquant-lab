"""
OnlineFleetBaseline — implementa FleetBaselineTracker usando o algoritmo de
Welford para média/variância incrementais.

Por que Welford e não recomputar mean/std do zero a cada leitura: é O(1)
por atualização (vs. O(n) recomputando sobre todo o histórico), numericamente
estável, e é literalmente "aprendizado online" no sentido exigido pela
Fase 5 — o estado (mean, M2, n) é atualizado incrementalmente conforme
novas leituras chegam, nunca recomputado em lote.
"""
from __future__ import annotations

from collections import defaultdict


class OnlineFleetBaseline:
    def __init__(self) -> None:
        # chave: (operating_condition, sensor_name) -> (n, mean, M2)
        self._state: dict[tuple[int, str], tuple[int, float, float]] = defaultdict(
            lambda: (0, 0.0, 0.0)
        )

    def update(self, operating_condition: int, sensor_values: dict[str, float]) -> None:
        for sensor, value in sensor_values.items():
            key = (operating_condition, sensor)
            n, mean, m2 = self._state[key]
            n += 1
            delta = value - mean
            mean += delta / n
            delta2 = value - mean
            m2 += delta * delta2
            self._state[key] = (n, mean, m2)

    def stats(self, operating_condition: int) -> dict[str, tuple[float, float]]:
        """Retorna {sensor: (mean, std)} para compatibilidade com a interface atual."""
        out: dict[str, tuple[float, float]] = {}
        for (cond, sensor), (n, mean, m2) in self._state.items():
            if cond != operating_condition:
                continue
            variance = m2 / n if n > 1 else 0.0
            std = variance**0.5 if variance > 0 else 1e-6  # evita divisão por zero no z-score
            out[sensor] = (mean, std)
        return out

    def stats_with_n(self, operating_condition: int) -> dict[str, tuple[float, float, int]]:
        """Retorna {sensor: (mean, std, n)} — n permite quantificar incerteza do baseline."""
        out: dict[str, tuple[float, float, int]] = {}
        for (cond, sensor), (n, mean, m2) in self._state.items():
            if cond != operating_condition:
                continue
            variance = m2 / n if n > 1 else 0.0
            std = variance**0.5 if variance > 0 else 1e-6
            out[sensor] = (mean, std, n)
        return out
