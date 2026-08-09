"""
ZScoreHealthIndexEstimator — Health Index como média ponderada dos
|z-scores| dos sensores mais acoplados à degradação (peso = coupling do
SensorSchema). HI cresce conforme os sensores se afastam do comportamento
"saudável" da frota (baseline online), sem exigir rótulo de falha para
funcionar — importante porque no mundo real não se sabe a priori quando
uma unidade vai falhar.

Incerteza do HI (v0.4.1+):
  - Se baseline_stats fornecer n (via stats_with_n), usa proxy 0.15 / sqrt(n_min).
  - Caso contrário (compatibilidade), mantém 0.15 fixo.
  Isso reduz a incerteza do HI conforme o baseline da frota acumula evidência,
  alinhado à hipótese H3 (decomposição aleatória/epistêmica) sem exigir
  um modelo bayesiano completo ainda.
"""
from __future__ import annotations

import math

from aeroquant.sensor_data.domain.value_objects import SensorSchema


class ZScoreHealthIndexEstimator:
    def __init__(self, schema: SensorSchema, coupling_threshold: float = 0.2) -> None:
        self._weights = {
            s.name: s.degradation_coupling
            for s in schema.sensors
            if s.degradation_coupling >= coupling_threshold
        }
        if not self._weights:
            raise ValueError(
                "Nenhum sensor com coupling >= coupling_threshold — verifique o schema"
            )

    def estimate(
        self,
        sensor_values: dict[str, float],
        baseline_stats: dict[str, tuple],
    ) -> tuple[float, float]:
        """
        baseline_stats pode ser:
          - {sensor: (mean, std)}           → incerteza fixa 0.15 (legado)
          - {sensor: (mean, std, n)}        → incerteza 0.15 / sqrt(n_min)
        """
        weighted_abs_z: list[float] = []
        weights: list[float] = []
        ns: list[int] = []

        for sensor, weight in self._weights.items():
            if sensor not in sensor_values or sensor not in baseline_stats:
                continue
            stats = baseline_stats[sensor]
            mean, std = stats[0], stats[1]
            n = int(stats[2]) if len(stats) >= 3 else None
            z = abs((sensor_values[sensor] - mean) / std)
            weighted_abs_z.append(z * weight)
            weights.append(weight)
            if n is not None:
                ns.append(n)

        if not weighted_abs_z:
            return 0.0, 1.0  # sem baseline suficiente ainda -> HI neutro, incerteza máxima

        health_index = sum(weighted_abs_z) / sum(weights)

        if ns:
            n_min = max(min(ns), 1)
            # Proxy de incerteza epistêmica do baseline: encolhe com evidência.
            # Floor em 0.02 evita colapso numérico; teto implícito em 0.15.
            uncertainty = min(0.15, 0.15 / math.sqrt(n_min))
            uncertainty = max(uncertainty, 0.02)
        else:
            uncertainty = 0.15

        return health_index, uncertainty
