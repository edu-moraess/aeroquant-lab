"""
ZScoreHealthIndexEstimator — Health Index como média ponderada dos
|z-scores| dos sensores mais acoplados à degradação (peso = coupling do
SensorSchema). HI cresce conforme os sensores se afastam do comportamento
"saudável" da frota (baseline online), sem exigir rótulo de falha para
funcionar — importante porque no mundo real não se sabe a priori quando
uma unidade vai falhar.
"""
from __future__ import annotations

import math

from aeroquant.sensor_data.domain.value_objects import SensorSchema


class ZScoreHealthIndexEstimator:
    def __init__(self, schema: SensorSchema, coupling_threshold: float = 0.2) -> None:
        self._weights = {
            s.name: s.degradation_coupling for s in schema.sensors if s.degradation_coupling >= coupling_threshold
        }
        if not self._weights:
            raise ValueError("Nenhum sensor com coupling >= coupling_threshold — verifique o schema")

    def estimate(
        self, sensor_values: dict[str, float], baseline_stats: dict[str, tuple[float, float]]
    ) -> tuple[float, float]:
        weighted_abs_z: list[float] = []
        weights: list[float] = []
        n_min = math.inf

        for sensor, weight in self._weights.items():
            if sensor not in sensor_values or sensor not in baseline_stats:
                continue
            mean, std = baseline_stats[sensor]
            z = abs((sensor_values[sensor] - mean) / std)
            weighted_abs_z.append(z * weight)
            weights.append(weight)

        if not weighted_abs_z:
            return 0.0, 1.0  # sem baseline suficiente ainda -> HI neutro, incerteza máxima

        health_index = sum(weighted_abs_z) / sum(weights)
        # incerteza do HI: proxy simples baseado em quão pouco dado de baseline existe.
        # Como não temos aqui o n exato por sensor (só mean/std), usamos uma
        # incerteza fixa moderada — refinar isso é candidato natural para a
        # Fase 6 quando um modelo de verdade substituir este baseline.
        uncertainty = 0.15
        return health_index, uncertainty
