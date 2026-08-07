"""
calibrate_failure_threshold — resolve um problema real encontrado nesta
sessão: usar failure_threshold=1.0 "chutado" fazia a extrapolação de RUL
já nascer perto do fim de vida, porque o Health Index (média ponderada de
|z-scores|) não tem nenhuma razão estatística para ficar limitado a
[0, 1] (um único sensor com |z|>1 já acontece ~32% das vezes por acaso).

A prática correta em PHM é calibrar o limiar de falha empiricamente a
partir de uma frota histórica de referência (run-to-failure) — exatamente
o que esta função faz: roda o MESMO estimador de HI, com o MESMO baseline
compartilhado, sobre várias unidades sintéticas até o fim de vida, e
calibra o limiar a partir da distribuição do HI no último ciclo de vida
de cada unidade.
"""
from __future__ import annotations

import numpy as np

from aeroquant.digital_twin.application.ports import FleetBaselineTracker, HealthIndexEstimator
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams, SensorSchema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)


def calibrate_failure_threshold(
    schema: SensorSchema,
    baseline_tracker: FleetBaselineTracker,
    hi_estimator: HealthIndexEstimator,
    n_calibration_units: int = 15,
    healthy_window_cycles: int = 20,
    lifetime_mean: int = 180,
    lifetime_std: int = 35,
    percentile: float = 50.0,
    seed: int = 2026,
) -> float:
    """Retorna o limiar de falha calibrado (percentil `percentile` do HI no
    último ciclo de vida, sobre `n_calibration_units` unidades sintéticas).

    IMPORTANTE: usa o MESMO `baseline_tracker` que será usado depois pelo
    Digital Twin em produção — a calibração e o uso real precisam
    compartilhar a mesma referência de "saudável", senão o limiar não
    significa nada.
    """
    generator = StochasticSensorGenerator()
    rng = np.random.default_rng(seed)
    lifetimes = np.clip(rng.normal(lifetime_mean, lifetime_std, size=n_calibration_units), 30, None).astype(int)

    end_of_life_his: list[float] = []
    for i, lifetime in enumerate(lifetimes):
        unit = Unit(unit_id=f"calib-unit-{i}", fleet_id="calibration", max_cycles=int(lifetime), fault_mode=FaultMode.GRADUAL)
        params = DegradationParams(seed=int(rng.integers(0, 2**31 - 1)), noise_std=0.012)
        readings = generator.generate_unit(unit, schema, params)

        last_hi = None
        for r in readings:
            if r.cycle <= healthy_window_cycles:
                baseline_tracker.update(r.operating_condition, r.values)
            stats = baseline_tracker.stats(r.operating_condition)
            hi, _ = hi_estimator.estimate(r.values, stats)
            last_hi = hi
        if last_hi is not None:
            end_of_life_his.append(last_hi)

    return float(np.percentile(end_of_life_his, percentile))
