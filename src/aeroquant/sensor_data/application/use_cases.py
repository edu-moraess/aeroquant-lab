"""
Use cases — orquestram DataGenerator/Repository via suas interfaces (ports),
nunca importando implementações concretas de infrastructure diretamente.
As implementações concretas são injetadas pelo chamador (composition root).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeroquant.sensor_data.application.ports import DataGenerator, SensorRepository
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams, SensorSchema


@dataclass
class FleetGenerationResult:
    n_units: int
    n_readings: int
    lifetime_mean: float
    lifetime_std: float


class GenerateSyntheticFleet:
    """Gera uma frota sintética de N unidades com vidas úteis variáveis."""

    def __init__(self, generator: DataGenerator, repository: SensorRepository) -> None:
        self._generator = generator
        self._repository = repository

    def run(
        self,
        schema: SensorSchema,
        params: DegradationParams,
        n_units: int,
        lifetime_mean: int = 200,
        lifetime_std: int = 40,
        fleet_id: str = "synthetic-fleet-01",
        seed: int | None = None,
    ) -> FleetGenerationResult:
        rng = np.random.default_rng(seed)
        lifetimes = np.clip(
            rng.normal(lifetime_mean, lifetime_std, size=n_units), 30, None
        ).astype(int)

        total_readings = 0
        for i, lifetime in enumerate(lifetimes):
            unit = Unit(
                unit_id=f"{fleet_id}-unit-{i + 1:04d}",
                fleet_id=fleet_id,
                max_cycles=int(lifetime),
                fault_mode=FaultMode.GRADUAL,
            )
            unit_params = params if params.seed is not None else _reseed(params, int(rng.integers(0, 2**31 - 1)))
            readings = self._generator.generate_unit(unit, schema, unit_params)
            self._repository.save(readings)
            total_readings += len(readings)

        return FleetGenerationResult(
            n_units=n_units,
            n_readings=total_readings,
            lifetime_mean=float(np.mean(lifetimes)),
            lifetime_std=float(np.std(lifetimes)),
        )


def _reseed(params: DegradationParams, seed: int) -> DegradationParams:
    from dataclasses import replace

    return replace(params, seed=seed)
