"""
Ports — Digital Twin Context. Três estimadores desacoplados de propósito:
Health Index, RUL e baseline de frota são conceitualmente independentes e
serão substituídos por implementações de ML na Fase 6 sem alterar
UpdateDigitalTwin (application/use_cases.py) nem o agregado de domínio.
"""
from __future__ import annotations

from typing import Protocol

from aeroquant.digital_twin.domain.entities import DigitalTwinSnapshot, DigitalTwinState
from aeroquant.sensor_data.domain.value_objects import RULEstimate


class FleetBaselineTracker(Protocol):
    """Mantém estatísticas (mean/std) de sensores 'saudáveis' da frota,
    atualizadas de forma incremental (aprendizado online) a cada leitura."""

    def update(self, operating_condition: int, sensor_values: dict[str, float]) -> None:
        ...

    def stats(self, operating_condition: int) -> dict[str, tuple[float, float]]:
        """Retorna {sensor: (mean, std)} para a condição operacional dada."""
        ...


class HealthIndexEstimator(Protocol):
    def estimate(
        self, sensor_values: dict[str, float], baseline_stats: dict[str, tuple[float, float]]
    ) -> tuple[float, float]:
        """Retorna (health_index, incerteza)."""
        ...


class RULEstimator(Protocol):
    def estimate(self, history: list[DigitalTwinSnapshot], failure_threshold: float) -> RULEstimate:
        ...


class DigitalTwinRepository(Protocol):
    def save(self, state: DigitalTwinState) -> None:
        ...

    def load(self, unit_id: str) -> DigitalTwinState:
        ...
