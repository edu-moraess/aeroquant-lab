"""InMemoryDigitalTwinRepository — suficiente para o MVP local (Fase 5).
Trocar por Postgres/TimescaleDB é uma troca de adapter (ver ports.py)."""
from __future__ import annotations

from aeroquant.digital_twin.domain.entities import DigitalTwinState


class InMemoryDigitalTwinRepository:
    def __init__(self) -> None:
        self._states: dict[str, DigitalTwinState] = {}

    def save(self, state: DigitalTwinState) -> None:
        self._states[state.unit_id] = state

    def load(self, unit_id: str) -> DigitalTwinState:
        if unit_id not in self._states:
            self._states[unit_id] = DigitalTwinState(unit_id=unit_id)
        return self._states[unit_id]
