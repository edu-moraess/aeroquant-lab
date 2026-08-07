"""
Entidades de domínio — Digital Twin Context.

Decisão de design: o Digital Twin CONSOME estimativas de Health Index e
RUL (via ports), ele mesmo não as calcula com regras de negócio complexas
— isso pertence aos estimadores (infrastructure) e, futuramente, ao RUL
Context (Fase 6, ML). O agregado aqui só garante consistência do histórico
(ciclos estritamente crescentes) e expõe leitura de tendência.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aeroquant.sensor_data.domain.value_objects import RULEstimate


@dataclass(frozen=True)
class DigitalTwinSnapshot:
    """Um "frame" do gêmeo digital em um ciclo específico."""

    cycle: int
    health_index: float
    health_index_uncertainty: float
    rul: RULEstimate
    is_anomaly: bool
    anomaly_reason: str | None = None


@dataclass
class DigitalTwinState:
    """Agregado raiz: estado acumulado do gêmeo digital de UMA unidade."""

    unit_id: str
    history: list[DigitalTwinSnapshot] = field(default_factory=list)

    def append(self, snapshot: DigitalTwinSnapshot) -> None:
        if self.history and snapshot.cycle <= self.history[-1].cycle:
            raise ValueError(
                f"Ciclo {snapshot.cycle} não é posterior ao último snapshot "
                f"({self.history[-1].cycle}) para a unidade {self.unit_id}"
            )
        self.history.append(snapshot)

    @property
    def latest(self) -> DigitalTwinSnapshot | None:
        return self.history[-1] if self.history else None

    @property
    def anomaly_cycles(self) -> list[int]:
        return [s.cycle for s in self.history if s.is_anomaly]
