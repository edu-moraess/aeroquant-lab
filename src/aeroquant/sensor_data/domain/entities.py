"""
Entidades de domínio — Sensor Data Context.

Regra de dependência (Clean Architecture): este módulo não importa nada
de infrastructure/ ou application/. Nenhuma dependência de numpy/pandas
aqui de propósito — o domínio deve ser testável sem qualquer biblioteca
de dados instalada.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FaultMode(str, Enum):
    """Modos de falha suportados pelo gerador sintético (Fase 4, Nível 1)."""

    NONE = "none"
    ABRUPT = "abrupt"
    INTERMITTENT = "intermittent"
    GRADUAL = "gradual"


@dataclass(frozen=True)
class SensorReading:
    """Uma leitura de sensores de uma unidade em um ciclo/timestamp específico.

    `values` usa o nome do sensor (definido no SensorSchema) como chave.
    """

    unit_id: str
    cycle: int
    operating_condition: int
    values: dict[str, float]

    def __post_init__(self) -> None:
        if self.cycle < 0:
            raise ValueError(f"cycle não pode ser negativo: {self.cycle}")
        if not self.values:
            raise ValueError("SensorReading precisa de pelo menos um valor de sensor")


@dataclass
class Unit:
    """Uma unidade monitorada (ex.: um motor turbofan específico de uma frota).

    `max_cycles` é o ciclo em que a unidade efetivamente falha — conhecido
    em dados sintéticos (usado para gerar o rótulo de RUL) e desconhecido
    em dados reais de teste (o que é justamente o que se quer prever).
    """

    unit_id: str
    fleet_id: str
    max_cycles: int
    fault_mode: FaultMode = FaultMode.GRADUAL
    metadata: dict[str, str] = field(default_factory=dict)

    def rul_at(self, cycle: int) -> int:
        """RUL verdadeiro (label) em um ciclo — só existe porque max_cycles é conhecido."""
        if cycle > self.max_cycles:
            raise ValueError(
                f"cycle {cycle} excede max_cycles {self.max_cycles} da unidade {self.unit_id}"
            )
        return self.max_cycles - cycle
