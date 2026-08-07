"""
Value Objects — Sensor Data Context.

O SensorSchema é a única fonte de verdade sobre quais sensores existem,
suas unidades e faixas válidas. Tanto o gerador sintético (Fase 4) quanto
o adaptador C-MAPSS (infrastructure/adapters) devem produzir dados que
respeitem o mesmo SensorSchema — é isso que viabiliza o treino híbrido
sintético→real sem retrabalho de schema.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorSpec:
    """Especificação de um único sensor."""

    name: str
    unit: str
    valid_min: float
    valid_max: float
    baseline: float  # valor esperado em condição saudável (health = 0)
    degradation_coupling: float  # o quanto esse sensor reage à degradação [0, 1]


@dataclass(frozen=True)
class SensorSchema:
    """Schema compartilhado entre gerador sintético e dados públicos (C-MAPSS)."""

    sensors: tuple[SensorSpec, ...]
    n_operating_conditions: int

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sensors)

    def spec_for(self, name: str) -> SensorSpec:
        for s in self.sensors:
            if s.name == name:
                return s
        raise KeyError(f"Sensor desconhecido no schema: {name}")


@dataclass(frozen=True)
class DegradationParams:
    """Parâmetros do processo estocástico de degradação (ver stochastic_generator.py).

    Estes são exatamente os parâmetros que a Fase 2 (H2) propõe variar
    sistematicamente para quantificar o gap sintético-real.
    """

    noise_std: float = 0.02          # ruído de medição (fração do baseline)
    drift_rate: float = 0.0005       # deriva lenta do sensor (não relacionada a falha)
    degradation_shape: float = 2.0   # shape do processo Gamma de degradação
    degradation_scale: float = 0.01  # scale do processo Gamma de degradação
    abrupt_fault_rate: float = 0.0   # taxa (Poisson) de falha abrupta por ciclo
    abrupt_fault_magnitude: float = 0.15  # magnitude do salto, fração do baseline
    intermittent_fault_prob: float = 0.0  # prob. de pulso intermitente por ciclo
    intermittent_fault_magnitude: float = 0.10
    seed: int | None = None


@dataclass(frozen=True)
class RULEstimate:
    """Estimativa de RUL com incerteza — usada a partir da Fase 6/RUL Context.

    Definido já na Fase 3 porque o ETL precisa gerar rótulos (`RUL` verdadeiro)
    no mesmo "formato" que o modelo vai aprender a prever.
    """

    point: float
    lower: float
    upper: float
    confidence: float = 0.90
