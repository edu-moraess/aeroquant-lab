"""
Ports — interfaces que a camada application depende, e que infrastructure
implementa (Dependency Inversion). Usamos typing.Protocol em vez de ABC
para não forçar herança — qualquer classe com esses métodos serve, o que
facilita duplos de teste sem framework de mocking.
"""
from __future__ import annotations

from typing import Protocol

from aeroquant.sensor_data.domain.entities import SensorReading, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams, SensorSchema


class DataGenerator(Protocol):
    """Gera séries de sensores sintéticas para uma unidade até sua falha."""

    def generate_unit(
        self, unit: Unit, schema: SensorSchema, params: DegradationParams
    ) -> list[SensorReading]:
        ...


class SensorRepository(Protocol):
    """Persistência de leituras de sensores — implementação concreta é um detalhe."""

    def save(self, readings: list[SensorReading]) -> None:
        ...

    def load(self, unit_id: str | None = None) -> list[SensorReading]:
        ...


class DatasetAdapter(Protocol):
    """Adapta um dataset público (ex.: C-MAPSS) para o SensorSchema comum."""

    def parse(self, filepath: str, schema: SensorSchema) -> list[SensorReading]:
        ...
