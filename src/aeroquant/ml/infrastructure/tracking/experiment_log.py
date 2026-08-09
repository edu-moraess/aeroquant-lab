"""Registro simples de experimentos."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    model: str
    architecture: str
    seed: int
    n_features: int
    sequence_length: int | None
    n_train: int
    n_val: int | None
    n_test: int
    epochs: int | None
    best_epoch: int | None
    rmse: float
    mae: float
    r2: float | None
    nasa_score: float
    bias: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentLog:
    def __init__(self) -> None:
        self._records: list[ExperimentRecord] = []

    def add(self, record: ExperimentRecord) -> None:
        self._records.append(record)

    def all(self) -> list[ExperimentRecord]:
        return list(self._records)

    def as_table_rows(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records]


def new_experiment_id(prefix: str = "exp") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
