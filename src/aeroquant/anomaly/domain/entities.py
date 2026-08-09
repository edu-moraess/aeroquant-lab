from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyPoint:
    unit_id: str
    cycle: int
    score: float
    is_anomaly: bool
    method: str
    reason: str | None = None


@dataclass
class AnomalyReport:
    method: str
    n_samples: int
    n_anomalies: int
    threshold: float
    points: list[AnomalyPoint]

    @property
    def rate(self) -> float:
        return self.n_anomalies / max(self.n_samples, 1)
