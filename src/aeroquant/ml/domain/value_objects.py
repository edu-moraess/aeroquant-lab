"""Value Objects — ML Context. RULMetrics fica aqui (domínio) porque é um
CONCEITO do domínio (o que significa avaliar um modelo de RUL), não um
detalhe de implementação — infrastructure/evaluation/metrics.py CALCULA
isto, mas não deveria ser dono da definição."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RULMetrics:
    rmse: float
    mae: float
    nasa_score: float
    n_samples: int
    interval_coverage_90: float | None = None