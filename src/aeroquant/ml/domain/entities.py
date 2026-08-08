"""Entidades de domínio — ML Context. Deliberadamente finas: o "modelo"
de verdade (o objeto sklearn ajustado) vive em infrastructure; aqui só
metadados e o contrato de o que significa um modelo treinado.

`predictor` é tipado como `Any` (não `sklearn.base.BaseEstimator`) de
propósito: o domínio não deveria importar sklearn — quem chama `.predict()`
sobre esse objeto é infrastructure, que sabe o tipo real. Mesma lógica de
usar `Protocol` em vez de herança nos outros contexts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aeroquant.ml.domain.value_objects import RULMetrics


@dataclass
class TrainedModel:
    name: str
    algorithm: str
    features_used: list[str]
    predictor: Any
    supports_uncertainty: bool = False


@dataclass
class ComparisonResult:
    baseline_name: str
    results: dict[str, RULMetrics] = field(default_factory=dict)  # nome do modelo -> métricas

    def ranked_by_rmse(self) -> list[tuple[str, float]]:
        return sorted(((name, m.rmse) for name, m in self.results.items()), key=lambda x: x[1])