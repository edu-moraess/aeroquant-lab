"""
DataQualityChecker — checagens declarativas simples, no espírito do
Great Expectations (que não pôde ser instalado neste container por falta
de acesso à rede). Quando o projeto migrar para um ambiente com internet,
estas mesmas checagens devem virar `Expectation`s do Great Expectations —
a lista abaixo já está estruturada para tornar essa migração mecânica.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from aeroquant.sensor_data.domain.value_objects import SensorSchema


@dataclass
class QualityIssue:
    check: str
    severity: str  # "error" | "warning"
    detail: str


@dataclass
class QualityReport:
    n_rows: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


class DataQualityChecker:
    def __init__(self, schema: SensorSchema) -> None:
        self._schema = schema

    def run(self, df: pd.DataFrame) -> QualityReport:
        issues: list[QualityIssue] = []
        issues += self._check_nulls(df)
        issues += self._check_ranges(df)
        issues += self._check_monotonic_cycles(df)
        issues += self._check_duplicates(df)
        return QualityReport(n_rows=len(df), issues=issues)

    def _check_nulls(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        null_rate = df.isnull().mean()
        for col, rate in null_rate.items():
            if rate > 0:
                severity = "error" if rate > 0.05 else "warning"
                issues.append(QualityIssue("null_rate", severity, f"{col}: {rate:.2%} nulo"))
        return issues

    def _check_ranges(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        for spec in self._schema.sensors:
            if spec.name not in df.columns:
                continue
            out_of_range = ((df[spec.name] < spec.valid_min) | (df[spec.name] > spec.valid_max)).sum()
            if out_of_range > 0:
                issues.append(
                    QualityIssue(
                        "range_violation", "error", f"{spec.name}: {out_of_range} leituras fora de [{spec.valid_min}, {spec.valid_max}]"
                    )
                )
        return issues

    def _check_monotonic_cycles(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues = []
        for unit_id, group in df.groupby("unit_id"):
            cycles = group["cycle"].to_numpy()
            if not (cycles[1:] > cycles[:-1]).all():
                issues.append(QualityIssue("monotonic_cycle", "error", f"{unit_id}: ciclos não estritamente crescentes"))
        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> list[QualityIssue]:
        n_dup = df.duplicated(subset=["unit_id", "cycle"]).sum()
        if n_dup > 0:
            return [QualityIssue("duplicates", "error", f"{n_dup} pares (unit_id, cycle) duplicados")]
        return []
