"""
CSVSensorRepository — implementação MVP do port SensorRepository.

Decisão: CSV em vez de Parquet/TimescaleDB nesta fase porque o container
de desenvolvimento atual não tem acesso à internet para instalar pyarrow
(pandas.to_parquet exige pyarrow ou fastparquet, nenhum dos dois presente).
A interface SensorRepository já isola essa decisão — trocar para
TimescaleDB/Parquet no ambiente com rede é uma troca de adapter, não uma
reescrita de domínio ou de use cases.
"""
from __future__ import annotations

import csv
import os

from aeroquant.sensor_data.domain.entities import SensorReading


class CSVSensorRepository:
    def __init__(self, path: str) -> None:
        self._path = path
        self._fieldnames: list[str] | None = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def save(self, readings: list[SensorReading]) -> None:
        if not readings:
            return
        sensor_names = sorted(readings[0].values.keys())
        fieldnames = ["unit_id", "cycle", "operating_condition", *sensor_names]
        file_exists = os.path.exists(self._path)

        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in readings:
                row = {"unit_id": r.unit_id, "cycle": r.cycle, "operating_condition": r.operating_condition}
                row.update(r.values)
                writer.writerow(row)

    def load(self, unit_id: str | None = None) -> list[SensorReading]:
        if not os.path.exists(self._path):
            return []
        out: list[SensorReading] = []
        with open(self._path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if unit_id is not None and row["unit_id"] != unit_id:
                    continue
                meta = {"unit_id", "cycle", "operating_condition"}
                values = {k: float(v) for k, v in row.items() if k not in meta}
                out.append(
                    SensorReading(
                        unit_id=row["unit_id"],
                        cycle=int(row["cycle"]),
                        operating_condition=int(row["operating_condition"]),
                        values=values,
                    )
                )
        return out
