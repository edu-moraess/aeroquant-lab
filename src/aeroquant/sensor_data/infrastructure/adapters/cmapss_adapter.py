"""
CMAPSSAdapter — converte os arquivos brutos do NASA C-MAPSS (train_FD00X.txt
/ test_FD00X.txt) para o SensorSchema comum usado pelo gerador sintético.

Formato conhecido do C-MAPSS (documentado pela NASA, sem cabeçalho):
  coluna 1        : unit number
  coluna 2        : time, in cycles
  colunas 3-5     : operational settings 1-3
  colunas 6-26    : sensor measurements 1-21
  separador       : espaço (possível espaço extra ao final de cada linha)

NOTA: validar com arquivos reais quando disponíveis no ambiente.
"""
from __future__ import annotations

import pandas as pd

from aeroquant.sensor_data.domain.entities import SensorReading
from aeroquant.sensor_data.domain.value_objects import SensorSchema

_N_OPERATIONAL_SETTINGS = 3
_N_SENSORS = 21


class CMAPSSAdapter:
    def parse(self, filepath: str, schema: SensorSchema) -> list[SensorReading]:
        df = self._read_raw(filepath)
        sensor_names = schema.names()
        if len(sensor_names) != _N_SENSORS:
            raise ValueError(
                f"Schema tem {len(sensor_names)} sensores, C-MAPSS tem {_N_SENSORS}. "
                "Use build_cmapss_like_schema() para garantir compatibilidade."
            )

        readings: list[SensorReading] = []
        for row in df.itertuples(index=False):
            values = {name: float(getattr(row, f"sensor_raw_{i + 1}")) for i, name in enumerate(sensor_names)}
            readings.append(
                SensorReading(
                    unit_id=f"cmapss-unit-{int(row.unit_number):04d}",
                    cycle=int(row.time_cycles),
                    operating_condition=self._encode_operating_condition(row),
                    values=values,
                )
            )
        return readings

    def to_dataframe(self, filepath: str, schema: SensorSchema) -> pd.DataFrame:
        """Converte arquivo C-MAPSS em DataFrame compatível com o ETL do projeto."""
        readings = self.parse(filepath, schema)
        rows = []
        for r in readings:
            row = {
                "unit_id": r.unit_id,
                "cycle": r.cycle,
                "operating_condition": r.operating_condition,
            }
            row.update(r.values)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def attach_train_rul(df: pd.DataFrame, max_rul_cap: int = 125) -> pd.DataFrame:
        """RUL piecewise para arquivos de treino (run-to-failure)."""
        out = df.copy()
        true_rul = out.groupby("unit_id")["cycle"].transform("max") - out["cycle"]
        out["rul"] = true_rul.clip(upper=max_rul_cap)
        return out

    @staticmethod
    def attach_test_rul(df: pd.DataFrame, rul_filepath: str, max_rul_cap: int = 125) -> pd.DataFrame:
        """Anexa RUL_FD00X.txt (uma linha por unit, RUL no último ciclo observado)."""
        rul_values = pd.read_csv(rul_filepath, sep=r"\s+", header=None, names=["rul_at_end"])
        units = sorted(df["unit_id"].unique())
        if len(units) != len(rul_values):
            raise ValueError(
                f"RUL file tem {len(rul_values)} linhas, dataset tem {len(units)} unidades."
            )
        mapping = {u: float(rul_values.iloc[i]["rul_at_end"]) for i, u in enumerate(units)}
        out = df.copy()
        max_cycle = out.groupby("unit_id")["cycle"].transform("max")
        end_rul = out["unit_id"].map(mapping)
        out["rul"] = (end_rul + (max_cycle - out["cycle"])).clip(upper=max_rul_cap)
        return out

    @staticmethod
    def _read_raw(filepath: str) -> pd.DataFrame:
        columns = (
            ["unit_number", "time_cycles"]
            + [f"op_setting_{i + 1}" for i in range(_N_OPERATIONAL_SETTINGS)]
            + [f"sensor_raw_{i + 1}" for i in range(_N_SENSORS)]
        )
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=columns)
        return df

    @staticmethod
    def _encode_operating_condition(row) -> int:
        # Placeholder: FD002/FD004 usam 6 regimes; com dados reais, preferir k-means.
        return 0
