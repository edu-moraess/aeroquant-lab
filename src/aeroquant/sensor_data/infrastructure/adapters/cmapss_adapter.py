"""
CMAPSSAdapter — converte os arquivos brutos do NASA C-MAPSS (train_FD00X.txt
/ test_FD00X.txt) para o SensorSchema comum usado pelo gerador sintético.

Formato conhecido do C-MAPSS (documentado pela NASA, sem cabeçalho):
  coluna 1        : unit number
  coluna 2        : time, in cycles
  colunas 3-5     : operational settings 1-3
  colunas 6-26    : sensor measurements 1-21
  separador       : espaço (possível espaço extra ao final de cada linha)

NOTA IMPORTANTE: este adaptador ainda não foi testado contra um arquivo
real, porque nenhum arquivo C-MAPSS foi enviado até este ponto do projeto
(o container de desenvolvimento não tem acesso à internet para baixá-lo).
A estrutura segue a documentação oficial do dataset; validar com
`python -m aeroquant.sensor_data.infrastructure.adapters.cmapss_adapter
<caminho_do_arquivo>` assim que o arquivo estiver disponível.
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
        # Placeholder simples: nos subconjuntos FD002/FD004 existem 6 regimes
        # operacionais discretos combináveis a partir de op_setting_1/2/3.
        # Uma vez com dados reais em mãos, isso deve ser substituído por
        # clustering (ex.: k-means com k=6) sobre (op_setting_1, op_setting_2,
        # op_setting_3), como é prática padrão na literatura de C-MAPSS.
        return 0
