"""
CMAPSSAdapter — NASA C-MAPSS (train/test_FD00X.txt) → DataFrame do projeto.

FD001/FD003: 1 regime operacional
FD002/FD004: 6 regimes (KMeans nos op_settings — fit só no treino)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroquant.sensor_data.domain.entities import SensorReading
from aeroquant.sensor_data.domain.value_objects import SensorSchema

_N_OPERATIONAL_SETTINGS = 3
_N_SENSORS = 21
_OP_COLS = [f"op_setting_{i + 1}" for i in range(_N_OPERATIONAL_SETTINGS)]
_SENSOR_RAW = [f"sensor_raw_{i + 1}" for i in range(_N_SENSORS)]


@dataclass
class OperatingConditionEncoder:
    """K-Means de regimes operacionais — fit só no treino (sem leakage)."""

    n_clusters: int = 6
    seed: int = 42
    _centers: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> OperatingConditionEncoder:
        from sklearn.cluster import KMeans

        X = df[_OP_COLS].to_numpy(dtype=float)
        if float(np.std(X)) < 1e-6:
            self.n_clusters = 1
            self._centers = X.mean(axis=0, keepdims=True)
            return self
        k = min(self.n_clusters, max(1, len(np.unique(X.round(4), axis=0))))
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        km.fit(X)
        self.n_clusters = int(k)
        self._centers = km.cluster_centers_
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._centers is None:
            raise RuntimeError("Encoder não foi fitado")
        X = df[_OP_COLS].to_numpy(dtype=float)
        d = ((X[:, None, :] - self._centers[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1).astype(int)


class CMAPSSAdapter:
    def parse(self, filepath: str, schema: SensorSchema) -> list[SensorReading]:
        df = self.to_dataframe(filepath, schema)
        sensor_names = schema.names()
        readings: list[SensorReading] = []
        for row in df.itertuples(index=False):
            values = {name: float(getattr(row, name)) for name in sensor_names}
            readings.append(
                SensorReading(
                    unit_id=str(row.unit_id),
                    cycle=int(row.cycle),
                    operating_condition=int(row.operating_condition),
                    values=values,
                )
            )
        return readings

    def to_dataframe(
        self,
        filepath: str,
        schema: SensorSchema,
        *,
        op_encoder: OperatingConditionEncoder | None = None,
    ) -> pd.DataFrame:
        sensor_names = schema.names()
        if len(sensor_names) != _N_SENSORS:
            raise ValueError(f"Schema tem {len(sensor_names)} sensores, C-MAPSS tem {_N_SENSORS}.")
        raw = self._read_raw(filepath)
        out = pd.DataFrame({
            "unit_id": raw["unit_number"].map(lambda u: f"cmapss-unit-{int(u):04d}"),
            "cycle": raw["time_cycles"].astype(int),
        })
        for i, name in enumerate(sensor_names):
            out[name] = raw[f"sensor_raw_{i + 1}"].astype(float)
        for c in _OP_COLS:
            out[c] = raw[c].astype(float)
        if op_encoder is not None:
            out["operating_condition"] = op_encoder.transform(out)
        else:
            out["operating_condition"] = 0
        return out

    @staticmethod
    def attach_train_rul(df: pd.DataFrame, max_rul_cap: int = 125) -> pd.DataFrame:
        out = df.copy()
        true_rul = out.groupby("unit_id")["cycle"].transform("max") - out["cycle"]
        out["rul"] = true_rul.clip(upper=max_rul_cap)
        return out

    @staticmethod
    def attach_test_rul(df: pd.DataFrame, rul_filepath: str, max_rul_cap: int = 125) -> pd.DataFrame:
        rul_values = pd.read_csv(rul_filepath, sep=r"\s+", header=None, names=["rul_at_end"])
        units = sorted(df["unit_id"].unique())
        if len(units) != len(rul_values):
            raise ValueError(f"RUL file tem {len(rul_values)} linhas, dataset tem {len(units)} unidades.")
        mapping = {u: float(rul_values.iloc[i]["rul_at_end"]) for i, u in enumerate(units)}
        out = df.copy()
        max_cycle = out.groupby("unit_id")["cycle"].transform("max")
        end_rul = out["unit_id"].map(mapping)
        out["rul"] = (end_rul + (max_cycle - out["cycle"])).clip(upper=max_rul_cap)
        return out

    @staticmethod
    def _read_raw(filepath: str) -> pd.DataFrame:
        columns = ["unit_number", "time_cycles"] + _OP_COLS + _SENSOR_RAW
        return pd.read_csv(filepath, sep=r"\s+", header=None, names=columns)
