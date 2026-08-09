"""
Janelas temporais por unidade — entrada para modelos sequenciais (MLP-seq / LSTM).

Cada amostra é (seq_len, n_features) com o RUL do *último* ciclo da janela.
Split deve continuar sendo por unit_id (nunca misturar unidades entre treino/teste).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    X: np.ndarray  # (n, seq_len, n_features)
    y: np.ndarray  # (n,)
    unit_ids: np.ndarray  # (n,) unit_id da janela
    feature_names: list[str]
    seq_len: int


def build_sequence_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seq_len: int = 30,
    target_col: str = "rul",
    stride: int = 1,
) -> SequenceDataset:
    """Gera janelas deslizantes por `unit_id`."""
    if seq_len < 2:
        raise ValueError("seq_len deve ser >= 2")

    xs: list[np.ndarray] = []
    ys: list[float] = []
    units: list[str] = []

    cols = list(feature_cols)
    for unit_id, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("cycle")
        feats = g[cols].to_numpy(dtype=float)
        targets = g[target_col].to_numpy(dtype=float)
        n = len(g)
        if n < seq_len:
            continue
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            xs.append(feats[start:end])
            ys.append(float(targets[end - 1]))
            units.append(str(unit_id))

    if not xs:
        empty = np.zeros((0, seq_len, len(cols)), dtype=float)
        return SequenceDataset(
            X=empty,
            y=np.zeros(0, dtype=float),
            unit_ids=np.array([], dtype=str),
            feature_names=cols,
            seq_len=seq_len,
        )

    return SequenceDataset(
        X=np.stack(xs, axis=0),
        y=np.asarray(ys, dtype=float),
        unit_ids=np.asarray(units),
        feature_names=cols,
        seq_len=seq_len,
    )


def split_sequences_by_unit(
    data: SequenceDataset,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[SequenceDataset, SequenceDataset]:
    """Split treino/teste por unidade (sem vazamento temporal entre units)."""
    unique = np.unique(data.unit_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_units = set(shuffled[:n_test].tolist())

    test_mask = np.array([u in test_units for u in data.unit_ids])
    train_mask = ~test_mask

    def _subset(mask: np.ndarray) -> SequenceDataset:
        return SequenceDataset(
            X=data.X[mask],
            y=data.y[mask],
            unit_ids=data.unit_ids[mask],
            feature_names=data.feature_names,
            seq_len=data.seq_len,
        )

    return _subset(train_mask), _subset(test_mask)
