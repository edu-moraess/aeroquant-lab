"""Split por unit_id — obrigatório para dados de degradação."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_unit(df: pd.DataFrame, test_fraction: float = 0.3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_ids = df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unit_ids)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_units = set(shuffled[:n_test])
    train_df = df[~df["unit_id"].isin(test_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()
    assert set(train_df["unit_id"]).isdisjoint(set(test_df["unit_id"]))
    return train_df, test_df


def split_by_unit_three_way(
    df: pd.DataFrame, *, val_fraction: float = 0.15, test_fraction: float = 0.20, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_fraction + test_fraction >= 0.95:
        raise ValueError("val_fraction + test_fraction deve deixar espaço para treino")
    unit_ids = np.array(df["unit_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unit_ids)
    n = len(shuffled)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 6)
    test_units = set(shuffled[:n_test])
    val_units = set(shuffled[n_test : n_test + n_val])
    train_units = set(shuffled[n_test + n_val :])
    train_df = df[df["unit_id"].isin(train_units)].copy()
    val_df = df[df["unit_id"].isin(val_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()
    assert train_units.isdisjoint(val_units) and train_units.isdisjoint(test_units) and val_units.isdisjoint(test_units)
    return train_df, val_df, test_df
