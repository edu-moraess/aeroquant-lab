"""
group_split — split de treino/teste por unit_id, NUNCA por linha aleatória.

Por que isso é obrigatório (não estilístico): linhas da mesma unidade são
temporalmente correlacionadas (ciclo N e N+1 do mesmo motor compartilham
quase todo o estado de degradação). Um split aleatório por linha vazaria
informação da mesma unidade entre treino e teste, inflando artificialmente
a performance reportada — exatamente o risco que a Fase 1 (auditoria) e a
Fase 2 (pergunta científica) já sinalizaram como obrigatório de evitar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_unit(
    df: pd.DataFrame, test_fraction: float = 0.3, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_ids = df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unit_ids)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_units = set(shuffled[:n_test])

    train_df = df[\~df["unit_id"].isin(test_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()

    # Invariante de sanidade: nenhuma unidade pode aparecer nos dois lados.
    assert train_df["unit_id"].isin(test_units).sum() == 0
    assert set(train_df["unit_id"]).isdisjoint(set(test_df["unit_id"]))

    return train_df, test_df