"""Política de ranking configurável para benchmark de modelos RUL."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RankingPolicy:
    nasa_weight: float = 1.0
    rmse_weight: float = 0.5
    abs_bias_weight: float = 0.25

    def score_row(self, row: pd.Series) -> float:
        nasa = float(row.get("NASA Score", row.get("nasa_score", 0.0)))
        rmse = float(row.get("RMSE", row.get("rmse", 0.0)))
        bias = abs(float(row.get("Bias", row.get("bias", 0.0))))
        return self.nasa_weight * nasa + self.rmse_weight * rmse + self.abs_bias_weight * bias

    def rank(self, table: pd.DataFrame) -> pd.DataFrame:
        out = table.copy()
        out["_rank_score"] = out.apply(self.score_row, axis=1)
        out = out.sort_values("_rank_score", ascending=True).reset_index(drop=True)
        out["Rank"] = range(1, len(out) + 1)
        cols = ["Rank"] + [c for c in out.columns if c not in ("Rank", "_rank_score")]
        return out[cols].drop(columns=["_rank_score"], errors="ignore")
