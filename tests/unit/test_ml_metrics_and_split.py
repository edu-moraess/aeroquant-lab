from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import (
    interval_coverage,
    mae,
    nasa_asymmetric_score,
    rmse,
)
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit


class TestMetrics(unittest.TestCase):
    def test_rmse_zero_for_perfect_prediction(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        self.assertEqual(rmse(y, y), 0.0)

    def test_mae_matches_manual_calculation(self) -> None:
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 15.0])
        self.assertAlmostEqual(mae(y_true, y_pred), (2.0 + 5.0) / 2, places=6)

    def test_nasa_score_penalizes_overestimation_more(self) -> None:
        """Propriedade central da métrica (Fase 2): para o mesmo |erro|,
        superestimar RUL (d>0) deve custar mais que subestimar (d<0)."""
        y_true = np.array([100.0])
        y_over = np.array([120.0])   # superestimou em 20
        y_under = np.array([80.0])   # subestimou em 20

        score_over = nasa_asymmetric_score(y_true, y_over)
        score_under = nasa_asymmetric_score(y_true, y_under)
        self.assertGreater(score_over, score_under)

    def test_nasa_score_zero_for_perfect_prediction(self) -> None:
        y = np.array([50.0, 80.0])
        self.assertAlmostEqual(nasa_asymmetric_score(y, y), 0.0, places=6)

    def test_interval_coverage_counts_correctly(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        lower = np.array([5.0, 25.0, 25.0])
        upper = np.array([15.0, 35.0, 35.0])
        # unidade 0: 10 in [5,15] OK | unidade 1: 20 NOT in [25,35] | unidade 2: 30 in [25,35] OK
        self.assertAlmostEqual(interval_coverage(y_true, lower, upper), 2 / 3, places=6)


class TestGroupSplit(unittest.TestCase):
    def test_no_unit_overlap_between_train_and_test(self) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [f"u{i}" for i in range(20) for _ in range(5)],
                "cycle": list(range(1, 6)) * 20,
                "rul": list(range(100, 95, -1)) * 20,
            }
        )
        train_df, test_df = split_by_unit(df, test_fraction=0.3, seed=1)
        self.assertTrue(set(train_df["unit_id"]).isdisjoint(set(test_df["unit_id"])))
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)

    def test_split_fraction_approximately_respected(self) -> None:
        df = pd.DataFrame({"unit_id": [f"u{i}" for i in range(100) for _ in range(3)]})
        train_df, test_df = split_by_unit(df, test_fraction=0.25, seed=2)
        n_test_units = test_df["unit_id"].nunique()
        self.assertEqual(n_test_units, 25)


if __name__ == "__main__":
    unittest.main()