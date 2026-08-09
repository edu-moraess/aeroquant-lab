import unittest

import numpy as np
import pandas as pd

from aeroquant.ml.infrastructure.evaluation.metrics import (
    bias, evaluate_by_rul_bucket, evaluate_extended, mae, nasa_asymmetric_score, r2_score, rmse,
)
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit, split_by_unit_three_way


class TestMetrics(unittest.TestCase):
    def test_rmse_zero_for_perfect_prediction(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        self.assertEqual(rmse(y, y), 0.0)

    def test_nasa_score_penalizes_overestimation_more(self) -> None:
        y_true = np.array([100.0])
        self.assertGreater(
            nasa_asymmetric_score(y_true, np.array([120.0])),
            nasa_asymmetric_score(y_true, np.array([80.0])),
        )

    def test_bias_positive_when_overestimating(self) -> None:
        self.assertGreater(bias(np.array([10.0, 20.0]), np.array([12.0, 25.0])), 0)

    def test_r2_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r2_score(y, y), 1.0, places=6)

    def test_extended_and_buckets(self) -> None:
        y_true = np.array([5.0, 15.0, 40.0, 80.0])
        y_pred = np.array([6.0, 12.0, 45.0, 70.0])
        self.assertEqual(evaluate_extended(y_true, y_pred).n_samples, 4)
        self.assertEqual(len(evaluate_by_rul_bucket(y_true, y_pred)), 4)


class TestGroupSplit(unittest.TestCase):
    def test_no_unit_overlap(self) -> None:
        df = pd.DataFrame({
            "unit_id": [f"u{i}" for i in range(20) for _ in range(5)],
            "cycle": list(range(1, 6)) * 20,
        })
        train_df, test_df = split_by_unit(df, test_fraction=0.3, seed=1)
        self.assertTrue(set(train_df["unit_id"]).isdisjoint(set(test_df["unit_id"])))

    def test_three_way_no_overlap(self) -> None:
        df = pd.DataFrame({"unit_id": [f"u{i}" for i in range(40) for _ in range(3)]})
        train, val, test = split_by_unit_three_way(df, seed=7)
        a, b, c = set(train["unit_id"]), set(val["unit_id"]), set(test["unit_id"])
        self.assertTrue(a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c))


if __name__ == "__main__":
    unittest.main()
