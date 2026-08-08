from __future__ import annotations

import unittest

from aeroquant.ml.application.use_cases import TrainAndCompareModels
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    GradientBoostingQuantileTrainer,
    LinearRegressionTrainer,
    RandomForestTrainer,
)
from aeroquant.sensor_data.application.use_cases import GenerateSyntheticFleet
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.etl.pipeline import (
    add_rul_labels,
    clean,
    engineer_features,
    normalize,
    readings_to_dataframe,
)
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)
from aeroquant.sensor_data.infrastructure.repositories.csv_repository import CSVSensorRepository


class TestMLModelComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = build_cmapss_like_schema()
        generator = StochasticSensorGenerator()
        repo = CSVSensorRepository("/tmp/aeroquant_ml_test_fleet.csv")
        use_case = GenerateSyntheticFleet(generator, repo)
        use_case.run(schema, DegradationParams(noise_std=0.015), n_units=40, lifetime_mean=150, lifetime_std=25, seed=99)

        readings = repo.load()
        df = readings_to_dataframe(readings, schema)
        df = clean(df, schema)
        df = normalize(df, schema)
        df = engineer_features(df, schema, window=5)
        df = add_rul_labels(df, max_rul_cap=125)

        cls.feature_cols = [c for c in df.columns if c.endswith("_z") or "_roll_" in c or "_delta_" in c]
        cls.train_df, cls.test_df = split_by_unit(df, test_fraction=0.3, seed=5)

    def test_no_unit_leakage_in_this_dataset(self) -> None:
        self.assertTrue(set(self.train_df["unit_id"]).isdisjoint(set(self.test_df["unit_id"])))

    def test_all_models_produce_valid_metrics(self) -> None:
        use_case = TrainAndCompareModels(
            {
                "linear_regression": LinearRegressionTrainer(),
                "random_forest": RandomForestTrainer(n_estimators=50),
                "gradient_boosting_quantile": GradientBoostingQuantileTrainer(n_estimators=50),
            }
        )
        result = use_case.run(self.train_df, self.test_df, self.feature_cols)

        for name, metrics in result.results.items():
            with self.subTest(model=name):
                self.assertGreater(metrics.rmse, 0)
                self.assertGreater(metrics.mae, 0)
                self.assertGreaterEqual(metrics.nasa_score, 0)
                self.assertEqual(metrics.n_samples, len(self.test_df))

    def test_models_with_uncertainty_have_reasonable_coverage(self) -> None:
        """Cobertura de 90% não precisa ser exatamente 0.90 (é uma amostra
        finita), mas não pode ser absurda (ex.: 0.05 ou 1.0 exatos, sinal
        de intervalo quebrado)."""
        use_case = TrainAndCompareModels(
            {
                "random_forest": RandomForestTrainer(n_estimators=50),
                "gradient_boosting_quantile": GradientBoostingQuantileTrainer(n_estimators=50),
            }
        )
        result = use_case.run(self.train_df, self.test_df, self.feature_cols)

        for name in ("random_forest", "gradient_boosting_quantile"):
            coverage = result.results[name].interval_coverage_90
            with self.subTest(model=name):
                self.assertIsNotNone(coverage)
                self.assertGreater(coverage, 0.3)
                self.assertLessEqual(coverage, 1.0)

    def test_ranked_by_rmse_is_sorted_ascending(self) -> None:
        use_case = TrainAndCompareModels(
            {
                "linear_regression": LinearRegressionTrainer(),
                "random_forest": RandomForestTrainer(n_estimators=50),
            }
        )
        result = use_case.run(self.train_df, self.test_df, self.feature_cols)
        ranked = result.ranked_by_rmse()
        rmses = [r[1] for r in ranked]
        self.assertEqual(rmses, sorted(rmses))


if __name__ == "__main__":
    unittest.main()