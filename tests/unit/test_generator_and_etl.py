"""
Testes de propriedade (no espírito do que `hypothesis` faria) escritos
com unittest puro, já que este container não tem acesso à rede para
instalar hypothesis/pytest. Quando o projeto migrar para um ambiente com
internet, estes casos devem virar `@given(...)` do hypothesis, testando
sobre uma faixa de seeds/params em vez de valores fixos.
"""
from __future__ import annotations

import unittest

import numpy as np

from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.etl.pipeline import (
    add_rul_labels,
    clean,
    engineer_features,
    normalize,
    readings_to_dataframe,
    select_features,
)
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)


class TestStochasticGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_cmapss_like_schema()
        self.generator = StochasticSensorGenerator()

    def test_generates_exactly_max_cycles_readings(self) -> None:
        unit = Unit(unit_id="u1", fleet_id="f1", max_cycles=150, fault_mode=FaultMode.GRADUAL)
        params = DegradationParams(seed=42)
        readings = self.generator.generate_unit(unit, self.schema, params)
        self.assertEqual(len(readings), 150)
        self.assertEqual(readings[0].cycle, 1)
        self.assertEqual(readings[-1].cycle, 150)

    def test_degradation_increases_sensor_dispersion_over_life(self) -> None:
        """Sensores fortemente acoplados à degradação devem se afastar mais
        do baseline perto do fim de vida do que no início (health monotônico)."""
        unit = Unit(unit_id="u1", fleet_id="f1", max_cycles=200, fault_mode=FaultMode.GRADUAL)
        params = DegradationParams(seed=7, noise_std=0.01)
        readings = self.generator.generate_unit(unit, self.schema, params)

        coupled_sensor = "sensor_4"  # coupling alto (0.45) no schema
        spec = self.schema.spec_for(coupled_sensor)
        early = np.mean([abs(r.values[coupled_sensor] - spec.baseline) for r in readings[:20]])
        late = np.mean([abs(r.values[coupled_sensor] - spec.baseline) for r in readings[-20:]])
        self.assertGreater(late, early)

    def test_values_respect_schema_bounds(self) -> None:
        unit = Unit(unit_id="u1", fleet_id="f1", max_cycles=100, fault_mode=FaultMode.GRADUAL)
        params = DegradationParams(seed=1, abrupt_fault_rate=0.02, intermittent_fault_prob=0.05)
        readings = self.generator.generate_unit(unit, self.schema, params)
        for r in readings:
            for name, value in r.values.items():
                spec = self.schema.spec_for(name)
                self.assertGreaterEqual(value, spec.valid_min)
                self.assertLessEqual(value, spec.valid_max)

    def test_reproducible_with_same_seed(self) -> None:
        unit = Unit(unit_id="u1", fleet_id="f1", max_cycles=80, fault_mode=FaultMode.GRADUAL)
        r1 = self.generator.generate_unit(unit, self.schema, DegradationParams(seed=99))
        r2 = self.generator.generate_unit(unit, self.schema, DegradationParams(seed=99))
        self.assertEqual(r1[10].values, r2[10].values)


class TestETLPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_cmapss_like_schema()
        generator = StochasticSensorGenerator()
        readings = []
        for i, lifetime in enumerate([120, 180, 90]):
            unit = Unit(unit_id=f"u{i}", fleet_id="f1", max_cycles=lifetime, fault_mode=FaultMode.GRADUAL)
            readings += generator.generate_unit(unit, self.schema, DegradationParams(seed=10 + i))
        self.df = readings_to_dataframe(readings, self.schema)

    def test_rul_is_capped_and_non_negative(self) -> None:
        df = add_rul_labels(self.df, max_rul_cap=125)
        self.assertTrue((df["rul"] >= 0).all())
        self.assertTrue((df["rul"] <= 125).all())
        # última leitura de cada unidade deve ter RUL = 0
        last_rows = df.sort_values("cycle").groupby("unit_id").tail(1)
        self.assertTrue((last_rows["rul"] == 0).all())

    def test_clean_has_no_nulls(self) -> None:
        df = clean(self.df, self.schema)
        sensor_cols = [s.name for s in self.schema.sensors]
        self.assertEqual(df[sensor_cols].isnull().sum().sum(), 0)

    def test_normalize_produces_zero_mean_per_condition(self) -> None:
        df = clean(self.df, self.schema)
        df = normalize(df, self.schema)
        for cond, group in df.groupby("operating_condition"):
            mean_z = group["sensor_4_z"].mean()
            self.assertAlmostEqual(mean_z, 0.0, delta=0.5)

    def test_feature_engineering_adds_expected_columns(self) -> None:
        df = clean(self.df, self.schema)
        df = normalize(df, self.schema)
        df = engineer_features(df, self.schema, window=5)
        self.assertIn("sensor_4_z_roll_mean5", df.columns)
        self.assertIn("sensor_4_z_delta_first", df.columns)

    def test_feature_selection_drops_weak_correlation(self) -> None:
        df = clean(self.df, self.schema)
        df = normalize(df, self.schema)
        df = engineer_features(df, self.schema, window=5)
        df = add_rul_labels(df)
        candidate_cols = [f"{s.name}_z" for s in self.schema.sensors]
        selected = select_features(df, candidate_cols)
        self.assertLessEqual(len(selected), len(candidate_cols))
        self.assertGreater(len(selected), 0)


if __name__ == "__main__":
    unittest.main()
