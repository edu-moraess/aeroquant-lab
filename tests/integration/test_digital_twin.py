from __future__ import annotations

import unittest

import numpy as np

from aeroquant.digital_twin.application.use_cases import UpdateDigitalTwin
from aeroquant.digital_twin.infrastructure.estimators.linear_extrapolation_rul import (
    LinearExtrapolationRULEstimator,
)
from aeroquant.digital_twin.infrastructure.estimators.welford_fleet_baseline import OnlineFleetBaseline
from aeroquant.digital_twin.infrastructure.estimators.zscore_health_index import ZScoreHealthIndexEstimator
from aeroquant.digital_twin.infrastructure.repositories.in_memory_repository import (
    InMemoryDigitalTwinRepository,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)


class TestOnlineFleetBaseline(unittest.TestCase):
    def test_matches_numpy_mean_std_after_full_pass(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(50, 5, size=200)

        tracker = OnlineFleetBaseline()
        for v in values:
            tracker.update(operating_condition=0, sensor_values={"s": float(v)})

        mean, std = tracker.stats(0)["s"]
        self.assertAlmostEqual(mean, float(np.mean(values)), places=6)
        self.assertAlmostEqual(std, float(np.std(values)), places=6)


class TestDigitalTwinIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_cmapss_like_schema()
        self.generator = StochasticSensorGenerator()

    def _build_dt(self) -> UpdateDigitalTwin:
        return UpdateDigitalTwin(
            baseline_tracker=OnlineFleetBaseline(),
            hi_estimator=ZScoreHealthIndexEstimator(self.schema, coupling_threshold=0.2),
            rul_estimator=LinearExtrapolationRULEstimator(min_points=8, window=40),
            repository=InMemoryDigitalTwinRepository(),
        )

    def test_rul_point_converges_towards_truth_near_end_of_life(self) -> None:
        unit = Unit(unit_id="u1", fleet_id="f1", max_cycles=160, fault_mode=FaultMode.GRADUAL)
        readings = self.generator.generate_unit(unit, self.schema, DegradationParams(seed=5, noise_std=0.01))

        dt = self._build_dt()
        errors_early: list[float] = []
        errors_late: list[float] = []
        for r in readings:
            snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values)
            true_rul = unit.max_cycles - r.cycle
            error = abs(snap.rul.point - true_rul)
            if r.cycle < 40:
                errors_early.append(error)
            elif r.cycle > 140:
                errors_late.append(error)

        # Erro médio perto do fim de vida deve ser bem menor que no início
        # (mais dados acumulados = extrapolação melhor) — não exige acerto
        # perfeito, só que a tendência de melhora exista, que é o requisito
        # real de um estimador online razoável.
        self.assertLess(np.mean(errors_late), np.mean(errors_early))

    def test_rul_uncertainty_shrinks_as_data_accumulates(self) -> None:
        unit = Unit(unit_id="u1", fleet_id="f1", max_cycles=150, fault_mode=FaultMode.GRADUAL)
        readings = self.generator.generate_unit(unit, self.schema, DegradationParams(seed=3, noise_std=0.01))

        dt = self._build_dt()
        widths_early: list[float] = []
        widths_late: list[float] = []
        for r in readings:
            snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values)
            width = snap.rul.upper - snap.rul.lower
            if 10 <= r.cycle < 30:
                widths_early.append(width)
            elif r.cycle > 120:
                widths_late.append(width)

        self.assertLess(np.mean(widths_late), np.mean(widths_early))

    def test_anomaly_flag_fires_after_injected_abrupt_fault(self) -> None:
        unit = Unit(unit_id="u1", fleet_id="f1", max_cycles=150, fault_mode=FaultMode.ABRUPT)
        # abrupt_fault_rate alto o bastante para garantir o evento numa vida curta
        params = DegradationParams(seed=11, noise_std=0.01, abrupt_fault_rate=0.03, abrupt_fault_magnitude=0.6)
        readings = self.generator.generate_unit(unit, self.schema, params)

        dt = self._build_dt()
        anomaly_cycles = []
        for r in readings:
            snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values)
            if snap.is_anomaly:
                anomaly_cycles.append(r.cycle)

        self.assertGreater(len(anomaly_cycles), 0, "esperava pelo menos uma anomalia detectada")


if __name__ == "__main__":
    unittest.main()
