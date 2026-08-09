"""
Monte Carlo de propagação de incerteza do RUL (Fase 8).

Objetivo científico (H3): separar, de forma empírica e reproduzível,

  - **Aleatória (aleatoric)**: variação induzida por ruído de sensor e
    stochasticidade do processo de degradação (seed do gerador).
  - **Epistêmica (epistemic)**: variação induzida por amostragem finita
    do baseline da frota (janela saudável / n do Welford) e pelo limiar
    de falha calibrado empiricamente.

Método:
  1. Gera N trajetórias sintéticas com parâmetros perturbados.
  2. Roda o Digital Twin (mesmo pipeline da Fase 5) em cada trajetória.
  3. Coleta RUL no ciclo de referência (fração da vida).
  4. Reporta quantis, média, desvio e uma decomposição empírica:

       Var_total ≈ Var_aleatoric + Var_epistemic

Honestidade: decomposição é aproximada (não Bayesiana formal).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeroquant.digital_twin.application.use_cases import UpdateDigitalTwin
from aeroquant.digital_twin.infrastructure.estimators.linear_extrapolation_rul import (
    LinearExtrapolationRULEstimator,
)
from aeroquant.digital_twin.infrastructure.estimators.threshold_calibration import (
    calibrate_failure_threshold,
)
from aeroquant.digital_twin.infrastructure.estimators.welford_fleet_baseline import (
    OnlineFleetBaseline,
)
from aeroquant.digital_twin.infrastructure.estimators.zscore_health_index import (
    ZScoreHealthIndexEstimator,
)
from aeroquant.digital_twin.infrastructure.repositories.in_memory_repository import (
    InMemoryDigitalTwinRepository,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)


@dataclass(frozen=True)
class MonteCarloRULResult:
    n_runs: int
    reference_cycle_fraction: float
    rul_samples: np.ndarray
    mean: float
    std: float
    q05: float
    q50: float
    q95: float
    var_total: float
    var_aleatoric: float
    var_epistemic: float
    true_rul_at_ref: float | None


def _run_single_dt(unit, readings, schema, failure_threshold, coupling=0.2, reference_cycle=None):
    hi_est = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling)
    dt = UpdateDigitalTwin(
        baseline_tracker=OnlineFleetBaseline(),
        hi_estimator=hi_est,
        rul_estimator=LinearExtrapolationRULEstimator(min_points=8, window=40),
        repository=InMemoryDigitalTwinRepository(),
        healthy_window_cycles=20,
    )
    last_rul = float("nan")
    target = reference_cycle if reference_cycle is not None else unit.max_cycles
    for r in readings:
        snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
        if r.cycle >= target:
            return float(snap.rul.point)
        last_rul = float(snap.rul.point)
    return last_rul


def run_monte_carlo_rul(
    n_runs: int = 40,
    base_seed: int = 42,
    max_cycles: int = 160,
    noise_std: float = 0.012,
    noise_std_jitter: float = 0.004,
    reference_cycle_fraction: float = 0.6,
    coupling: float = 0.2,
    n_calibration_units: int = 12,
) -> MonteCarloRULResult:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    ref_cycle = max(int(max_cycles * reference_cycle_fraction), 10)
    true_rul = float(max_cycles - ref_cycle)

    def _calibrate(seed_offset: int, percentile: float = 50.0) -> float:
        hi = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling)
        return calibrate_failure_threshold(
            schema, OnlineFleetBaseline(), hi,
            n_calibration_units=n_calibration_units,
            seed=2026 + seed_offset,
            percentile=percentile,
        )

    total_ruls = []
    rng = np.random.default_rng(base_seed)
    for i in range(n_runs):
        seed_i = base_seed + 1000 + i
        noise_i = float(np.clip(noise_std + rng.normal(0, noise_std_jitter), 0.001, 0.05))
        unit = Unit(unit_id=f"mc-{i:04d}", fleet_id="mc", max_cycles=max_cycles, fault_mode=FaultMode.GRADUAL)
        params = DegradationParams(seed=seed_i, noise_std=noise_i, abrupt_fault_rate=0.0, abrupt_fault_magnitude=0.0)
        readings = generator.generate_unit(unit, schema, params)
        thr = _calibrate(seed_offset=i * 17)
        total_ruls.append(_run_single_dt(unit, readings, schema, thr, coupling, ref_cycle))

    total = np.array(total_ruls, dtype=float)
    total = total[np.isfinite(total)]

    fixed_thr = _calibrate(seed_offset=0)
    aleatoric_ruls = []
    for i in range(n_runs):
        seed_i = base_seed + 2000 + i
        noise_i = float(np.clip(noise_std + rng.normal(0, noise_std_jitter), 0.001, 0.05))
        unit = Unit(unit_id=f"mc-a-{i:04d}", fleet_id="mc-a", max_cycles=max_cycles, fault_mode=FaultMode.GRADUAL)
        params = DegradationParams(seed=seed_i, noise_std=noise_i, abrupt_fault_rate=0.0, abrupt_fault_magnitude=0.0)
        readings = generator.generate_unit(unit, schema, params)
        aleatoric_ruls.append(_run_single_dt(unit, readings, schema, fixed_thr, coupling, ref_cycle))

    aleatoric = np.array(aleatoric_ruls, dtype=float)
    aleatoric = aleatoric[np.isfinite(aleatoric)]

    unit_fixed = Unit(unit_id="mc-epi-fixed", fleet_id="mc-e", max_cycles=max_cycles, fault_mode=FaultMode.GRADUAL)
    params_fixed = DegradationParams(seed=base_seed + 9999, noise_std=noise_std, abrupt_fault_rate=0.0, abrupt_fault_magnitude=0.0)
    readings_fixed = generator.generate_unit(unit_fixed, schema, params_fixed)
    epistemic_ruls = []
    percentiles = np.linspace(30.0, 70.0, n_runs)
    for i in range(n_runs):
        thr = _calibrate(seed_offset=i * 31 + 7, percentile=float(percentiles[i]))
        epistemic_ruls.append(_run_single_dt(unit_fixed, readings_fixed, schema, thr, coupling, ref_cycle))

    epistemic = np.array(epistemic_ruls, dtype=float)
    epistemic = epistemic[np.isfinite(epistemic)]

    var_total = float(np.var(total)) if len(total) > 1 else 0.0
    var_aleatoric = float(np.var(aleatoric)) if len(aleatoric) > 1 else 0.0
    var_epistemic = float(np.var(epistemic)) if len(epistemic) > 1 else 0.0

    return MonteCarloRULResult(
        n_runs=len(total),
        reference_cycle_fraction=reference_cycle_fraction,
        rul_samples=total,
        mean=float(np.mean(total)) if len(total) else float("nan"),
        std=float(np.std(total)) if len(total) else float("nan"),
        q05=float(np.percentile(total, 5)) if len(total) else float("nan"),
        q50=float(np.percentile(total, 50)) if len(total) else float("nan"),
        q95=float(np.percentile(total, 95)) if len(total) else float("nan"),
        var_total=var_total,
        var_aleatoric=var_aleatoric,
        var_epistemic=var_epistemic,
        true_rul_at_ref=true_rul,
    )
