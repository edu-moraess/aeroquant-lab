"""
Demonstração fim-a-fim da Fase 5 (Digital Twin): simula streaming
cycle-by-cycle de uma unidade sintética (com uma falha abrupta injetada)
pelo Digital Twin, e plota RUL previsto (com intervalo) vs. RUL verdadeiro.

Rodar com: PYTHONPATH=src python3 scripts/demo_digital_twin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aeroquant.digital_twin.application.use_cases import UpdateDigitalTwin
from aeroquant.digital_twin.infrastructure.estimators.linear_extrapolation_rul import (
    LinearExtrapolationRULEstimator,
)
from aeroquant.digital_twin.infrastructure.estimators.welford_fleet_baseline import OnlineFleetBaseline
from aeroquant.digital_twin.infrastructure.estimators.zscore_health_index import ZScoreHealthIndexEstimator
from aeroquant.digital_twin.infrastructure.repositories.in_memory_repository import (
    InMemoryDigitalTwinRepository,
)
from aeroquant.digital_twin.infrastructure.estimators.threshold_calibration import (
    calibrate_failure_threshold,
)
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()

    unit = Unit(unit_id="dt-demo-unit", fleet_id="f1", max_cycles=170, fault_mode=FaultMode.ABRUPT)
    params = DegradationParams(seed=42, noise_std=0.012, abrupt_fault_rate=0.006, abrupt_fault_magnitude=0.5)
    readings = generator.generate_unit(unit, schema, params)

    # Calibra o limiar de falha empiricamente (ver threshold_calibration.py)
    # ANTES de instanciar o baseline que o Digital Twin vai usar de verdade,
    # mas usando um baseline_tracker separado dedicado à calibração — a
    # frota de calibração não deve contaminar o baseline da unidade
    # monitorada em produção.
    baseline_tracker = OnlineFleetBaseline()
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=0.2)
    failure_threshold = calibrate_failure_threshold(
        schema, OnlineFleetBaseline(), hi_estimator, n_calibration_units=15, percentile=50.0
    )
    print(f"Limiar de falha calibrado (mediana do HI em fim-de-vida, frota de referência): {failure_threshold:.3f}")

    dt = UpdateDigitalTwin(
        baseline_tracker=baseline_tracker,
        hi_estimator=hi_estimator,
        rul_estimator=LinearExtrapolationRULEstimator(min_points=8, window=40),
        repository=InMemoryDigitalTwinRepository(),
        healthy_window_cycles=20,
    )

    cycles, true_rul, pred_rul, lower, upper, anomaly_cycles = [], [], [], [], [], []
    for r in readings:
        snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold=failure_threshold)
        cycles.append(r.cycle)
        true_rul.append(unit.max_cycles - r.cycle)
        pred_rul.append(snap.rul.point)
        lower.append(snap.rul.lower)
        upper.append(snap.rul.upper)
        if snap.is_anomaly:
            anomaly_cycles.append(r.cycle)

    print(f"Ciclos de vida: {unit.max_cycles}")
    print(f"Anomalias detectadas em: {anomaly_cycles}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cycles, true_rul, label="RUL verdadeiro", color="black", linewidth=2)
    ax.plot(cycles, pred_rul, label="RUL previsto (Digital Twin)", color="tab:blue")
    ax.fill_between(cycles, lower, upper, color="tab:blue", alpha=0.2, label="Intervalo de predição (90%)")
    for c in anomaly_cycles:
        ax.axvline(c, color="tab:red", linestyle="--", alpha=0.6)
    if anomaly_cycles:
        ax.axvline(anomaly_cycles[0], color="tab:red", linestyle="--", alpha=0.6, label="Anomalia detectada")

    ax.set_xlabel("Ciclo")
    ax.set_ylabel("RUL")
    ax.set_ylim(0, 250)
    ax.set_title("Digital Twin — RUL previsto vs. verdadeiro (baseline estatístico)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = OUT_DIR / "digital_twin_rul_tracking.png"
    fig.savefig(out_path, dpi=130)
    print(f"Gráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
