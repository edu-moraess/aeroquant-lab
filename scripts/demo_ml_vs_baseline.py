"""
Comparação justa entre os modelos de ML (Fase 6, batch — preveem RUL a
partir de um snapshot de features) e o baseline estatístico da Fase 5
(streaming — extrapolação linear do Health Index sobre o histórico
acumulado). Mesmas unidades de teste, mesma métrica (RMSE/MAE/NASA score).

Por que os dois modos de avaliação são diferentes mas o resultado é
comparável: o que importa para a Fase 12 (Validação Científica) é "dado
o que cada abordagem sabe até o ciclo N, qual o erro do RUL previsto para
o ciclo N" — não como cada uma chegou lá internamente.

Rodar com: PYTHONPATH=src python3 scripts/demo_ml_vs_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aeroquant.digital_twin.application.use_cases import UpdateDigitalTwin
from aeroquant.digital_twin.infrastructure.estimators.linear_extrapolation_rul import (
    LinearExtrapolationRULEstimator,
)
from aeroquant.digital_twin.infrastructure.estimators.threshold_calibration import (
    calibrate_failure_threshold,
)
from aeroquant.digital_twin.infrastructure.estimators.welford_fleet_baseline import OnlineFleetBaseline
from aeroquant.digital_twin.infrastructure.estimators.zscore_health_index import ZScoreHealthIndexEstimator
from aeroquant.digital_twin.infrastructure.repositories.in_memory_repository import (
    InMemoryDigitalTwinRepository,
)
from aeroquant.ml.application.use_cases import TrainAndCompareModels
from aeroquant.ml.infrastructure.evaluation.metrics import evaluate
from aeroquant.ml.infrastructure.splitting.group_split import split_by_unit
from aeroquant.ml.infrastructure.trainers.sklearn_trainers import (
    GradientBoostingQuantileTrainer,
    LinearRegressionTrainer,
    RandomForestTrainer,
)
from aeroquant.sensor_data.application.use_cases import GenerateSyntheticFleet
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
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

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()

    # 1) Frota (60 unidades) — mesma fonte de dados para ML e para o baseline.
    repo = CSVSensorRepository(str(OUT_DIR / "ml_fleet.csv"))
    GenerateSyntheticFleet(generator, repo).run(
        schema, DegradationParams(noise_std=0.015), n_units=60, lifetime_mean=170, lifetime_std=30, seed=2026
    )
    readings = repo.load()
    df = readings_to_dataframe(readings, schema)
    df = clean(df, schema)
    df = normalize(df, schema)
    df = engineer_features(df, schema, window=5)
    df = add_rul_labels(df, max_rul_cap=125)

    feature_cols = [c for c in df.columns if c.endswith("_z") or "_roll_" in c or "_delta_" in c]
    train_df, test_df = split_by_unit(df, test_fraction=0.3, seed=7)
    print(f"Treino: {train_df['unit_id'].nunique()} unidades | Teste: {test_df['unit_id'].nunique()} unidades")

    # 2) Modelos de ML (batch)
    ml_use_case = TrainAndCompareModels(
        {
            "linear_regression": LinearRegressionTrainer(),
            "random_forest": RandomForestTrainer(n_estimators=200),
            "gradient_boosting_quantile": GradientBoostingQuantileTrainer(n_estimators=200),
        }
    )
    ml_result = ml_use_case.run(train_df, test_df, feature_cols)

    # 3) Baseline da Fase 5 (streaming) — reavaliado nas MESMAS unidades de teste.
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=0.2)
    failure_threshold = calibrate_failure_threshold(schema, OnlineFleetBaseline(), hi_estimator, n_calibration_units=15)

    baseline_true, baseline_pred = [], []
    for unit_id in test_df["unit_id"].unique():
        unit_rows = test_df[test_df["unit_id"] == unit_id].sort_values("cycle")
        max_cycle = int(unit_rows["cycle"].max())
        unit = Unit(unit_id=str(unit_id), fleet_id="test", max_cycles=max_cycle, fault_mode=FaultMode.GRADUAL)

        dt = UpdateDigitalTwin(
            baseline_tracker=OnlineFleetBaseline(),
            hi_estimator=hi_estimator,
            rul_estimator=LinearExtrapolationRULEstimator(min_points=8, window=40),
            repository=InMemoryDigitalTwinRepository(),
            healthy_window_cycles=20,
        )
        for _, row in unit_rows.iterrows():
            sensor_values = {s.name: row[s.name] for s in schema.sensors}
            snap = dt.ingest(unit.unit_id, int(row["cycle"]), int(row["operating_condition"]), sensor_values, failure_threshold)
            baseline_true.append(row["rul"])
            baseline_pred.append(snap.rul.point)

    baseline_metrics = evaluate(np.array(baseline_true), np.array(baseline_pred))

    # 4) Comparação final
    print("\n=== Comparação final (mesmas unidades de teste) ===")
    print(f"{'Modelo':<32}{'RMSE':>10}{'MAE':>10}{'NASA score':>16}")
    print(f"{'baseline (Fase 5, streaming)':<32}{baseline_metrics.rmse:>10.2f}{baseline_metrics.mae:>10.2f}{baseline_metrics.nasa_score:>16.3e}")
    for name, m in ml_result.results.items():
        print(f"{name:<32}{m.rmse:>10.2f}{m.mae:>10.2f}{m.nasa_score:>16.3e}")
    print(
        "\nNota sobre o NASA score: é uma soma de exponenciais — extremamente\n"
        "sensível a poucas previsões catastroficamente erradas (é assim que a\n"
        "métrica foi desenhada: punir subestimação perigosa de forma severa).\n"
        "O baseline tem pelo menos uma previsão muito ruim numa cauda de vida\n"
        "(ver docs/architecture/fase5-digital-twin.md) que domina totalmente\n"
        "a soma. RMSE/MAE são mais interpretáveis para leitura rápida aqui;\n"
        "o NASA score gigante É o achado (não um erro de cálculo) — mostra\n"
        "que o baseline comete erros ocasionais muito mais graves que os\n"
        "modelos de ML, mesmo quando o RMSE médio já indica isso.\n"
    )

    best_ml = min(ml_result.results.items(), key=lambda kv: kv[1].rmse)
    verdict = "SUPERA" if best_ml[1].rmse < baseline_metrics.rmse else "NÃO supera"
    print(f"\nMelhor modelo de ML ({best_ml[0]}) {verdict} o baseline da Fase 5 em RMSE.")

    _plot_comparison(baseline_metrics, ml_result)


def _plot_comparison(baseline_metrics, ml_result) -> None:
    names = ["baseline (Fase 5)"] + list(ml_result.results.keys())
    rmses = [baseline_metrics.rmse] + [m.rmse for m in ml_result.results.values()]
    colors = ["tab:gray"] + ["tab:blue"] * len(ml_result.results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, rmses, color=colors)
    ax.set_ylabel("RMSE (ciclos)")
    ax.set_title("RUL: Baseline (Fase 5) vs. Modelos de ML (Fase 6) — mesmas unidades de teste")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out_path = OUT_DIR / "ml_vs_baseline_comparison.png"
    fig.savefig(out_path, dpi=130)
    print(f"Gráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()