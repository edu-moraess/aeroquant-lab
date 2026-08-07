"""
Demonstração fim-a-fim da Fase 3 + Fase 4 (Nível 1):
gerar frota sintética -> checar qualidade -> ETL -> features prontas para ML.

Rodar com: PYTHONPATH=src python3 scripts/demo_end_to_end.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aeroquant.sensor_data.application.use_cases import GenerateSyntheticFleet
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
from aeroquant.sensor_data.infrastructure.quality.checks import DataQualityChecker
from aeroquant.sensor_data.infrastructure.repositories.csv_repository import CSVSensorRepository

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    repo = CSVSensorRepository(str(OUT_DIR / "synthetic_fleet.csv"))

    use_case = GenerateSyntheticFleet(generator, repo)
    params = DegradationParams(noise_std=0.015, abrupt_fault_rate=0.001, intermittent_fault_prob=0.01)
    result = use_case.run(schema, params, n_units=30, lifetime_mean=180, lifetime_std=35, seed=123)

    print("=== Geração sintética ===")
    print(f"Unidades: {result.n_units} | Leituras: {result.n_readings}")
    print(f"Vida útil média: {result.lifetime_mean:.1f} ciclos (desvio: {result.lifetime_std:.1f})")

    readings = repo.load()
    df = readings_to_dataframe(readings, schema)

    print("\n=== Qualidade dos dados ===")
    checker = DataQualityChecker(schema)
    report = checker.run(df)
    print(f"Linhas: {report.n_rows} | Passou: {report.passed} | Issues: {len(report.issues)}")
    for issue in report.issues[:10]:
        print(f"  [{issue.severity}] {issue.check}: {issue.detail}")

    print("\n=== ETL ===")
    df = clean(df, schema)
    df = normalize(df, schema)
    df = engineer_features(df, schema, window=5)
    df = add_rul_labels(df, max_rul_cap=125)
    candidate_cols = [f"{s.name}_z" for s in schema.sensors]
    selected = select_features(df, candidate_cols, min_abs_corr_with_rul=0.05)
    print(f"Features candidatas: {len(candidate_cols)} | Selecionadas (|corr(RUL)|>=0.05): {len(selected)}")
    print(f"Selecionadas: {selected}")

    df.to_csv(OUT_DIR / "featurized_dataset.csv", index=False)
    print(f"\nDataset final: {df.shape[0]} linhas x {df.shape[1]} colunas -> {OUT_DIR / 'featurized_dataset.csv'}")

    _plot_sample_unit(df, schema)


def _plot_sample_unit(df, schema) -> None:
    unit_id = df["unit_id"].iloc[0]
    unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(unit_df["cycle"], unit_df["sensor_4"], label="sensor_4 (bruto)", color="tab:blue")
    axes[0].plot(unit_df["cycle"], unit_df["sensor_2"], label="sensor_2 (bruto)", color="tab:orange", alpha=0.7)
    axes[0].set_ylabel("Valor do sensor")
    axes[0].set_title(f"Trajetória de degradação sintética — {unit_id}")
    axes[0].legend(loc="upper right")

    axes[1].plot(unit_df["cycle"], unit_df["rul"], color="tab:red")
    axes[1].set_ylabel("RUL (rótulo, cap=125)")
    axes[1].set_xlabel("Ciclo")

    fig.tight_layout()
    out_path = OUT_DIR / "sample_unit_degradation.png"
    fig.savefig(out_path, dpi=130)
    print(f"Gráfico salvo em: {out_path}")


if __name__ == "__main__":
    main()
