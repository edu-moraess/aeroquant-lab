"""
Demo Fase 8 — Monte Carlo de RUL com decomposição empírica de incerteza.

Rodar:
    PYTHONPATH=src python3 scripts/demo_monte_carlo_rul.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aeroquant.uncertainty.monte_carlo_rul import run_monte_carlo_rul

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    print("Monte Carlo RUL (Fase 8) — propagação de incerteza...")
    result = run_monte_carlo_rul(
        n_runs=30,
        max_cycles=160,
        reference_cycle_fraction=0.6,
        n_calibration_units=10,
        base_seed=42,
    )

    print(f"\nRuns válidos: {result.n_runs}")
    print(f"Ciclo de referência: {result.reference_cycle_fraction:.0%} da vida")
    print(f"RUL verdadeiro no ref: {result.true_rul_at_ref:.1f} ciclos")
    print(f"RUL previsto  mean±std: {result.mean:.1f} ± {result.std:.1f}")
    print(f"Quantis 5/50/95%: {result.q05:.1f} / {result.q50:.1f} / {result.q95:.1f}")
    print(f"\nVariância total:     {result.var_total:.2f}")
    print(f"  ≈ aleatória:       {result.var_aleatoric:.2f}  ({100*result.var_aleatoric/max(result.var_total,1e-9):.0f}%)")
    print(f"  ≈ epistêmica:      {result.var_epistemic:.2f}  ({100*result.var_epistemic/max(result.var_total,1e-9):.0f}%)")
    print(
        "\nNota: decomposição empírica (não Bayesiana). "
        "Aleatória = ruído/seed; epistêmica = limiar de falha (percentil de calibração)."
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(result.rul_samples, bins=15, color="steelblue", edgecolor="white")
    axes[0].axvline(result.true_rul_at_ref, color="crimson", ls="--", label="RUL verdadeiro")
    axes[0].axvline(result.mean, color="orange", ls="-", label="média MC")
    axes[0].set_xlabel("RUL previsto (ciclos)")
    axes[0].set_ylabel("contagem")
    axes[0].set_title("Distribuição Monte Carlo do RUL")
    axes[0].legend()

    labels = ["total", "aleatória", "epistêmica"]
    vals = [result.var_total, result.var_aleatoric, result.var_epistemic]
    axes[1].bar(labels, vals, color=["#64748b", "#38bdf8", "#f97316"])
    axes[1].set_ylabel("variância")
    axes[1].set_title("Decomposição empírica de variância (H3)")
    fig.tight_layout()
    out = OUT / "monte_carlo_rul.png"
    fig.savefig(out, dpi=130)
    print(f"\nGráfico: {out}")


if __name__ == "__main__":
    main()
