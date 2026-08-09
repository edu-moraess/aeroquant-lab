"""
Página Streamlit — Monte Carlo RUL (Fase 8).

Propagação de incerteza com decomposição empírica aleatória vs epistêmica.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import plotly.graph_objects as go
import streamlit as st

from aeroquant.uncertainty.monte_carlo_rul import run_monte_carlo_rul

st.set_page_config(page_title="Monte Carlo RUL — AeroQuant Lab", page_icon="🎲", layout="wide")

st.title("Monte Carlo RUL — Fase 8")
st.caption(
    "Propagação de incerteza do RUL via Monte Carlo. "
    "Decomposição empírica: **aleatória** (ruído/seed de degradação) vs "
    "**epistêmica** (limiar de falha calibrado em percentis 30–70). "
    "Não é Bayesiana formal — baseline metodológico para H3."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    n_runs = st.slider("Trajetórias MC", 8, 50, 20, 2)
with c2:
    max_cycles = st.slider("Vida útil (ciclos)", 80, 220, 140, 10)
with c3:
    frac = st.slider("Fração de vida (ref.)", 0.4, 0.8, 0.6, 0.05)
with c4:
    seed = st.number_input("Seed", value=42, step=1)

st.warning(
    "Cada execução recalibra limiares e gera trajetórias — pode levar 30–90s "
    "dependendo de n_runs. Use valores menores para exploração rápida."
)

if st.button("Executar Monte Carlo", type="primary"):
    with st.spinner("Simulando trajetórias + decomposição de variância..."):
        result = run_monte_carlo_rul(
            n_runs=int(n_runs),
            max_cycles=int(max_cycles),
            reference_cycle_fraction=float(frac),
            base_seed=int(seed),
            n_calibration_units=8,
        )
        st.session_state["mc_result"] = result

if "mc_result" in st.session_state:
    r = st.session_state["mc_result"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("RUL médio MC", f"{r.mean:.1f}")
    k2.metric("Desvio", f"{r.std:.1f}")
    k3.metric("Q05–Q95", f"{r.q05:.0f}–{r.q95:.0f}")
    k4.metric("RUL verdadeiro", f"{r.true_rul_at_ref:.0f}")
    k5.metric("Runs", r.n_runs)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=r.rul_samples, nbinsx=20, marker_color="#38bdf8", name="RUL MC"))
        fig.add_vline(x=r.true_rul_at_ref, line_dash="dash", line_color="#ef4444", annotation_text="verdadeiro")
        fig.add_vline(x=r.mean, line_dash="solid", line_color="#f97316", annotation_text="média")
        fig.update_layout(
            height=380, template="plotly_dark", title="Distribuição Monte Carlo do RUL",
            xaxis_title="RUL previsto (ciclos)", margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        labels = ["total", "aleatória", "epistêmica"]
        vals = [r.var_total, r.var_aleatoric, r.var_epistemic]
        colors = ["#64748b", "#38bdf8", "#f97316"]
        fig2 = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors))
        fig2.update_layout(
            height=380, template="plotly_dark", title="Decomposição empírica de variância (H3)",
            yaxis_title="variância", margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    tot = max(r.var_total, 1e-12)
    st.markdown(
        f"**Variância total** = {r.var_total:.1f} · "
        f"aleatória {r.var_aleatoric:.1f} ({100*r.var_aleatoric/tot:.0f}%) · "
        f"epistêmica {r.var_epistemic:.1f} ({100*r.var_epistemic/tot:.0f}%)"
    )
    st.caption(
        "Aleatória: ruído de sensor + seed do processo Gamma. "
        "Epistêmica: percentil do limiar de falha calibrado (30–70). "
        "Soma das partes não precisa fechar 100% da total (fontes não ortogonais)."
    )
else:
    st.info("Configure parâmetros e clique em **Executar Monte Carlo**.")
