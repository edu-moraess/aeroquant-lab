"""
Página Streamlit — ML RUL (Fase 6 experimental).

Comparação interativa dos 3 modelos scikit-learn da Fase 6 sobre frota
sintética. Não substitui scripts/demo_ml_vs_baseline.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboards"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ml_experiment import run_ml_experiment

st.set_page_config(page_title="ML RUL — AeroQuant Lab", page_icon="🧠", layout="wide")

st.title("ML RUL — Fase 6 (experimental)")
st.caption(
    "Comparação Linear Regression · Random Forest · Gradient Boosting Quantile "
    "sobre frota **sintética**. Split por unidade (sem vazamento temporal). "
    "Dados reais C-MAPSS ainda não carregados."
)

c1, c2, c3 = st.columns(3)
with c1:
    n_units = st.slider("Unidades na frota", 12, 60, 28, 4)
with c2:
    seed = st.number_input("Seed", value=2026, step=1)
with c3:
    n_estimators = st.slider("n_estimators (RF/GBM)", 30, 150, 60, 10)

if st.button("Treinar e comparar", type="primary") or "ml_page_result" in st.session_state:
    if st.session_state.get("ml_page_params") != (n_units, seed, n_estimators):
        with st.spinner("Gerando frota + treinando modelos..."):
            st.session_state["ml_page_result"] = run_ml_experiment(
                n_units=int(n_units),
                seed=int(seed),
                n_estimators=int(n_estimators),
            )
            st.session_state["ml_page_params"] = (n_units, seed, n_estimators)

    res = st.session_state["ml_page_result"]

    st.markdown(
        f"**Treino:** {res.n_train_units} unidades · **Teste:** {res.n_test_units} · "
        f"**Features:** {res.n_features} · **Melhor (RMSE):** `{res.best_model_name}`"
    )
    st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_sc = go.Figure()
        fig_sc.add_trace(
            go.Scatter(
                x=res.test_true,
                y=res.test_pred_best,
                mode="markers",
                marker=dict(size=5, opacity=0.45, color="#38bdf8"),
                name="predições",
            )
        )
        mx = float(max(res.test_true.max(), res.test_pred_best.max()))
        fig_sc.add_trace(
            go.Scatter(
                x=[0, mx], y=[0, mx], mode="lines",
                line=dict(color="#94a3b8", dash="dash"), name="ideal",
            )
        )
        fig_sc.update_layout(
            height=400, template="plotly_dark",
            title=f"Predito × Verdadeiro — {res.best_model_name}",
            xaxis_title="RUL verdadeiro", yaxis_title="RUL previsto",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        if res.feature_importance is not None:
            fig_fi = go.Figure(
                go.Bar(
                    x=res.feature_importance["importance"][::-1],
                    y=res.feature_importance["feature"][::-1],
                    orientation="h", marker_color="#f97316",
                )
            )
            fig_fi.update_layout(
                height=400, template="plotly_dark",
                title="Importância de features (top 15)",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="importância",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Importância disponível para RF e GBM.")

    residual = res.test_pred_best - res.test_true
    fig_hist = go.Figure(go.Histogram(x=residual, nbinsx=40, marker_color="#a78bfa"))
    fig_hist.add_vline(x=0, line_dash="dot", line_color="#94a3b8")
    fig_hist.update_layout(
        height=300, template="plotly_dark",
        title="Distribuição do residual (pred − true)",
        xaxis_title="ciclos", margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "Nota: métrica NASA é assimétrica (penaliza superestimação de RUL). "
        "Comparação formal streaming baseline vs ML: `scripts/demo_ml_vs_baseline.py`."
    )
else:
    st.info("Ajuste os parâmetros e clique em **Treinar e comparar**.")
