"""
AeroQuant Lab — Dashboard unificado (Streamlit).

Uma única aplicação. Sem multipage no sidebar.
Abas: Digital Twin · Fleet · ML RUL · Monte Carlo · Sobre.

    streamlit run dashboards/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboards"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
from aeroquant.uncertainty.monte_carlo_rul import run_monte_carlo_rul

from ml_experiment import run_ml_experiment

REPO_URL = "https://github.com/edu-moraess/aeroquant-lab"

st.set_page_config(
    page_title="AeroQuant Lab",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stMetric {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; border-radius: 8px; padding: 0.6rem;
    }
    .block-container { padding-top: 1.0rem; }
    section[data-testid="stSidebar"] { border-right: 1px solid #1e293b; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AeroQuant Lab")
st.caption(
    "Digital Twin · RUL com incerteza · ML · Monte Carlo — "
    "monitoramento de saúde de motores turbofan (C-MAPSS-like)."
)

with st.sidebar:
    st.markdown("### Simulação")
    max_cycles = st.slider("Vida útil (ciclos)", 60, 300, 170, 10)
    noise_std = st.slider("Ruído do sensor (σ)", 0.0, 0.05, 0.012, 0.001)
    abrupt_rate = st.slider("Taxa de falha abrupta", 0.0, 0.02, 0.005, 0.001)
    seed = st.number_input("Seed", value=42, step=1)

    st.markdown("### Digital Twin")
    coupling_threshold = st.slider("Limiar de acoplamento HI", 0.05, 0.5, 0.2, 0.05)
    confidence = st.select_slider(
        "Confiança do intervalo RUL",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x * 100)}%",
    )

    st.markdown("### Frota")
    n_units_fleet = st.slider("Unidades na frota", 8, 40, 16, 2)

    st.divider()
    st.caption(f"[Repositório]({REPO_URL})")
    st.caption("Python · Clean Architecture · DDD")

schema = build_cmapss_like_schema()
generator = StochasticSensorGenerator()


def _build_dt(coupling: float):
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling)
    failure_threshold = calibrate_failure_threshold(
        schema, OnlineFleetBaseline(), hi_estimator
    )
    dt = UpdateDigitalTwin(
        baseline_tracker=OnlineFleetBaseline(),
        hi_estimator=hi_estimator,
        rul_estimator=LinearExtrapolationRULEstimator(),
        repository=InMemoryDigitalTwinRepository(),
    )
    return dt, failure_threshold


tab_twin, tab_fleet, tab_ml, tab_mc, tab_about = st.tabs(
    ["Digital Twin", "Fleet View", "ML RUL", "Monte Carlo", "Sobre"]
)

with tab_twin:
    unit = Unit(
        unit_id="dashboard-unit", fleet_id="f1",
        max_cycles=int(max_cycles), fault_mode=FaultMode.ABRUPT,
    )
    params = DegradationParams(
        seed=int(seed), noise_std=noise_std,
        abrupt_fault_rate=abrupt_rate, abrupt_fault_magnitude=0.5,
    )
    readings = generator.generate_unit(unit, schema, params)
    dt, failure_threshold = _build_dt(coupling_threshold)

    rows = []
    last_snap = None
    for r in readings:
        snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
        last_snap = snap
        rows.append({
            "cycle": r.cycle, "true_rul": unit.max_cycles - r.cycle,
            "predicted_rul": snap.rul.point, "rul_lower": snap.rul.lower,
            "rul_upper": snap.rul.upper, "health_index": snap.health_index,
            "anomaly": snap.is_anomaly,
        })
    df = pd.DataFrame(rows)
    df["residual"] = df["predicted_rul"] - df["true_rul"]
    df["uncertainty_half"] = (df["rul_upper"] - df["rul_lower"]) / 2.0

    rul_width = float(df["rul_upper"].iloc[-1] - df["rul_lower"].iloc[-1])
    hi_now = float(last_snap.health_index) if last_snap else float("nan")
    mae = float(np.mean(np.abs(df["residual"])))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("RUL previsto", f"{df['predicted_rul'].iloc[-1]:.0f}")
    k2.metric("Health Index", f"{hi_now:.2f}")
    k3.metric("Incerteza RUL (±)", f"{rul_width / 2:.0f}")
    k4.metric("Anomalias", int(df["anomaly"].sum()))
    k5.metric("MAE residual", f"{mae:.1f}")
    st.caption(f"Limiar calibrado: **{failure_threshold:.3f}** · Confiança: **{int(confidence * 100)}%**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RUL previsto × verdadeiro")
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(x=df["cycle"], y=df["true_rul"], name="True RUL", line=dict(color="#38bdf8", width=2.5)))
        fig_rul.add_trace(go.Scatter(x=df["cycle"], y=df["predicted_rul"], name="Predicted RUL", line=dict(color="#f97316", width=2.5)))
        fig_rul.add_trace(go.Scatter(x=df["cycle"], y=df["rul_upper"], fill=None, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_rul.add_trace(go.Scatter(x=df["cycle"], y=df["rul_lower"], fill="tonexty", fillcolor="rgba(249, 115, 22, 0.18)", name=f"Intervalo {int(confidence * 100)}%", line=dict(width=0)))
        fig_rul.update_layout(height=400, template="plotly_dark", yaxis_title="ciclos restantes", xaxis_title="ciclo", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_rul, use_container_width=True)
    with col2:
        st.subheader("Health Index + anomalias")
        anom = df[df["anomaly"]]
        fig_hi = go.Figure()
        fig_hi.add_trace(go.Scatter(x=df["cycle"], y=df["health_index"], name="Health Index", line=dict(color="#22d3ee", width=2)))
        if len(anom):
            fig_hi.add_trace(go.Scatter(x=anom["cycle"], y=anom["health_index"], mode="markers", name="Anomalia", marker=dict(color="#ef4444", size=9, symbol="x")))
        fig_hi.add_hline(y=failure_threshold, line_dash="dash", line_color="#94a3b8", annotation_text="limiar calibrado")
        fig_hi.update_layout(height=400, template="plotly_dark", yaxis_title="HI", xaxis_title="ciclo", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_hi, use_container_width=True)

    st.subheader("Diagnóstico de predição")
    c3, c4 = st.columns(2)
    with c3:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=df["cycle"], y=df["residual"], mode="lines+markers", name="Residual", line=dict(color="#a78bfa", width=1.5), marker=dict(size=4)))
        fig_res.add_hline(y=0, line_dash="dot", line_color="#64748b")
        fig_res.update_layout(height=320, template="plotly_dark", title="Residual RUL", yaxis_title="ciclos", xaxis_title="ciclo", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_res, use_container_width=True)
    with c4:
        fig_unc = go.Figure()
        fig_unc.add_trace(go.Scatter(x=df["cycle"], y=df["uncertainty_half"], name="Meia-largura", fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.15)", line=dict(color="#38bdf8", width=2)))
        fig_unc.update_layout(height=320, template="plotly_dark", title="Incerteza OLS", yaxis_title="± ciclos", xaxis_title="ciclo", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_unc, use_container_width=True)

    st.subheader("Sensores")
    sensor_names = schema.names()
    default_sensors = [s.name for s in schema.sensors if s.degradation_coupling >= 0.3][:6]
    chosen = st.multiselect("Sensores (por acoplamento)", options=sensor_names, default=default_sensors)
    if chosen:
        fig_s = go.Figure()
        for name in chosen:
            spec = schema.spec_for(name)
            vals = [r.values[name] for r in readings]
            fig_s.add_trace(go.Scatter(x=df["cycle"], y=vals, name=f"{name} (κ={spec.degradation_coupling:.2f})", line=dict(width=1.4)))
        fig_s.update_layout(height=360, template="plotly_dark", yaxis_title="valor", xaxis_title="ciclo", margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_s, use_container_width=True)

with tab_fleet:
    st.subheader("Saúde agregada da frota")
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling_threshold)
    failure_threshold = calibrate_failure_threshold(schema, OnlineFleetBaseline(), hi_estimator)
    fleet_rows, final_rul = [], []
    for i in range(int(n_units_fleet)):
        u = Unit(unit_id=f"U{i:03d}", fleet_id="fleet-f1", max_cycles=150,
                 fault_mode=FaultMode.ABRUPT if i % 3 == 0 else FaultMode.GRADUAL)
        p = DegradationParams(seed=int(seed) + i, noise_std=noise_std,
                              abrupt_fault_rate=abrupt_rate, abrupt_fault_magnitude=0.5)
        rdgs = generator.generate_unit(u, schema, p)
        dt = UpdateDigitalTwin(baseline_tracker=OnlineFleetBaseline(), hi_estimator=hi_estimator,
                               rul_estimator=LinearExtrapolationRULEstimator(),
                               repository=InMemoryDigitalTwinRepository())
        last_hi = None
        for r in rdgs:
            snap = dt.ingest(u.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
            last_hi = snap.health_index
            fleet_rows.append({"unit_id": u.unit_id, "cycle": r.cycle, "health_index": snap.health_index,
                               "true_rul": u.max_cycles - r.cycle, "pred_rul": snap.rul.point})
        final_rul.append({"unit_id": u.unit_id, "final_hi": last_hi, "max_cycles": u.max_cycles,
                          "fault_mode": u.fault_mode.name})
    fleet_df = pd.DataFrame(fleet_rows)
    final_df = pd.DataFrame(final_rul)
    hm = fleet_df.pivot(index="unit_id", columns="cycle", values="health_index")
    fig_hm = go.Figure(go.Heatmap(
        z=hm.values, x=hm.columns, y=hm.index,
        colorscale=[[0.0, "#0ea5e9"], [0.4, "#22d3ee"], [0.6, "#facc15"], [0.8, "#f97316"], [1.0, "#dc2626"]],
        zmid=failure_threshold, colorbar=dict(title="HI"),
    ))
    fig_hm.update_layout(height=440, template="plotly_dark",
                         title=f"Heatmap Health Index · limiar ≈ {failure_threshold:.3f}",
                         xaxis_title="ciclo", yaxis_title="unidade", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_hm, use_container_width=True)
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Ranking — mais degradadas")
        last_hi = fleet_df.groupby("unit_id")["health_index"].last().sort_values(ascending=False).head(12).reset_index()
        last_hi.columns = ["unidade", "HI final"]
        st.dataframe(last_hi, use_container_width=True, hide_index=True)
    with c6:
        st.subheader("Distribuição de HI final")
        fig_hist = px.histogram(final_df, x="final_hi", nbins=15, color="fault_mode", barmode="overlay", opacity=0.75,
                                color_discrete_map={"ABRUPT": "#ef4444", "GRADUAL": "#38bdf8"}, template="plotly_dark")
        fig_hist.add_vline(x=failure_threshold, line_dash="dash", line_color="#94a3b8")
        fig_hist.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="HI final", yaxis_title="contagem")
        st.plotly_chart(fig_hist, use_container_width=True)
    st.subheader("Trajetórias médias por modo de falha")
    fleet_df = fleet_df.merge(final_df[["unit_id", "fault_mode"]], on="unit_id", how="left")
    traj = fleet_df.groupby(["fault_mode", "cycle"])["health_index"].agg(["mean", "std"]).reset_index()
    fig_traj = go.Figure()
    for mode, color in [("ABRUPT", "#ef4444"), ("GRADUAL", "#38bdf8")]:
        sub = traj[traj["fault_mode"] == mode]
        if sub.empty:
            continue
        fig_traj.add_trace(go.Scatter(x=sub["cycle"], y=sub["mean"], name=f"{mode} (média)", line=dict(color=color, width=2.5)))
        fig_traj.add_trace(go.Scatter(x=sub["cycle"], y=sub["mean"] + sub["std"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_traj.add_trace(go.Scatter(x=sub["cycle"], y=sub["mean"] - sub["std"], fill="tonexty",
                                      fillcolor=("rgba(239,68,68,0.12)" if mode == "ABRUPT" else "rgba(56,189,248,0.12)"),
                                      name=f"{mode} ±1σ", line=dict(width=0)))
    fig_traj.add_hline(y=failure_threshold, line_dash="dash", line_color="#94a3b8")
    fig_traj.update_layout(height=360, template="plotly_dark", xaxis_title="ciclo", yaxis_title="HI médio",
                           margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_traj, use_container_width=True)

with tab_ml:
    st.subheader("Comparação de modelos de RUL")
    st.caption("Linear · Random Forest · Gradient Boosting Quantile. Split por unidade. Dados sintéticos.")
    m1, m2, m3 = st.columns(3)
    with m1:
        ml_units = st.slider("Unidades na frota ML", 12, 48, 24, 4, key="ml_u")
    with m2:
        ml_seed = st.number_input("Seed ML", value=2026, step=1, key="ml_s")
    with m3:
        ml_trees = st.slider("n_estimators", 30, 120, 60, 10, key="ml_t")
    if st.button("Treinar e comparar", type="primary", key="ml_btn"):
        with st.spinner("Gerando frota + treinando modelos..."):
            st.session_state["ml_result"] = run_ml_experiment(
                n_units=int(ml_units), seed=int(ml_seed), noise_std=noise_std, n_estimators=int(ml_trees),
            )
    if "ml_result" in st.session_state:
        res = st.session_state["ml_result"]
        st.markdown(f"**Treino:** {res.n_train_units} un. · **Teste:** {res.n_test_units} un. · **Features:** {res.n_features} · **Melhor:** `{res.best_model_name}`")
        st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)
        ca, cb = st.columns(2)
        with ca:
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=res.test_true, y=res.test_pred_best, mode="markers", marker=dict(size=5, opacity=0.45, color="#38bdf8"), name="predições"))
            mx = float(max(res.test_true.max(), res.test_pred_best.max()))
            fig_sc.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#94a3b8", dash="dash"), name="ideal"))
            fig_sc.update_layout(height=380, template="plotly_dark", title=f"Predito × Verdadeiro — {res.best_model_name}", xaxis_title="RUL verdadeiro", yaxis_title="RUL previsto", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_sc, use_container_width=True)
        with cb:
            if res.feature_importance is not None:
                fig_fi = go.Figure(go.Bar(x=res.feature_importance["importance"][::-1], y=res.feature_importance["feature"][::-1], orientation="h", marker_color="#f97316"))
                fig_fi.update_layout(height=380, template="plotly_dark", title="Importância de features (top 15)", margin=dict(l=10, r=10, t=40, b=10), xaxis_title="importância")
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Importância disponível para RF e GBM.")
        residual = res.test_pred_best - res.test_true
        fig_rh = go.Figure(go.Histogram(x=residual, nbinsx=40, marker_color="#a78bfa"))
        fig_rh.add_vline(x=0, line_dash="dot", line_color="#94a3b8")
        fig_rh.update_layout(height=280, template="plotly_dark", title="Residual (pred − true)", xaxis_title="ciclos", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_rh, use_container_width=True)
    else:
        st.info("Clique em **Treinar e comparar** para executar o experimento.")

with tab_mc:
    st.subheader("Propagação de incerteza do RUL")
    st.caption("Decomposição empírica: **aleatória** (ruído/seed) vs **epistêmica** (percentil do limiar). Baseline H3 — não Bayesiana formal.")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        mc_runs = st.slider("Trajetórias", 8, 40, 16, 2, key="mc_r")
    with mc2:
        mc_life = st.slider("Vida útil MC", 80, 220, 140, 10, key="mc_l")
    with mc3:
        mc_frac = st.slider("Fração de vida (ref.)", 0.4, 0.8, 0.6, 0.05, key="mc_f")
    with mc4:
        mc_seed = st.number_input("Seed MC", value=42, step=1, key="mc_s")
    if st.button("Executar Monte Carlo", type="primary", key="mc_btn"):
        with st.spinner("Simulando trajetórias + decomposição de variância..."):
            st.session_state["mc_result"] = run_monte_carlo_rul(
                n_runs=int(mc_runs), max_cycles=int(mc_life),
                reference_cycle_fraction=float(mc_frac), base_seed=int(mc_seed),
                noise_std=noise_std, n_calibration_units=8,
            )
    if "mc_result" in st.session_state:
        r = st.session_state["mc_result"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("RUL médio", f"{r.mean:.1f}")
        k2.metric("Desvio", f"{r.std:.1f}")
        k3.metric("Q05–Q95", f"{r.q05:.0f}–{r.q95:.0f}")
        k4.metric("RUL verdadeiro", f"{r.true_rul_at_ref:.0f}")
        k5.metric("Runs", r.n_runs)
        xa, xb = st.columns(2)
        with xa:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=r.rul_samples, nbinsx=20, marker_color="#38bdf8", name="RUL MC"))
            fig.add_vline(x=r.true_rul_at_ref, line_dash="dash", line_color="#ef4444", annotation_text="verdadeiro")
            fig.add_vline(x=r.mean, line_dash="solid", line_color="#f97316", annotation_text="média")
            fig.update_layout(height=360, template="plotly_dark", title="Distribuição Monte Carlo do RUL", xaxis_title="RUL previsto (ciclos)", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with xb:
            fig2 = go.Figure(go.Bar(x=["total", "aleatória", "epistêmica"], y=[r.var_total, r.var_aleatoric, r.var_epistemic], marker_color=["#64748b", "#38bdf8", "#f97316"]))
            fig2.update_layout(height=360, template="plotly_dark", title="Decomposição empírica de variância (H3)", yaxis_title="variância", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        tot = max(r.var_total, 1e-12)
        st.markdown(f"**Variância total** = {r.var_total:.1f} · aleatória {r.var_aleatoric:.1f} ({100 * r.var_aleatoric / tot:.0f}%) · epistêmica {r.var_epistemic:.1f} ({100 * r.var_epistemic / tot:.0f}%)")
        st.caption("Aleatória: ruído + seed Gamma. Epistêmica: percentil do limiar (30–70). Fontes não ortogonais.")
    else:
        st.info("Configure parâmetros e clique em **Executar Monte Carlo**.")

with tab_about:
    st.subheader("AeroQuant Lab")
    st.markdown(f"""Plataforma de pesquisa em **Python** para monitoramento inteligente da saúde de aeronaves (turbofan C-MAPSS-like). Foco: **Digital Twin** + **RUL com incerteza** + manutenção preditiva.\n\n[Repositório GitHub]({REPO_URL}) · licença MIT.""")
    st.subheader("Status das fases")
    st.dataframe(pd.DataFrame([
        {"Fase": "1 — Arquitetura", "Status": "Concluída"},
        {"Fase": "2 — Pergunta científica", "Status": "Concluída"},
        {"Fase": "3 — Engenharia de dados", "Status": "Concluída"},
        {"Fase": "4 — Dados sintéticos", "Status": "Concluída (C-MAPSS real pendente)"},
        {"Fase": "5 — Digital Twin", "Status": "Concluída"},
        {"Fase": "6 — Machine Learning RUL", "Status": "Concluída (sklearn)"},
        {"Fase": "7 — Computer Vision", "Status": "Planejada"},
        {"Fase": "8 — Monte Carlo", "Status": "Concluída"},
        {"Fase": "9 — XAI (SHAP)", "Status": "Próxima"},
        {"Fase": "10 — Dashboard", "Status": "Esta aplicação"},
        {"Fase": "11 — MLOps", "Status": "Parcial"},
        {"Fase": "12 — Validação científica", "Status": "Planejada"},
        {"Fase": "13 — Publicação", "Status": "Planejada"},
    ]), use_container_width=True, hide_index=True)
    st.subheader("Arquitetura")
    st.markdown("""Clean Architecture por **Bounded Context** (DDD):\n\n- `sensor_data` — gerador estocástico, ETL, schema C-MAPSS-like\n- `digital_twin` — Welford baseline, HI z-score, RUL OLS, limiar calibrado\n- `ml` — trainers sklearn, split por unidade, métricas NASA\n- `uncertainty` — Monte Carlo com decomposição aleatória/epistêmica""")
    st.subheader("Limitações (honestidade científica)")
    st.markdown("""- Dados reais C-MAPSS ainda não carregados — adapter existe, validação pendente.\n- Decomposição de incerteza é **empírica**, não Bayesiana formal.\n- Repositório do Digital Twin é in-memory (demos e testes).\n- Deep learning (LSTM) e XAI (SHAP) ainda não integrados neste dashboard.""")
