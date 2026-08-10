"""
AeroQuant Lab — Dashboard unificado.

Abas: Digital Twin · Fleet · ML clássico · Neural Net · Anomalias · Monte Carlo · Risk

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
from aeroquant.xai.shap_explainer import explain_model
from anomaly_experiment import run_anomaly_experiment
from methodology_panel import render_methodology_sidebar
from ml_experiment import run_ml_experiment
from nn_experiment import run_nn_experiment
from seq_experiment import run_seq_experiment
from ui_theme import apply_global_css, get_theme, methodology_block, plotly_layout

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian",
        "toggleSpikelines", "zoomIn2d", "zoomOut2d",
    ],
    "toImageButtonOptions": {"format": "png", "filename": "aeroquant", "scale": 2},
}

REPO_URL = "https://github.com/edu-moraess/aeroquant-lab"

st.set_page_config(
    page_title="AeroQuant Lab",
    page_icon="AQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME = get_theme()
apply_global_css(THEME)

st.title("AeroQuant Lab")
st.caption("Digital Twin · RUL · ML · Neural Net · Anomalias · Monte Carlo · Risk")

_hero = _ROOT / "docs" / "assets" / "hero_3d.png"
if _hero.exists():
    st.image(str(_hero), width="stretch")
st.divider()

with st.sidebar:
    st.markdown("**Simulação**")
    max_cycles = st.slider("Vida útil (ciclos)", 60, 300, 170, 10)
    noise_std = st.slider("Ruído σ", 0.0, 0.05, 0.012, 0.001)
    abrupt_rate = st.slider("Falha abrupta", 0.0, 0.02, 0.005, 0.001)
    seed = st.number_input("Seed", value=42, step=1)

    st.markdown("**Modelo**")
    coupling_threshold = st.slider("Acoplamento HI", 0.05, 0.5, 0.2, 0.05)
    confidence = st.select_slider(
        "Confiança RUL (IC)",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x * 100)}%",
        help="Nível do intervalo de predição OLS do RUL.",
    )

    st.markdown("**Frota**")
    n_units_fleet = st.slider("Unidades", 8, 40, 16, 2)

    st.divider()
    render_methodology_sidebar()
    st.caption(f"[GitHub]({REPO_URL})")

schema = build_cmapss_like_schema()
generator = StochasticSensorGenerator()


def _build_dt(coupling: float, conf: float = 0.90):
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling)
    failure_threshold = calibrate_failure_threshold(
        schema, OnlineFleetBaseline(), hi_estimator
    )
    dt = UpdateDigitalTwin(
        baseline_tracker=OnlineFleetBaseline(),
        hi_estimator=hi_estimator,
        rul_estimator=LinearExtrapolationRULEstimator(confidence=float(conf)),
        repository=InMemoryDigitalTwinRepository(),
    )
    return dt, failure_threshold


def _safe_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


tab_twin, tab_fleet, tab_ml, tab_nn, tab_anom, tab_mc, tab_risk = st.tabs(
    ["Digital Twin", "Fleet", "ML clássico", "Neural Net", "Anomalias", "Monte Carlo", "Risk"]
)

with tab_twin:
    methodology_block(
        info="Monitoramento de uma unidade ao longo da vida útil sintética.",
        method="HI = z-score ponderado. RUL por extrapolação linear do HI até limiar calibrado, com IC OLS.",
        interpretation="RUL decresce com a degradação. Anomalias marcam saltos no HI. A banda alarga com a extrapolação.",
        limitations="Dados sintéticos. Acoplamento heurístico. Incerteza do HI data-driven (Welford n).",
        label="Sobre este painel",
    )

    unit = Unit(unit_id="dashboard-unit", fleet_id="f1", max_cycles=int(max_cycles), fault_mode=FaultMode.ABRUPT)
    params = DegradationParams(seed=int(seed), noise_std=noise_std, abrupt_fault_rate=abrupt_rate, abrupt_fault_magnitude=0.5)
    readings = generator.generate_unit(unit, schema, params)
    dt, failure_threshold = _build_dt(coupling_threshold, confidence)

    rows, last_snap = [], None
    for r in readings:
        snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
        last_snap = snap
        rows.append({
            "cycle": r.cycle, "true_rul": unit.max_cycles - r.cycle,
            "predicted_rul": snap.rul.point, "rul_lower": snap.rul.lower,
            "rul_upper": snap.rul.upper, "health_index": snap.health_index, "anomaly": snap.is_anomaly,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("Dados insuficientes para esta configuração.")
    else:
        df["residual"] = df["predicted_rul"] - df["true_rul"]
        df["uncertainty_half"] = (df["rul_upper"] - df["rul_lower"]) / 2.0
        rul_width = _safe_float(df["rul_upper"].iloc[-1] - df["rul_lower"].iloc[-1], 0.0)
        hi_now = _safe_float(last_snap.health_index if last_snap else None)
        mae = _safe_float(np.mean(np.abs(df["residual"])))

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("RUL", f"{_safe_float(df['predicted_rul'].iloc[-1]):.0f}")
        k2.metric("Health Index", f"{hi_now:.2f}")
        k3.metric("Incerteza (±)", f"{rul_width / 2:.0f}")
        k4.metric("Anomalias", int(df["anomaly"].sum()))
        k5.metric("MAE", f"{mae:.1f}")
        st.caption(f"Limiar: {failure_threshold:.3f} · Confiança: {int(confidence * 100)}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RUL**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["true_rul"], name="Verdadeiro", line=dict(color=THEME["SERIES_A"], width=2)))
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["predicted_rul"], name="Previsto", line=dict(color=THEME["SERIES_B"], width=2)))
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["rul_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["rul_lower"], fill="tonexty", fillcolor=THEME["FILL_ACCENT"], name=f"IC {int(confidence * 100)}%", line=dict(width=0)))
            fig.update_layout(**plotly_layout(THEME, height=360, x_title="Ciclo", y_title="Ciclos restantes"))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        with col2:
            st.markdown("**Health Index**")
            anom = df[df["anomaly"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["health_index"], name="HI", line=dict(color=THEME["SERIES_C"], width=2)))
            if len(anom):
                fig.add_trace(go.Scatter(x=anom["cycle"], y=anom["health_index"], mode="markers", name="Anomalia", marker=dict(color=THEME["ERROR"], size=8, symbol="x")))
            fig.add_hline(y=failure_threshold, line_dash="dash", line_color=THEME["SERIES_MUTED"], annotation_text="limiar", annotation_font_color=THEME["TEXT_SECONDARY"])
            fig.update_layout(**plotly_layout(THEME, height=360, x_title="Ciclo", y_title="HI"))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Residual**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["residual"], mode="lines", line=dict(color=THEME["SERIES_D"], width=1.5)))
            fig.add_hline(y=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=280, x_title="Ciclo", y_title="Ciclos", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        with c4:
            st.markdown("**Incerteza OLS**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["uncertainty_half"], fill="tozeroy", fillcolor=THEME["FILL_ACCENT"], line=dict(color=THEME["SERIES_A"], width=1.5)))
            fig.update_layout(**plotly_layout(THEME, height=280, x_title="Ciclo", y_title="± ciclos", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)

        st.markdown("**Sensores**")
        default_sensors = [s.name for s in schema.sensors if s.degradation_coupling >= 0.3][:5]
        chosen = st.multiselect("Sensores", options=schema.names(), default=default_sensors, label_visibility="collapsed")
        if chosen:
            fig = go.Figure()
            for name in chosen:
                spec = schema.spec_for(name)
                vals = [r.values[name] for r in readings]
                fig.add_trace(go.Scatter(x=df["cycle"], y=vals, name=f"{name} (κ={spec.degradation_coupling:.2f})", line=dict(width=1.3)))
            fig.update_layout(**plotly_layout(THEME, height=300, x_title="Ciclo", y_title="Valor"))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)

with tab_fleet:
    methodology_block(
        info="Saúde agregada da frota sintética.",
        method="Cada unidade usa o mesmo Digital Twin. Heatmap = HI por ciclo; ranking = HI final.",
        interpretation="Tons quentes = HI acima do limiar. ABRUPT e GRADUAL têm trajetórias distintas.",
        limitations="Vida fixa de 150 ciclos por unidade (sintético).",
        label="Sobre este painel",
    )
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling_threshold)
    failure_threshold = calibrate_failure_threshold(schema, OnlineFleetBaseline(), hi_estimator)
    fleet_rows, final_rul = [], []
    for i in range(int(n_units_fleet)):
        u = Unit(unit_id=f"U{i:03d}", fleet_id="fleet-f1", max_cycles=150,
                 fault_mode=FaultMode.ABRUPT if i % 3 == 0 else FaultMode.GRADUAL)
        p = DegradationParams(seed=int(seed) + i, noise_std=noise_std, abrupt_fault_rate=abrupt_rate, abrupt_fault_magnitude=0.5)
        rdgs = generator.generate_unit(u, schema, p)
        dt = UpdateDigitalTwin(baseline_tracker=OnlineFleetBaseline(), hi_estimator=hi_estimator,
                               rul_estimator=LinearExtrapolationRULEstimator(confidence=float(confidence)),
                               repository=InMemoryDigitalTwinRepository())
        last_hi = None
        for r in rdgs:
            snap = dt.ingest(u.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
            last_hi = snap.health_index
            fleet_rows.append({"unit_id": u.unit_id, "cycle": r.cycle, "health_index": snap.health_index})
        final_rul.append({"unit_id": u.unit_id, "final_hi": last_hi, "fault_mode": u.fault_mode.name})

    fleet_df = pd.DataFrame(fleet_rows)
    final_df = pd.DataFrame(final_rul)
    if fleet_df.empty:
        st.warning("Dados insuficientes para a frota.")
    else:
        st.markdown("**Heatmap HI**")
        hm = fleet_df.pivot(index="unit_id", columns="cycle", values="health_index")
        colorscale = [[0.0, THEME["SERIES_A"]], [0.45, THEME["SERIES_C"]], [0.65, THEME["WARNING"]], [0.85, THEME["SERIES_B"]], [1.0, THEME["ERROR"]]]
        fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale=colorscale, zmid=failure_threshold, colorbar=dict(title="HI", thickness=12)))
        fig.update_layout(**plotly_layout(THEME, height=380, title=f"Limiar ≈ {failure_threshold:.3f}", x_title="Ciclo", y_title="Unidade", show_legend=False))
        st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**Ranking**")
            rank = fleet_df.groupby("unit_id")["health_index"].last().sort_values(ascending=False).head(12).reset_index()
            rank.columns = ["Unidade", "HI"]
            rank["HI"] = rank["HI"].round(3)
            st.dataframe(rank, width='stretch', hide_index=True, height=300)
        with c6:
            st.markdown("**Distribuição**")
            fig = px.histogram(final_df, x="final_hi", nbins=14, color="fault_mode", barmode="overlay", opacity=0.7,
                               color_discrete_map={"ABRUPT": THEME["ERROR"], "GRADUAL": THEME["SERIES_A"]})
            fig.add_vline(x=failure_threshold, line_dash="dash", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=300, x_title="HI final", y_title="Contagem"))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)

        st.markdown("**Trajetória por modo**")
        fleet_df = fleet_df.merge(final_df[["unit_id", "fault_mode"]], on="unit_id", how="left")
        traj = fleet_df.groupby(["fault_mode", "cycle"])["health_index"].agg(["mean", "std"]).reset_index()
        fig = go.Figure()
        for mode, key in [("ABRUPT", "ERROR"), ("GRADUAL", "SERIES_A")]:
            sub = traj[traj["fault_mode"] == mode]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(x=sub["cycle"], y=sub["mean"], name=mode, line=dict(color=THEME[key], width=2)))
            fig.add_trace(go.Scatter(x=sub["cycle"], y=sub["mean"] + sub["std"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=sub["cycle"], y=sub["mean"] - sub["std"], fill="tonexty",
                                     fillcolor=THEME["FILL_WARN"] if mode == "ABRUPT" else THEME["FILL_ACCENT"], name=f"{mode} ±1σ", line=dict(width=0)))
        fig.add_hline(y=failure_threshold, line_dash="dash", line_color=THEME["SERIES_MUTED"])
        fig.update_layout(**plotly_layout(THEME, height=320, x_title="Ciclo", y_title="HI médio"))
        st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
