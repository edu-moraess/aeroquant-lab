"""
AeroQuant Lab — Dashboard unificado.

Abas: Digital Twin · Fleet · ML clássico · Neural Net · Monte Carlo
Tema: nativo do Streamlit (⋮ → Settings → Theme).

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
st.caption("Digital Twin · RUL · ML · Neural Net · Monte Carlo")

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


tab_twin, tab_fleet, tab_ml, tab_nn, tab_mc = st.tabs(
    ["Digital Twin", "Fleet", "ML clássico", "Neural Net", "Monte Carlo"]
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
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        with col2:
            st.markdown("**Health Index**")
            anom = df[df["anomaly"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["health_index"], name="HI", line=dict(color=THEME["SERIES_C"], width=2)))
            if len(anom):
                fig.add_trace(go.Scatter(x=anom["cycle"], y=anom["health_index"], mode="markers", name="Anomalia", marker=dict(color=THEME["ERROR"], size=8, symbol="x")))
            fig.add_hline(y=failure_threshold, line_dash="dash", line_color=THEME["SERIES_MUTED"], annotation_text="limiar", annotation_font_color=THEME["TEXT_SECONDARY"])
            fig.update_layout(**plotly_layout(THEME, height=360, x_title="Ciclo", y_title="HI"))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Residual**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["residual"], mode="lines", line=dict(color=THEME["SERIES_D"], width=1.5)))
            fig.add_hline(y=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=280, x_title="Ciclo", y_title="Ciclos", show_legend=False))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        with c4:
            st.markdown("**Incerteza OLS**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["uncertainty_half"], fill="tozeroy", fillcolor=THEME["FILL_ACCENT"], line=dict(color=THEME["SERIES_A"], width=1.5)))
            fig.update_layout(**plotly_layout(THEME, height=280, x_title="Ciclo", y_title="± ciclos", show_legend=False))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

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
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

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
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**Ranking**")
            rank = fleet_df.groupby("unit_id")["health_index"].last().sort_values(ascending=False).head(12).reset_index()
            rank.columns = ["Unidade", "HI"]
            rank["HI"] = rank["HI"].round(3)
            st.dataframe(rank, use_container_width=True, hide_index=True, height=300)
        with c6:
            st.markdown("**Distribuição**")
            fig = px.histogram(final_df, x="final_hi", nbins=14, color="fault_mode", barmode="overlay", opacity=0.7,
                               color_discrete_map={"ABRUPT": THEME["ERROR"], "GRADUAL": THEME["SERIES_A"]})
            fig.add_vline(x=failure_threshold, line_dash="dash", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=300, x_title="HI final", y_title="Contagem"))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

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
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

with tab_ml:
    methodology_block(
        info="Comparação de modelos de RUL em frota sintética.",
        method="Linear, Random Forest, GBM Quantile e MLP. Split por unidade. RMSE, MAE, NASA score.",
        interpretation="Menor RMSE = melhor ajuste. NASA penaliza superestimação. SHAP explica o modelo, não a física.",
        limitations="Só dados sintéticos. LSTM sequencial ainda não integrado nesta aba.",
        label="Sobre este painel",
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        ml_units = st.slider("Unidades", 12, 48, 24, 4, key="ml_u")
    with m2:
        ml_seed = st.number_input("Seed", value=2026, step=1, key="ml_s")
    with m3:
        ml_trees = st.slider("Árvores", 30, 120, 60, 10, key="ml_t")
    if st.button("Treinar", type="primary", key="ml_btn"):
        with st.spinner("Treinando..."):
            try:
                st.session_state["ml_result"] = run_ml_experiment(n_units=int(ml_units), seed=int(ml_seed), noise_std=noise_std, n_estimators=int(ml_trees))
                st.session_state.pop("shap_exp", None)
            except Exception as e:
                st.error("Falha no treino.")
                st.caption(str(e))
    if "ml_result" in st.session_state:
        res = st.session_state["ml_result"]
        st.caption(f"Treino {res.n_train_units} · Teste {res.n_test_units} · Features {res.n_features} · Melhor: {res.best_model_name}")
        st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Predito × Verdadeiro**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred_best, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name="Predições"))
            mx = float(max(res.test_true.max(), res.test_pred_best.max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal"))
            fig.update_layout(**plotly_layout(THEME, height=340, x_title="RUL verdadeiro", y_title="RUL previsto"))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        with cb:
            st.markdown("**Importância**")
            if res.feature_importance is not None:
                fig = go.Figure(go.Bar(x=res.feature_importance["importance"][::-1], y=res.feature_importance["feature"][::-1], orientation="h", marker_color=THEME["SERIES_B"]))
                fig.update_layout(**plotly_layout(THEME, height=340, x_title="Importância", show_legend=False))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            else:
                st.info("Disponível para RF e GBM.")
        residual = res.test_pred_best - res.test_true
        st.markdown("**Residual**")
        fig = go.Figure(go.Histogram(x=residual, nbinsx=36, marker_color=THEME["SERIES_D"]))
        fig.add_vline(x=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
        fig.update_layout(**plotly_layout(THEME, height=240, x_title="pred − true", show_legend=False))
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        st.markdown("**SHAP**")
        if res.trained_model is not None and res.X_test is not None:
            if st.button("Calcular SHAP", key="shap_btn"):
                with st.spinner("SHAP..."):
                    try:
                        st.session_state["shap_exp"] = explain_model(res.trained_model, res.X_test, max_samples=min(150, len(res.X_test)), local_index=0)
                    except Exception as e:
                        st.error("SHAP indisponível para este modelo.")
                        st.caption(str(e))
            if "shap_exp" in st.session_state:
                exp = st.session_state["shap_exp"]
                st.caption(f"{exp.method} · n={exp.n_samples_explained} · base={exp.base_value:.2f}")
                sx, sy = st.columns(2)
                with sx:
                    fig = go.Figure(go.Bar(x=exp.feature_importance["mean_abs_shap"][::-1], y=exp.feature_importance["feature"][::-1], orientation="h", marker_color=THEME["SERIES_C"]))
                    fig.update_layout(**plotly_layout(THEME, height=360, title="|SHAP| médio", x_title="mean |SHAP|", show_legend=False))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                with sy:
                    if exp.local_shap is not None:
                        colors = [THEME["ERROR"] if v < 0 else THEME["SUCCESS"] for v in exp.local_shap["shap_value"][::-1]]
                        fig = go.Figure(go.Bar(x=exp.local_shap["shap_value"][::-1], y=exp.local_shap["feature"][::-1], orientation="h", marker_color=colors))
                        fig.update_layout(**plotly_layout(THEME, height=360, title="Local", x_title="SHAP", show_legend=False))
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                    else:
                        st.info("Local requer TreeSHAP.")
        else:
            st.caption("Treine um modelo para habilitar SHAP.")
    else:
        st.info("Clique em **Treinar** para comparar modelos.")

with tab_nn:
    methodology_block(
        info="Redes neurais para RUL: MLP tabular e modelos sequenciais (janelas temporais).",
        method="MLP tabular: features engenheiradas. Sequence MLP: janelas (T×F) achatadas + MLP. LSTM: sequências com PyTorch (se instalado). Split por unidade.",
        interpretation="Compare RMSE/MAE entre modos. Sequence models capturam dependência temporal explícita; LSTM exige torch.",
        limitations="Dados sintéticos. LSTM opcional (torch). Sem Transformer nesta versão.",
        label="Sobre este painel",
    )

    mode = st.radio(
        "Modo",
        options=["MLP tabular", "Sequence MLP", "LSTM"],
        horizontal=True,
        key="nn_mode",
        help="LSTM requer PyTorch. No Streamlit Cloud use Sequence MLP se torch não estiver disponível.",
    )

    if mode == "MLP tabular":
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            nn_units = st.slider("Unidades", 12, 40, 24, 4, key="nn_u")
        with n2:
            nn_seed = st.number_input("Seed", value=2026, step=1, key="nn_s")
        with n3:
            nn_arch = st.selectbox("Arquitetura", options=["(32,)", "(64, 32)", "(128, 64)", "(64, 32, 16)"], index=1, key="nn_a")
        with n4:
            nn_iter = st.slider("Max iterações", 50, 400, 200, 25, key="nn_i")
        compare_bl = st.checkbox("Comparar com Linear e RF", value=True, key="nn_cmp")
        if st.button("Treinar MLP", type="primary", key="nn_btn"):
            hidden = tuple(int(x.strip()) for x in nn_arch.strip("()").split(",") if x.strip())
            with st.spinner("Treinando MLP tabular..."):
                try:
                    st.session_state["nn_result"] = run_nn_experiment(
                        n_units=int(nn_units), seed=int(nn_seed), noise_std=noise_std,
                        hidden=hidden, max_iter=int(nn_iter), compare_baselines=bool(compare_bl),
                    )
                    st.session_state.pop("seq_result", None)
                except Exception as e:
                    st.error("Falha no treino da rede neural.")
                    st.caption(str(e))
        if "nn_result" in st.session_state:
            res = st.session_state["nn_result"]
            st.caption(f"{res.architecture} · épocas {res.n_epochs} · treino {res.n_train_units} · teste {res.n_test_units} · features {res.n_features}")
            st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Predito × Verdadeiro (MLP)**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred_mlp, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name="MLP"))
                mx = float(max(res.test_true.max(), res.test_pred_mlp.max()))
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal"))
                fig.update_layout(**plotly_layout(THEME, height=340, x_title="RUL verdadeiro", y_title="RUL previsto"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            with c2:
                st.markdown("**Curva de perda**")
                if res.loss_curve:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=res.loss_curve, mode="lines", line=dict(color=THEME["SERIES_B"], width=2), name="loss"))
                    fig.update_layout(**plotly_layout(THEME, height=340, x_title="Época", y_title="Loss", show_legend=False))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                else:
                    st.info("Curva de perda indisponível.")
            residual = res.test_pred_mlp - res.test_true
            st.markdown("**Residual**")
            fig = go.Figure(go.Histogram(x=residual, nbinsx=36, marker_color=THEME["SERIES_D"]))
            fig.add_vline(x=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=240, x_title="pred − true", show_legend=False))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info("Configure e clique em **Treinar MLP**.")

    else:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            seq_units = st.slider("Unidades", 12, 36, 20, 4, key="seq_u")
        with s2:
            seq_len = st.slider("Janela T", 10, 50, 30, 5, key="seq_t")
        with s3:
            seq_seed = st.number_input("Seed", value=2026, step=1, key="seq_s")
        with s4:
            if mode == "LSTM":
                seq_epochs = st.slider("Épocas LSTM", 10, 80, 30, 5, key="seq_e")
            else:
                seq_iter = st.slider("Max iterações", 50, 300, 150, 25, key="seq_i")

        label = "Treinar LSTM" if mode == "LSTM" else "Treinar Sequence MLP"
        if st.button(label, type="primary", key="seq_btn"):
            with st.spinner("Treinando modelo sequencial..."):
                try:
                    kwargs = dict(
                        n_units=int(seq_units),
                        seed=int(seq_seed),
                        noise_std=noise_std,
                        seq_len=int(seq_len),
                        stride=2,
                        model="lstm" if mode == "LSTM" else "sequence_mlp",
                    )
                    if mode == "LSTM":
                        kwargs["lstm_epochs"] = int(seq_epochs)
                    else:
                        kwargs["max_iter"] = int(seq_iter)
                    st.session_state["seq_result"] = run_seq_experiment(**kwargs)
                    st.session_state.pop("nn_result", None)
                except Exception as e:
                    st.error("Falha no treino sequencial.")
                    st.caption(str(e))

        if "seq_result" in st.session_state:
            res = st.session_state["seq_result"]
            st.caption(
                f"{res.algorithm} · T={res.seq_len} · épocas {res.n_epochs} · "
                f"janelas treino {res.n_train_windows} · teste {res.n_test_windows} · "
                f"features {res.n_features} · torch={'sim' if res.torch_available else 'não'}"
            )
            st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Predito × Verdadeiro**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name=res.model_name))
                mx = float(max(res.test_true.max(), res.test_pred.max()))
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal"))
                fig.update_layout(**plotly_layout(THEME, height=340, x_title="RUL verdadeiro", y_title="RUL previsto"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            with c2:
                st.markdown("**Curva de perda**")
                if res.loss_curve:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=res.loss_curve, mode="lines", line=dict(color=THEME["SERIES_C"], width=2), name="loss"))
                    fig.update_layout(**plotly_layout(THEME, height=340, x_title="Época", y_title="Loss", show_legend=False))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                else:
                    st.info("Curva de perda indisponível.")
            residual = res.test_pred - res.test_true
            st.markdown("**Residual**")
            fig = go.Figure(go.Histogram(x=residual, nbinsx=36, marker_color=THEME["SERIES_D"]))
            fig.add_vline(x=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=240, x_title="pred − true", show_legend=False))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info(f"Configure e clique em **{label}**.")
            if mode == "LSTM":
                st.caption("LSTM requer o pacote `torch`. Se falhar no Cloud, use Sequence MLP.")

with tab_mc:
    methodology_block(
        info="Propagação de incerteza do RUL.",
        method="Aleatória: ruído/seed, limiar fixo. Epistêmica: trajetória fixa, percentil do limiar (30–70).",
        interpretation="Distribuição do RUL e variâncias aproximam H3. Não é Bayesiana formal.",
        limitations="Fontes não ortogonais. Custo cresce com n_runs.",
        label="Sobre este painel",
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        mc_runs = st.slider("Trajetórias", 8, 40, 16, 2, key="mc_r")
    with mc2:
        mc_life = st.slider("Vida útil", 80, 220, 140, 10, key="mc_l")
    with mc3:
        mc_frac = st.slider("Fração de vida", 0.4, 0.8, 0.6, 0.05, key="mc_f")
    with mc4:
        mc_seed = st.number_input("Seed", value=42, step=1, key="mc_s")
    if st.button("Executar", type="primary", key="mc_btn"):
        with st.spinner("Monte Carlo..."):
            try:
                st.session_state["mc_result"] = run_monte_carlo_rul(
                    n_runs=int(mc_runs), max_cycles=int(mc_life), reference_cycle_fraction=float(mc_frac),
                    base_seed=int(mc_seed), noise_std=noise_std, n_calibration_units=8)
            except Exception as e:
                st.error("Falha no Monte Carlo.")
                st.caption(str(e))
    if "mc_result" in st.session_state:
        r = st.session_state["mc_result"]
        if r.n_runs < 1 or not np.isfinite(r.mean):
            st.warning("Dados insuficientes para a distribuição de RUL.")
        else:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("RUL médio", f"{r.mean:.1f}")
            k2.metric("Desvio", f"{r.std:.1f}")
            k3.metric("Q05–Q95", f"{r.q05:.0f}–{r.q95:.0f}")
            k4.metric("Verdadeiro", f"{r.true_rul_at_ref:.0f}")
            k5.metric("Runs", r.n_runs)
            xa, xb = st.columns(2)
            with xa:
                st.markdown("**Distribuição**")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=r.rul_samples, nbinsx=18, marker_color=THEME["SERIES_A"]))
                fig.add_vline(x=r.true_rul_at_ref, line_dash="dash", line_color=THEME["ERROR"], annotation_text="verdadeiro", annotation_font_color=THEME["TEXT_SECONDARY"])
                fig.add_vline(x=r.mean, line_dash="solid", line_color=THEME["SERIES_B"], annotation_text="média", annotation_font_color=THEME["TEXT_SECONDARY"])
                fig.update_layout(**plotly_layout(THEME, height=320, x_title="RUL (ciclos)", show_legend=False))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            with xb:
                st.markdown("**Variância**")
                fig = go.Figure(go.Bar(x=["Total", "Aleatória", "Epistêmica"], y=[r.var_total, r.var_aleatoric, r.var_epistemic],
                                       marker_color=[THEME["SERIES_MUTED"], THEME["SERIES_A"], THEME["SERIES_B"]]))
                fig.update_layout(**plotly_layout(THEME, height=320, y_title="Variância", show_legend=False))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            tot = max(r.var_total, 1e-12)
            st.caption(f"Total {r.var_total:.1f} · Aleatória {r.var_aleatoric:.1f} ({100*r.var_aleatoric/tot:.0f}%) · Epistêmica {r.var_epistemic:.1f} ({100*r.var_epistemic/tot:.0f}%)")
    else:
        st.info("Configure e clique em **Executar**.")
