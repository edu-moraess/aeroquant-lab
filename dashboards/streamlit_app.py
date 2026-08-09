"""
Dashboard AeroQuant Lab — Fase 10 (Dashboard) alinhada às Fases 1–5:
frota sintética estilo C-MAPSS + Digital Twin com quantificação de incerteza.

Validação local:
    pip install -r requirements/dashboard.txt   # ou requirements.txt (Cloud)
    streamlit run dashboards/streamlit_app.py

Componentes atuais (honestidade científica — apenas o que existe):
- Digital Twin unitário: KPIs, RUL com banda OLS, Health Index + anomalias,
  residual predicted−true, evolução da incerteza e explorador de sensores.
- Fleet View: heatmap HI, ranking de degradação, distribuição de RUL final,
  trajetórias agregadas.
- Sobre: status das fases, arquitetura Clean/DDD e limitações explícitas.

Fases 6+ (ML RUL, CV, XAI, Monte Carlo) permanecem fora do dashboard
até implementadas e testadas — sem placeholders enganosos.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
from aeroquant.sensor_data.domain.entities import FaultMode, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)

REPO_URL = "https://github.com/edu-moraess/aeroquant-lab"

# ---------------------------------------------------------------------------
# Tema visual — estética técnica / aeroespacial (dark-friendly)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroQuant Lab",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stMetric { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border: 1px solid #334155; border-radius: 8px; padding: 0.6rem; }
    .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AeroQuant Lab — Digital Twin & Frota Sintética")
st.caption(
    "Monitoramento de saúde de motores turbofan (benchmark C-MAPSS-like) com "
    "Digital Twin, Health Index z-score ponderado e RUL por extrapolação linear "
    "com intervalo de predição OLS. Incerteza é cidadã de primeira classe."
)

with st.expander("Arquitetura do pipeline (Fases 1–5)"):
    st.markdown(
        f"""
        1. **Gerador estocástico** (`StochasticSensorGenerator`) — 21 sensores,
           degradação via processo Gamma, deriva de calibração, ruído gaussiano
           e falhas abruptas/intermitentes.
        2. **Baseline online** (Welford) da frota saudável.
        3. **Health Index** = z-score ponderado pelo acoplamento de cada sensor
           à degradação; anomalias detectadas nos incrementos do HI.
        4. **RUL** por extrapolação linear do HI até limiar de falha
           **calibrado empiricamente** (não arbitrário), com banda de incerteza OLS.
        5. Clean Architecture por Bounded Context (`sensor_data`, `digital_twin`).

        Repositório: [{REPO_URL}]({REPO_URL})
        """
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Simulação unitária")
    max_cycles = st.slider("Vida útil da unidade (ciclos)", 60, 300, 170, 10)
    noise_std = st.slider("Ruído do sensor (σ)", 0.0, 0.05, 0.012, 0.001)
    abrupt_rate = st.slider("Taxa de falha abrupta", 0.0, 0.02, 0.005, 0.001)
    seed = st.number_input("Seed unitária", value=42, step=1)

    st.header("Digital Twin")
    coupling_threshold = st.slider("Limiar de acoplamento HI", 0.05, 0.5, 0.2, 0.05)
    confidence = st.select_slider(
        "Confiança do intervalo RUL",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x * 100)}%",
    )

    st.header("Fleet View")
    n_units_fleet = st.slider("Unidades na frota", 8, 40, 20, 2)
    fleet_seed = st.number_input("Seed da frota", value=7, step=1)

    st.divider()
    st.caption("Apenas Fases 1–5. ML (Fase 6+) fora deste dashboard.")

# ---------------------------------------------------------------------------
# Componentes compartilhados
# ---------------------------------------------------------------------------
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
    return dt, failure_threshold, hi_estimator


# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
tab_twin, tab_fleet, tab_about = st.tabs(
    ["Digital Twin (unitário)", "Fleet View", "Sobre o Projeto"]
)

# ===========================================================================
# Aba 1 — Digital Twin unitário
# ===========================================================================
with tab_twin:
    unit = Unit(
        unit_id="dashboard-unit",
        fleet_id="f1",
        max_cycles=int(max_cycles),
        fault_mode=FaultMode.ABRUPT,
    )
    params = DegradationParams(
        seed=int(seed),
        noise_std=noise_std,
        abrupt_fault_rate=abrupt_rate,
        abrupt_fault_magnitude=0.5,
    )
    readings = generator.generate_unit(unit, schema, params)

    dt, failure_threshold, _ = _build_dt(coupling_threshold)

    rows = []
    last_snap = None
    for r in readings:
        snap = dt.ingest(
            unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold
        )
        last_snap = snap
        rows.append(
            {
                "cycle": r.cycle,
                "true_rul": unit.max_cycles - r.cycle,
                "predicted_rul": snap.rul.point,
                "rul_lower": snap.rul.lower,
                "rul_upper": snap.rul.upper,
                "health_index": snap.health_index,
                "anomaly": snap.is_anomaly,
            }
        )
    df = pd.DataFrame(rows)
    df["residual"] = df["predicted_rul"] - df["true_rul"]
    df["uncertainty_half"] = (df["rul_upper"] - df["rul_lower"]) / 2.0

    # KPIs
    rul_width = float(df["rul_upper"].iloc[-1] - df["rul_lower"].iloc[-1])
    hi_now = float(last_snap.health_index) if last_snap else float("nan")
    mae = float(np.mean(np.abs(df["residual"])))
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        st.metric(
            "RUL previsto (ciclos)",
            f"{df['predicted_rul'].iloc[-1]:.0f}",
            delta=f"{-df['predicted_rul'].diff().iloc[-1]:.1f}",
        )
    with col_k2:
        st.metric("Health Index", f"{hi_now:.2f}", delta=f"{hi_now - df['health_index'].iloc[0]:.2f}")
    with col_k3:
        st.metric("Incerteza RUL (±)", f"{rul_width / 2:.0f}")
    with col_k4:
        st.metric("Anomalias", int(df["anomaly"].sum()))
    with col_k5:
        st.metric("MAE residual", f"{mae:.1f}")

    st.caption(
        f"Limiar de falha calibrado empiricamente: **{failure_threshold:.3f}** · "
        f"Confiança do intervalo: **{int(confidence * 100)}%**"
    )

    # --- RUL + HI lado a lado ------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RUL previsto × verdadeiro + banda de incerteza")
        fig_rul = go.Figure()
        fig_rul.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df["true_rul"],
                name="True RUL",
                line=dict(color="#38bdf8", width=2.5),
            )
        )
        fig_rul.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df["predicted_rul"],
                name="Predicted RUL",
                line=dict(color="#f97316", width=2.5),
            )
        )
        fig_rul.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df["rul_upper"],
                fill=None,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_rul.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df["rul_lower"],
                fill="tonexty",
                fillcolor="rgba(249, 115, 22, 0.18)",
                name=f"Intervalo {int(confidence * 100)}%",
                line=dict(width=0),
            )
        )
        fig_rul.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="ciclos restantes",
            xaxis_title="ciclo",
            template="plotly_dark",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_rul, use_container_width=True)

    with col2:
        st.subheader("Health Index + anomalias + limiar calibrado")
        anom = df[df["anomaly"]]
        fig_hi = go.Figure()
        fig_hi.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df["health_index"],
                name="Health Index",
                line=dict(color="#22d3ee", width=2),
            )
        )
        if len(anom):
            fig_hi.add_trace(
                go.Scatter(
                    x=anom["cycle"],
                    y=anom["health_index"],
                    mode="markers",
                    name="Anomalia",
                    marker=dict(color="#ef4444", size=9, symbol="x"),
                )
            )
        fig_hi.add_hline(
            y=failure_threshold,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text="limiar calibrado",
            annotation_position="top left",
        )
        fig_hi.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="HI",
            xaxis_title="ciclo",
            template="plotly_dark",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_hi, use_container_width=True)

    # --- Análise de residual e dinâmica de incerteza -------------------------
    st.subheader("Diagnóstico de predição")
    c3, c4 = st.columns(2)
    with c3:
        fig_res = go.Figure()
        fig_res.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df["residual"],
                mode="lines+markers",
                name="Residual (pred − true)",
                line=dict(color="#a78bfa", width=1.5),
                marker=dict(size=4),
            )
        )
        fig_res.add_hline(y=0, line_dash="dot", line_color="#64748b")
        fig_res.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="residual (ciclos)",
            xaxis_title="ciclo",
            template="plotly_dark",
            title="Residual RUL ao longo da vida",
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with c4:
        fig_unc = go.Figure()
        fig_unc.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df["uncertainty_half"],
                name="Meia-largura do intervalo",
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color="#38bdf8", width=2),
            )
        )
        fig_unc.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="± ciclos",
            xaxis_title="ciclo",
            template="plotly_dark",
            title="Evolução da incerteza do RUL (OLS)",
        )
        st.plotly_chart(fig_unc, use_container_width=True)

    # --- Explorador de sensores ----------------------------------------------
    st.subheader("Explorador de sensores brutos")
    sensor_names = schema.names()
    default_sensors = [
        s.name for s in schema.sensors if s.degradation_coupling >= 0.3
    ][:6]
    chosen = st.multiselect(
        "Sensores (ordenados por acoplamento à degradação)",
        options=sensor_names,
        default=default_sensors,
    )
    if chosen:
        fig_s = go.Figure()
        for name in chosen:
            spec = schema.spec_for(name)
            vals = [r.values[name] for r in readings]
            fig_s.add_trace(
                go.Scatter(
                    x=df["cycle"],
                    y=vals,
                    name=f"{name} (κ={spec.degradation_coupling:.2f})",
                    line=dict(width=1.4),
                )
            )
        fig_s.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="valor do sensor",
            xaxis_title="ciclo",
            template="plotly_dark",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("Selecione ao menos um sensor.")

# ===========================================================================
# Aba 2 — Fleet View
# ===========================================================================
with tab_fleet:
    st.subheader("Saúde agregada da frota")

    hi_estimator = ZScoreHealthIndexEstimator(
        schema, coupling_threshold=coupling_threshold
    )
    failure_threshold = calibrate_failure_threshold(
        schema, OnlineFleetBaseline(), hi_estimator
    )

    fleet_rows = []
    final_rul = []
    for i in range(int(n_units_fleet)):
        u = Unit(
            unit_id=f"U{i:03d}",
            fleet_id="fleet-f1",
            max_cycles=150,
            fault_mode=FaultMode.ABRUPT if i % 3 == 0 else FaultMode.GRADUAL,
        )
        p = DegradationParams(
            seed=int(fleet_seed) + i,
            noise_std=noise_std,
            abrupt_fault_rate=abrupt_rate,
            abrupt_fault_magnitude=0.5,
        )
        rdgs = generator.generate_unit(u, schema, p)
        dt = UpdateDigitalTwin(
            baseline_tracker=OnlineFleetBaseline(),
            hi_estimator=hi_estimator,
            rul_estimator=LinearExtrapolationRULEstimator(),
            repository=InMemoryDigitalTwinRepository(),
        )
        last_hi = None
        for r in rdgs:
            snap = dt.ingest(
                u.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold
            )
            last_hi = snap.health_index
            fleet_rows.append(
                {
                    "unit_id": u.unit_id,
                    "cycle": r.cycle,
                    "health_index": snap.health_index,
                    "true_rul": u.max_cycles - r.cycle,
                    "pred_rul": snap.rul.point,
                }
            )
        final_rul.append(
            {
                "unit_id": u.unit_id,
                "final_hi": last_hi,
                "max_cycles": u.max_cycles,
                "fault_mode": u.fault_mode.name,
            }
        )

    fleet_df = pd.DataFrame(fleet_rows)
    final_df = pd.DataFrame(final_rul)

    # Heatmap
    hm = fleet_df.pivot(index="unit_id", columns="cycle", values="health_index")
    fig_hm = go.Figure(
        go.Heatmap(
            z=hm.values,
            x=hm.columns,
            y=hm.index,
            colorscale=[
                [0.0, "#0ea5e9"],
                [0.4, "#22d3ee"],
                [0.6, "#facc15"],
                [0.8, "#f97316"],
                [1.0, "#dc2626"],
            ],
            zmid=failure_threshold,
            colorbar=dict(title="HI"),
        )
    )
    fig_hm.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="ciclo",
        yaxis_title="unidade",
        template="plotly_dark",
        title=f"Heatmap Health Index · limiar calibrado ≈ {failure_threshold:.3f}",
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(
        "Tons amarelo→vermelho indicam HI acima do limiar de falha calibrado empiricamente."
    )

    # Ranking + distribuição
    c5, c6 = st.columns([1, 1])
    with c5:
        st.subheader("Ranking — unidades mais degradadas (último ciclo)")
        last_hi = (
            fleet_df.groupby("unit_id")["health_index"]
            .last()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        last_hi.columns = ["unidade", "HI final"]
        st.dataframe(last_hi, use_container_width=True, hide_index=True)

    with c6:
        st.subheader("Distribuição de HI final da frota")
        fig_hist = px.histogram(
            final_df,
            x="final_hi",
            nbins=15,
            color="fault_mode",
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={"ABRUPT": "#ef4444", "GRADUAL": "#38bdf8"},
            template="plotly_dark",
        )
        fig_hist.add_vline(
            x=failure_threshold,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text="limiar",
        )
        fig_hist.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="HI final",
            yaxis_title="contagem",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Trajetórias médias por modo de falha
    st.subheader("Trajetórias médias de HI por modo de falha")
    fleet_df = fleet_df.merge(
        final_df[["unit_id", "fault_mode"]], on="unit_id", how="left"
    )
    traj = (
        fleet_df.groupby(["fault_mode", "cycle"])["health_index"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig_traj = go.Figure()
    for mode, color in [("ABRUPT", "#ef4444"), ("GRADUAL", "#38bdf8")]:
        sub = traj[traj["fault_mode"] == mode]
        if sub.empty:
            continue
        fig_traj.add_trace(
            go.Scatter(
                x=sub["cycle"],
                y=sub["mean"],
                name=f"{mode} (média)",
                line=dict(color=color, width=2.5),
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=sub["cycle"],
                y=sub["mean"] + sub["std"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=sub["cycle"],
                y=sub["mean"] - sub["std"],
                fill="tonexty",
                fillcolor=(
                    "rgba(239,68,68,0.12)" if mode == "ABRUPT" else "rgba(56,189,248,0.12)"
                ),
                name=f"{mode} ±1σ",
                line=dict(width=0),
            )
        )
    fig_traj.add_hline(
        y=failure_threshold, line_dash="dash", line_color="#94a3b8"
    )
    fig_traj.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="ciclo",
        yaxis_title="HI médio",
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_traj, use_container_width=True)

# ===========================================================================
# Aba 3 — Sobre
# ===========================================================================
with tab_about:
    st.subheader("AeroQuant Lab")
    st.markdown(
        f"""
        Plataforma de pesquisa em **Python 3.12** para monitoramento inteligente
        da saúde de aeronaves (caso de estudo: motores turbofan estilo NASA
        C-MAPSS). Foco científico: **Digital Twin** + **RUL com quantificação
        de incerteza** + manutenção preditiva.

        [Repositório GitHub]({REPO_URL}) · licença MIT.
        """
    )

    st.subheader("Status das fases (fonte: README + código real)")
    status_df = pd.DataFrame(
        [
            {"Fase": "1 — Auditoria & Arquitetura", "Status": "Concluída"},
            {"Fase": "2 — Pergunta Científica", "Status": "Concluída"},
            {"Fase": "3 — Engenharia de Dados", "Status": "Concluída"},
            {"Fase": "4 — Dados Sintéticos", "Status": "Concluída"},
            {"Fase": "5 — Digital Twin", "Status": "Concluída"},
            {"Fase": "6 — Machine Learning (RUL)", "Status": "Implementada (comparação baseline)"},
            {"Fase": "7 — Computer Vision", "Status": "Planejada"},
            {"Fase": "8 — Simulação Monte Carlo", "Status": "Planejada"},
            {"Fase": "9 — Explicabilidade (XAI)", "Status": "Planejada"},
            {"Fase": "10 — Dashboard", "Status": "Esta página (Plotly + Streamlit)"},
            {"Fase": "11 — MLOps", "Status": "Parcial (Docker/CI)"},
            {"Fase": "12 — Validação Científica", "Status": "Planejada"},
            {"Fase": "13 — Publicação", "Status": "Planejada"},
        ]
    )
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.subheader("Arquitetura")
    st.markdown(
        """
        Clean Architecture por **Bounded Context** (DDD):

        - `sensor_data` — gerador estocástico, ETL, schema C-MAPSS-like, qualidade.
        - `digital_twin` — baseline Welford online, HI z-score ponderado,
          RUL por extrapolação linear + intervalo OLS, calibração empírica
          do limiar de falha, repositório em memória.
        - Use-case central: `UpdateDigitalTwin`.
        - API FastAPI parcial em `src/aeroquant/api`.
        """
    )

    st.subheader("Limitações explícitas (honestidade científica)")
    st.markdown(
        """
        - Dados reais C-MAPSS ainda não carregados (`data/external/` aguarda upload);
          adapter existe, validação empírica pendente.
        - Incerteza do Health Index ainda fixa (0.15); limiar de acoplamento é
          heurístico — refinamentos na linha de ML (Fase 6+).
        - Repositório do Digital Twin é in-memory (adequado a demos e testes).
        - Este dashboard **não** expõe modelos de ML de RUL nem XAI até que
          estejam integrados e testados de ponta a ponta.
        - Referências: `docs/science/REFERENCES.md`.
        """
    )
