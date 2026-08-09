"""
Dashboard AeroQuant Lab — Cobre a Fase 10 (Dashboard) para o que já existe
de verdade (Fases 1–5: frota sintética + Digital Twin).

Este arquivo FOI executado e validado localmente (agosto/2026):
    pip install -r requirements/dashboard.txt
    streamlit run dashboards/streamlit_app.py

Componentes:
- Digital Twin (real): KPIs, RUL com banda de incerteza (Plotly), Health
  Index com marcadores de anomalia e explorador de sensores brutos.
- Fleet View: matriz de calor do Health Index por unidade e ciclo, com
  ranking das unidades mais degradadas.
- Sobre o Projeto: status das fases, arquitetura e link para o repo.

Sensores/RUL de ML (Fase 6), Computer Vision (Fase 7) e explicabilidade
(Fase 9) continuam fora deste dashboard — propositalmente, para não fingir
funcionalidades que não existem.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import plotly.graph_objects as go
import pandas as pd
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

st.set_page_config(page_title="AeroQuant Lab", page_icon="✈️", layout="wide")
st.title("AeroQuant Lab — Frota Sintética & Digital Twin")
st.caption(
    "Digital Twin com quantificação de incerteza sobre dados sintéticos "
    "estilo C-MAPSS. Fases 1–5 implementadas; ML (Fase 6+): pendente."
)

with st.expander("Como funciona?"):
    st.markdown(
        """
        1. **Gerador sintético** (`StochasticSensorGenerator`) produz leituras
           de até 21 sensores com degradação via processo Gamma, deriva de
           calibração, ruído gaussiano e falhas abruptas/intermitentes.
        2. O **Digital Twin** mantém um baseline saudável da frota
           (Welford online), calcula o **Health Index** (z-score ponderado
           pelo acoplamento de cada sensor à degradação), detecta anomalias
           nos incrementos do HI e estima o **RUL** por extrapolação linear
           com intervalo de predição OLS — a incerteza é cidadã de primeira
           classe, não pós-processamento.
        3. O limiar de falha é **calibrado empiricamente** contra uma frota
           de referência, em vez de fixado arbitrariamente.

        Arquitetura completa (Clean Architecture por Bounded Context) e
        roadmap das 13 fases no [repositório]({repo}).
        """.format(repo=REPO_URL)
    )

# ---------------------------------------------------------------------------
# Sidebar — Parâmetros agrupados
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Simulação")
    max_cycles = st.slider("Vida útil da unidade (ciclos)", 60, 300, 170)
    noise_std = st.slider("Ruído do sensor", 0.0, 0.05, 0.012)
    abrupt_rate = st.slider("Taxa de falha abrupta", 0.0, 0.02, 0.005)
    seed = st.number_input("Seed", value=42)

    st.header("Digital Twin")
    coupling_threshold = st.slider("Limiar de acoplamento do HI", 0.05, 0.5, 0.2, 0.05)
    confidence = st.select_slider(
        "Confiança do intervalo de RUL",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x * 100)}%",
    )

    st.header("Frota (Fleet View)")
    n_units_fleet = st.slider("Unidades na frota", 10, 60, 25)
    fleet_seed = st.number_input("Seed da frota", value=7)

# ---------------------------------------------------------------------------
# Construção compartilhada dos componentes do Digital Twin
# ---------------------------------------------------------------------------
schema = build_cmapss_like_schema()
generator = StochasticSensorGenerator()

def _build_dt(coupling: float, conf: float):
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling)
    failure_threshold = calibrate_failure_threshold(schema, OnlineFleetBaseline(), hi_estimator)
    dt = UpdateDigitalTwin(
        baseline_tracker=OnlineFleetBaseline(),
        hi_estimator=hi_estimator,
        rul_estimator=LinearExtrapolationRULEstimator(),
        repository=InMemoryDigitalTwinRepository(),
    )
    return dt, failure_threshold

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
tab_twin, tab_fleet, tab_about = st.tabs(
    ["Digital Twin (real)", "Fleet View", "Sobre o Projeto"]
)

# ===========================================================================
# Aba 1 — Digital Twin
# ===========================================================================
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

    dt, failure_threshold = _build_dt(coupling_threshold, confidence)

    rows, last_snap = [], None
    for r in readings:
        snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
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

    # --- KPIs no topo -------------------------------------------------------
    rul_width = float(df["rul_upper"].iloc[-1] - df["rul_lower"].iloc[-1])
    hi_at_failure = float(last_snap.health_index) if last_snap else float("nan")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("RUL previsto (ciclos)", f"{df['predicted_rul'].iloc[-1]:.0f}",
                  delta=-df["predicted_rul"].diff().iloc[-1])
    with col_k2:
        st.metric("Health Index atual", f"{hi_at_failure:.2f}",
                  delta=f"{hi_at_failure - df['health_index'].iloc[0]:.2f}")
    with col_k3:
        st.metric("Incerteza do RUL (±)", f"{rul_width / 2:.0f}")
    with col_k4:
        st.metric("Anomalias detectadas", int(df["anomaly"].sum()))

    st.caption(
        f"Limiar de falha calibrado empiricamente: {failure_threshold:.3f} · "
        f"Confiança do intervalo: {int(confidence * 100)}%"
    )

    # --- RUL com banda de incerteza -----------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RUL: previsto vs. verdadeiro")
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(x=df["cycle"], y=df["true_rul"],
                                     name="True RUL", line=dict(color="#1f3b73")))
        fig_rul.add_trace(go.Scatter(x=df["cycle"], y=df["predicted_rul"],
                                     name="Predicted RUL", line=dict(color="#e07b39")))
        fig_rul.add_trace(go.Scatter(
            x=df["cycle"], y=df["rul_upper"], fill=None,
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_rul.add_trace(go.Scatter(
            x=df["cycle"], y=df["rul_lower"], fill="tonexty",
            fillcolor="rgba(31,119,180,0.2)", name=f"Intervalo {int(confidence * 100)}%"))
        fig_rul.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0),
                              yaxis_title="ciclos restantes")
        st.plotly_chart(fig_rul, use_container_width=True)

    with col2:
        st.subheader("Health Index")
        anom = df[df["anomaly"]]
        fig_hi = go.Figure()
        fig_hi.add_trace(go.Scatter(x=df["cycle"], y=df["health_index"],
                                    name="Health Index", line=dict(color="#1f77b4")))
        if len(anom):
            fig_hi.add_trace(go.Scatter(x=anom["cycle"], y=anom["health_index"],
                                        mode="markers", name="Anomalia",
                                        marker=dict(color="#d62728", size=8)))
        fig_hi.add_hline(y=failure_threshold, line_dash="dash", line_color="gray",
                         annotation_text="limiar calibrado")
        fig_hi.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0),
                             yaxis_title="HI")
        st.plotly_chart(fig_hi, use_container_width=True)

    # --- Explorador de sensores brutos ---------------------------------------
    st.subheader("Explorador de sensores")
    sensor_names = schema.names()
    default_sensors = [s.name for s in schema.sensors if s.degradation_coupling >= 0.3]
    chosen = st.multiselect("Sensores (coloridos pelo acoplamento à degradação)",
                            options=sensor_names, default=default_sensors)
    if chosen:
        fig_s = go.Figure()
        for name in chosen:
            spec = schema.spec_for(name)
            fig_s.add_trace(go.Scatter(
                x=df["cycle"], y=[r.values[name] for r in readings],
                name=f"{name} (acoplamento {spec.degradation_coupling:.2f})",
                line=dict(width=1)))
        fig_s.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0),
                            yaxis_title="valor do sensor")
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("Selecione ao menos um sensor para visualizar os sinais brutos.")

# ===========================================================================
# Aba 2 — Fleet View
# ===========================================================================
with tab_fleet:
    st.subheader("Saúde da frota (Heatmap HI × ciclo × unidade)")
    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=coupling_threshold)
    failure_threshold = calibrate_failure_threshold(schema, OnlineFleetBaseline(), hi_estimator)

    fleet_rows = []
    for i in range(int(n_units_fleet)):
        u = Unit(unit_id=f"fleet-unit-{i:03d}", fleet_id="fleet-f1",
                 max_cycles=150,
                 fault_mode=FaultMode.ABRUPT if i % 3 == 0 else FaultMode.GRADUAL)
        p = DegradationParams(seed=int(fleet_seed) + i, noise_std=noise_std,
                              abrupt_fault_rate=abrupt_rate, abrupt_fault_magnitude=0.5)
        rdgs = generator.generate_unit(u, schema, p)
        dt = UpdateDigitalTwin(
            baseline_tracker=OnlineFleetBaseline(), hi_estimator=hi_estimator,
            rul_estimator=LinearExtrapolationRULEstimator(),
            repository=InMemoryDigitalTwinRepository(),
        )
        for r in rdgs:
            snap = dt.ingest(u.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
            fleet_rows.append({"unit_id": u.unit_id, "cycle": r.cycle,
                               "health_index": snap.health_index,
                               "rul": u.max_cycles - r.cycle})
    fleet_df = pd.DataFrame(fleet_rows)

    hm = fleet_df.pivot(index="unit_id", columns="cycle", values="health_index")
    fig_hm = go.Figure(go.Heatmap(
        z=hm.values, x=hm.columns, y=hm.index,
        colorscale=[[0, "#3b82f6"], [0.5, "#facc15"], [1, "#dc2626"]],
        zmid=failure_threshold))
    fig_hm.add_hline(y=0, line_width=0)
    fig_hm.update_layout(
        height=450, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="ciclo", yaxis_title="unidade",
        coloraxis_colorbar_title_text="HI")
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(f"Faixa amarela/vermelha indica HI acima do limiar calibrado ({failure_threshold:.3f}).")

    st.subheader("Ranking — unidades mais degradadas")
    last_hi = fleet_df.groupby("unit_id")["health_index"].last().sort_values(ascending=False)
    top = last_hi.head(10).reset_index()
    top.columns = ["unidade", "HI no último ciclo"]
    st.dataframe(top, use_container_width=True, hide_index=True)

# ===========================================================================
# Aba 3 — Sobre o Projeto
# ===========================================================================
with tab_about:
    st.subheader("AeroQuant Lab")
    st.markdown(
        f"Plataforma de pesquisa em Python 3.12 para monitoramento da saúde "
        f"de aeronaves com **Digital Twin**, predição de **RUL** com "
        f"quantificação de incerteza e manutenção preditiva, usando motores "
        f"turbofan (benchmark NASA C-MAPSS) como caso de estudo. "
        f"[Repositório]({REPO_URL}) (MIT)."
    )

    st.subheader("Status das fases")
    st.dataframe(pd.DataFrame([
        {"Fase": "1 — Auditoria & Arquitetura", "Status": "Concluída"},
        {"Fase": "2 — Pergunta Científica", "Status": "Concluída"},
        {"Fase": "3 — Engenharia de Dados", "Status": "Concluída"},
        {"Fase": "4 — Dados Sintéticos", "Status": "Concluída"},
        {"Fase": "5 — Digital Twin", "Status": "Concluída"},
        {"Fase": "6 — Machine Learning (RUL)", "Status": "Planejada"},
        {"Fase": "7 — Computer Vision", "Status": "Planejada"},
        {"Fase": "8 — Simulação Monte Carlo", "Status": "Planejada"},
        {"Fase": "9 — Explicabilidade (XAI)", "Status": "Planejada"},
        {"Fase": "10 — Dashboard", "Status": "Em andamento (esta página)"},
        {"Fase": "11 — MLOps", "Status": "Planejada"},
        {"Fase": "12 — Validação Científica", "Status": "Planejada"},
        {"Fase": "13 — Publicação", "Status": "Planejada"},
    ]), use_container_width=True, hide_index=True)

    st.subheader("Arquitetura")
    st.markdown(
        "O código segue **Clean Architecture por Bounded Context** (DDD): "
        "os contextos `sensor_data` (gerador + ETL + qualidade) e "
        "`digital_twin` (baseline online Welford, Health Index por z-score "
        "ponderado, RUL por extrapolação linear com intervalo OLS e "
        "calibração empírica do limiar) são interligados por um use case "
        "`UpdateDigitalTwin` e expostos por uma API FastAPI parcial."
    )

    st.subheader("Limitações conhecidas (honestidade científica)")
    st.markdown(
        "- Sem dados reais C-MAPSS ainda — o adapter existe, mas nunca foi "
        "validado contra arquivo real (pasta `data/external/` aguarda upload).\n"
        "- A incerteza do Health Index é fixa (0.15) e o limiar de acoplamento "
        "é heurístico — refinamentos previstos na Fase 6.\n"
        "- O repositório do Digital Twin é em memória, adequado para demos "
        "locais.\n"
        "- Referências bibliográficas deliberadamente não listadas: ver "
        "`docs/science/REFERENCES.md` no repositório."
    )
