"""
AVISO: streamlit NÃO está instalado neste container (sem acesso à rede) —
este arquivo foi escrito com cuidado sintático mas NÃO foi executado. Antes
de confiar nele:

    pip install -r requirements/dashboard.txt
    streamlit run dashboards/streamlit_app.py

Cobre um subconjunto da Fase 10 (Dashboard) — só a parte que já existe de
verdade (frota sintética + Digital Twin). Sensores/RUL de ML (Fase 6),
Computer Vision (Fase 7) e explicabilidade (Fase 9) ficam como abas
"Em breve" até essas fases serem implementadas — propositalmente, para não
fingir uma funcionalidade que não existe.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

st.set_page_config(page_title="AeroQuant Lab", layout="wide")
st.title("AeroQuant Lab — Frota Sintética & Digital Twin")
st.caption("Fases 1-5 implementadas. CV / XAI / dashboards de ML: pendentes (Fase 6+).")

with st.sidebar:
    st.header("Parâmetros")
    max_cycles = st.slider("Vida útil da unidade (ciclos)", 60, 300, 170)
    noise_std = st.slider("Ruído do sensor", 0.0, 0.05, 0.012)
    abrupt_rate = st.slider("Taxa de falha abrupta", 0.0, 0.02, 0.005)
    seed = st.number_input("Seed", value=42)

tab_twin, tab_pending = st.tabs(["Digital Twin (real)", "Em breve"])

with tab_twin:
    schema = build_cmapss_like_schema()
    generator = StochasticSensorGenerator()
    unit = Unit(unit_id="dashboard-unit", fleet_id="f1", max_cycles=int(max_cycles), fault_mode=FaultMode.ABRUPT)
    params = DegradationParams(seed=int(seed), noise_std=noise_std, abrupt_fault_rate=abrupt_rate, abrupt_fault_magnitude=0.5)
    readings = generator.generate_unit(unit, schema, params)

    hi_estimator = ZScoreHealthIndexEstimator(schema, coupling_threshold=0.2)
    failure_threshold = calibrate_failure_threshold(schema, OnlineFleetBaseline(), hi_estimator)
    dt = UpdateDigitalTwin(
        baseline_tracker=OnlineFleetBaseline(),
        hi_estimator=hi_estimator,
        rul_estimator=LinearExtrapolationRULEstimator(),
        repository=InMemoryDigitalTwinRepository(),
    )

    rows = []
    for r in readings:
        snap = dt.ingest(unit.unit_id, r.cycle, r.operating_condition, r.values, failure_threshold)
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RUL: previsto vs. verdadeiro")
        st.line_chart(df.set_index("cycle")[["true_rul", "predicted_rul"]])
    with col2:
        st.subheader("Health Index")
        st.line_chart(df.set_index("cycle")[["health_index"]])

    n_anomalies = int(df["anomaly"].sum())
    st.metric("Anomalias detectadas", n_anomalies)
    if n_anomalies:
        st.dataframe(df[df["anomaly"]][["cycle", "health_index"]])

with tab_pending:
    st.info(
        "RUL via Machine Learning (Fase 6), inspeção visual (Fase 7), "
        "explicabilidade SHAP (Fase 9) e MLOps completo (Fase 11) ainda "
        "não foram implementados — este dashboard só mostra o que existe "
        "de verdade até a Fase 5."
    )
