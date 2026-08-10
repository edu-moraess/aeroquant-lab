"""Painel Risk — incerteza + maintenance intelligence + alertas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from uncertainty_risk_experiment import run_uncertainty_risk_experiment


def render_risk_tab(*, noise_std: float, THEME: dict, plotly_layout, methodology_block, PLOTLY_CONFIG: dict) -> None:
    methodology_block(
        info="Maintenance intelligence: RUL uncertainty → risk level → alertas.",
        method="Métodos: RF tree quantiles, residual-based, GBM quantile, Monte Carlo DT. Risk: P(RUL < threshold) + Expected RUL. Alertas: Webhook/Telegram.",
        interpretation="LOW/MEDIUM/HIGH/CRITICAL são parâmetros de engenharia, não normas aeronáuticas.",
        limitations="Dados sintéticos/C-MAPSS. Intervalos residual-based assumem resíduos aproximadamente i.i.d.",
        label="Sobre este painel",
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        risk_method = st.selectbox(
            "Método de incerteza",
            options=["random_forest", "residual", "gbm_quantile", "monte_carlo"],
            format_func=lambda x: {
                "random_forest": "Random Forest quantiles",
                "residual": "Residual-based",
                "gbm_quantile": "GBM Quantile",
                "monte_carlo": "Monte Carlo (DT)",
            }[x],
            key="risk_m",
        )
    with r2:
        risk_units = st.slider("Unidades", 12, 36, 24, 4, key="risk_u")
    with r3:
        risk_thr = st.slider("Maintenance threshold", 10, 80, 30, 5, key="risk_t")
    with r4:
        risk_seed = st.number_input("Seed", value=2026, step=1, key="risk_s")

    if st.button("Avaliar risco", type="primary", key="risk_btn"):
        with st.spinner("Estimando incerteza e risco..."):
            try:
                st.session_state["risk_result"] = run_uncertainty_risk_experiment(
                    n_units=int(risk_units),
                    seed=int(risk_seed),
                    noise_std=noise_std,
                    maintenance_threshold=float(risk_thr),
                    method=risk_method,
                )
            except Exception as e:
                st.error("Falha na avaliação de risco.")
                st.caption(str(e))

    if "risk_result" not in st.session_state:
        st.info("Configure o método e clique em **Avaliar risco**.")
        render_alerts_section()
        return

    res = st.session_state["risk_result"]
    risk = res.risk
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Expected RUL", f"{risk.expected_rul:.1f}")
    c2.metric("P10", f"{risk.p10:.0f}")
    c3.metric("P50", f"{risk.p50:.0f}")
    c4.metric("P90", f"{risk.p90:.0f}")
    c5.metric(f"P(RUL<{int(risk.maintenance_threshold)})", f"{100 * risk.prob_below_threshold:.0f}%")
    c6.metric("Risk level", risk.level)
    st.caption(risk.rationale)
    st.caption(res.protocol_note)
    st.dataframe(pd.DataFrame([res.metrics_row]), width='stretch', hide_index=True)

    u1, u2 = st.columns(2)
    with u1:
        st.markdown("**Predicted vs Actual RUL**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res.y_true, y=res.y_pred, mode="markers",
            marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name="pred",
        ))
        if len(res.y_true) > 1:
            mx = float(max(float(res.y_true.max()), float(res.y_pred.max())))
            fig.add_trace(go.Scatter(
                x=[0, mx], y=[0, mx], mode="lines",
                line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal",
            ))
        fig.update_layout(**plotly_layout(THEME, height=340, x_title="True RUL", y_title="Predicted RUL"))
        st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
    with u2:
        st.markdown("**Prediction interval (P10–P90)**")
        fig = go.Figure()
        order = np.argsort(res.y_true)
        yt = res.y_true[order]
        fig.add_trace(go.Scatter(x=yt, y=res.p90[order], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=yt, y=res.p10[order], fill="tonexty", fillcolor=THEME["FILL_ACCENT"],
            name="P10–P90", line=dict(width=0),
        ))
        fig.add_trace(go.Scatter(
            x=yt, y=res.y_pred[order], mode="lines",
            line=dict(color=THEME["SERIES_B"], width=2), name="Expected",
        ))
        fig.add_hline(
            y=float(risk.maintenance_threshold), line_dash="dash",
            line_color=THEME["ERROR"], annotation_text="threshold",
        )
        fig.update_layout(**plotly_layout(THEME, height=340, x_title="True RUL (sorted)", y_title="RUL"))
        st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)

    render_alerts_section()


def render_alerts_section() -> None:
    """Configuração e disparo de alertas Webhook / Telegram para risco CRITICAL/HIGH."""
    st.markdown("---")
    st.subheader("Alertas em tempo real")
    st.caption(
        "Dispara notificações quando o nível de risco é **CRITICAL** ou **HIGH**. "
        "Credenciais via campos abaixo ou variáveis de ambiente "
        "`AEROQUANT_WEBHOOK_URL`, `AEROQUANT_TELEGRAM_BOT_TOKEN`, `AEROQUANT_TELEGRAM_CHAT_ID`."
    )

    from aeroquant.alerts import AlertDispatcher, AlertEvent
    from aeroquant.risk.assessment import RiskAssessment

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        webhook_url = st.text_input(
            "Webhook URL",
            value=st.session_state.get("alert_webhook", ""),
            placeholder="https://hooks.slack.com/services/... ou endpoint genérico",
            key="alert_webhook_in",
            help="POST JSON com campos text + alert",
        )
        st.session_state["alert_webhook"] = webhook_url
    with cfg2:
        tg_token = st.text_input(
            "Telegram Bot Token",
            value=st.session_state.get("alert_tg_token", ""),
            type="password",
            key="alert_tg_token_in",
        )
        st.session_state["alert_tg_token"] = tg_token
        tg_chat = st.text_input(
            "Telegram Chat ID",
            value=st.session_state.get("alert_tg_chat", ""),
            key="alert_tg_chat_in",
            help="ID do chat/grupo (número ou @channel)",
        )
        st.session_state["alert_tg_chat"] = tg_chat

    triggers = st.multiselect(
        "Níveis que disparam alerta",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH"],
        key="alert_triggers",
    )

    dispatcher = AlertDispatcher.from_env(
        webhook_url=webhook_url or None,
        telegram_token=tg_token or None,
        telegram_chat_id=tg_chat or None,
        trigger_levels=frozenset(triggers) if triggers else frozenset({"CRITICAL", "HIGH"}),
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        test_btn = st.button("Enviar alerta de teste", key="alert_test_btn")
    with b2:
        risk_btn = st.button(
            "Alertar com último Risk",
            key="alert_risk_btn",
            disabled="risk_result" not in st.session_state,
        )
    with b3:
        st.caption(f"Canais ativos: {len(dispatcher.channels)}")

    if test_btn:
        event = AlertEvent(
            level="CRITICAL",
            title="Teste AeroQuant Lab",
            message="Alerta de teste — módulo de notificações operacional.",
            unit_id="test-unit-0001",
            expected_rul=12.0,
            p10=5.0,
            p50=12.0,
            p90=22.0,
            maintenance_threshold=30.0,
            prob_below_threshold=0.85,
        )
        results = dispatcher.dispatch(event, force=True)
        for r in results:
            if r.ok:
                st.success(f"{r.channel}: OK ({r.status_code or '—'})")
            else:
                st.warning(f"{r.channel}: {r.detail}")

    if risk_btn and "risk_result" in st.session_state:
        risk: RiskAssessment = st.session_state["risk_result"].risk
        results = dispatcher.dispatch_risk(risk, unit_id="fleet-eval", force=False)
        for r in results:
            if r.ok:
                st.success(f"{r.channel}: {r.detail if r.channel == 'policy' else 'enviado'}")
            else:
                st.warning(f"{r.channel}: {r.detail}")

    with st.expander("Como configurar Telegram"):
        st.markdown(
            """
1. Abra o [@BotFather](https://t.me/BotFather) no Telegram → `/newbot` → copie o **token**.
2. Envie uma mensagem ao bot; obtenha o **chat_id** via  
   `https://api.telegram.org/bot<TOKEN>/getUpdates` (campo `chat.id`).
3. Cole token e chat_id acima (ou em secrets do Streamlit Cloud).

**Webhook:** qualquer endpoint que aceite `POST` JSON  
`{"text": "...", "alert": { ... }}` (Slack Incoming Webhooks funcionam no campo `text`).
"""
        )
