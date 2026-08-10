"""01 — COMMAND CENTER: visão de frota integrada."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_command_center(*, THEME: dict, plotly_layout, PLOTLY_CONFIG: dict) -> None:
    st.markdown("### Command Center")
    st.caption("Aircraft Health Monitoring · frota · decision support")

    snapshots = st.session_state.get("fleet_snapshots")
    if not snapshots:
        st.info(
            "Gere a frota demo abaixo ou execute Model Lab para popular o Command Center. "
            "Pipeline: Health → RUL → Monte Carlo → Risk → Decision."
        )
        if st.button("Gerar frota demo (Digital Twin)", type="primary", key="cc_demo"):
            with st.spinner("Simulando frota e pipeline integrado..."):
                st.session_state["fleet_snapshots"] = _demo_fleet()
                st.rerun()
        return

    units = snapshots
    n = len(units)
    avg_health = float(np.mean([u.health.overall_score for u in units]))
    avg_rul = float(np.mean([u.expected_rul for u in units]))
    n_crit = sum(1 for u in units if u.risk.level in ("CRITICAL", "HIGH"))
    n_anom = sum(1 for u in units if u.anomaly_severity in ("WARNING", "CRITICAL"))
    avg_pf = float(np.mean([u.prob_fail_30 for u in units]))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Fleet Health", f"{avg_health:.0f}/100")
    k2.metric("Avg RUL (P50)", f"{avg_rul:.0f} cyc")
    k3.metric("At Risk (H/C)", f"{n_crit}/{n}")
    k4.metric("Critical Anomalies", str(n_anom))
    k5.metric("P(fail≤30) avg", f"{100*avg_pf:.0f}%")

    st.markdown("#### Most Critical Aircraft")
    ranked = sorted(units, key=lambda u: u.risk.score, reverse=True)
    rows = [{
        "Unit": u.unit_id,
        "Health": round(u.health.overall_score, 1),
        "RUL P50": round(u.expected_rul, 1),
        "P10": round(u.p10, 1),
        "P90": round(u.p90, 1),
        "P(fail≤30)": f"{100*u.prob_fail_30:.0f}%",
        "Risk": u.risk.level,
        "Score": round(u.risk.score, 1),
        "Action": u.decision.action,
        "Anomaly": u.anomaly_severity,
    } for u in ranked[:12]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        counts = pd.Series([u.risk.level for u in units]).value_counts().reindex(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
        fig = go.Figure(go.Bar(
            x=list(counts.index), y=list(counts.values),
            marker_color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        ))
        fig.update_layout(**plotly_layout(THEME, height=280, x_title="Risk", y_title="Aircraft"))
        st.plotly_chart(fig, width="stretch", theme="streamlit", config=PLOTLY_CONFIG)
    with c2:
        fig = go.Figure(go.Scatter(
            x=[u.expected_rul for u in units],
            y=[u.risk.score for u in units],
            mode="markers+text",
            text=[u.unit_id[-4:] for u in units],
            textposition="top center",
            marker=dict(size=10, color=[u.health.overall_score for u in units],
                        colorscale="RdYlGn", showscale=True, colorbar=dict(title="Health")),
        ))
        fig.update_layout(**plotly_layout(THEME, height=280, x_title="Expected RUL", y_title="Risk Score"))
        st.plotly_chart(fig, width="stretch", theme="streamlit", config=PLOTLY_CONFIG)

    st.markdown("#### Aircraft Detail")
    sel = st.selectbox("Unit", [u.unit_id for u in ranked], key="cc_unit")
    u = next(x for x in units if x.unit_id == sel)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Health", f"{u.health.overall_score:.0f}")
    d2.metric("RUL P10/P50/P90", f"{u.p10:.0f} / {u.expected_rul:.0f} / {u.p90:.0f}")
    d3.metric("Risk", f"{u.risk.level} ({u.risk.score:.0f})")
    d4.metric("Action", u.decision.action)
    st.caption(u.risk.rationale)
    st.caption(f"**Decision:** {u.decision.action} · {u.decision.urgency} · {u.decision.window}")
    st.caption(u.decision.reason)
    st.caption(u.decision.disclaimer)
    if u.health.systems:
        st.dataframe(pd.DataFrame([
            {"System": s.name, "Score": round(s.score, 1), "Trend": round(s.trend, 1),
             "Status": s.status} for s in u.health.systems
        ]), width="stretch", hide_index=True)
    st.dataframe(pd.DataFrame([
        {"Driver": d.name, "Points": round(d.contribution, 1)} for d in u.risk.drivers
    ]), width="stretch", hide_index=True)


def _demo_fleet(n: int = 16, seed: int = 2026):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from aeroquant.platform.pipeline import build_unit_snapshot
    from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema

    rng = np.random.default_rng(seed)
    schema = build_cmapss_like_schema()
    snaps = []
    for i in range(n):
        life = int(rng.integers(80, 200))
        cycles = np.arange(1, life + 1)
        data = {"unit_id": f"AC-{i+1:03d}", "cycle": cycles}
        for s in schema.names()[:8]:
            data[f"{s}_z"] = rng.normal(0, 0.5, size=life) + cycles / life * rng.uniform(0.5, 2.5)
        df = pd.DataFrame(data)
        progress = df["cycle"].iloc[-1] / max(life, 1)
        expected = max(5.0, life * (1 - progress) * rng.uniform(0.7, 1.1))
        width = rng.uniform(15, 40)
        sev = rng.choice(["NORMAL", "WATCH", "WARNING", "CRITICAL"], p=[0.4, 0.3, 0.2, 0.1])
        snaps.append(build_unit_snapshot(
            unit_id=f"AC-{i+1:03d}", unit_df=df,
            expected_rul=float(expected),
            p10=float(max(1.0, expected - width * 0.5)),
            p50=float(expected),
            p90=float(expected + width * 0.5),
            anomaly_severity=str(sev),
            late_failure_risk=0.55,
            n_mc=2000, seed=seed + i,
        ))
    return snaps
