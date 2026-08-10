"""09 — VALIDATION: bias, calibration, failure regions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_validation_panel(*, THEME: dict, plotly_layout, PLOTLY_CONFIG: dict) -> None:
    st.markdown("### Model Validation")
    st.caption("Bias · Calibration · Failure-region · unit_id split")
    res = st.session_state.get("ml_result")
    if res is None:
        st.info("Treine um modelo em **ML clássico** para popular esta página.")
        return
    st.markdown(f"**Best model:** {res.best_model_name}")
    st.caption(getattr(res, "protocol_note", ""))
    br = getattr(res, "bias_report", None)
    if br is not None:
        st.warning(br.bias_message)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean error (raw)", f"{br.mean_error:+.2f}")
        c2.metric("Median error", f"{br.median_error:+.2f}")
        c3.metric("Overestimation", f"{100*br.overestimation_rate:.0f}%")
        c4.metric("Late Failure Risk", f"{100*br.late_failure_risk:.0f}%")
    cal = getattr(res, "calibration_report", None)
    if cal is not None:
        st.caption(cal.message)
        c1, c2, c3 = st.columns(3)
        c1.metric("Coverage P10–P90", f"{100*cal.interval_coverage:.1f}%")
        c2.metric("P90 coverage", f"{100*cal.p90_coverage:.1f}%")
        c3.metric("Mean width", f"{cal.mean_width:.1f}")
        if getattr(res, "p10", None) is not None:
            order = np.argsort(res.test_true)
            yt = res.test_true[order]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yt, y=np.asarray(res.p90)[order], name="P90"))
            fig.add_trace(go.Scatter(x=yt, y=np.asarray(res.p10)[order], name="P10"))
            fig.add_trace(go.Scatter(x=yt, y=res.test_pred_best[order], name="Corrected P50"))
            fig.add_trace(go.Scatter(x=yt, y=yt, name="Ideal", line=dict(dash="dash")))
            fig.update_layout(**plotly_layout(THEME, height=340, x_title="True RUL", y_title="Predicted"))
            st.plotly_chart(fig, width="stretch", theme="streamlit", config=PLOTLY_CONFIG)
    fr = getattr(res, "failure_regions", None)
    if fr:
        st.markdown("#### Failure-region metrics (raw)")
        st.dataframe(pd.DataFrame(fr), width="stretch", hide_index=True)
    st.dataframe(getattr(res, "ranked_table", pd.DataFrame()), width="stretch", hide_index=True)
