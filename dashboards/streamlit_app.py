"""AeroQuant Lab — PHM platform loader with sidebar navigation.

    streamlit run dashboards/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_dir = Path(__file__).resolve().parent
_ROOT = _dir.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_dir))

import streamlit as st

st.set_page_config(
    page_title="AeroQuant Lab",
    page_icon="AQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "01 · Command Center",
    "02 · Digital Twin",
    "03 · Health / Fleet",
    "04 · Anomaly Engine",
    "05 · RUL / Model Lab",
    "06 · Neural Net",
    "07 · Monte Carlo",
    "08 · Risk Intelligence",
    "09 · Validation",
    "10 · Explainability",
    "11 · Methodology",
]

with st.sidebar:
    st.markdown("### AeroQuant Lab")
    st.caption("PHM · Digital Twin · Decision Support")
    page = st.radio("Navegação", PAGES, key="nav_page")
    st.divider()

_a = (_dir / "_app_body_a.py").read_text()
_b = (_dir / "_app_body_b.py").read_text()

_a = _a.replace(
    """st.set_page_config(
    page_title="AeroQuant Lab",
    page_icon="AQ",
    layout="wide",
    initial_sidebar_state="expanded",
)
""",
    "",
)

_OLD = '''tab_twin, tab_fleet, tab_ml, tab_nn, tab_anom, tab_mc, tab_risk = st.tabs(\n    ["Digital Twin", "Fleet", "ML clássico", "Neural Net", "Anomalias", "Monte Carlo", "Risk"]\n)'''

_NEW = '''_nav = st.session_state.get("nav_page", "05 · RUL / Model Lab")
if _nav.startswith("01"):
    from command_center import render_command_center
    render_command_center(THEME=THEME, plotly_layout=plotly_layout, PLOTLY_CONFIG=PLOTLY_CONFIG)
    st.stop()
if _nav.startswith("09"):
    from validation_panel import render_validation_panel
    render_validation_panel(THEME=THEME, plotly_layout=plotly_layout, PLOTLY_CONFIG=PLOTLY_CONFIG)
    st.stop()
if _nav.startswith("11"):
    st.markdown("### Methodology & Limitations")
    render_methodology_sidebar()
    st.markdown("**Fluxo PHM:** Data → Features → Health → Anomaly → RUL (+bias) → Uncertainty → MC → Risk → Decision.")
    st.markdown("**Positive bias:** superestimação de RUL documentada; correção y_corr = y_hat - bias.")
    st.stop()
if _nav.startswith("10"):
    st.markdown("### Explainability")
    if "ml_result" in st.session_state and getattr(st.session_state["ml_result"], "feature_importance", None) is not None:
        st.dataframe(st.session_state["ml_result"].feature_importance, width="stretch", hide_index=True)
    else:
        st.info("Treine um modelo no Model Lab para ver importâncias / SHAP.")
    st.stop()

tab_cmd, tab_twin, tab_fleet, tab_ml, tab_nn, tab_anom, tab_mc, tab_risk = st.tabs(
    ["Command Center", "Digital Twin", "Fleet", "ML clássico", "Neural Net", "Anomalias", "Monte Carlo", "Risk"]
)
with tab_cmd:
    from command_center import render_command_center
    render_command_center(THEME=THEME, plotly_layout=plotly_layout, PLOTLY_CONFIG=PLOTLY_CONFIG)
'''

if _OLD in _a:
    _a = _a.replace(_OLD, _NEW)

_a = _a.replace(
    'st.title("AeroQuant Lab")\nst.caption("Digital Twin · RUL · ML · Neural Net · Anomalias · Monte Carlo · Risk")',
    'st.title("AeroQuant Lab")\nst.caption("PHM platform · Command Center · RUL · Risk · Decision")',
)

exec(compile(_a + _b, str(_dir / "streamlit_app_full.py"), "exec"), globals())
