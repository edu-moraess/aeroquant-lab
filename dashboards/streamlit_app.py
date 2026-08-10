"""AeroQuant Lab — PHM platform (navegação só pela sidebar).

    streamlit run dashboards/streamlit_app.py
"""
from __future__ import annotations

import re
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
    st.caption("PHM · Predictive Maintenance")
    st.radio("Módulo", PAGES, key="nav_page")
    st.divider()
    st.caption("Parâmetros de simulação")

_a = (_dir / "_app_body_a.py").read_text()
_b = (_dir / "_app_body_b.py").read_text()

_a = re.sub(r"st\.set_page_config\([\s\S]*?\)\s*", "", _a, count=1)

if "st.tabs(" in _a or "st.tabs(" in _b:
    raise RuntimeError("Navegação por abas ainda presente — use apenas a sidebar.")

for old in (
    'st.caption("Command Center · Digital Twin · RUL · ML · Neural Net · Anomalia · MC · Risk · Decision")',
    'st.caption("Digital Twin · RUL · ML · Neural Net · Anomalias · Monte Carlo · Risk")',
    'st.caption("PHM platform · Command Center · RUL · Risk · Decision")',
):
    _a = _a.replace(old, 'st.caption("Navegue pelos módulos na sidebar.")')

exec(compile(_a + "\n" + _b, str(_dir / "streamlit_app_full.py"), "exec"), globals())
