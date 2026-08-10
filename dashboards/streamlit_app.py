"""AeroQuant Lab — navegação exclusiva pela sidebar (sem abas horizontais).

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


def _strip_page_config(src: str) -> str:
    return re.sub(r"st\.set_page_config\([\s\S]*?\)\s*", "", src, count=1)


def _tabs_to_sidebar_routing(src_a: str, src_b: str) -> str:
    text = src_a + "\n" + src_b

    text = re.sub(
        r"[a-z_]+(?:,\s*[a-z_]+)*\s*=\s*st\.tabs\(\s*\[[\s\S]*?\]\s*\)\s*",
        '_PAGE = st.session_state.get("nav_page", "01 · Command Center")\n\n',
        text,
        count=1,
    )
    text = re.sub(r"st\.tabs\(\s*\[[\s\S]*?\]\s*\)", "None", text)

    for old, new in [
        ("with tab_cmd:", 'if _PAGE.startswith("01"):'),
        ("with tab_twin:", 'elif _PAGE.startswith("02"):'),
        ("with tab_fleet:", 'elif _PAGE.startswith("03"):'),
        ("with tab_anom:", 'elif _PAGE.startswith("04"):'),
        ("with tab_ml:", 'elif _PAGE.startswith("05"):'),
        ("with tab_nn:", 'elif _PAGE.startswith("06"):'),
        ("with tab_mc:", 'elif _PAGE.startswith("07"):'),
        ("with tab_risk:", 'elif _PAGE.startswith("08"):'),
    ]:
        text = text.replace(old, new)

    # Injetar Command Center se ausente
    if 'startswith("01")' not in text:
        text = text.replace(
            '_PAGE = st.session_state.get("nav_page", "01 · Command Center")\n\n',
            '_PAGE = st.session_state.get("nav_page", "01 · Command Center")\n\n'
            'if _PAGE.startswith("01"):\n'
            '    from command_center import render_command_center\n'
            '    render_command_center(THEME=THEME, plotly_layout=plotly_layout, PLOTLY_CONFIG=PLOTLY_CONFIG)\n\n',
            1,
        )

    # Evitar elif órfão: primeiro bloco de página deve ser if
    text = re.sub(
        r"(_PAGE = st\.session_state\.get\(\"nav_page\".*?\)\n\n)elif _PAGE",
        r"\1if _PAGE",
        text,
        count=1,
        flags=re.S,
    )

    if 'startswith("09")' not in text:
        text += '''

elif _PAGE.startswith("09"):
    from validation_panel import render_validation_panel
    render_validation_panel(THEME=THEME, plotly_layout=plotly_layout, PLOTLY_CONFIG=PLOTLY_CONFIG)

elif _PAGE.startswith("10"):
    st.markdown("### Explainability")
    res = st.session_state.get("ml_result")
    if res is not None and getattr(res, "feature_importance", None) is not None:
        st.dataframe(res.feature_importance, width="stretch", hide_index=True)
        st.caption("Importâncias = contribuição preditiva, não causalidade.")
    else:
        st.info("Treine um modelo em **05 · RUL / Model Lab**.")

elif _PAGE.startswith("11"):
    st.markdown("### Methodology & Limitations")
    render_methodology_sidebar()
    st.markdown("**Fluxo:** Data → Health → Anomaly → RUL (+bias) → MC → Risk → Decision.")
    st.caption("Decision support — não é autoridade de aeronavegabilidade.")
'''

    text = text.replace(
        'st.caption("Digital Twin · RUL · ML · Neural Net · Anomalias · Monte Carlo · Risk")',
        'st.caption("Navegue pelos módulos na sidebar.")',
    )
    return text


_a = _strip_page_config((_dir / "_app_body_a.py").read_text())
_b = (_dir / "_app_body_b.py").read_text()
_code = _tabs_to_sidebar_routing(_a, _b)
exec(compile(_code, str(_dir / "streamlit_app_full.py"), "exec"), globals())
