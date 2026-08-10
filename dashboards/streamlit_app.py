"""AeroQuant Lab — Dashboard unificado (loader).

    streamlit run dashboards/streamlit_app.py
"""
from pathlib import Path
import sys

_dir = Path(__file__).resolve().parent
_ROOT = _dir.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_dir))

_a = (_dir / "_app_body_a.py").read_text()
_b = (_dir / "_app_body_b.py").read_text()

# Inject Command Center as first tab if not already present
_OLD = '''tab_twin, tab_fleet, tab_ml, tab_nn, tab_anom, tab_mc, tab_risk = st.tabs(\n    ["Digital Twin", "Fleet", "ML clássico", "Neural Net", "Anomalias", "Monte Carlo", "Risk"]\n)'''
_NEW = '''tab_cmd, tab_twin, tab_fleet, tab_ml, tab_nn, tab_anom, tab_mc, tab_risk = st.tabs(\n    ["Command Center", "Digital Twin", "Fleet", "ML clássico", "Neural Net", "Anomalias", "Monte Carlo", "Risk"]\n)\n\nwith tab_cmd:\n    from command_center import render_command_center\n    render_command_center(THEME=THEME, plotly_layout=plotly_layout, PLOTLY_CONFIG=PLOTLY_CONFIG)\n'''
if "tab_cmd" not in _a and _OLD in _a:
    _a = _a.replace(_OLD, _NEW)
    _a = _a.replace(
        "Digital Twin · RUL · ML · Neural Net · Anomalias · Monte Carlo · Risk",
        "Command Center · Health · RUL · Risk · Decision",
    )

_code = _a + _b
exec(compile(_code, str(_dir / "streamlit_app_full.py"), "exec"), globals())
