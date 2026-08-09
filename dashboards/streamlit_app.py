"""AeroQuant Lab — Dashboard unificado (loader).

    streamlit run dashboards/streamlit_app.py
"""
from pathlib import Path

_dir = Path(__file__).resolve().parent
_code = (_dir / "_app_body_a.py").read_text() + (_dir / "_app_body_b.py").read_text()
exec(compile(_code, str(_dir / "streamlit_app_full.py"), "exec"), globals())
