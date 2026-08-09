"""
Tema centralizado do AeroQuant Lab (Light / Dark).

Todos os componentes visuais (CSS, Plotly, cores de série) devem consumir
estes tokens — nunca hex soltos espalhados no app.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

LIGHT_THEME: dict[str, str] = {
    "BACKGROUND": "#FFFFFF",
    "SURFACE": "#F7F8FA",
    "SURFACE_SECONDARY": "#EEF1F5",
    "TEXT_PRIMARY": "#0F172A",
    "TEXT_SECONDARY": "#475569",
    "BORDER": "#E2E8F0",
    "ACCENT": "#2563EB",
    "ACCENT_2": "#0D9488",
    "SUCCESS": "#16A34A",
    "WARNING": "#D97706",
    "ERROR": "#DC2626",
    "GRID": "#E2E8F0",
    "SERIES_A": "#2563EB",
    "SERIES_B": "#EA580C",
    "SERIES_C": "#0D9488",
    "SERIES_D": "#7C3AED",
    "SERIES_MUTED": "#64748B",
    "FILL_ACCENT": "rgba(37, 99, 235, 0.12)",
    "FILL_WARN": "rgba(220, 38, 38, 0.10)",
    "FILL_OK": "rgba(22, 163, 74, 0.10)",
}

DARK_THEME: dict[str, str] = {
    "BACKGROUND": "#0B1220",
    "SURFACE": "#111827",
    "SURFACE_SECONDARY": "#1F2937",
    "TEXT_PRIMARY": "#E5E7EB",
    "TEXT_SECONDARY": "#9CA3AF",
    "BORDER": "#374151",
    "ACCENT": "#60A5FA",
    "ACCENT_2": "#2DD4BF",
    "SUCCESS": "#4ADE80",
    "WARNING": "#FBBF24",
    "ERROR": "#F87171",
    "GRID": "#1F2937",
    "SERIES_A": "#60A5FA",
    "SERIES_B": "#FB923C",
    "SERIES_C": "#2DD4BF",
    "SERIES_D": "#A78BFA",
    "SERIES_MUTED": "#94A3B8",
    "FILL_ACCENT": "rgba(96, 165, 250, 0.18)",
    "FILL_WARN": "rgba(248, 113, 113, 0.15)",
    "FILL_OK": "rgba(74, 222, 128, 0.12)",
}


def detect_base_theme() -> str:
    """Retorna 'light' ou 'dark'.

    Prioridade: override manual (sidebar) → tema nativo Streamlit → dark.
    """
    override = st.session_state.get("_aq_theme_base")
    if override in ("light", "dark"):
        return override
    try:
        theme = st.context.theme
        base = getattr(theme, "type", None) or getattr(theme, "base", None)
        if base:
            b = str(base).lower()
            if b in ("light", "dark"):
                return b
    except Exception:
        pass
    return "dark"


def get_theme() -> dict[str, str]:
    return LIGHT_THEME if detect_base_theme() == "light" else DARK_THEME


def apply_global_css(theme: dict[str, str] | None = None) -> None:
    """CSS que acompanha o tema ativo. Não esconde o menu do Streamlit."""
    t = theme or get_theme()
    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {t["BACKGROUND"]} !important;
        }}
        section[data-testid="stSidebar"] > div:first-child {{
            background-color: {t["SURFACE"]} !important;
        }}
        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }}
        div[data-testid="stMetric"] {{
            background: {t["SURFACE"]};
            border: 1px solid {t["BORDER"]};
            border-radius: 6px;
            padding: 0.65rem 0.85rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {t["TEXT_SECONDARY"]} !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {t["TEXT_PRIMARY"]} !important;
            font-size: 1.35rem !important;
            font-weight: 600 !important;
        }}
        section[data-testid="stSidebar"] {{
            border-right: 1px solid {t["BORDER"]};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}
        button[data-baseweb="tab"] {{
            font-size: 0.9rem;
            font-weight: 500;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {t["BORDER"]};
            border-radius: 6px;
        }}
        details[data-testid="stExpander"] {{
            border: 1px solid {t["BORDER"]};
            border-radius: 6px;
            background: {t["SURFACE"]};
        }}
        h1 {{ font-size: 1.55rem !important; font-weight: 600 !important; margin-bottom: 0.15rem !important; }}
        h2, h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
        .stCaption {{ color: {t["TEXT_SECONDARY"]} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(
    theme: dict[str, str] | None = None,
    *,
    height: int = 360,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = True,
) -> dict[str, Any]:
    """Layout Plotly consistente com o tema ativo."""
    t = theme or get_theme()
    return dict(
        height=height,
        paper_bgcolor=t["BACKGROUND"],
        plot_bgcolor=t["SURFACE"],
        font=dict(color=t["TEXT_PRIMARY"], size=12, family="Inter, system-ui, sans-serif"),
        title=dict(text=title or "", font=dict(size=13, color=t["TEXT_PRIMARY"])) if title else None,
        margin=dict(l=48, r=16, t=40 if title else 24, b=44),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=t["TEXT_SECONDARY"]),
        )
        if show_legend
        else dict(traceorder="normal"),
        xaxis=dict(
            title=dict(text=x_title or "", font=dict(size=11, color=t["TEXT_SECONDARY"])),
            gridcolor=t["GRID"],
            zerolinecolor=t["BORDER"],
            linecolor=t["BORDER"],
            tickfont=dict(color=t["TEXT_SECONDARY"], size=10),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text=y_title or "", font=dict(size=11, color=t["TEXT_SECONDARY"])),
            gridcolor=t["GRID"],
            zerolinecolor=t["BORDER"],
            linecolor=t["BORDER"],
            tickfont=dict(color=t["TEXT_SECONDARY"], size=10),
            showgrid=True,
        ),
        hoverlabel=dict(
            bgcolor=t["SURFACE_SECONDARY"],
            font_color=t["TEXT_PRIMARY"],
            bordercolor=t["BORDER"],
        ),
    )


def methodology_block(
    info: str,
    method: str,
    interpretation: str,
    limitations: str | None = None,
    *,
    label: str = "Metodologia",
) -> None:
    """Expander: Informações · Metodologia · Interpretação · Limitações."""
    with st.expander(label, expanded=False):
        st.markdown(f"**Informações**  \n{info}")
        st.markdown(f"**Metodologia**  \n{method}")
        st.markdown(f"**Interpretação**  \n{interpretation}")
        if limitations:
            st.markdown(f"**Limitações**  \n{limitations}")
