"""
Tema centralizado — acompanha o tema nativo do Streamlit (Settings → Theme).

Não há seletor customizado na sidebar. Cores vêm de tokens LIGHT/DARK
detectados via st.context.theme.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

LIGHT_THEME: dict[str, str] = {
    "BACKGROUND": "#FFFFFF",
    "SURFACE": "#F8FAFC",
    "SURFACE_SECONDARY": "#F1F5F9",
    "TEXT_PRIMARY": "#0F172A",
    "TEXT_SECONDARY": "#64748B",
    "BORDER": "#E2E8F0",
    "ACCENT": "#1D4ED8",
    "SUCCESS": "#15803D",
    "WARNING": "#B45309",
    "ERROR": "#B91C1C",
    "GRID": "#E2E8F0",
    "SERIES_A": "#1D4ED8",
    "SERIES_B": "#C2410C",
    "SERIES_C": "#0F766E",
    "SERIES_D": "#6D28D9",
    "SERIES_MUTED": "#64748B",
    "FILL_ACCENT": "rgba(29, 78, 216, 0.10)",
    "FILL_WARN": "rgba(185, 28, 28, 0.08)",
}

DARK_THEME: dict[str, str] = {
    "BACKGROUND": "#0F172A",
    "SURFACE": "#1E293B",
    "SURFACE_SECONDARY": "#334155",
    "TEXT_PRIMARY": "#F1F5F9",
    "TEXT_SECONDARY": "#94A3B8",
    "BORDER": "#334155",
    "ACCENT": "#60A5FA",
    "SUCCESS": "#4ADE80",
    "WARNING": "#FBBF24",
    "ERROR": "#F87171",
    "GRID": "#334155",
    "SERIES_A": "#60A5FA",
    "SERIES_B": "#FB923C",
    "SERIES_C": "#2DD4BF",
    "SERIES_D": "#A78BFA",
    "SERIES_MUTED": "#94A3B8",
    "FILL_ACCENT": "rgba(96, 165, 250, 0.16)",
    "FILL_WARN": "rgba(248, 113, 113, 0.12)",
}


def detect_base_theme() -> str:
    """Usa apenas o tema nativo do Streamlit (menu ⋮ → Settings → Theme)."""
    try:
        theme = st.context.theme
        base = getattr(theme, "type", None) or getattr(theme, "base", None)
        if base and str(base).lower() in ("light", "dark"):
            return str(base).lower()
    except Exception:
        pass
    return "dark"


def get_theme() -> dict[str, str]:
    return LIGHT_THEME if detect_base_theme() == "light" else DARK_THEME


def apply_global_css(theme: dict[str, str] | None = None) -> None:
    """CSS leve. Não esconde o menu do Streamlit."""
    t = theme or get_theme()
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}
        div[data-testid="stMetric"] {{
            background: {t["SURFACE"]};
            border: 1px solid {t["BORDER"]};
            border-radius: 6px;
            padding: 0.55rem 0.75rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {t["TEXT_SECONDARY"]} !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {t["TEXT_PRIMARY"]} !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
        }}
        section[data-testid="stSidebar"] {{
            border-right: 1px solid {t["BORDER"]};
        }}
        button[data-baseweb="tab"] {{
            font-size: 0.88rem;
            font-weight: 500;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {t["BORDER"]};
            border-radius: 6px;
        }}
        details {{
            border: 1px solid {t["BORDER"]};
            border-radius: 6px;
            background: {t["SURFACE"]};
        }}
        h1 {{
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
            margin-bottom: 0.1rem !important;
        }}
        h2, h3 {{
            font-size: 1.0rem !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(
    theme: dict[str, str] | None = None,
    *,
    height: int = 340,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = True,
) -> dict[str, Any]:
    t = theme or get_theme()
    layout: dict[str, Any] = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["SURFACE"],
        font=dict(color=t["TEXT_PRIMARY"], size=12, family="Inter, system-ui, sans-serif"),
        margin=dict(l=48, r=12, t=36 if title else 20, b=40),
        xaxis=dict(
            title=dict(text=x_title or "", font=dict(size=11, color=t["TEXT_SECONDARY"])),
            gridcolor=t["GRID"],
            zerolinecolor=t["BORDER"],
            linecolor=t["BORDER"],
            tickfont=dict(color=t["TEXT_SECONDARY"], size=10),
        ),
        yaxis=dict(
            title=dict(text=y_title or "", font=dict(size=11, color=t["TEXT_SECONDARY"])),
            gridcolor=t["GRID"],
            zerolinecolor=t["BORDER"],
            linecolor=t["BORDER"],
            tickfont=dict(color=t["TEXT_SECONDARY"], size=10),
        ),
        hoverlabel=dict(
            bgcolor=t["SURFACE_SECONDARY"],
            font_color=t["TEXT_PRIMARY"],
            bordercolor=t["BORDER"],
        ),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color=t["TEXT_PRIMARY"]))
    if show_legend:
        layout["legend"] = dict(
            orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=t["TEXT_SECONDARY"]),
        )
    else:
        layout["showlegend"] = False
    return layout


def methodology_block(
    info: str,
    method: str,
    interpretation: str,
    limitations: str | None = None,
    *,
    label: str = "Metodologia",
) -> None:
    with st.expander(label, expanded=False):
        st.markdown(f"**O que é**  \n{info}")
        st.markdown(f"**Como calcula**  \n{method}")
        st.markdown(f"**Como interpretar**  \n{interpretation}")
        if limitations:
            st.markdown(f"**Limitações**  \n{limitations}")
