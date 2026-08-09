"""
Helpers de UI. Tema visual = nativo do Streamlit.
Modebar Plotly transparente e canto superior direito.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

SERIES = {
    "A": "#2563EB",
    "B": "#EA580C",
    "C": "#0D9488",
    "D": "#7C3AED",
    "MUTED": "#94A3B8",
    "ERROR": "#DC2626",
    "SUCCESS": "#16A34A",
    "FILL": "rgba(37, 99, 235, 0.15)",
    "FILL_WARN": "rgba(220, 38, 38, 0.12)",
}

PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
        "zoomIn2d",
        "zoomOut2d",
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "aeroquant",
        "height": None,
        "width": None,
        "scale": 2,
    },
}


def get_theme() -> dict[str, str]:
    return {
        "SERIES_A": SERIES["A"],
        "SERIES_B": SERIES["B"],
        "SERIES_C": SERIES["C"],
        "SERIES_D": SERIES["D"],
        "SERIES_MUTED": SERIES["MUTED"],
        "ERROR": SERIES["ERROR"],
        "SUCCESS": SERIES["SUCCESS"],
        "WARNING": SERIES["B"],
        "FILL_ACCENT": SERIES["FILL"],
        "FILL_WARN": SERIES["FILL_WARN"],
        "TEXT_PRIMARY": "",
        "TEXT_SECONDARY": SERIES["MUTED"],
        "BACKGROUND": "rgba(0,0,0,0)",
        "SURFACE": "rgba(0,0,0,0)",
        "SURFACE_SECONDARY": "rgba(0,0,0,0)",
        "BORDER": SERIES["MUTED"],
        "GRID": SERIES["MUTED"],
        "ACCENT": SERIES["A"],
    }


def apply_global_css(theme: dict[str, str] | None = None) -> None:
    """CSS mínimo: modebar transparente no canto superior direito."""
    st.markdown(
        """
        <style>
        .js-plotly-plot .modebar {
            top: 2px !important;
            right: 2px !important;
            left: auto !important;
            background: transparent !important;
            backdrop-filter: none !important;
        }
        .js-plotly-plot .modebar-group {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            margin-left: 2px !important;
        }
        .js-plotly-plot .modebar-btn {
            background: transparent !important;
            border: none !important;
        }
        .js-plotly-plot .modebar-btn:hover {
            background: rgba(128, 128, 128, 0.18) !important;
        }
        .js-plotly-plot .modebar-btn path {
            fill-opacity: 0.55 !important;
        }
        .js-plotly-plot .modebar-btn:hover path {
            fill-opacity: 0.9 !important;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        h1 {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.1rem !important;
        }
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
    layout: dict[str, Any] = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=28, t=44 if title else 32, b=40),
        font=dict(size=12),
        xaxis=dict(title=x_title or "", showgrid=True, zeroline=False),
        yaxis=dict(title=y_title or "", showgrid=True, zeroline=False),
        modebar=dict(
            orientation="v",
            bgcolor="rgba(0,0,0,0)",
            color="rgba(128,128,128,0.7)",
            activecolor="rgba(128,128,128,1)",
        ),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=13), x=0, xanchor="left")
    if show_legend:
        layout["legend"] = dict(
            orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", borderwidth=0,
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
    label: str = "Sobre este painel",
) -> None:
    with st.expander(label, expanded=False):
        st.markdown(f"**O que é**  \n{info}")
        st.markdown(f"**Como calcula**  \n{method}")
        st.markdown(f"**Como interpretar**  \n{interpretation}")
        if limitations:
            st.markdown(f"**Limitações**  \n{limitations}")
