"""
Helpers de UI. Tema visual = 100% nativo do Streamlit.
Cores de série fixas (legíveis em Light e Dark).
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


def plotly_layout(
    *,
    height: int = 340,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = True,
) -> dict[str, Any]:
    """Layout Plotly mínimo. Fundo transparente — o Streamlit aplica o tema."""
    layout: dict[str, Any] = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=12, t=36 if title else 20, b=40),
        font=dict(size=12),
        xaxis=dict(title=x_title or "", showgrid=True, zeroline=False),
        yaxis=dict(title=y_title or "", showgrid=True, zeroline=False),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=13))
    if show_legend:
        layout["legend"] = dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)")
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
