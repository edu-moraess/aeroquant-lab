"""Reexport — compatibilidade com imports antigos `from theme import ...`."""
from ui_theme import (  # noqa: F401
    PLOTLY_CONFIG,
    SERIES,
    apply_global_css,
    get_theme,
    methodology_block,
    plotly_layout,
)

__all__ = [
    "PLOTLY_CONFIG",
    "SERIES",
    "apply_global_css",
    "get_theme",
    "methodology_block",
    "plotly_layout",
]
