"""Bloco Methodology & Limitations para a sidebar."""
from __future__ import annotations

import streamlit as st


def render_methodology_sidebar() -> None:
    with st.sidebar.expander("Methodology & Limitations", expanded=False):
        st.markdown(
            """
**Dados**  
Frota sintética C-MAPSS-like (degradação + falhas). Não é telemetria real.

**Target RUL**  
Ciclos restantes até falha, com cap piecewise-linear (default 125).

**Split**  
Por `unit_id` (engines). Train / val / test sem overlap.  
Normalização: mean/std **somente do treino**.

**Sequence length**  
`T = 30` → últimos 30 ciclos para estimar o RUL **atual** (sem futuro).

**NASA Score**  
Penaliza superestimar RUL mais que subestimar (d = pred − true).

**Risk thresholds**  
Parâmetros de engenharia configuráveis — **não** normas aeronáuticas.

**Limitações**  
- Resultados de simulação ≠ operação real  
- Possível dataset shift em dados reais  
- Incerteza residual é aproximação, não calibração formal  
- Não usar para decisão de segurança operacional
            """.strip()
        )
