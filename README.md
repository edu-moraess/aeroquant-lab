# AeroQuant Lab

![AeroQuant Lab — Digital Twin & Aircraft Health Monitoring](https://raw.githubusercontent.com/edu-moraess/aeroquant-lab/main/docs/assets/hero_3d.png)

Plataforma técnica de pesquisa para **Aircraft Health Monitoring**, **Predictive Maintenance** e **Remaining Useful Life (RUL)**.

Não é um produto de certificação aeronáutica. Resultados de simulação **não** representam desempenho operacional real de uma aeronave.

---

## O que é

AeroQuant Lab implementa um pipeline de engenharia:

```
Sensor Data
    → Digital Twin (Health Index + RUL)
    → Feature Engineering (leakage-free)
    → ML / Neural Networks
    → RUL Prediction
    → Uncertainty (P10 / P50 / P90)
    → Monte Carlo
    → Risk Assessment
    → Maintenance Intelligence
```

## Arquitetura

| Bounded Context | Responsabilidade |
|-----------------|------------------|
| `sensor_data` | Geração sintética C-MAPSS-like, ETL, features |
| `digital_twin` | HI (z-score), RUL por extrapolação linear + IC |
| `ml` | Trainers, métricas, split por unit_id, sequências |
| `anomaly` | Isolation Forest, residual z-score |
| `uncertainty` | Monte Carlo RUL |
| `risk` | LOW/MEDIUM/HIGH/CRITICAL + P(RUL < threshold) |
| `xai` | SHAP |
| `dashboards` | Streamlit unificado |

## Data Pipeline

1. Geração sintética (degradação + falhas)
2. Clean por unidade
3. **Split por unit_id** (sem overlap de engines)
4. **Normalize**: fit só no treino
5. Features: z-score, rolling causal, delta
6. RUL label com cap piecewise-linear (default 125)

### Split

```
~65% TRAIN · ~15% VALIDATION · ~20% TEST
```

Janelas do mesmo engine **nunca** cruzam treino e teste.

## Modelos

| Família | Modelos |
|---------|---------|
| Tabular | Linear, Random Forest, GBM Quantile, MLP |
| Sequencial | Sequence MLP, LSTM*, Transformer* |
| Baseline DT | Extrapolação linear do HI |

\* `torch` opcional.

**Ranking default:** `NASA + 0.5·RMSE + 0.25·|Bias|`

NASA Score penaliza superestimar RUL mais que subestimar.

## Uncertainty & Risk

- Residual-based, RF quantiles, GBM quantile, Monte Carlo DT
- Saídas: Expected RUL, P10, P50, P90, P(RUL < threshold)
- Levels: LOW · MEDIUM · HIGH · CRITICAL (thresholds configuráveis)

## Dashboard

```bash
streamlit run dashboards/streamlit_app.py
```

Abas: Digital Twin · Fleet · ML clássico · Neural Net · Anomalias · Monte Carlo · Risk

Deploy: https://aeroquant-lab.streamlit.app/

## Instalação

```bash
git clone https://github.com/edu-moraess/aeroquant-lab
cd aeroquant-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboards/streamlit_app.py
```

## Estrutura

```
src/aeroquant/{sensor_data,digital_twin,ml,anomaly,uncertainty,risk,xai}/
dashboards/
tests/unit/
docs/
```

## Limitações

1. Dados sintéticos ≠ operação real
2. Sem certificação de segurança operacional
3. Dataset shift possível em dados reais
4. Incerteza residual é aproximação
5. Risk thresholds são parâmetros de engenharia
6. C-MAPSS real: adapter existe; upload pendente

## Documentação

- `docs/AUDIT_REPORT_P0_P1.md` — leakage, métricas, split
- `docs/AUDIT_REPORT_P2_P3_P4.md` — tracking, uncertainty, risk
- `docs/RELATORIO_STATUS.md` — o que funciona e por quê
- `docs/roadmap.md`

## Prioridade de qualidade

```
CORRECTNESS → REPRODUCIBILITY → VALIDATION → UNCERTAINTY → RISK → UI
```

Projeto de pesquisa e demonstração de pipeline. Sem garantia de uso operacional.
