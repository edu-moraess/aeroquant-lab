# Relatório de status — AeroQuant Lab

## O que funciona e por que faz sentido

### Digital Twin
- Health Index por z-score + baseline Welford
- RUL por extrapolação linear do HI com IC OLS
- Baseline interpretável antes de ML black-box

### Pipeline
- Split por unit_id (sem leakage entre engines)
- Normalização fit só no treino
- Features rolling causais

### ML / NN
- Linear, RF, GBM, MLP, Sequence MLP, LSTM/Transformer
- Ranking NASA-first

### Uncertainty → Risk
- P10/P50/P90 + P(RUL < threshold)
- Levels LOW…CRITICAL com thresholds configuráveis

### UI
- Tabs unificadas + Methodology na sidebar
- Diagnósticos de residual e painel Risk

## O que ainda não é
- Certificação / manutenção real
- Validação C-MAPSS completa
- Calibração formal de intervalos

## Princípio
CORRECTNESS → REPRODUCIBILITY → VALIDATION → UNCERTAINTY → RISK → UI
