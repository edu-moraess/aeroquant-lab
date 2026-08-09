# Roadmap

Baseado no master prompt original (13 fases) e na priorização decidida na
Fase 1 (núcleo = RUL + Digital Twin).

## Concluído

- [x] Fase 1 — Auditoria crítica e arquitetura
- [x] Fase 2 — Pergunta científica, hipóteses, venues de publicação
- [x] Fase 3 — Engenharia de dados (ETL, qualidade)
- [x] Fase 4, Nível 1 — Gerador sintético
- [x] Fase 5 — Digital Twin com baseline estatístico (HI, RUL+incerteza, anomalia)
- [x] Fase 6 — Machine Learning para RUL (Linear / RF / GBM quantile / MLP)
- [x] Fase 8 — Monte Carlo RUL (decomposição empírica aleatória/epistêmica)
- [x] Fase 9 — XAI (TreeSHAP sobre RF/GBM + explicação local/global no dashboard)
- [x] Fase 10 — Dashboard unificado (Streamlit)
- [x] MLP + Sequence MLP + LSTM (torch opcional)
- [x] Anomaly Detection (Isolation Forest + residual z-score)

## Bloqueado (depende de ação externa)

- [ ] Fase 4, Nível 2 — Ingestão de dados públicos C-MAPSS
- [ ] Fase 4, Nível 3 — Treino híbrido sintético→real

## Próximo

- [ ] Transformer para RUL
- [~] Decomposição de incerteza (H3) — parcial: HI data-driven, quantile ML, Monte Carlo
- [ ] Fase 12 — Validação científica formal

## Menor prioridade no núcleo atual

- [ ] Fase 7 — Computer Vision
- [ ] Fase 11 — MLOps completo (MLflow/DVC)
- [ ] Fase 13 — Publicação

## Decisões a revisitar

- Licença MIT default.
- `n_operating_conditions=1` no gerador antes do benchmark FD002/FD004.
