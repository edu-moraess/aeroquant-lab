# Roadmap

Baseado no master prompt original (13 fases) e na priorização decidida na
Fase 1 (núcleo = RUL + Digital Twin).

## Concluído

- [x] Fase 1 — Auditoria crítica e arquitetura
- [x] Fase 2 — Pergunta científica, hipóteses, venues de publicação
- [x] Fase 3 — Engenharia de dados (ETL, qualidade)
- [x] Fase 4, Nível 1 — Gerador sintético
- [x] Fase 5 — Digital Twin com baseline estatístico (HI, RUL+incerteza, anomalia)
- [x] Fase 6 — Machine Learning para RUL: 3 modelos scikit-learn (Linear
      Regression, Random Forest, Gradient Boosting quantile) comparados
      contra o baseline da Fase 5 nas mesmas unidades de teste.
      `gradient_boosting_quantile` venceu (RMSE 22.3 vs. 155.8 do baseline).
      Deep learning (LSTM/Transformer) NÃO implementado — ver item pendente.
- [x] Fase 8 — Monte Carlo RUL com decomposição empírica aleatória/epistêmica
- [x] Fase 10 — Dashboard (Streamlit + Plotly + multipage ML/MC)

## Bloqueado (depende de ação externa)

- [ ] Fase 4, Nível 2 — Ingestão de dados públicos C-MAPSS
      **Bloqueio**: arquivos precisam ser enviados manualmente (`data/external/README.md`)
- [ ] Fase 4, Nível 3 — Treino híbrido sintético→real
      **Bloqueio**: depende do Nível 2

## Próximo (não bloqueado, pode começar já)

- [ ] Deep learning para RUL (LSTM/Transformer) — torch disponível em ambientes com rede.
- [~] Decomposição de incerteza (H3) — progresso: HI data-driven (Welford n);
      quantile/ensemble ML; Monte Carlo Fase 8 com var_aleatoric vs var_epistemic.
      Falta formalismo Bayesiano completo.
- [ ] Anomaly Detection como Bounded Context próprio
- [ ] Fase 9 — Explainability (SHAP/Captum sobre modelos da Fase 6)
- [ ] Fase 12 — Validação científica formal

## Independente / menor prioridade no núcleo atual

- [ ] Fase 7 — Computer Vision (bloqueada por falta de dataset real licenciado)
- [ ] Fase 11 — MLOps completo (CI existe; MLflow/DVC não integrados)

## Final

- [ ] Fase 13 — Publicação (ver `papers/README.md`)

## Decisões a revisitar

- Licença (`LICENSE`): MIT default, não decisão institucional confirmada.
- `n_operating_conditions=1` no gerador — precisa condition-dependent real
  antes do benchmark FD002/FD004.
