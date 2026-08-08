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
      Deep learning (LSTM/Transformer) NÃO implementado — sem rede para
      instalar `torch` neste ambiente, ver item pendente abaixo.

## Bloqueado (depende de ação externa)

- [ ] Fase 4, Nível 2 — Ingestão de dados públicos C-MAPSS
      **Bloqueio**: arquivos precisam ser enviados manualmente (`data/external/README.md`)
- [ ] Fase 4, Nível 3 — Treino híbrido sintético→real
      **Bloqueio**: depende do Nível 2

## Próximo (não bloqueado, pode começar já)

- [ ] Deep learning para RUL (LSTM/Transformer) — bloqueado só por falta de
      rede neste ambiente específico, não cientificamente. Rodar em
      ambiente com `pip install torch` (ex.: Claude Code local).
- [ ] Decomposição de incerteza aleatória/epistêmica de verdade (H3) — hoje
      só temos incerteza por quantile regression (Gradient Boosting) e
      variância de ensemble (Random Forest), nenhuma decomposição explícita.
- [ ] Anomaly Detection como Bounded Context próprio (hoje é regra simples dentro do Digital Twin)

## Depende de dados reais ou de mais maturidade

- [ ] Fase 9 — Explainability (SHAP/Captum sobre os modelos da Fase 6 — os modelos já existem, isso pode começar)
- [ ] Fase 12 — Validação científica formal (benchmark, ablação, robustez) — a comparação da Fase 6 é um primeiro passo, não a validação completa

## Independente, mas de menor prioridade dado o núcleo escolhido

- [ ] Fase 7 — Computer Vision (bloqueada cientificamente por falta de dataset real licenciado — ver Fase 1, seção 1.3)
- [ ] Fase 8 — Simulação Monte Carlo
- [ ] Fase 10 — Dashboard de produção (hoje existe um esqueleto Streamlit não testado)
- [ ] Fase 11 — MLOps completo (CI existe; MLflow/DVC não integrados)

## Final

- [ ] Fase 13 — Publicação (ver `papers/README.md` — nada escrito ainda, corretamente)

## Decisões que vão precisar ser revisitadas

- Licença (`LICENSE`): MIT é o default assumido, não uma decisão institucional confirmada.
- `n_operating_conditions=1` no gerador (ver `docs/architecture/fase5-digital-twin.md`) — precisa virar condition-dependent de verdade antes do benchmark contra FD002/FD004.