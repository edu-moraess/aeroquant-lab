# Fase 5 — Digital Twin: Arquitetura e Decisões

Ver diagrama de sequência: `assets/diagrams/digital_twin_sequence.mmd`.

## Por que o Digital Twin não calcula HI/RUL diretamente

`UpdateDigitalTwin` (application/use_cases.py) orquestra três estimadores
desacoplados via ports (`HealthIndexEstimator`, `RULEstimator`,
`FleetBaselineTracker`) — o agregado de domínio (`DigitalTwinState`) só
garante consistência do histórico. Isso significa que a Fase 6 pode trocar
`ZScoreHealthIndexEstimator`/`LinearExtrapolationRULEstimator` por modelos
de deep learning sem tocar no use case nem no domínio — só uma troca de
`infrastructure`.

## Três bugs reais encontrados e corrigidos nesta fase

Detalhados em `CHANGELOG.md`, resumo aqui:

1. **Baseline contaminado pela própria degradação** — corrigido travando a
   atualização do baseline a uma janela inicial de vida assumida-saudável
   (`healthy_window_cycles`).
2. **Baseline por condição operacional sub-amostrado** — o gerador não
   modela comportamento condition-dependent real; `n_operating_conditions`
   default reduzido de 6 para 1 até isso ser implementado de verdade.
3. **Limiar de falha arbitrário (`failure_threshold=1.0`)** — o Health
   Index (soma ponderada de |z-scores|) não tem relação com a escala
   [0,1]. Corrigido com calibração empírica (`threshold_calibration.py`)
   contra uma frota de referência sintética.

## Por que o baseline de RUL ainda é "ruidoso" mesmo depois das correções

Ver `outputs/digital_twin_rul_tracking.png` (gerado por
`scripts/demo_digital_twin.py`): o RUL previsto segue a tendência geral do
RUL verdadeiro, mas com variância visível — isso é esperado e não é um bug
a mais: `LinearExtrapolationRULEstimator` é deliberadamente um baseline
estatístico simples (regressão linear sobre uma janela do Health Index).
Ele existe para ser superado pela Fase 6, não para ser perfeito. Se ele já
fosse preciso o suficiente, isso enfraqueceria — não fortaleceria — a
justificativa científica de investir em modelos de ML mais complexos
(Fase 12 exige exatamente essa comparação).

## O que falta para a Fase 5 ser considerada "completa" no sentido do master prompt

- [ ] Persistência real (Postgres/TimescaleDB) — hoje é `InMemoryDigitalTwinRepository`.
- [ ] Calibração de `failure_threshold` usando dados reais (C-MAPSS) em vez de só frota sintética.
- [ ] Decomposição explícita de incerteza aleatória vs. epistêmica (H3 da Fase 2) — o estimador atual retorna só um intervalo agregado.
- [ ] Anomaly Detection como Bounded Context próprio (hoje é uma regra simples dentro do Digital Twin Context).
