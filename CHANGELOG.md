# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

## [0.4.1] — Digital Twin: incerteza do HI data-driven + Dashboard Fase 10

### Adicionado
- `OnlineFleetBaseline.stats_with_n()` — expõe o contador Welford `n` por sensor/condição.
- `ZScoreHealthIndexEstimator` passa a reduzir a incerteza do Health Index com a evidência do baseline (`0.15 / sqrt(n_min)`, floor 0.02) quando `n` está disponível; mantém 0.15 fixo em modo legado (compatibilidade).
- `UpdateDigitalTwin` prefere `stats_with_n` via duck-typing.
- Dashboard Streamlit (Fase 10): `requirements.txt` na raiz (corrige plotly no Cloud), residual RUL, evolução da incerteza OLS, heatmap de frota, trajetórias por modo de falha, ranking e histograma.

### Motivação científica
- Aproxima a hipótese H3 (decomposição de incerteza) sem exigir modelo bayesiano completo: a incerteza do HI deixa de ser constante arbitrária e passa a refletir quanta evidência o baseline da frota já acumulou.

## [0.4.0] — Fase 6 (Machine Learning)

### Adicionado
- ML Context: `TrainAndCompareModels` (use case), 3 trainers scikit-learn (`LinearRegressionTrainer`, `RandomForestTrainer`, `GradientBoostingQuantileTrainer`), métricas (`RMSE`, `MAE`, NASA asymmetric score), split por unidade (`split_by_unit`) sem vazamento.
- `GradientBoostingQuantileTrainer` produz intervalo de predição real (quantile loss, alpha 0.05/0.5/0.95) — primeira quantificação de incerteza aprendida do projeto (vs. o intervalo OLS do baseline).
- Comparação final justa: os 3 modelos de ML avaliados nas MESMAS unidades de teste que o baseline da Fase 5, em modo streaming, mesma métrica. Resultado: `gradient_boosting_quantile` supera o baseline em RMSE por \~7x (22.3 vs. 155.8).
- 11 novos testes (7 unit + 4 integration) — 24 no total no projeto.
- PyTorch/deep learning (LSTM/Transformer, previstos na arquitetura da Fase 1) NÃO foram implementados — sem acesso à rede para instalar `torch` neste ambiente. Documentado como próximo passo, não fabricado.

### Corrigido
- **Violação de Clean Architecture própria**: `ml/domain/entities.py` importava de `ml/infrastructure/` (direção errada). Corrigido movendo `RULMetrics` para `domain/value_objects.py` — infrastructure agora importa de domain, não o contrário.
- **Baseline da Fase 5 "explodindo" em extrapolação**: ao comparar contra ML pela primeira vez, ficou evidente que `LinearExtrapolationRULEstimator` podia projetar RUL de milhares de ciclos quando a inclinação da regressão local ficava muito pequena (mas positiva) — erro de inversão de regressão sem limite. Corrigido com um teto de extrapolação (3x os ciclos já observados da unidade). RMSE do baseline caiu de 3815 para 155.8 (ainda pior que ML, mas agora um número real, não uma explosão numérica).
- **Overflow na métrica NASA**: `exp()` de um erro muito grande (antes da correção acima) virava `inf`, quebrando a comparação. Adicionado clipping de segurança em `nasa_asymmetric_score`.
- **Teste com propriedade estatística boa demais para ser verdade**: `test_rul_uncertainty_shrinks_as_data_accumulates` assumia que a incerteza do baseline encolhe monotonicamente até o fim da vida. Achado real: perto do fim de vida, sensores de alto acoplamento saturam no teto do range a cada ciclo (Health Index fica momentaneamente achatado), a regressão local perde inclinação detectável, e o estimador cai — corretamente — no fallback "não extrapolar". O teste foi reescrito para comparar início vs. meio de vida (a propriedade real que importa).

## [0.3.0] — Fase 5 (Digital Twin)

### Adicionado
- Digital Twin Context completo: `DigitalTwinState`/`DigitalTwinSnapshot` (domínio), `UpdateDigitalTwin` (use case), estimadores de Health Index (z-score ponderado por acoplamento), RUL (extrapolação linear com intervalo de predição OLS) e baseline de frota online (Welford).
- Calibração empírica do limiar de falha (`calibrate_failure_threshold`) a partir de uma frota de referência sintética.
- Detecção de anomalia por desvio de HI em relação à tendência recente (z-score de incrementos).
- 4 testes de integração cobrindo convergência de RUL, encolhimento de incerteza e detecção de anomalia contra falha abrupta injetada.

### Corrigido
- **Contaminação do baseline**: o baseline de frota estava sendo atualizado com leituras já degradadas da própria unidade monitorada, fazendo a "média saudável" perseguir o sinal de degradação e o Health Index cair perto do fim de vida (o oposto do esperado). Corrigido travando a atualização do baseline a uma janela inicial de vida assumida-saudável.
- **Baseline por condição operacional sub-amostrado**: com 6 regimes operacionais simulados mas nenhum comportamento condition-dependent real nos sensores, o baseline por condição ficava instável (poucas amostras por regime). Reduzido o padrão para 1 condição até o gerador ganhar comportamento condition-dependent de verdade.
- **Limiar de falha arbitrário**: `failure_threshold=1.0` fixo não tinha relação com a escala real do Health Index (soma ponderada de z-scores, não limitada a [0,1]), fazendo o RUL previsto colapsar perto de zero desde o início. Substituído por calibração empírica contra uma frota de referência.

## [0.2.0] — Fase 3 (Engenharia de Dados) + Fase 4 Nível 1 (Dados Sintéticos)

### Adicionado
- Sensor Data Context completo em Clean Architecture (domain/application/infrastructure).
- `StochasticSensorGenerator`: degradação via processo Gamma, deriva de sensor, ruído gaussiano, falha abrupta (Bernoulli) e intermitente.
- Pipeline de ETL: limpeza, normalização por condição operacional, feature engineering (rolling stats + delta), rótulo de RUL piecewise-linear, seleção de variáveis.
- `DataQualityChecker` (substituto local do Great Expectations, indisponível offline).
- `CMAPSSAdapter` para o formato NASA C-MAPSS (não testado contra arquivo real — nenhum enviado até esta versão).
- 9 testes de propriedade do gerador e do pipeline de ETL.

### Corrigido
- **Saturação prematura de sensores**: o efeito de degradação escalava pelo baseline do sensor em vez do range válido (`valid_max - baseline`), fazendo sensores de alto acoplamento saturarem no teto do range aos \~15% da vida útil. Corrigido para escalar pelo headroom real.

## [0.1.0] — Fase 1 (Arquitetura) + Fase 2 (Pergunta Científica)

### Adicionado
- Auditoria crítica do escopo original (13 fases) e recomendação de priorização: núcleo científico = RUL + Digital Twin, demais fases como suporte.
- Arquitetura completa: Clean Architecture + DDD, 9 bounded contexts, stack tecnológica justificada.
- Pergunta científica, hipóteses (H1–H3), objetivos, contribuição científica posicionada como metodológica (não "mais uma arquitetura de RUL"), lacunas na literatura e mapeamento de venues de publicação ativos (PHM Society NA/Europe 2026).

## Pendente (não implementado — não fabricado como se existisse)
- Fase 4, Níveis 2–3: ingestão real do C-MAPSS e treino híbrido (bloqueado por acesso à rede — arquivos precisam ser enviados manualmente).
- Fases 7–13: Computer Vision, Simulação Monte Carlo, XAI, Dashboard de produção, MLOps completo (CI/CD real, MLflow, DVC), Validação Científica formal, Publicação.
