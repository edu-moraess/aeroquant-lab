# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

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
- **Saturação prematura de sensores**: o efeito de degradação escalava pelo baseline do sensor em vez do range válido (`valid_max - baseline`), fazendo sensores de alto acoplamento saturarem no teto do range aos ~15% da vida útil. Corrigido para escalar pelo headroom real.

## [0.1.0] — Fase 1 (Arquitetura) + Fase 2 (Pergunta Científica)

### Adicionado
- Auditoria crítica do escopo original (13 fases) e recomendação de priorização: núcleo científico = RUL + Digital Twin, demais fases como suporte.
- Arquitetura completa: Clean Architecture + DDD, 9 bounded contexts, stack tecnológica justificada.
- Pergunta científica, hipóteses (H1–H3), objetivos, contribuição científica posicionada como metodológica (não "mais uma arquitetura de RUL"), lacunas na literatura e mapeamento de venues de publicação ativos (PHM Society NA/Europe 2026).

## Pendente (não implementado — não fabricado como se existisse)
- Fase 4, Níveis 2–3: ingestão real do C-MAPSS e treino híbrido (bloqueado por acesso à rede — arquivos precisam ser enviados manualmente).
- Fase 6: modelos de Machine Learning para RUL/anomalia/classificação (o baseline estatístico da Fase 5 existe justamente para ser o piso de comparação).
- Fases 7–13: Computer Vision, Simulação Monte Carlo, XAI, Dashboard de produção, MLOps completo (CI/CD real, MLflow, DVC), Validação Científica formal, Publicação.
