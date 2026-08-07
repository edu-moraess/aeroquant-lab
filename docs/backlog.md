# Backlog

Itens menores que não justificam uma entrada própria no `roadmap.md`, mas
que ficaram anotados durante o desenvolvimento das Fases 1–5.

## Dívida técnica conhecida

- [ ] `CSVSensorRepository` é append-only sem proteção contra duplicação em
      re-execuções (causou um falso-positivo de qualidade de dados durante
      o desenvolvimento — ver histórico de sessão). Adicionar opção de
      overwrite explícito ou detecção de re-execução.
- [ ] `ZScoreHealthIndexEstimator.estimate` retorna uma incerteza fixa
      (`0.15`) em vez de calculada — está marcado no próprio código como
      candidato a refinamento na Fase 6.
- [ ] `CMAPSSAdapter._encode_operating_condition` é um placeholder (sempre
      retorna `0`) — precisa de clustering real (k-means, k=6) assim que
      houver arquivo C-MAPSS real para calibrar contra.
- [ ] `LinearExtrapolationRULEstimator` pode gerar intervalos muito largos
      quando a inclinação estimada da regressão fica perto de zero
      (instabilidade conhecida de inversão de regressão) — funcionalmente
      correto (reflete incerteza real), mas vale considerar um teto de
      largura de intervalo mais elegante que o comportamento atual.

## Melhorias de infraestrutura

- [ ] Trocar `InMemoryDigitalTwinRepository` por persistência real antes de
      qualquer uso além de demos locais.
- [ ] Validar de fato `src/aeroquant/api/main.py` e `dashboards/streamlit_app.py`
      assim que houver ambiente com rede — atualmente escritos mas não
      executados.
- [ ] Popular `requirements/mlops.txt` de verdade (MLflow, DVC) quando a
      Fase 6 começar a gerar experimentos para versionar.

## Perguntas em aberto (não são bugs, são decisões pendentes)

- [ ] Vale a pena migrar comentários/docstrings para inglês se o objetivo
      é publicação internacional (Fase 13)? Hoje está em português por
      consistência com o restante da comunicação do projeto.
- [ ] `failure_threshold` calibrado por percentil 50 (mediana) — vale testar
      sensibilidade a essa escolha (25/75) antes de fixar como padrão.
