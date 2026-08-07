# Guia do Pesquisador

Este guia assume que você já leu `docs/science/fase2-pergunta-cientifica.md`
(problema de pesquisa, hipóteses H1–H3, contribuição científica posicionada).

## O que já dá para testar cientificamente hoje

- **H3 (incerteza como decisão)** — parcialmente testável: o baseline
  `LinearExtrapolationRULEstimator` já produz intervalo de predição real
  (não decorativo — cresce com a distância de extrapolação, propriedade
  verificada em `tests/integration/test_digital_twin.py::test_rul_uncertainty_shrinks_as_data_accumulates`).
  Falta comparar decisões de manutenção derivadas desse intervalo contra
  decisões derivadas de estimativa pontual (a parte "decisão" de H3).
- **H1 e H2 (transferência híbrida, quantificação do gap sintético-real)**
  — ainda não testáveis: dependem dos dados C-MAPSS reais (`data/external/`,
  pendente de upload) e de um modelo de ML (Fase 6) para comparar contra
  o baseline sintético-only.

## Como reproduzir os resultados que já existem

```bash
PYTHONPATH=src python3 scripts/demo_digital_twin.py
```

Isso regenera `outputs/digital_twin_rul_tracking.png` com uma seed fixa
(42) — mesmo resultado sempre. Para testar sensibilidade a parâmetros do
gerador (relevante para H2 assim que os dados reais chegarem), edite
`config/generator.yaml` ou passe outros valores de `DegradationParams`
diretamente.

## Rastreabilidade de decisões científicas

Cada bug corrigido na Fase 5 tem uma explicação da causa raiz em
`CHANGELOG.md` — isso não é só "log de engenharia", é material direto para
a seção de Métodos/Limitações de um eventual manuscrito: por exemplo, o
bug do "baseline contaminado" é evidência empírica de por que separar
período de referência saudável de monitoramento contínuo é necessário, não
só uma boa prática arbitrária.

## Checklist antes de reivindicar qualquer resultado como "científico"

Da Fase 12 (`docs/architecture/fase1-auditoria-arquitetura.md`, seção 5):
- [ ] Comparação com baseline (existe: `LinearExtrapolationRULEstimator`)
- [ ] Validação estatística
- [ ] Validação temporal (split por unidade/tempo, nunca k-fold aleatório — ver aviso em `docs/science/fase2-pergunta-cientifica.md`)
- [ ] Benchmark com datasets públicos (bloqueado até `data/external/` ser populado)
- [ ] Estudos de ablação
- [ ] Quantificação de incerteza (parcial — só HI/RUL agregados, falta decomposição aleatória/epistêmica)
- [ ] Avaliação de robustez

Nenhum desses está marcado como completo — é intencional, refletindo o
estado real do projeto na Fase 5 de 13.
