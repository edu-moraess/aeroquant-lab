# models/

Placeholder intencional. Nenhum modelo treinado é versionado aqui ainda.

## O que existe hoje (Fase 6)

Os modelos de Machine Learning são treinados e avaliados **em tempo de execução**
pelos scripts e testes (não serializados em disco nesta versão):

- `LinearRegressionTrainer`
- `RandomForestTrainer`
- `GradientBoostingQuantileTrainer` (com intervalos de quantis)

O artefato de comparação gerado por `scripts/demo_ml_vs_baseline.py` fica em
`outputs/ml_vs_baseline_comparison.png`.

## O que NÃO existe (e não deve ser inventado)

- Arquivos `.pkl` / `.joblib` / `.pt` de modelos treinados.
- Checkpoints de deep learning.
- Artefatos de MLflow ou DVC.

Quando a Fase 6 for estendida com persistência real (ou quando deep learning
for adicionado), este diretório passará a receber os artefatos versionados
e este README será atualizado.

## Limitações explícitas da rodada atual

- Modelos existem apenas como objetos em memória durante a execução do
  script, não como arquivo carregável.
- Treino sobre dados reais (C-MAPSS) — tudo acima é só frota sintética.
- Deep learning (LSTM/Transformer) — sem `torch` neste ambiente sem rede.