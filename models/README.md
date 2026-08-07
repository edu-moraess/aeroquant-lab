# Modelos Treinados

**Status: PENDENTE — Fase 6 (Machine Learning) ainda não foi implementada.**

Este diretório propositalmente NÃO contém nenhum modelo (nem placeholder
binário fake) — um arquivo `.pt`/`.pkl` vazio ou fabricado seria pior que
não ter nada, porque criaria a ilusão de que existe um modelo treinado.

## O que vai existir aqui quando a Fase 6 acontecer

```
models/
├── baseline/               # o próprio baseline estatístico da Fase 5
│   └── linear_rul_config.yaml   # não é "modelo" no sentido de artefato binário — é config
├── rul/
│   ├── lstm_v1/
│   │   ├── model.pt
│   │   ├── mlflow_run_id.txt
│   │   └── metrics.json
│   └── ...
└── registry -> aponta para o MLflow Model Registry (ver docs/architecture/)
```

## O baseline que EXISTE hoje

O baseline oficial de comparação (`comparação com baseline`, exigido pela
Fase 12) já está implementado e testado — não é um placeholder, é código
real: `src/aeroquant/digital_twin/infrastructure/estimators/linear_extrapolation_rul.py`.
Qualquer modelo de ML da Fase 6 precisa superar esse baseline nas mesmas
métricas para justificar a complexidade adicional.
