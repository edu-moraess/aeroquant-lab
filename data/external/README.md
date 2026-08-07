# Dados Externos — NASA C-MAPSS

**Status: PENDENTE.** Nenhum arquivo real foi baixado ou processado ainda —
o ambiente de desenvolvimento original não tinha acesso à rede.

## O que colocar aqui

Baixe do repositório oficial NASA Prognostics Data Repository e coloque
nesta pasta, sem alterar os nomes:

```
data/external/
├── train_FD001.txt   train_FD002.txt   train_FD003.txt   train_FD004.txt
├── test_FD001.txt    test_FD002.txt    test_FD003.txt    test_FD004.txt
└── RUL_FD001.txt      RUL_FD002.txt     RUL_FD003.txt     RUL_FD004.txt
```

## Depois de colocar os arquivos

O adaptador já está pronto em
`src/aeroquant/sensor_data/infrastructure/adapters/cmapss_adapter.py`
(`CMAPSSAdapter.parse(filepath, schema)`), mas **não foi testado contra um
arquivo real** — ele segue a documentação oficial do formato, não uma
validação empírica. Rode e inspecione manualmente antes de confiar:

```python
from aeroquant.sensor_data.infrastructure.adapters.cmapss_adapter import CMAPSSAdapter
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema

schema = build_cmapss_like_schema()
readings = CMAPSSAdapter().parse("data/external/train_FD001.txt", schema)
print(len(readings), readings[0])
```

Pontos a validar manualmente na primeira execução:
1. Contagem de unidades e ciclos bate com a documentação (FD001: 100 unidades de treino).
2. `_encode_operating_condition` está hardcoded para retornar `0` — para FD002/FD004 (6 regimes reais), isso precisa virar clustering sobre os 3 `op_setting`s antes de qualquer resultado ser confiável.
