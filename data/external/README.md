# Dados Externos — NASA C-MAPSS

## Download

```bash
python scripts/download_cmapss.py
```

Fonte: `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`

## Subsets (todos disponíveis após o download)

| Subset | Train engines | Test | Regimes | Falhas | Tamanho approx. |
|--------|---------------|------|---------|--------|-----------------|
| FD001 | 100 | 100 | 1 | 1 (HPC) | ~3.5 MB train |
| **FD002** | **260** | **259** | **6** | 1 (HPC) | **~9 MB** |
| FD003 | 100 | 100 | 1 | 2 | ~4 MB |
| **FD004** | **249** | **248** | **6** | **2** | **~10 MB** |

FD002 e FD004 são os datasets grandes multi-regime do C-MAPSS clássico.

## Protocolo no AeroQuant Lab

1. `OperatingConditionEncoder` (KMeans) **fit só no treino**
2. `fit_normalize_stats` **só no treino**
3. Features rolling causais
4. Avaliação no test oficial + `RUL_FD00X.txt`
5. Ranking NASA-first

## Uso

```python
from cmapss_experiment import run_cmapss_experiment, list_available_subsets
print(list_available_subsets())
res = run_cmapss_experiment(subset="FD002")  # multi-regime grande
print(res.ranked_table)
```

## Nota

C-MAPSS é **simulação NASA**, não telemetria de frota comercial.
N-CMAPSS (HDF5, GB) ainda não está integrado — próximo passo opcional.
