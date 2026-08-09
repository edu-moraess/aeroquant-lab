# Dados Externos — NASA C-MAPSS

```bash
python scripts/download_cmapss.py
```

Fonte: `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`

C-MAPSS é **simulação NASA** (benchmark PHM), não telemetria de frota comercial.

Uso:
```python
from cmapss_experiment import run_cmapss_experiment
res = run_cmapss_experiment(subset="FD001")
print(res.ranked_table)
```
