# NASA C-MAPSS (no repositório / auto-download)

## Arquivos de treino

Coloque (ou baixe) em `data/external/`:

```
train_FD001.txt … train_FD004.txt
test_FD001.txt  … test_FD004.txt
RUL_FD001.txt   … RUL_FD004.txt
```

```bash
python scripts/download_cmapss.py
```

Fonte: https://data.nasa.gov/docs/legacy/CMAPSSData.zip

O experimento `cmapss_experiment` chama o download automaticamente se `train_FD001.txt` não existir.

| Subset | Train engines | Regimes |
|--------|---------------|--------|
| FD001 | 100 | 1 |
| FD002 | 260 | 6 |
| FD003 | 100 | 1 |
| FD004 | 249 | 6 |
