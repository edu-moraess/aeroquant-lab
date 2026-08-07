# API — Documentação

Implementação: `src/aeroquant/api/main.py` (FastAPI). **Não testada neste
ambiente** (sem fastapi/uvicorn instalados, sem rede) — validar
manualmente antes de uso real. Ver aviso no topo do próprio arquivo.

## Rodando

```bash
pip install -r requirements/api.txt
PYTHONPATH=src uvicorn aeroquant.api.main:app --reload
# Documentação interativa (Swagger) gerada automaticamente pelo FastAPI:
# http://localhost:8000/docs
```

## Endpoints

### `GET /health`
Health check simples. Retorna `{"status": "ok"}`.

### `POST /fleet/generate`
Gera uma frota sintética (não persiste em disco além de um CSV temporário).

**Body:**
```json
{
  "n_units": 30,
  "lifetime_mean": 180,
  "lifetime_std": 35,
  "seed": 42
}
```

**Resposta:**
```json
{
  "n_units": 30,
  "n_readings": 5616,
  "lifetime_mean": 187.2,
  "lifetime_std": 31.6
}
```

### `POST /digital-twin/ingest`
Alimenta uma leitura de sensores no Digital Twin (estado em memória,
compartilhado entre requisições — trocar por persistência real antes de
uso multi-usuário) e retorna o snapshot atualizado.

**Body:**
```json
{
  "unit_id": "aircraft-001",
  "cycle": 42,
  "operating_condition": 0,
  "sensor_values": {"sensor_1": 518.7, "sensor_2": 642.1, "...": "..."}
}
```

**Resposta:**
```json
{
  "cycle": 42,
  "health_index": 0.31,
  "rul_point": 118.4,
  "rul_lower": 95.2,
  "rul_upper": 141.6,
  "is_anomaly": false,
  "anomaly_reason": null
}
```

## Limitações conhecidas desta versão da API

- Estado do Digital Twin é global em memória — reiniciar o processo perde
  todo o histórico. Aceitável para demo, não para produção.
- Não há autenticação/autorização.
- `sensor_values` precisa conter os 21 sensores esperados pelo schema
  (`build_cmapss_like_schema()`) — não há validação amigável de schema
  incompleto ainda (Pydantic vai levantar um erro genérico).
