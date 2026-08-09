# AeroQuant Lab — Auditoria técnica (P0/P1)

## Problemas encontrados

### P0 — Segurança metodológica

1. **Data leakage na normalização (CRÍTICO)**  
   `normalize()` era aplicado no DataFrame completo *antes* do split train/test.  
   **Correção:** `fit_normalize_stats` só no treino; `apply_normalize` em val/test.

2. **Sequence experiment com o mesmo leakage** — corrigido.

3. **Ausência de validation set** — `split_by_unit_three_way` disponível; benchmark usa 3-way.

4. **NASA Score** — fórmula correta (d=pred−true; /10 vs /13). Testes preservados.

5. **Target alignment** — RUL do último ciclo da janela (sem futuro). OK.

### P1 — Avaliação

6. **Métricas estendidas:** R², Bias, P50/P90/P95 Abs Err, buckets por faixa de RUL.
7. **Ranking configurável** (NASA-first).
8. **Protocolo documentado** entre tabular e sequencial (N pode diferir).

## Status de leakage

| Risco | Status |
|-------|--------|
| Split por linha | Mitigado (unit_id) |
| Scale com stats do teste | **Corrigido** |
| Target futuro na janela | OK |
| Rolling com futuro | OK (causal) |

## Próximos: P2 (val loss/checkpoint) → P3 (incerteza) → P4 (risk) → P5 (UI) → P6 (docs)

**Limitação:** dados sintéticos ≠ operação real.
