# AeroQuant Lab — Arquitetura PHM Integrada

## Fluxo

DATA → FEATURES → HEALTH STATE → ANOMALY → RUL (+ bias correction) → UNCERTAINTY → MONTE CARLO → RISK → DECISION → COMMAND CENTER

## Positive bias

Mean error histórico ≈ **+3.7 ciclos** (superestima RUL).
Correção: `ŷ_corr = ŷ − bias` + bandas P10/P90.
**Late Failure Risk** = taxa de overestimation.

## Módulos

- `platform/health_state.py`, `risk_engine.py`, `pipeline.py`
- `prognostics/bias_correction.py`
- `decision/maintenance.py`
- `dashboards/command_center.py`

## Limitações

Decision support apenas. C-MAPSS/sintético. MC log-normal aproximado.
