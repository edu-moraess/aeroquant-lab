# Guia do Desenvolvedor

## Visão rápida do que existe

| Bounded Context | Status | Localização |
|---|---|---|
| Sensor Data | Completo, testado | `src/aeroquant/sensor_data/` |
| Digital Twin | Completo, testado | `src/aeroquant/digital_twin/` |
| RUL/Anomaly (ML) | Não implementado | Fase 6 |
| Vision Inspection | Não implementado | Fase 7 |
| Simulation (Monte Carlo) | Não implementado | Fase 8 |
| Explainability | Não implementado | Fase 9 |
| Model Lifecycle (MLOps) | Parcial (config CI/Docker existe, MLflow/DVC não integrados) | Fase 11 |
| Presentation (API/Dashboard) | Escrito, não testado (sem rede no dev original) | `src/aeroquant/api/`, `dashboards/` |

## Convenções

- **Type hints em tudo.** `from __future__ import annotations` no topo de cada módulo.
- **Protocol, não ABC**, para ports — permite duck typing em testes sem herança forçada.
- **Dataclasses no domínio**, nunca Pydantic (domínio não deve depender de bibliotecas de infraestrutura — Pydantic é aceitável em `api/`, onde já é fronteira de serialização).
- **Toda função de ETL é pura** (`etl/pipeline.py`) — recebe DataFrame, devolve DataFrame, sem I/O e sem estado.
- **Nomes de arquivo em inglês, comentários e docstrings em português** — reflete como o projeto foi conduzido até aqui; mantenha para consistência, a menos que o projeto vire público internacionalmente (nesse caso, ver `docs/roadmap.md`).

## Como adicionar um novo estimador (ex.: RUL via LSTM na Fase 6)

1. Implemente a interface `RULEstimator` (`digital_twin/application/ports.py`) — só precisa do método `estimate(history, failure_threshold) -> RULEstimate`.
2. Coloque em `digital_twin/infrastructure/estimators/lstm_rul_estimator.py`.
3. Escreva testes em `tests/integration/` comparando contra o baseline (`LinearExtrapolationRULEstimator`) nas mesmas unidades sintéticas/reais — isso é literalmente o requisito da Fase 12.
4. Troque a injeção em `UpdateDigitalTwin(...)` — nenhuma outra linha do use case ou do domínio deveria precisar mudar. Se precisar mudar, é sinal de vazamento de responsabilidade — pare e reavalie a interface antes de continuar.

## Debugging: lições aprendidas nesta sessão (evite repetir)

Ver `CHANGELOG.md` para os 3 bugs reais encontrados. O padrão comum nos
três: **um valor que "parecia razoável" (baseline como escala, 6 condições
porque o C-MAPSS real tem 6, `failure_threshold=1.0` porque HI "deveria"
ser [0,1]) não foi validado empiricamente antes de assumir como correto.**
Sempre que introduzir uma constante ou faixa de valor nova, gere dados
sintéticos e PLOTE antes de assumir que está certo — foi assim que os três
bugs foram achados aqui, não por inspeção de código.

## Rodando localmente

```bash
bash scripts/install.sh
source .venv/bin/activate
PYTHONPATH=src python3 scripts/demo_end_to_end.py
PYTHONPATH=src python3 scripts/demo_digital_twin.py
```
