# Alertas em tempo real (Webhook / Telegram)

Módulo: `aeroquant.alerts`

## Canais

| Canal | Env / UI | Payload |
|-------|----------|--------|
| Webhook | `AEROQUANT_WEBHOOK_URL` | POST JSON `{text, alert}` |
| Telegram | `AEROQUANT_TELEGRAM_BOT_TOKEN` + `AEROQUANT_TELEGRAM_CHAT_ID` | Bot API `sendMessage` HTML |

## Política

Por padrão só **CRITICAL** e **HIGH** disparam envio (configurável na UI Risk).

## Integração

```python
from aeroquant.alerts import AlertDispatcher
from aeroquant.risk.assessment import assess_risk

risk = assess_risk(point_estimate=12.0, maintenance_threshold=30.0)
dispatcher = AlertDispatcher.from_env()
results = dispatcher.dispatch_risk(risk, unit_id="eng-0001")
```

UI: aba **Risk** → seção **Alertas em tempo real**.
