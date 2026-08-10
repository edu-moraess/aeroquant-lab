# Alertas em tempo real (Webhook / WhatsApp)

Módulo: `aeroquant.alerts` — **Telegram removido**.

## Canais

| Canal | Configuração | Uso |
|-------|--------------|-----|
| Webhook | `AEROQUANT_WEBHOOK_URL` | POST JSON / Slack |
| WhatsApp CallMeBot | `AEROQUANT_WHATSAPP_PHONE` + `AEROQUANT_WHATSAPP_APIKEY` | Demo / pessoal |
| WhatsApp Cloud API | `AEROQUANT_WA_TOKEN` + `AEROQUANT_WA_PHONE_NUMBER_ID` + phone | Produção Meta |

## Política

Só **CRITICAL** e **HIGH** disparam por padrão (ajustável na UI).

## UI

Aba **Risk** → **Alertas em tempo real**.
