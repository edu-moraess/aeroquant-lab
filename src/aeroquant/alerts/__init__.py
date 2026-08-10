"""Alertas em tempo real (Webhook / Telegram) para falhas críticas de turbofan."""
from aeroquant.alerts.channels import TelegramChannel, WebhookChannel
from aeroquant.alerts.dispatcher import AlertDispatcher, risk_to_alert
from aeroquant.alerts.domain import AlertDispatchResult, AlertEvent

__all__ = [
    "AlertDispatcher",
    "AlertDispatchResult",
    "AlertEvent",
    "TelegramChannel",
    "WebhookChannel",
    "risk_to_alert",
]
