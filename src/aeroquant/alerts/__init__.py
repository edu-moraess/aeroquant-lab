"""Alertas em tempo real (Webhook / WhatsApp) para falhas críticas de turbofan."""
from aeroquant.alerts.channels import (
    WebhookChannel,
    WhatsAppCallMeBotChannel,
    WhatsAppCloudAPIChannel,
)
from aeroquant.alerts.dispatcher import AlertDispatcher, risk_to_alert
from aeroquant.alerts.domain import AlertDispatchResult, AlertEvent

__all__ = [
    "AlertDispatcher",
    "AlertDispatchResult",
    "AlertEvent",
    "WebhookChannel",
    "WhatsAppCallMeBotChannel",
    "WhatsAppCloudAPIChannel",
    "risk_to_alert",
]
