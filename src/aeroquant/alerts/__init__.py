"""Alertas em tempo real (Webhook / WhatsApp Cloud API)."""
from aeroquant.alerts.channels import WebhookChannel, WhatsAppCloudAPIChannel
from aeroquant.alerts.dispatcher import AlertDispatcher, risk_to_alert
from aeroquant.alerts.domain import AlertDispatchResult, AlertEvent

__all__ = [
    "AlertDispatcher",
    "AlertDispatchResult",
    "AlertEvent",
    "WebhookChannel",
    "WhatsAppCloudAPIChannel",
    "risk_to_alert",
]
