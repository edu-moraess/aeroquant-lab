"""Dispatcher: RiskAssessment → AlertEvent → Webhook / WhatsApp Cloud."""
from __future__ import annotations

from aeroquant.alerts.channels import AlertChannel, WebhookChannel, WhatsAppCloudAPIChannel
from aeroquant.alerts.domain import AlertDispatchResult, AlertEvent
from aeroquant.risk.assessment import RiskAssessment

DEFAULT_TRIGGER_LEVELS = frozenset({"CRITICAL", "HIGH"})


def risk_to_alert(
    risk: RiskAssessment, *, unit_id: str = "fleet", title: str | None = None,
) -> AlertEvent:
    level = risk.level.upper()
    return AlertEvent(
        level=level,
        title=title or f"Turbofan risk: {level}",
        message=risk.rationale,
        unit_id=unit_id,
        expected_rul=risk.expected_rul,
        p10=risk.p10,
        p50=risk.p50,
        p90=risk.p90,
        maintenance_threshold=risk.maintenance_threshold,
        prob_below_threshold=risk.prob_below_threshold,
        source="AeroQuant Lab · Predictive Maintenance",
    )


class AlertDispatcher:
    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        trigger_levels: frozenset[str] | None = None,
    ) -> None:
        self.channels = channels or []
        self.trigger_levels = trigger_levels or DEFAULT_TRIGGER_LEVELS

    @classmethod
    def from_env(
        cls,
        *,
        webhook_url: str | None = None,
        whatsapp_phone: str | None = None,
        wa_token: str | None = None,
        wa_phone_number_id: str | None = None,
        trigger_levels: frozenset[str] | None = None,
    ) -> AlertDispatcher:
        channels: list[AlertChannel] = []
        wh = WebhookChannel(url=webhook_url)
        if wh.url:
            channels.append(wh)
        cloud = WhatsAppCloudAPIChannel(
            token=wa_token, phone_number_id=wa_phone_number_id, to_phone=whatsapp_phone,
        )
        if cloud.token and cloud.phone_number_id and cloud.to_phone:
            channels.append(cloud)
        return cls(channels=channels, trigger_levels=trigger_levels)

    def should_dispatch(self, event: AlertEvent) -> bool:
        return event.level.upper() in self.trigger_levels

    def dispatch(self, event: AlertEvent, *, force: bool = False) -> list[AlertDispatchResult]:
        if not force and not self.should_dispatch(event):
            return [AlertDispatchResult(
                channel="policy", ok=True,
                detail=f"Nível {event.level} fora dos triggers; não enviado.",
            )]
        if not self.channels:
            return [AlertDispatchResult(
                channel="none", ok=False,
                detail="Nenhum canal configurado (Webhook e/ou WhatsApp Cloud API).",
            )]
        return [ch.send(event) for ch in self.channels]

    def dispatch_risk(self, risk: RiskAssessment, *, unit_id: str = "fleet", force: bool = False):
        return self.dispatch(risk_to_alert(risk, unit_id=unit_id), force=force)
