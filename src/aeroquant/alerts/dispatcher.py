"""Dispatcher: transforma RiskAssessment → AlertEvent e envia aos canais."""
from __future__ import annotations

from aeroquant.alerts.channels import (
    AlertChannel,
    WebhookChannel,
    WhatsAppCallMeBotChannel,
    WhatsAppCloudAPIChannel,
)
from aeroquant.alerts.domain import AlertDispatchResult, AlertEvent
from aeroquant.risk.assessment import RiskAssessment

DEFAULT_TRIGGER_LEVELS = frozenset({"CRITICAL", "HIGH"})


def risk_to_alert(
    risk: RiskAssessment,
    *,
    unit_id: str = "fleet",
    title: str | None = None,
) -> AlertEvent:
    level = risk.level.upper()
    title = title or f"Turbofan risk: {level}"
    return AlertEvent(
        level=level,
        title=title,
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
    """Orquestra canais de alerta com filtro por severidade."""

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
        whatsapp_apikey: str | None = None,
        wa_token: str | None = None,
        wa_phone_number_id: str | None = None,
        trigger_levels: frozenset[str] | None = None,
    ) -> AlertDispatcher:
        channels: list[AlertChannel] = []
        wh = WebhookChannel(url=webhook_url)
        if wh.url:
            channels.append(wh)

        cmb = WhatsAppCallMeBotChannel(phone=whatsapp_phone, apikey=whatsapp_apikey)
        if cmb.phone and cmb.apikey:
            channels.append(cmb)
        else:
            cloud = WhatsAppCloudAPIChannel(
                token=wa_token,
                phone_number_id=wa_phone_number_id,
                to_phone=whatsapp_phone,
            )
            if cloud.token and cloud.phone_number_id and cloud.to_phone:
                channels.append(cloud)

        return cls(channels=channels, trigger_levels=trigger_levels)

    def should_dispatch(self, event: AlertEvent) -> bool:
        return event.level.upper() in self.trigger_levels

    def dispatch(
        self,
        event: AlertEvent,
        *,
        force: bool = False,
    ) -> list[AlertDispatchResult]:
        if not force and not self.should_dispatch(event):
            return [
                AlertDispatchResult(
                    channel="policy",
                    ok=True,
                    detail=f"Nível {event.level} fora dos triggers {sorted(self.trigger_levels)}; não enviado.",
                )
            ]
        if not self.channels:
            return [
                AlertDispatchResult(
                    channel="none",
                    ok=False,
                    detail="Nenhum canal configurado (Webhook e/ou WhatsApp).",
                )
            ]
        return [ch.send(event) for ch in self.channels]

    def dispatch_risk(
        self,
        risk: RiskAssessment,
        *,
        unit_id: str = "fleet",
        force: bool = False,
    ) -> list[AlertDispatchResult]:
        event = risk_to_alert(risk, unit_id=unit_id)
        return self.dispatch(event, force=force)
