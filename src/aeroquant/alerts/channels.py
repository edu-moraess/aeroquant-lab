"""Canais: Webhook HTTP e WhatsApp Meta Cloud API (sem CallMeBot/Telegram)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from aeroquant.alerts.domain import AlertDispatchResult, AlertEvent


class AlertChannel(Protocol):
    name: str

    def send(self, event: AlertEvent) -> AlertDispatchResult: ...


class WebhookChannel:
    name = "webhook"

    def __init__(self, url: str | None = None, timeout: float = 12.0) -> None:
        self.url = (url or os.environ.get("AEROQUANT_WEBHOOK_URL") or "").strip()
        self.timeout = timeout

    def send(self, event: AlertEvent) -> AlertDispatchResult:
        if not self.url:
            return AlertDispatchResult(
                channel=self.name, ok=False,
                detail="Webhook URL não configurada (AEROQUANT_WEBHOOK_URL).",
            )
        payload = {
            "text": f"[{event.level}] {event.title} — {event.unit_id}: {event.message}",
            "alert": event.to_dict(),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "AeroQuantLab-Alerts/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()[:500].decode("utf-8", errors="replace")
                return AlertDispatchResult(
                    channel=self.name, ok=200 <= resp.status < 300,
                    detail=body or f"HTTP {resp.status}", status_code=resp.status,
                )
        except urllib.error.HTTPError as e:
            return AlertDispatchResult(channel=self.name, ok=False, detail=str(e), status_code=e.code)
        except Exception as e:  # noqa: BLE001
            return AlertDispatchResult(channel=self.name, ok=False, detail=str(e))


class WhatsAppCloudAPIChannel:
    name = "whatsapp_cloud"

    def __init__(
        self,
        token: str | None = None,
        phone_number_id: str | None = None,
        to_phone: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.token = (token or os.environ.get("AEROQUANT_WA_TOKEN") or "").strip()
        self.phone_number_id = (
            phone_number_id or os.environ.get("AEROQUANT_WA_PHONE_NUMBER_ID") or ""
        ).strip()
        raw = to_phone or os.environ.get("AEROQUANT_WHATSAPP_PHONE") or ""
        self.to_phone = "".join(c for c in raw if c.isdigit())
        self.timeout = timeout

    def send(self, event: AlertEvent) -> AlertDispatchResult:
        if not self.token or not self.phone_number_id or not self.to_phone:
            return AlertDispatchResult(
                channel=self.name, ok=False,
                detail="WhatsApp Cloud API não configurada (token + phone_number_id + to).",
            )
        url = f"https://graph.facebook.com/v19.0/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": self.to_phone,
            "type": "text",
            "text": {"preview_url": False, "body": event.plain_text()[:4096]},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "AeroQuantLab-Alerts/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()[:800].decode("utf-8", errors="replace")
                return AlertDispatchResult(
                    channel=self.name, ok=200 <= resp.status < 300,
                    detail=body, status_code=resp.status,
                )
        except urllib.error.HTTPError as e:
            err = e.read()[:400].decode("utf-8", errors="replace") if e.fp else str(e)
            return AlertDispatchResult(channel=self.name, ok=False, detail=err, status_code=e.code)
        except Exception as e:  # noqa: BLE001
            return AlertDispatchResult(channel=self.name, ok=False, detail=str(e))
