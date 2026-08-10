"""Canais de notificação: Webhook HTTP e Telegram Bot API."""
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
    """POST JSON para URL configurável (Slack-compatible / genérico)."""

    name = "webhook"

    def __init__(self, url: str | None = None, timeout: float = 12.0) -> None:
        self.url = (url or os.environ.get("AEROQUANT_WEBHOOK_URL") or "").strip()
        self.timeout = timeout

    def send(self, event: AlertEvent) -> AlertDispatchResult:
        if not self.url:
            return AlertDispatchResult(
                channel=self.name,
                ok=False,
                detail="Webhook URL não configurada (AEROQUANT_WEBHOOK_URL).",
            )
        payload = {
            "text": f"[{event.level}] {event.title} — {event.unit_id}: {event.message}",
            "alert": event.to_dict(),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "AeroQuantLab-Alerts/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()[:500].decode("utf-8", errors="replace")
                return AlertDispatchResult(
                    channel=self.name,
                    ok=200 <= resp.status < 300,
                    detail=body or f"HTTP {resp.status}",
                    status_code=resp.status,
                )
        except urllib.error.HTTPError as e:
            return AlertDispatchResult(
                channel=self.name, ok=False, detail=str(e), status_code=e.code
            )
        except Exception as e:  # noqa: BLE001
            return AlertDispatchResult(channel=self.name, ok=False, detail=str(e))


class TelegramChannel:
    """Envia mensagem via Bot API (sendMessage)."""

    name = "telegram"

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout: float = 12.0,
    ) -> None:
        self.bot_token = (bot_token or os.environ.get("AEROQUANT_TELEGRAM_BOT_TOKEN") or "").strip()
        self.chat_id = (chat_id or os.environ.get("AEROQUANT_TELEGRAM_CHAT_ID") or "").strip()
        self.timeout = timeout

    def send(self, event: AlertEvent) -> AlertDispatchResult:
        if not self.bot_token or not self.chat_id:
            return AlertDispatchResult(
                channel=self.name,
                ok=False,
                detail="Telegram não configurado (AEROQUANT_TELEGRAM_BOT_TOKEN / AEROQUANT_TELEGRAM_CHAT_ID).",
            )
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": event.telegram_html(),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "AeroQuantLab-Alerts/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()[:800].decode("utf-8", errors="replace")
                ok = 200 <= resp.status < 300
                return AlertDispatchResult(
                    channel=self.name, ok=ok, detail=body, status_code=resp.status
                )
        except urllib.error.HTTPError as e:
            err_body = e.read()[:400].decode("utf-8", errors="replace") if e.fp else str(e)
            return AlertDispatchResult(
                channel=self.name, ok=False, detail=err_body, status_code=e.code
            )
        except Exception as e:  # noqa: BLE001
            return AlertDispatchResult(channel=self.name, ok=False, detail=str(e))
