"""Domínio de alertas de falha crítica (turbofan / RUL)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AlertEvent:
    """Evento de alerta emitido pelo sistema de health monitoring."""

    level: str  # CRITICAL | HIGH | MEDIUM | LOW | INFO
    title: str
    message: str
    unit_id: str = "fleet"
    expected_rul: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    maintenance_threshold: float | None = None
    prob_below_threshold: float | None = None
    source: str = "AeroQuant Lab"
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def telegram_html(self) -> str:
        emoji = {
            "CRITICAL": "🚨",
            "HIGH": "⚠️",
            "MEDIUM": "🟠",
            "LOW": "🟢",
            "INFO": "ℹ️",
        }.get(self.level, "📢")
        lines = [
            f"{emoji} <b>{self.level}</b> — {self.title}",
            f"<code>{self.unit_id}</code>",
            self.message,
        ]
        if self.expected_rul is not None:
            lines.append(f"RUL esperado: <b>{self.expected_rul:.1f}</b> ciclos")
        if self.p10 is not None and self.p90 is not None:
            lines.append(f"P10–P90: {self.p10:.0f} – {self.p90:.0f}")
        if self.prob_below_threshold is not None and self.maintenance_threshold is not None:
            lines.append(
                f"P(RUL &lt; {self.maintenance_threshold:.0f}) = {100 * self.prob_below_threshold:.0f}%"
            )
        lines.append(f"<i>{self.source} · {self.timestamp_utc}</i>")
        return "\n".join(lines)


@dataclass(frozen=True)
class AlertDispatchResult:
    channel: str
    ok: bool
    detail: str
    status_code: int | None = None
