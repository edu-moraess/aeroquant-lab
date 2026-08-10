"""Maintenance Decision Engine — decision support (não comando operacional)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceRecommendation:
    action: str
    urgency: str
    window: str
    reason: str
    disclaimer: str = (
        "Decision support apenas — não substitui procedimentos de manutenção "
        "nem autoridade de aeronavegabilidade."
    )


def recommend_maintenance(
    *,
    risk_level: str,
    expected_rul: float,
    p10: float,
    prob_fail_30: float,
    anomaly_severity: str = "NORMAL",
    health_score: float = 70.0,
) -> MaintenanceRecommendation:
    risk = risk_level.upper()
    anom = anomaly_severity.upper()
    if risk == "CRITICAL" or p10 < 10 or (anom == "CRITICAL" and expected_rul < 30):
        return MaintenanceRecommendation(
            action="URGENT INSPECTION", urgency="CRITICAL",
            window="Imediato / próximos 5–10 ciclos",
            reason=f"Risk={risk}, P10={p10:.0f}, P(fail≤30)={100*prob_fail_30:.0f}%, anomaly={anom}, health={health_score:.0f}.",
        )
    if risk == "HIGH" or expected_rul < 30 or anom in ("WARNING", "CRITICAL"):
        return MaintenanceRecommendation(
            action="SCHEDULE MAINTENANCE", urgency="HIGH",
            window="Próximos 10–20 ciclos",
            reason=f"Risk={risk}, expected RUL={expected_rul:.0f}, anomaly={anom}, health={health_score:.0f}.",
        )
    if risk == "MEDIUM" or expected_rul < 60 or anom == "WATCH":
        return MaintenanceRecommendation(
            action="INSPECT", urgency="MEDIUM",
            window="Próximos 20–40 ciclos",
            reason=f"Degradação moderada (RUL≈{expected_rul:.0f}, risk={risk}).",
        )
    if health_score < 80 or anom == "WATCH":
        return MaintenanceRecommendation(
            action="MONITOR", urgency="LOW",
            window="Rotina / next A-check window",
            reason=f"Health={health_score:.0f}, anomaly={anom}; acompanhar tendência.",
        )
    return MaintenanceRecommendation(
        action="NO ACTION", urgency="NONE", window="—",
        reason="Estado dentro de faixa nominal para o horizonte avaliado.",
    )
