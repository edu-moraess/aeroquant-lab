"""Risk Engine — Failure × Health × Anomaly × Uncertainty × Late-failure bias."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDriver:
    name: str
    contribution: float


@dataclass(frozen=True)
class IntegratedRisk:
    score: float
    level: str
    drivers: list[RiskDriver]
    rationale: str


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_integrated_risk(
    *,
    prob_fail_horizon: float,
    health_score: float,
    anomaly_severity: str = "NORMAL",
    interval_width: float = 20.0,
    expected_rul: float = 50.0,
    late_failure_risk: float = 0.5,
) -> IntegratedRisk:
    fail_pts = float(np_clip(prob_fail_horizon * 30.0, 0, 30))
    health_pts = float(np_clip((100.0 - health_score) / 100.0 * 25.0, 0, 25))
    anom_pts = float({"NORMAL": 0, "WATCH": 8, "WARNING": 16, "CRITICAL": 25}.get(anomaly_severity.upper(), 8))
    unc_pts = float(np_clip(interval_width / 80.0 * 15.0, 0, 15))
    late_pts = float(np_clip(late_failure_risk * 15.0, 0, 15))
    rul_pts = float(np_clip((40.0 - min(expected_rul, 40.0)) / 40.0 * 10.0, 0, 10))
    score = min(100.0, fail_pts + health_pts + anom_pts + unc_pts + late_pts + rul_pts)
    drivers = sorted([
        RiskDriver("Failure probability", fail_pts),
        RiskDriver("Health degradation", health_pts),
        RiskDriver("Anomaly severity", anom_pts),
        RiskDriver("RUL uncertainty", unc_pts),
        RiskDriver("Late failure risk (bias)", late_pts),
        RiskDriver("Low expected RUL", rul_pts),
    ], key=lambda d: d.contribution, reverse=True)
    level = "CRITICAL" if score >= 75 else "HIGH" if score >= 55 else "MEDIUM" if score >= 35 else "LOW"
    top = ", ".join(f"{d.name} +{d.contribution:.0f}" for d in drivers[:3])
    return IntegratedRisk(score, level, drivers, f"Risk={score:.0f} ({level}). Main drivers: {top}.")
