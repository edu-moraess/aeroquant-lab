"""Pipeline: Health → RUL → MC → Risk → Decision."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aeroquant.decision.maintenance import MaintenanceRecommendation, recommend_maintenance
from aeroquant.platform.health_state import AircraftHealthState, estimate_health_from_z
from aeroquant.platform.risk_engine import IntegratedRisk, compute_integrated_risk
from aeroquant.prognostics.bias_correction import (
    BiasReport, apply_bias_correction, compute_bias_report, fit_bias_correction,
)


@dataclass
class UnitSnapshot:
    unit_id: str
    health: AircraftHealthState
    expected_rul: float
    p10: float
    p50: float
    p90: float
    prob_fail_30: float
    risk: IntegratedRisk
    decision: MaintenanceRecommendation
    anomaly_severity: str = "NORMAL"


@dataclass
class FleetPipelineResult:
    units: list[UnitSnapshot]
    bias_report: BiasReport | None
    failure_regions: list[dict] = field(default_factory=list)
    protocol_note: str = ""
    n_simulations: int = 0


def monte_carlo_rul_samples(
    expected: float, p10: float, p90: float, n: int = 5000, seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    med = max(expected, 1e-3)
    if p90 > p10 > 0:
        sigma = max((np.log(p90) - np.log(max(p10, 1e-3))) / (2 * 1.2816), 0.05)
    else:
        sigma = 0.25
    return rng.lognormal(np.log(med), sigma, size=n)


def build_unit_snapshot(
    *,
    unit_id: str,
    unit_df: pd.DataFrame,
    expected_rul: float,
    p10: float,
    p50: float,
    p90: float,
    anomaly_severity: str = "NORMAL",
    late_failure_risk: float = 0.5,
    n_mc: int = 3000,
    seed: int = 42,
) -> UnitSnapshot:
    health = estimate_health_from_z(unit_df, unit_id=unit_id)
    samples = monte_carlo_rul_samples(expected_rul, p10, p90, n=n_mc, seed=seed)
    prob_30 = float(np.mean(samples <= 30.0))
    risk = compute_integrated_risk(
        prob_fail_horizon=prob_30,
        health_score=health.overall_score,
        anomaly_severity=anomaly_severity,
        interval_width=float(p90 - p10),
        expected_rul=expected_rul,
        late_failure_risk=late_failure_risk,
    )
    decision = recommend_maintenance(
        risk_level=risk.level,
        expected_rul=expected_rul,
        p10=p10,
        prob_fail_30=prob_30,
        anomaly_severity=anomaly_severity,
        health_score=health.overall_score,
    )
    return UnitSnapshot(
        unit_id=unit_id, health=health, expected_rul=expected_rul,
        p10=p10, p50=p50, p90=p90, prob_fail_30=prob_30,
        risk=risk, decision=decision, anomaly_severity=anomaly_severity,
    )


def apply_pipeline_bias_correction(y_true, y_pred):
    report = compute_bias_report(y_true, y_pred)
    bias = fit_bias_correction(y_true, y_pred)
    resid = np.asarray(y_pred, float) - np.asarray(y_true, float)
    resid = resid[np.isfinite(resid)]
    std = float(np.std(resid)) if len(resid) > 2 else 10.0
    corr = apply_bias_correction(y_pred, bias, residual_std=std)
    return corr.point, corr.p10, corr.p90, report, bias
