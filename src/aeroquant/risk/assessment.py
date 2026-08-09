"""Classificação de risco de manutenção a partir de RUL e distribuição."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskThresholds:
    high_rul: float = 30.0
    medium_rul: float = 60.0
    critical_prob: float = 0.25


@dataclass(frozen=True)
class RiskAssessment:
    expected_rul: float
    p10: float
    p50: float
    p90: float
    maintenance_threshold: float
    prob_below_threshold: float
    level: str
    rationale: str


def assess_risk(
    rul_samples: np.ndarray | None = None,
    *,
    point_estimate: float | None = None,
    maintenance_threshold: float = 30.0,
    thresholds: RiskThresholds | None = None,
) -> RiskAssessment:
    thresholds = thresholds or RiskThresholds()
    if rul_samples is not None and len(rul_samples) > 0:
        samples = np.asarray(rul_samples, dtype=float)
        samples = samples[np.isfinite(samples)]
        expected = float(np.mean(samples))
        p10 = float(np.percentile(samples, 10))
        p50 = float(np.percentile(samples, 50))
        p90 = float(np.percentile(samples, 90))
        prob = float(np.mean(samples < maintenance_threshold))
    else:
        if point_estimate is None or not np.isfinite(point_estimate):
            raise ValueError("Informe rul_samples ou point_estimate")
        expected = float(point_estimate)
        p10 = p50 = p90 = expected
        prob = 1.0 if expected < maintenance_threshold else 0.0

    if prob >= thresholds.critical_prob and expected < thresholds.high_rul:
        level = "CRITICAL"
        rationale = f"High probability ({100*prob:.0f}%) of RUL < {maintenance_threshold:.0f} and expected RUL ({expected:.1f}) is low."
    elif expected < thresholds.high_rul or prob >= thresholds.critical_prob:
        level = "HIGH"
        rationale = f"Expected RUL = {expected:.1f} cycles (< {thresholds.high_rul:.0f}) or P(RUL < {maintenance_threshold:.0f}) = {100*prob:.0f}%."
    elif expected < thresholds.medium_rul:
        level = "MEDIUM"
        rationale = f"Expected RUL = {expected:.1f} cycles (between {thresholds.high_rul:.0f} and {thresholds.medium_rul:.0f})."
    else:
        level = "LOW"
        rationale = f"Expected RUL = {expected:.1f} cycles (> {thresholds.medium_rul:.0f}); P(RUL < {maintenance_threshold:.0f}) = {100*prob:.0f}%."

    return RiskAssessment(
        expected_rul=expected, p10=p10, p50=p50, p90=p90,
        maintenance_threshold=maintenance_threshold,
        prob_below_threshold=prob, level=level, rationale=rationale,
    )
