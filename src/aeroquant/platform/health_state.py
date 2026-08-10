"""Aircraft State Estimation — Health Index multi-sistema (0–100)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_SYSTEM_SENSORS = {
    "Thermal": ["sensor_02", "sensor_03", "sensor_04", "sensor_11", "sensor_17"],
    "Pressure": ["sensor_07", "sensor_09", "sensor_12", "sensor_16"],
    "Flow": ["sensor_08", "sensor_13", "sensor_14", "sensor_20", "sensor_21"],
    "Core": ["sensor_05", "sensor_06", "sensor_10", "sensor_15", "sensor_18", "sensor_19"],
}


@dataclass
class SystemHealth:
    name: str
    score: float
    trend: float
    status: str
    top_sensors: list[str]


@dataclass
class AircraftHealthState:
    unit_id: str
    overall_score: float
    systems: list[SystemHealth]
    status: str


def _status_from_score(score: float) -> str:
    if score >= 80:
        return "NORMAL"
    if score >= 60:
        return "WATCH"
    if score >= 40:
        return "WARNING"
    return "CRITICAL"


def estimate_health_from_z(
    unit_df: pd.DataFrame, *, unit_id: str, z_cols: list[str] | None = None,
) -> AircraftHealthState:
    z_cols = z_cols or [c for c in unit_df.columns if c.endswith("_z")]
    if not z_cols or unit_df.empty:
        return AircraftHealthState(unit_id, 50.0, [], "WATCH")
    last = unit_df.iloc[-1]
    systems: list[SystemHealth] = []
    scores = []
    for sys_name, sensors in _SYSTEM_SENSORS.items():
        cols = [c for c in z_cols if any(c.startswith(s) for s in sensors)]
        if not cols:
            continue
        abs_z = np.nanmean([abs(float(last[c])) for c in cols if pd.notna(last.get(c))])
        score = float(np.clip(100.0 * (1.0 - abs_z / 3.0), 0.0, 100.0))
        if len(unit_df) >= 6:
            prev = unit_df.iloc[-6:-1]
            prev_z = np.nanmean([np.nanmean(np.abs(prev[c].to_numpy(dtype=float))) for c in cols])
            prev_score = float(np.clip(100.0 * (1.0 - prev_z / 3.0), 0.0, 100.0))
            trend = score - prev_score
        else:
            trend = 0.0
        top = sorted(cols, key=lambda c: abs(float(last.get(c, 0) or 0)), reverse=True)[:3]
        systems.append(SystemHealth(sys_name, score, trend, _status_from_score(score), top))
        scores.append(score)
    overall = float(np.mean(scores)) if scores else 50.0
    return AircraftHealthState(unit_id, overall, systems, _status_from_score(overall))
