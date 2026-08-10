"""Constrói fleet_snapshots a partir de resultados de treino (ML / C-MAPSS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aeroquant.platform.pipeline import build_unit_snapshot


def snapshots_from_predictions(
    *,
    unit_ids,
    y_true,
    y_pred,
    p10=None,
    p90=None,
    feature_df=None,
    late_failure_risk: float = 0.5,
    seed: int = 42,
) -> list:
    unit_ids = np.asarray(unit_ids)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if p10 is None:
        p10 = np.clip(y_pred * 0.75, 0, None)
    if p90 is None:
        p90 = y_pred * 1.25
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    df = pd.DataFrame({"unit_id": unit_ids, "y_true": y_true, "y_pred": y_pred, "p10": p10, "p90": p90})
    if feature_df is not None:
        for c in [c for c in feature_df.columns if c.endswith("_z")][:12]:
            df[c] = feature_df[c].to_numpy()
    snaps = []
    for i, uid in enumerate(sorted(df["unit_id"].unique())):
        sub = df[df["unit_id"] == uid]
        last = sub.iloc[-1]
        udata = {"unit_id": uid, "cycle": np.arange(1, len(sub) + 1)}
        zcols = [c for c in sub.columns if c.endswith("_z")]
        if zcols:
            for c in zcols:
                udata[c] = sub[c].to_numpy()
        else:
            udata["sensor_02_z"] = (sub["y_pred"] - sub["y_true"]).to_numpy() / 10.0
        expected = float(last["y_pred"])
        snaps.append(build_unit_snapshot(
            unit_id=str(uid), unit_df=pd.DataFrame(udata),
            expected_rul=expected, p10=float(last["p10"]), p50=expected, p90=float(last["p90"]),
            anomaly_severity="WARNING" if expected < 30 else ("WATCH" if expected < 60 else "NORMAL"),
            late_failure_risk=late_failure_risk, n_mc=1500, seed=seed + i,
        ))
    return snaps
