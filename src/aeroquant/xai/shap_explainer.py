"""
Explicabilidade SHAP para modelos de RUL (Fase 9).

Usa TreeExplainer para Random Forest / Gradient Boosting (exato e rápido)
e fallback de permutation importance quando o modelo não é árvore.

Retorna:
  - mean |SHAP| por feature (importância global)
  - valores SHAP de uma amostra (explicação local)
  - base value do modelo

Honestidade: SHAP explica o *modelo*, não o processo físico de degradação.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore


@dataclass(frozen=True)
class SHAPExplanation:
    feature_importance: pd.DataFrame  # feature, mean_abs_shap
    local_shap: pd.DataFrame | None  # feature, shap_value (uma amostra)
    base_value: float
    method: str  # "tree_shap" | "permutation"
    n_samples_explained: int


def explain_model(
    model,
    X: pd.DataFrame,
    *,
    max_samples: int = 200,
    local_index: int = 0,
    seed: int = 0,
) -> SHAPExplanation:
    """Explica `model` sobre o DataFrame de features `X`."""
    estimator = getattr(model, "predictor", model)
    if isinstance(estimator, dict):
        estimator = estimator.get(0.5) or next(iter(estimator.values()))

    X_work = X.copy()
    if len(X_work) > max_samples:
        X_work = X_work.sample(n=max_samples, random_state=seed)

    feature_names = list(X_work.columns)

    if shap is not None and _is_tree_model(estimator):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_work)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        importance = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        idx = min(local_index, len(X_work) - 1)
        local = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": shap_values[idx],
                "feature_value": X_work.iloc[idx].to_numpy(),
            }
        ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        return SHAPExplanation(
            feature_importance=importance.head(20),
            local_shap=local.head(15),
            base_value=base,
            method="tree_shap",
            n_samples_explained=len(X_work),
        )

    from sklearn.inspection import permutation_importance

    y_dummy = estimator.predict(X_work)
    result = permutation_importance(
        estimator, X_work, y_dummy, n_repeats=5, random_state=seed,
        scoring="neg_mean_squared_error",
    )
    importance = (
        pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": result.importances_mean}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return SHAPExplanation(
        feature_importance=importance.head(20),
        local_shap=None,
        base_value=float(np.mean(y_dummy)),
        method="permutation",
        n_samples_explained=len(X_work),
    )


def _is_tree_model(estimator) -> bool:
    name = type(estimator).__name__.lower()
    return any(
        k in name
        for k in ("forest", "boosting", "tree", "xgb", "lgbm", "catboost", "histogram")
    )
