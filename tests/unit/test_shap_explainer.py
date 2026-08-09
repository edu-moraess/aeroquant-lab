"""Testes de propriedade do explainer SHAP (Fase 9)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from aeroquant.xai.shap_explainer import explain_model


class TestSHAPExplainer(unittest.TestCase):
    def test_tree_shap_on_rf(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(80, 5)), columns=[f"f{i}" for i in range(5)])
        y = X["f0"] * 2 + X["f1"] + rng.normal(scale=0.1, size=80)
        model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0)
        model.fit(X, y)
        exp = explain_model(model, X, max_samples=40)
        self.assertEqual(exp.method, "tree_shap")
        self.assertGreater(len(exp.feature_importance), 0)
        self.assertIsNotNone(exp.local_shap)
        top = exp.feature_importance.iloc[0]["feature"]
        self.assertIn(top, ("f0", "f1"))


if __name__ == "__main__":
    unittest.main()
