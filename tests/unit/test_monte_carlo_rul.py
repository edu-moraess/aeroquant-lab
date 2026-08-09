"""Testes de propriedade do Monte Carlo de RUL (Fase 8)."""
from __future__ import annotations

import unittest

from aeroquant.uncertainty.monte_carlo_rul import run_monte_carlo_rul


class TestMonteCarloRUL(unittest.TestCase):
    def test_returns_finite_distribution(self):
        r = run_monte_carlo_rul(n_runs=4, max_cycles=80, n_calibration_units=4, base_seed=1)
        self.assertGreaterEqual(r.n_runs, 1)
        self.assertTrue(all(x == x for x in r.rul_samples))  # no NaN
        self.assertGreaterEqual(r.q95, r.q05)
        self.assertGreaterEqual(r.var_total, 0.0)
        self.assertGreaterEqual(r.var_aleatoric, 0.0)
        self.assertGreaterEqual(r.var_epistemic, 0.0)

    def test_true_rul_consistent_with_fraction(self):
        r = run_monte_carlo_rul(
            n_runs=3, max_cycles=100, reference_cycle_fraction=0.5, n_calibration_units=3
        )
        self.assertEqual(r.true_rul_at_ref, 50.0)


if __name__ == "__main__":
    unittest.main()
