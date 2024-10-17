import unittest

import numpy as np

from scipy import stats

from gEconpy.model.model import autocovariance_matrix
from gEconpy.parser.parse_distributions import CompositeDistribution
from gEconpy.utilities import (
    get_shock_std_priors_from_hyperpriors,
)


class TestExtractShockStd(unittest.TestCase):
    def setUp(self):
        self.shocks = ["epsilon_A", "epsilon_B", "epsilon_C"]
        self.shock_priors = {
            "epsilon_A": CompositeDistribution(
                stats.norm, loc=stats.norm(0, 1), scale=stats.gamma(2, 1)
            ),
            "epsilon_B": CompositeDistribution(
                stats.norm, loc=0, scale=stats.gamma(2, 1)
            ),
            "epsilon_C": CompositeDistribution(
                stats.norm, loc=0, scale=stats.gamma(2, 1)
            ),
        }
        self.hyper_priors = {
            "sigma_A": ("epsilon_A", "scale", stats.gamma(2, 1)),
            "mu_A": ("epsilon_A", "loc", stats.norm(0, 1)),
            "sigma_B": ("epsilon_B", "scale", stats.gamma(2, 1)),
            "sigma_C": ("epsilon_C", "scale", stats.gamma(2, 1)),
        }

    def test_raises_on_invalid_out_keys(self):
        with self.assertRaises(ValueError):
            get_shock_std_priors_from_hyperpriors(
                self.shocks, self.hyper_priors, out_keys="invalid_argument"
            )

    def test_extract_with_parent_keys(self):
        shock_std = get_shock_std_priors_from_hyperpriors(
            self.shocks, self.hyper_priors, out_keys="parent"
        )
        self.assertTrue(all([shock in shock_std for shock in self.shocks]))
        self.assertEqual(len(shock_std), len(self.shocks))

    def test_extract_with_param_keys(self):
        shock_std = get_shock_std_priors_from_hyperpriors(
            self.shocks, self.hyper_priors, out_keys="param"
        )
        self.assertTrue(
            all([key in shock_std for key in self.hyper_priors.keys() if key != "mu_A"])
        )
        self.assertTrue("mu_A" not in shock_std)
        self.assertEqual(len(shock_std), len(self.shocks))


if __name__ == "__main__":
    unittest.main()
