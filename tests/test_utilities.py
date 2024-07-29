import unittest

import numpy as np

from scipy import stats

from gEconpy.parser.parse_distributions import CompositeDistribution
from gEconpy.shared.utilities import (
    build_Q_matrix,
    compute_autocorrelation_matrix,
    get_shock_std_priors_from_hyperpriors,
)


class TestBuildQMatrix(unittest.TestCase):
    def setUp(self):
        self.shocks = ["epsilon_A", "epsilon_B", "epsilon_C"]
        self.shock_std_priors = {
            "epsilon_A": stats.gamma(2, 1),
            "epsilon_B": stats.gamma(2, 1),
            "epsilon_C": stats.gamma(2, 1),
        }

    def test_passing_both_args_raises(self):
        with self.assertRaises(ValueError):
            build_Q_matrix(
                model_shocks=self.shocks,
                shock_dict={"epsilon_A": 3},
                shock_cov_matrix=np.eye(3),
                shock_std_priors=self.shock_std_priors,
            )

    def test_not_positive_semidef_raises(self):
        cov_mat = np.random.normal(size=(3, 3))
        with self.assertRaises(np.linalg.LinAlgError):
            build_Q_matrix(model_shocks=self.shocks, shock_cov_matrix=cov_mat)

    def test_cov_matrix_bad_shape_raises(self):
        cov_mat = np.random.normal(size=(3, 2))
        with self.assertRaises(ValueError):
            build_Q_matrix(model_shocks=self.shocks, shock_cov_matrix=cov_mat)

    def test_build_from_dictionary(self):
        Q = build_Q_matrix(
            model_shocks=self.shocks, shock_std_priors=None, shock_dict={"epsilon_A": 3}
        )

        expected_Q = np.array([[3, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])

        self.assertTrue(np.allclose(Q, expected_Q))

    def test_build_from_priors(self):
        Q = build_Q_matrix(
            model_shocks=self.shocks, shock_std_priors=self.shock_std_priors
        )
        expected_Q = np.eye(3)
        for i, shock_d in enumerate(self.shock_std_priors.values()):
            expected_Q[i, i] = shock_d.mean()

        self.assertTrue(np.allclose(Q, expected_Q))

    def test_build_from_mixed(self):
        Q = build_Q_matrix(
            model_shocks=self.shocks,
            shock_std_priors=self.shock_std_priors,
            shock_dict={"epsilon_B": 100},
        )

        expected_Q = np.eye(3)
        for i, shock_d in enumerate(self.shock_std_priors.values()):
            expected_Q[i, i] = shock_d.mean()

        expected_Q[1, 1] = 100
        self.assertTrue(np.allclose(Q, expected_Q))


class TestComputeAutocorrelation(unittest.TestCase):
    def test_compute_autocorrelation_matrix(self):
        A = np.eye(5)
        L = np.random.normal(size=(5, 5))
        Q = L @ L.T

        acorr = compute_autocorrelation_matrix(A, Q, n_lags=10)
        self.assertEqual(acorr.shape, (5, 10))


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
