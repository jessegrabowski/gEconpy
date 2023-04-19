import unittest
from gEconpy.shared.utilities import build_Q_matrix, compute_autocorrelation_matrix
from gEconpy.parser.parse_distributions import CompositeDistribution
from scipy import stats

import numpy as np

from pathlib import Path
ROOT = Path(__file__).parent.absolute()


class TestBuildQMatrix(unittest.TestCase):
    def setUp(self):
        self.shocks = ['epsilon_A', 'epsilon_B', 'epsilon_C']
        self.shock_priors = {'epsilon_A': CompositeDistribution(stats.norm, loc=0, scale=stats.gamma(2, 1)),
                             'epsilon_B': CompositeDistribution(stats.norm, loc=0, scale=stats.gamma(2, 1)),
                             'epsilon_C': CompositeDistribution(stats.norm, loc=0, scale=stats.gamma(2, 1))}

    def test_passing_both_args_raises(self):
        with self.assertRaises(ValueError):
            build_Q_matrix(model_shocks=self.shocks,
                           shock_dict={'epsilon_A':3},
                           shock_cov_matrix=np.eye(3),
                           shock_priors=self.shock_priors)

    def test_not_positive_semidef_raises(self):
        cov_mat = np.random.normal(size=(3, 3))
        with self.assertRaises(np.linalg.LinAlgError):
            build_Q_matrix(model_shocks=self.shocks,
                           shock_cov_matrix=cov_mat)

    def test_cov_matrix_bad_shape_raises(self):
        cov_mat = np.random.normal(size=(3,2))
        with self.assertRaises(ValueError):
            build_Q_matrix(model_shocks=self.shocks,
                           shock_cov_matrix=cov_mat)

    def test_build_from_dictionary(self):
        Q = build_Q_matrix(model_shocks=self.shocks,
                           shock_priors=None,
                           shock_dict={'epsilon_A':3})

        expected_Q = np.array([[3, 0,   0  ],
                               [0, 0.01, 0  ],
                               [0, 0,   0.01]])

        self.assertTrue(np.allclose(Q, expected_Q))

    def test_build_from_priors(self):
        Q = build_Q_matrix(model_shocks=self.shocks,
                           shock_priors=self.shock_priors)
        expected_Q = np.eye(3)
        for i, shock_d in enumerate(self.shock_priors.values()):
            expected_Q[i, i] = shock_d.rv_params['scale'].mean()

        self.assertTrue(np.allclose(Q, expected_Q))

    def test_build_from_mixed(self):
        Q = build_Q_matrix(model_shocks=self.shocks,
                           shock_priors=self.shock_priors,
                           shock_dict={'epsilon_B':100})

        expected_Q = np.eye(3)
        for i, shock_d in enumerate(self.shock_priors.values()):
            expected_Q[i, i] = shock_d.rv_params['scale'].mean()

        expected_Q[1, 1] = 100
        self.assertTrue(np.allclose(Q, expected_Q))

class TestComputeAutocorrelation(unittest.TestCase):

    def test_compute_autocorrelation_matrix(self):
        A = np.eye(5)
        L = np.random.normal(size=(5, 5))
        Q = L @ L.T

        acorr = compute_autocorrelation_matrix(A, Q, n_lags=10)
        self.assertEqual(acorr.shape, (5, 10))



if __name__ == '__main__':
    unittest.main()
