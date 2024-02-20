import os
import unittest

from pathlib import Path

import numpy as np

from gEconpy.classes.model import gEconModel
from gEconpy.estimation.estimation_utilities import (
    build_system_matrices,
    check_bk_condition,
    extract_sparse_data_from_model,
)
from gEconpy.estimation.kalman_filter import kalman_filter, univariate_kalman_filter

ROOT = Path(__file__).parent.absolute()


class BasicFunctionalityTests(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join(
            ROOT, "Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"
        )
        self.model = gEconModel(file_path, verbose=False)
        self.model.steady_state(verbose=False)
        self.model.solve_model(verbose=False)

    def test_extract_system_matrics(self):
        param_dict = self.model.free_param_dict

        sparse_data = extract_sparse_data_from_model(
            self.model, params_to_estimate=["theta"]
        )
        A, B, C, D = build_system_matrices(
            param_dict, sparse_data, vars_to_estimate=["theta"]
        )

        system = self.model.build_perturbation_matrices(
            np.fromiter(
                (self.model.free_param_dict | self.model.calib_param_dict).values(),
                dtype="float",
            ),
            np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
        )

        self.assertTrue(np.allclose(A, system[0]))
        self.assertTrue(np.allclose(B, system[1]))
        self.assertTrue(np.allclose(C, system[2]))
        self.assertTrue(np.allclose(D, system[3]))

        self.assertTrue(check_bk_condition(A, B, C, tol=1e-8))


class KalmanFilterTest(unittest.TestCase):
    def test_likelihood(self):
        # Test against an AR(1) model with rho=0.8
        # Expected value comes from statsmodels

        expected = np.array(
            [
                -1.42976416,
                -1.41893853,
                -1.63893853,
                -1.89893853,
                -2.19893853,
                -2.53893853,
                -2.91893853,
                -3.33893853,
                -3.79893853,
                -4.29893853,
            ]
        )
        data = np.arange(10, dtype="float64")[:, None]

        a0 = np.array([[0.0]])
        P0 = np.array([[1000000.0]])
        T = np.array([[0.8]])
        Z = np.array([[1.0]])
        R = np.array([[1.0]])
        H = np.array([[0.0]])
        Q = np.array([[1.0]])

        *_, ll_obs = kalman_filter(data, T, Z, R, H, Q, a0, P0)

        # The first observation is different from statsmodels because they apply some adjustment for the diffuse
        # initialization.
        self.assertTrue(np.allclose(expected[1:], ll_obs[1:]))

    def test_likelihood_with_missing(self):
        # Test against an AR(1) model with rho=0.8
        # Expected value comes from statsmodels

        expected = np.array(
            [
                -1.42976416,
                -1.41893853,
                -1.63893853,
                -1.89893853,
                -2.19893853,
                0.0,
                -4.77409153,
                -3.33893853,
                -3.79893853,
                -4.29893853,
            ]
        )

        data = np.arange(10, dtype="float64")[:, None]
        data[5] = np.nan

        a0 = np.array([[0.0]])
        P0 = np.array([[1000000.0]])
        T = np.array([[0.8]])
        Z = np.array([[1.0]])
        R = np.array([[1.0]])
        H = np.array([[0.0]])
        Q = np.array([[1.0]])

        *_, ll_obs = kalman_filter(data, T, Z, R, H, Q, a0, P0)

        # The first observation is different from statsmodels because they apply some adjustment for the diffuse
        # initialization.
        self.assertTrue(np.allclose(expected[1:], ll_obs[1:]))


class UnivariateKalmanFilterTest(unittest.TestCase):
    def test_likelihood(self):
        # Test against an AR(1) model with rho=0.8
        # Expected value comes from statsmodels

        expected = np.array(
            [
                -1.42976416,
                -1.41893853,
                -1.63893853,
                -1.89893853,
                -2.19893853,
                -2.53893853,
                -2.91893853,
                -3.33893853,
                -3.79893853,
                -4.29893853,
            ]
        )
        data = np.arange(10, dtype="float64")[:, None]

        a0 = np.array([[0.0]])
        P0 = np.array([[1000000.0]])
        T = np.array([[0.8]])
        Z = np.array([[1.0]])
        R = np.array([[1.0]])
        H = np.array([[0.0]])
        Q = np.array([[1.0]])

        *_, ll_obs = univariate_kalman_filter(data, T, Z, R, H, Q, a0, P0)

        # The first observation is different from statsmodels because they apply some adjustment for the diffuse
        # initialization.
        self.assertTrue(np.allclose(expected[1:], ll_obs[1:]))

    def test_likelihood_with_missing(self):
        # Test against an AR(1) model with rho=0.8
        # Expected value comes from statsmodels

        expected = np.array(
            [
                -1.42976416,
                -1.41893853,
                -1.63893853,
                -1.89893853,
                -2.19893853,
                0.0,
                -4.77409153,
                -3.33893853,
                -3.79893853,
                -4.29893853,
            ]
        )

        data = np.arange(10, dtype="float64")[:, None]
        data[5] = np.nan

        a0 = np.array([[0.0]])
        P0 = np.array([[1000000.0]])
        T = np.array([[0.8]])
        Z = np.array([[1.0]])
        R = np.array([[1.0]])
        H = np.array([[0.0]])
        Q = np.array([[1.0]])

        *_, ll_obs = univariate_kalman_filter(data, T, Z, R, H, Q, a0, P0)

        # The first observation is different from statsmodels because they apply some adjustment for the diffuse
        # initialization.
        self.assertTrue(np.allclose(expected[1:], ll_obs[1:]))


class TestModelEstimation(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join(
            ROOT, "Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"
        )
        self.model = gEconModel(file_path, verbose=False)
        self.model.steady_state(verbose=False)
        self.model.solve_model(verbose=False)

        self.data = (
            self.model.simulate(simulation_length=100, n_simulations=1)
            .xs(axis=1, level=1, key=0)
            .T
        )

    def filter_random_sample(self):
        pass


if __name__ == "__main__":
    unittest.main()
