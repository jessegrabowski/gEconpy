import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gEconpy import gEconModel
from gEconpy.sampling import prior_solvability_check
from gEconpy.sampling.prior_utilities import (
    get_initial_time_index,
    kalman_filter_from_prior,
    simulate_trajectories_from_prior,
)

ROOT = Path(__file__).parent.absolute()


class TestPriorSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = os.path.join(ROOT, "Test GCNs/Full_New_Keyensian.gcn")
        cls.model = gEconModel(file_path, verbose=False)

        # Add some priors
        cls.model.param_priors["alpha"] = stats.beta(a=3, b=1)
        cls.model.param_priors["rho_technology"] = stats.beta(a=1, b=3)
        cls.model.param_priors["rho_preference"] = stats.beta(a=1, b=3)

        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)

    def test_sample_solvability_cycle_reduction(self):
        data = prior_solvability_check(self.model, n_samples=100, pert_solver="cycle_reduction")

        self.assertEqual(data.shape[0], 100)

    def test_sample_solvability_gensys(self):
        data = prior_solvability_check(self.model, n_samples=100, pert_solver="gensys")

        self.assertEqual(data.shape[0], 100)

    def test_invalid_solver_raises(self):
        with self.assertRaises(ValueError):
            data = prior_solvability_check(self.model, n_samples=1, pert_solver="invalid")


class TestGetInitialTime(unittest.TestCase):
    def test_integer_index(self):
        df = pd.DataFrame(np.random.normal(size=100))
        initial_index = get_initial_time_index(df)

        self.assertEqual(initial_index, -1)

    def test_monthly_period_index(self):
        index = pd.date_range(start="1900-02-01", periods=100, freq="MS")
        df = pd.DataFrame(np.random.normal(size=100), index=index)
        initial_index = get_initial_time_index(df)

        self.assertEqual(initial_index, np.array(pd.to_datetime("1900-01-01"), dtype="datetime64"))

    def test_quarterly_period_index(self):
        index = pd.date_range(start="1900-04-01", periods=100, freq="QS")
        df = pd.DataFrame(np.random.normal(size=100), index=index)
        initial_index = get_initial_time_index(df)

        self.assertEqual(initial_index, np.array(pd.to_datetime("1900-01-01"), dtype="datetime64"))

    def test_annual_period_index(self):
        index = pd.date_range(start="1901-01-01", periods=100, freq="YS")
        df = pd.DataFrame(np.random.normal(size=100), index=index)
        initial_index = get_initial_time_index(df)

        self.assertEqual(initial_index, np.array(pd.to_datetime("1900-01-01"), dtype="datetime64"))


class TestSimulateTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = os.path.join(ROOT, "Test GCNs/RBC_Linearized.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)

    def test_simulate_trajectories(self):
        data = simulate_trajectories_from_prior(
            self.model, n_simulations=10, n_samples=10, simulation_length=10
        )

        self.assertEqual(data.index.values.tolist(), [x.base_name for x in self.model.variables])
        self.assertTrue(data.shape == (self.model.n_variables, 10 * 10 * 10))

    def test_pert_kwargs(self):
        data = simulate_trajectories_from_prior(
            self.model,
            n_simulations=10,
            n_samples=10,
            simulation_length=10,
            pert_kwargs={"solver": "gensys"},
        )
        self.assertIsNotNone(data)


class TestKalmanFilterFromPrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = os.path.join(ROOT, "Test GCNs/RBC_Linearized.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)

    def test_univariate_filter(self):
        data = self.model.simulate(simulation_length=100, n_simulations=1).T.droplevel(1)[["Y"]]
        kf_output = kalman_filter_from_prior(
            self.model, data, n_samples=10, filter_type="univariate"
        )

        self.assertIsNotNone(kf_output)


if __name__ == "main":
    unittest.main()
