import os
import unittest
from pathlib import Path

import numpy as np

from gEconpy.classes.model import gEconModel
from gEconpy.estimation.estimate import build_and_solve, build_Q_and_H
from gEconpy.estimation.estimation_utilities import extract_sparse_data_from_model

ROOT = Path(__file__).parent.absolute()


class TestEstimationHelpers(unittest.TestCase):
    def setUp(self) -> None:
        file_path = os.path.join(
            ROOT, "Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"
        )
        self.model = gEconModel(file_path, verbose=False)
        self.model.steady_state(verbose=False)
        self.model.solve_model(verbose=False)

    def test_build_and_solve(self):
        param_dict = self.model.free_param_dict
        to_estimate = list(param_dict.to_string().keys())
        sparse_data = extract_sparse_data_from_model(
            self.model, params_to_estimate=to_estimate
        )

        T, R, success = build_and_solve(param_dict, sparse_data, to_estimate)

        self.assertTrue(np.allclose(T, self.model.T.values))
        self.assertTrue(np.allclose(R, self.model.R.values))

    def test_build_Q_and_R(self):
        shock_names = [x.base_name for x in self.model.shocks]
        state_sigmas = dict(zip(shock_names, [0.1] * self.model.n_shocks))
        observed_vars = list(self.model.steady_state_dict.keys())
        n = len(observed_vars)

        Q, H = build_Q_and_H(
            state_sigmas,
            shock_variables=shock_names,
            obs_variables=observed_vars,
            obs_sigmas=None,
        )

        Q_result = np.array([[0.1]])
        H_result = np.zeros((n, n))

        self.assertTrue(np.allclose(Q, Q_result))
        self.assertTrue(np.allclose(H, H_result))


if __name__ == "__main__":
    unittest.main()
