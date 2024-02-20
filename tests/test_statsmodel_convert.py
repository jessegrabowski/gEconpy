import os
import unittest

from pathlib import Path
from warnings import catch_warnings, simplefilter

from gEconpy import compile_to_statsmodels, gEconModel
from gEconpy.estimation.transformers import IntervalTransformer, PositiveTransformer

ROOT = Path(__file__).parent.absolute()


class TestStatsModelConversion(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        file_path = os.path.join(
            ROOT, "Test GCNs/One_Block_Simple_1_w_Distributions.gcn"
        )
        self.model = gEconModel(file_path, verbose=False)
        self.model.steady_state(verbose=False)
        self.model.solve_model(verbose=False)

        self.data = self.model.simulate(simulation_length=100, n_simulations=1)
        self.data = self.data.droplevel(axis=1, level=1).T[["C"]]

    def test_conversion(self):
        MLEModel = compile_to_statsmodels(self.model)
        self.assertIsNotNone(MLEModel)

    def test_mle_fit(self):
        param_start_dict = {
            "alpha": 0.33,
            "gamma": 2.0,
            "rho": 0.85,
        }

        shock_start_dict = {"epsilon": 0.5}

        # The slope parameter controls the steepness of the gradient around 0 (lower slope = more gentle gradient)
        param_transforms = {
            "alpha": IntervalTransformer(low=1e-4, high=0.99),
            "gamma": IntervalTransformer(low=1.001, high=20),
            "rho": IntervalTransformer(low=1e-4, high=0.99),
        }

        MLEModel = compile_to_statsmodels(self.model)
        initial_params = self.model.free_param_dict.copy()
        mle_mod = MLEModel(
            self.data,
            param_start_dict=param_start_dict,
            shock_start_dict=shock_start_dict,
            noise_start_dict=None,
            param_transforms=param_transforms,
            shock_transforms=None,  # If None, will automatically transform to positive values only
            noise_transforms=None,  # If None, will automatically transform to positive values only
            initialization="stationary",
        )

        # This shouldn't succeed -- catch the warning
        with catch_warnings():
            simplefilter("ignore")
            mle_res = mle_mod.fit(method="lbfgs", maxiter=10, disp=0)

        # Final estimates in the mle_res object are the same as are saved in the model object
        for param in mle_res.params.index:
            if param in self.model.free_param_dict:
                self.assertEqual(
                    mle_res.params[param], self.model.free_param_dict[param], msg=param
                )

        # Check that parameters were changed
        for param in ["alpha", "gamma", "rho"]:
            self.assertNotEqual(
                self.model.free_param_dict[param], initial_params[param], msg=param
            )

        # Make sure parameters not given start values were not changed
        for param in ["beta", "delta"]:
            self.assertEqual(
                self.model.free_param_dict[param], initial_params[param], msg=param
            )

    def test_mle_fit_MAP(self):
        param_start_dict = {
            "alpha": 0.33,
            "gamma": 2.0,
            "rho": 0.85,
        }

        shock_start_dict = {"epsilon": 0.5}

        # The slope parameter controls the steepness of the gradient around 0 (lower slope = more gentle gradient)
        param_transforms = {
            "alpha": IntervalTransformer(low=1e-4, high=0.99, slope=1),
            "gamma": PositiveTransformer(),
            "rho": IntervalTransformer(low=1e-4, high=0.99, slope=1),
        }

        MLEModel = compile_to_statsmodels(self.model)
        initial_params = self.model.free_param_dict.copy()

        mle_mod = MLEModel(
            self.data,
            param_start_dict=param_start_dict,
            shock_start_dict=shock_start_dict,
            noise_start_dict=None,
            param_transforms=param_transforms,
            shock_transforms=None,  # If None, will automatically transform to positive values only
            noise_transforms=None,  # If None, will automatically transform to positive values only
            initialization="stationary",
            fit_MAP=True,
        )

        # This shouldn't succeed -- catch the warning
        with catch_warnings():
            simplefilter("ignore")
            mle_res = mle_mod.fit(method="lbfgs", maxiter=10, disp=0)

        # Final estimates in the mle_res object are the same as are saved in the model object
        for param in mle_res.params.index:
            if param in self.model.free_param_dict:
                self.assertEqual(
                    mle_res.params[param], self.model.free_param_dict[param], msg=param
                )

        # Check that parameters were changed
        for param in ["alpha", "gamma", "rho"]:
            self.assertNotEqual(
                self.model.free_param_dict[param], initial_params[param], msg=param
            )

        # Make sure parameters not given start values were not changed
        for param in ["beta", "delta"]:
            self.assertEqual(
                self.model.free_param_dict[param], initial_params[param], msg=param
            )

        self.assertIsNotNone(mle_res)


if __name__ == "main":
    unittest.main()
