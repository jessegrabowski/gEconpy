import os
import unittest
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from gEconpy.classes.model import gEconModel
from gEconpy.plotting import (
    plot_covariance_matrix,
    plot_eigenvalues,
    plot_irf,
    plot_prior_solvability,
    plot_simulation,
    prepare_gridspec_figure,
)
from gEconpy.plotting.plotting import (
    plot_acf,
    plot_corner,
    plot_heatmap,
    plot_kalman_filter,
)
from gEconpy.sampling import kalman_filter_from_posterior, prior_solvability_check

ROOT = Path(__file__).parent.absolute()


class TestUtilities(unittest.TestCase):
    def test_prepare_gridspec_figure_square(self):
        gs, locs = prepare_gridspec_figure(n_cols=3, n_plots=9)
        self.assertTrue(len(locs) == 9)

    def test_prepare_gridspec_figure_tall(self):
        gs, locs = prepare_gridspec_figure(n_cols=2, n_plots=9)
        self.assertTrue(len(locs) == 9)
        self.assertEqual(locs[-1][0], slice(8, 10, None))
        self.assertEqual(locs[-1][1], slice(1, 3, None))

    def test_prepare_gridspec_figure_wide(self):
        gs, locs = prepare_gridspec_figure(n_cols=4, n_plots=9)
        self.assertTrue(len(locs) == 9)
        self.assertEqual(locs[-1][0], slice(4, 6, None))
        self.assertEqual(locs[-1][1], slice(3, 5, None))


class TestPlotSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = os.path.join(ROOT, "Test GCNs/RBC_Linearized.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.data = cls.model.simulate(simulation_length=100, n_simulations=1)

    def test_plot_simulation_defaults(self):
        fig = plot_simulation(self.data)

        self.assertEqual(len(fig.axes), self.model.n_variables)
        plt.close()

    def test_plot_simulation_vars_to_plot(self):
        fig = plot_simulation(self.data, vars_to_plot=["Y", "C"])

        self.assertEqual(len(fig.axes), 2)
        plt.close()

    def test_var_not_found_raises(self):
        with self.assertRaises(ValueError) as error:
            fig = plot_simulation(self.data, vars_to_plot=["Y", "C", "Invalid"])
        error_msg = error.exception
        self.assertEqual(str(error_msg), "Invalid not found among model variables.")

    def test_plot_simulation_with_ci(self):
        fig = plot_simulation(self.data, ci=0.95)

        self.assertEqual(len(fig.axes), self.model.n_variables)
        plt.close()

    def test_plot_simulation_aesthetic_params(self):
        fig = plot_simulation(
            self.data, cmap="YlGn", figsize=(14, 4), dpi=100, fill_color="brickred"
        )

        self.assertEqual(len(fig.axes), self.model.n_variables)
        self.assertEqual(fig.get_dpi(), 100)
        self.assertEqual(fig.get_figwidth(), 14)
        self.assertEqual(fig.get_figheight(), 4)

        plt.close()


class TestIRFPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = os.path.join(ROOT, "Test GCNs/Full_New_Keyensian.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.irf = cls.model.impulse_response_function(simulation_length=100, shock_size=0.1)

    def test_plot_irf_defaults(self):
        fig = plot_irf(self.irf)

        self.assertEqual(len(fig.axes), self.model.n_variables)
        self.assertEqual(len(fig.axes[0].get_lines()), self.model.n_shocks)
        plt.close()

    def test_plot_irf_one_shock(self):
        with self.assertRaises(ValueError):
            fig = plot_irf(self.irf, shocks_to_plot="epsilon_A")

        fig = plot_irf(self.irf, shocks_to_plot=["epsilon_Y"])
        self.assertEqual(len(fig.axes), self.model.n_variables)
        self.assertEqual(len(fig.axes[0].get_lines()), 1)
        plt.close()

    def test_plot_irf_one_variable(self):
        with self.assertRaises(ValueError):
            fig = plot_irf(self.irf, vars_to_plot="Y")

        fig = plot_irf(self.irf, vars_to_plot=["Y"])
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].get_lines()), self.model.n_shocks)
        plt.close()

    def test_var_not_found_raises(self):
        with self.assertRaises(ValueError) as error:
            fig = plot_irf(self.irf, vars_to_plot=["Y", "C", "Invalid"])
        error_msg = error.exception
        self.assertEqual(str(error_msg), "Invalid not found among simulated impulse responses.")
        plt.close()

    def test_shock_not_found_raises(self):
        with self.assertRaises(ValueError) as error:
            fig = plot_irf(
                self.irf, vars_to_plot=["Y", "C"], shocks_to_plot=["epsilon_Y", "Invalid"]
            )
        error_msg = error.exception
        self.assertEqual(
            str(error_msg), "Invalid not found among shocks used in impulse response data."
        )

    def test_legend(self):
        fig = plot_irf(self.irf, vars_to_plot=["Y", "C"], shocks_to_plot=["epsilon_Y"], legend=True)
        self.assertIsNotNone(fig.axes[0].get_legend())
        self.assertIsNone(fig.axes[1].get_legend())
        plt.close()


class TestPriorSolvabilityPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_w_Distributions.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.data = prior_solvability_check(cls.model, n_samples=1_000)

    def test_plot_with_defaults(self):
        fig = plot_prior_solvability(self.data)

        n_priors = len(self.model.param_priors)
        self.assertTrue(len(fig.axes) == n_priors**2)
        plot_idxs = np.arange(n_priors**2).reshape((n_priors, n_priors))
        upper_idxs = plot_idxs[np.triu_indices_from(plot_idxs, 1)]
        lower_idxs = plot_idxs[np.tril_indices_from(plot_idxs)]
        for idx in upper_idxs:
            self.assertTrue(not fig.axes[idx].get_visible())
        for idx in lower_idxs:
            self.assertTrue(fig.axes[idx].get_visible())
        plt.close()

    def test_plot_with_vars_to_plot(self):
        fig = plot_prior_solvability(self.data, params_to_plot=["alpha", "gamma"])
        n_priors = 2

        self.assertTrue(len(fig.axes) == n_priors**2)
        plot_idxs = np.arange(n_priors**2).reshape((n_priors, n_priors))
        upper_idxs = plot_idxs[np.triu_indices_from(plot_idxs, 1)]
        lower_idxs = plot_idxs[np.tril_indices_from(plot_idxs)]
        for idx in upper_idxs:
            self.assertTrue(not fig.axes[idx].get_visible())
        for idx in lower_idxs:
            self.assertTrue(fig.axes[idx].get_visible())
        plt.close()

    def test_raises_if_param_not_found(self):
        with self.assertRaises(ValueError) as error:
            plot_prior_solvability(self.data, params_to_plot=["alpha", "beta"])

        msg = str(error.exception)
        self.assertEqual(
            msg, 'Cannot plot parameter "beta", it was not found in the provided data.'
        )


class TestPlotEigenvalues(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)

    def test_plot_with_defaults(self):
        fig = plot_eigenvalues(self.model)
        from matplotlib.collections import PathCollection

        scatter_points = fig.axes[0].findobj(PathCollection)[0].get_offsets().data
        data = self.model.check_bk_condition(return_value="df", verbose=False)

        n_finite = (data["Modulus"] < 1.5).sum()
        self.assertEqual(n_finite, scatter_points.shape[0])
        plt.close()

    def test_plot_with_aesthetic_params(self):
        fig = plot_eigenvalues(self.model, dpi=144, figsize=(2, 2))

        self.assertEqual(fig.get_figwidth(), 2)
        self.assertEqual(fig.get_figheight(), 2)
        self.assertEqual(fig.dpi, 144)
        plt.close()


class TestPlotCovarianceMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.cov_matrix = cls.model.compute_stationary_covariance_matrix(
            shock_cov_matrix=np.eye(1) * 0.01
        )

    def test_plot_with_defaults(self):
        fig = plot_covariance_matrix(self.cov_matrix)
        self.assertIsNotNone(fig)
        plt.close()

    def test_plot_heatmap_with_defaults(self):
        fig = plot_heatmap(self.cov_matrix)
        self.assertIsNotNone(fig)
        plt.close()

    def test_annotation_kwargs(self):
        fig = plot_covariance_matrix(
            self.cov_matrix, annotation_kwargs={"threshold": 0.5, "fontsize": 5}
        )
        self.assertIsNotNone(fig)
        plt.close()

    def test_heatmap_kwargs(self):
        fig = plot_covariance_matrix(
            self.cov_matrix, heatmap_kwargs={"interpolation": "antialiased"}
        )
        self.assertIsNotNone(fig)
        plt.close()


class TestPlotACF(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.acf = cls.model.compute_autocorrelation_matrix(shock_cov_matrix=np.eye(1) * 0.01)

    def test_plot_with_defaults(self):
        fig = plot_acf(self.acf)
        self.assertEqual(len(fig.axes), self.model.n_variables)
        plt.close()

    def test_plot_with_subset(self):
        fig = plot_acf(self.acf, vars_to_plot=["C", "K", "A"])
        self.assertEqual(len(fig.axes), 3)
        plt.close()

    def test_invalid_var_raises(self):
        with self.assertRaises(ValueError) as error:
            fig = plot_acf(self.acf, vars_to_plot=["K", "C", "Invalid"])
        msg = str(error.exception)
        self.assertEqual(
            msg, "Can not plot variable Invalid, it was not found in the provided covariance matrix"
        )


class TestPostEstimationPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_w_Distributions.gcn")
        cls.model = gEconModel(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)

        cls.data = cls.model.simulate(simulation_length=100, n_simulations=1)
        cls.data = cls.data.droplevel(axis=1, level=1).T[["C"]]

        cls.idata = cls.model.fit(
            cls.data,
            filter_type="univariate",
            draws=36,
            n_walkers=36,
            return_inferencedata=True,
            burn_in=0,
            verbose=False,
            compute_sampler_stats=False,
        )

    def test_plot_corner_with_defaults(self):
        fig = plot_corner(self.idata)
        self.assertIsNotNone(fig)
        plt.close()

    def test_plot_kalman_with_defaults(self):
        posterior = self.idata.posterior.stack(sample=["chain", "draw"])
        conditional_posterior = kalman_filter_from_posterior(
            self.model, self.data, posterior, n_samples=10
        )

        fig = plot_kalman_filter(conditional_posterior, self.data, kalman_output="predicted")
        self.assertIsNotNone(fig)
        plt.close()

        fig = plot_kalman_filter(conditional_posterior, self.data, kalman_output="filtered")
        self.assertIsNotNone(fig)
        plt.close()

        fig = plot_kalman_filter(conditional_posterior, self.data, kalman_output="smoothed")
        self.assertIsNotNone(fig)
        plt.close()

    def test_plot_kalman_raises_on_invalid_args(self):
        with self.assertRaises(ValueError):
            fig = plot_kalman_filter(self.idata, self.data, kalman_output="invalid")


if __name__ == "__main__":
    unittest.main()
