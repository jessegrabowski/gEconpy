import unittest
import os

import numpy as np

from gEconpy.classes.model import gEconModel
from gEconpy.plotting import prepare_gridspec_figure, plot_simulation, plot_irf, plot_prior_solvability, \
    plot_eigenvalues, plot_covariance_matrix

from pathlib import Path

from gEconpy.sampling import prior_solvability_check

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
        cls.model = gEconModel(os.path.join(ROOT, 'test GCNs/RBC_Linearized.gcn'), verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.data = cls.model.simulate(simulation_length=100, n_simulations=1)

    def test_plot_simulation_defaults(self):
        fig = plot_simulation(self.data)

        self.assertEqual(len(fig.axes), self.model.n_variables)

    def test_plot_simulation_vars_to_plot(self):
        fig = plot_simulation(self.data, vars_to_plot=['Y', 'C'])

        self.assertEqual(len(fig.axes), 2)

    def test_var_not_found_raises(self):
        with self.assertRaises(ValueError) as error:
            fig = plot_simulation(self.data, vars_to_plot=['Y', 'C', 'Invalid'])
        error_msg = error.exception
        self.assertEqual(str(error_msg), 'Invalid not found among model variables.')

    def test_plot_simulation_with_ci(self):
        fig = plot_simulation(self.data, ci=0.95)

        self.assertEqual(len(fig.axes), self.model.n_variables)

    def test_plot_simulation_aesthetic_params(self):
        fig = plot_simulation(self.data, cmap='YlGn', figsize=(14, 4), dpi=100, fill_color='brickred')

        self.assertEqual(len(fig.axes), self.model.n_variables)
        self.assertEqual(fig.get_dpi(), 100)
        self.assertEqual(fig.get_figwidth(), 14)
        self.assertEqual(fig.get_figheight(), 4)


class TestIRFPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = gEconModel(os.path.join(ROOT, 'test GCNs/Full_New_Keyensian.gcn'), verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.irf = cls.model.impulse_response_function(simulation_length=100, shock_size=0.1)

    def test_plot_irf_defaults(self):
        fig = plot_irf(self.irf)

        self.assertEqual(len(fig.axes), self.model.n_variables)
        self.assertEqual(len(fig.axes[0].get_lines()), self.model.n_shocks)

    def test_plot_irf_one_shock(self):
        with self.assertRaises(ValueError):
            fig = plot_irf(self.irf, shocks_to_plot='epsilon_A')

        fig = plot_irf(self.irf, shocks_to_plot=['epsilon_Y'])
        self.assertEqual(len(fig.axes), self.model.n_variables)
        self.assertEqual(len(fig.axes[0].get_lines()), 1)

    def test_plot_irf_one_variable(self):
        with self.assertRaises(ValueError):
            fig = plot_irf(self.irf, vars_to_plot='Y')

        fig = plot_irf(self.irf, vars_to_plot=['Y'])
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].get_lines()), self.model.n_shocks)

    def test_var_not_found_raises(self):
        with self.assertRaises(ValueError) as error:
            fig = plot_irf(self.irf, vars_to_plot=['Y', 'C', 'Invalid'])
        error_msg = error.exception
        self.assertEqual(str(error_msg), 'Invalid not found among simulated impulse responses.')

    def test_shock_not_found_raises(self):
        with self.assertRaises(ValueError) as error:
            fig = plot_irf(self.irf, vars_to_plot=['Y', 'C'], shocks_to_plot=['epsilon_Y', 'Invalid'])
        error_msg = error.exception
        self.assertEqual(str(error_msg), 'Invalid not found among shocks used in impulse response data.')

    def test_legend(self):
        fig = plot_irf(self.irf, vars_to_plot=['Y', 'C'], shocks_to_plot=['epsilon_Y'], legend=True)
        self.assertIsNotNone(fig.axes[0].get_legend())
        self.assertIsNone(fig.axes[1].get_legend())


class TestPriorSolvabilityPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = gEconModel(os.path.join(ROOT, 'test GCNs/One_Block_Simple_1_w_Distributions.gcn'), verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.data = prior_solvability_check(cls.model, n_samples=1_000)

    def test_plot_with_defaults(self):
        fig = plot_prior_solvability(self.data)

        n_priors = len(self.model.param_priors)
        self.assertTrue(len(fig.axes) == n_priors ** 2)
        plot_idxs = np.arange(n_priors ** 2).reshape((n_priors, n_priors))
        upper_idxs = plot_idxs[np.triu_indices_from(plot_idxs, 1)]
        lower_idxs = plot_idxs[np.tril_indices_from(plot_idxs)]
        for idx in upper_idxs:
            self.assertTrue(not fig.axes[idx].get_visible())
        for idx in lower_idxs:
            self.assertTrue(fig.axes[idx].get_visible())

    def test_plot_with_vars_to_plot(self):
        fig = plot_prior_solvability(self.data, params_to_plot=['alpha', 'gamma'])
        n_priors = 2

        self.assertTrue(len(fig.axes) == n_priors ** 2)
        plot_idxs = np.arange(n_priors ** 2).reshape((n_priors, n_priors))
        upper_idxs = plot_idxs[np.triu_indices_from(plot_idxs, 1)]
        lower_idxs = plot_idxs[np.tril_indices_from(plot_idxs)]
        for idx in upper_idxs:
            self.assertTrue(not fig.axes[idx].get_visible())
        for idx in lower_idxs:
            self.assertTrue(fig.axes[idx].get_visible())

    def test_raises_if_param_not_found(self):
        with self.assertRaises(ValueError) as error:
            plot_prior_solvability(self.data, params_to_plot=['alpha', 'beta'])

        msg = str(error.exception)
        self.assertEqual(msg, 'Cannot plot parameter "beta", it was not found in the provided data.')


class TestPlotEigenvalues(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = gEconModel(os.path.join(ROOT, 'test GCNs/One_Block_Simple_1.gcn'), verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)

    def test_plot_with_defaults(self):
        fig = plot_eigenvalues(self.model)
        from matplotlib.collections import PathCollection
        scatter_points = fig.axes[0].findobj(PathCollection)[0].get_offsets().data
        data = self.model.check_bk_condition(return_value='df', verbose=False)

        n_finite = (data['Modulus'] < 1.5).sum()
        self.assertEqual(n_finite, scatter_points.shape[0])

    def test_plot_with_aesthetic_params(self):
        fig = plot_eigenvalues(self.model, dpi=144, figsize=(2,2))

        self.assertEqual(fig.get_figwidth(), 2)
        self.assertEqual(fig.get_figheight(), 2)
        self.assertEqual(fig.dpi, 144)


class TestPlotCovarianceMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = gEconModel(os.path.join(ROOT, 'test GCNs/One_Block_Simple_1.gcn'), verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.cov_matrix = cls.model.compute_stationary_covariance_matrix(shock_cov_matrix=np.eye(1) * 0.01)

    def test_plot_with_defaults(self):
        fig = plot_covariance_matrix(self.cov_matrix)





if __name__ == '__main__':
    unittest.main()
