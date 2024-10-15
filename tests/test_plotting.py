import os
import unittest

from pathlib import Path

import numpy as np
import pytest

from matplotlib import pyplot as plt

from gEconpy.model.build import model_from_gcn
from gEconpy.model.model import (
    autocorrelation_matrix,
    bk_condition,
    impulse_response_function,
    simulate,
    stationary_covariance_matrix,
)
from gEconpy.plotting import (
    plot_covariance_matrix,
    plot_eigenvalues,
    plot_irf,
    plot_simulation,
    prepare_gridspec_figure,
)
from gEconpy.plotting.plotting import (
    plot_acf,
    plot_heatmap,
)

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
        cls.model = model_from_gcn(file_path, verbose=False)
        cls.data = simulate(
            cls.model, simulation_length=100, n_simulations=1000, shock_std=0.1
        )

    def test_plot_simulation_defaults(self):
        fig = plot_simulation(self.data)

        self.assertEqual(len(fig.axes), len(self.model.variables))
        plt.close()

    def test_plot_simulation_vars_to_plot(self):
        fig = plot_simulation(self.data, vars_to_plot=["Y", "C"])

        self.assertEqual(len(fig.axes), 2)
        plt.close()

    def test_var_not_found_raises(self):
        with self.assertRaises(ValueError) as error:
            plot_simulation(self.data, vars_to_plot=["Y", "C", "Invalid"])
        error_msg = error.exception
        self.assertEqual(str(error_msg), "Invalid not found among model variables.")

    def test_plot_simulation_with_ci(self):
        fig = plot_simulation(self.data, ci=0.95)

        self.assertEqual(len(fig.axes), len(self.model.variables))
        plt.close()

    def test_plot_simulation_aesthetic_params(self):
        fig = plot_simulation(
            self.data, cmap="YlGn", figsize=(14, 4), dpi=100, fill_color="brickred"
        )

        self.assertEqual(len(fig.axes), len(self.model.variables))
        self.assertEqual(fig.get_dpi(), 100)
        self.assertEqual(fig.get_figwidth(), 14)
        self.assertEqual(fig.get_figheight(), 4)
        plt.close()


@pytest.fixture(scope="session")
def irf_setup():
    file_path = os.path.join(ROOT, "Test GCNs/Full_New_Keynesian.gcn")

    model = model_from_gcn(file_path, verbose=False)
    model.steady_state(verbose=False)
    model.solve_model(verbose=False)
    irf = impulse_response_function(
        model,
        simulation_length=100,
        shock_size=0.1,
        return_individual_shocks=True,
    )

    return model, irf


def test_plot_irf_defaults(irf_setup):
    model, irf = irf_setup
    fig = plot_irf(irf, legend=True)

    assert len(fig.axes) == len(model.variables)
    assert len(fig.axes[0].get_lines()) == len(model.shocks)

    plt.close()


@pytest.mark.parametrize(
    "shocks_to_plot", ["epsilon_Y", ["epsilon_Y"]], ids=["str", "list"]
)
def test_plot_irf_one_shock(irf_setup, shocks_to_plot):
    model, irf = irf_setup
    fig = plot_irf(irf, shocks_to_plot=shocks_to_plot)

    assert len(fig.axes) == len(model.variables)
    assert len(fig.axes[0].get_lines()) == 1

    plt.close()


def test_plot_irf_one_variable(irf_setup):
    model, irf = irf_setup
    fig = plot_irf(irf, vars_to_plot="Y")

    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) == len(model.shocks)

    plt.close()


def test_var_not_found_raises(irf_setup):
    model, irf = irf_setup

    with pytest.raises(
        ValueError, match="Invalid not found among simulated impulse responses."
    ):
        plot_irf(irf, vars_to_plot=["Y", "C", "Invalid"])

    plt.close()


def test_shock_not_found_raises(irf_setup):
    model, irf = irf_setup

    with pytest.raises(
        ValueError,
        match="Invalid not found among shocks used in impulse response data.",
    ):
        plot_irf(
            irf,
            vars_to_plot=["Y", "C"],
            shocks_to_plot=["epsilon_Y", "Invalid"],
        )


def test_legend(irf_setup):
    model, irf = irf_setup

    fig = plot_irf(
        irf, vars_to_plot=["Y", "C"], shocks_to_plot=["epsilon_Y"], legend=True
    )
    assert all(axis.get_legend() is None for axis in fig.axes)
    assert len(fig.figure.legends) == 1
    plt.close()


class TestPlotEigenvalues(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
        cls.model = model_from_gcn(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)

    def test_plot_with_defaults(self):
        fig = plot_eigenvalues(self.model)
        from matplotlib.collections import PathCollection

        scatter_points = fig.axes[0].findobj(PathCollection)[0].get_offsets().data
        data = bk_condition(self.model, return_value="dataframe", verbose=False)

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
        cls.model = model_from_gcn(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.cov_matrix = stationary_covariance_matrix(
            cls.model, shock_cov_matrix=np.eye(1) * 0.01, return_df=True
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
        cls.model = model_from_gcn(file_path, verbose=False)
        cls.model.steady_state(verbose=False)
        cls.model.solve_model(verbose=False)
        cls.acf = autocorrelation_matrix(
            cls.model, shock_cov_matrix=np.eye(1) * 0.01, return_xr=True
        )

    def test_plot_with_defaults(self):
        fig = plot_acf(self.acf)
        self.assertEqual(len(fig.axes), len(self.model.variables))
        for axis, variable in zip(fig.axes, self.model.variables):
            assert axis.get_title() == variable.base_name

        plt.close()

    def test_plot_with_subset(self):
        fig = plot_acf(self.acf, vars_to_plot=["C", "K", "A"])
        self.assertEqual(len(fig.axes), 3)
        for axis, variable in zip(fig.axes, ["C", "K", "A"]):
            assert axis.get_title() == variable

        plt.close()

    def test_invalid_var_raises(self):
        with self.assertRaises(ValueError) as error:
            plot_acf(self.acf, vars_to_plot=["K", "C", "Invalid"])
        msg = str(error.exception)
        self.assertEqual(
            msg,
            "Can not plot variable Invalid, it was not found in the provided covariance matrix",
        )


if __name__ == "__main__":
    unittest.main()
