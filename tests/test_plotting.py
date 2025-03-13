import unittest
import warnings
from typing import Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from gEconpy.model.model import (
    autocorrelation_matrix,
    check_bk_condition,
    impulse_response_function,
    simulate,
    stationary_covariance_matrix,
)
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.plotting import (
    plot_acf,
    plot_covariance_matrix,
    plot_eigenvalues,
    plot_heatmap,
    plot_irf,
    plot_kalman_filter,
    plot_priors,
    plot_simulation,
    prepare_gridspec_figure,
)
from tests.utilities.shared_fixtures import (
    load_and_cache_model,
    load_and_cache_statespace,
)


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
        cls.model = load_and_cache_model("rbc_linearized.gcn", backend="numpy")
        cls.data = simulate(
            cls.model,
            simulation_length=100,
            n_simulations=1000,
            shock_std=0.1,
            solver="gensys",
            verbose=False,
            steady_state_kwargs={"verbose": False, "progressbar": False},
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
        plt.close()

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
    model = load_and_cache_model("full_nk.gcn", backend="numpy")
    model.steady_state(verbose=False)
    T, R = model.solve_model(verbose=False, solver="gensys")
    irf = impulse_response_function(
        model,
        T=T,
        R=R,
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


def test_plot_irf_raises_if_var_not_found(irf_setup):
    model, irf = irf_setup

    with pytest.raises(
        ValueError, match="Invalid not found among simulated impulse responses."
    ):
        plot_irf(irf, vars_to_plot=["Y", "C", "Invalid"])

    plt.close()


def test_plot_irf_raises_if_shock_not_found(irf_setup):
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
    plt.close()


def test_plot_irf_legend(irf_setup):
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
        cls.model = load_and_cache_model("one_block_1.gcn", backend="numpy")

    def test_plot_with_defaults(self):
        from matplotlib.collections import PathCollection

        fig = plot_eigenvalues(
            self.model,
            linearize_model_kwargs={
                "verbose": False,
                "steady_state_kwargs": {"progressbar": False, "verbose": False},
            },
        )

        scatter_points = fig.axes[0].findobj(PathCollection)[0].get_offsets().data
        data = check_bk_condition(
            self.model,
            return_value="dataframe",
            verbose=False,
            steady_state_kwargs={"progressbar": False, "verbose": False},
        )

        n_finite = (data["Modulus"] < 1.5).sum()
        self.assertEqual(n_finite, scatter_points.shape[0])
        plt.close()

    def test_plot_with_aesthetic_params(self):
        fig = plot_eigenvalues(
            self.model,
            dpi=144,
            figsize=(2, 2),
            linearize_model_kwargs={
                "verbose": False,
                "steady_state_kwargs": {"progressbar": False, "verbose": False},
            },
        )

        self.assertEqual(fig.get_figwidth(), 2)
        self.assertEqual(fig.get_figheight(), 2)
        self.assertEqual(fig.dpi, 144)
        plt.close()


class TestPlotCovarianceMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_and_cache_model("one_block_1.gcn", backend="numpy")

        cls.cov_matrix = stationary_covariance_matrix(
            cls.model,
            shock_cov_matrix=np.eye(1) * 0.01,
            return_df=True,
            verbose=False,
            steady_state_kwargs={"progressbar": False, "verbose": False},
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
        cls.model = load_and_cache_model("one_block_1.gcn", backend="numpy")

        cls.acf = autocorrelation_matrix(
            cls.model,
            shock_cov_matrix=np.eye(1) * 0.01,
            return_xr=True,
            verbose=False,
            steady_state_kwargs={"progressbar": False, "verbose": False},
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
        plt.close()


@pytest.fixture(scope="session")
def ss_mod() -> DSGEStateSpace:
    model = load_and_cache_statespace("rbc_linearized.gcn")
    model.configure(
        observed_states=["Y", "C", "L"],
        measurement_error=["Y", "C", "L"],
        full_shock_covaraince=False,
        solver="gensys",
        mode="FAST_RUN",
        verbose=False,
    )

    return model


@pytest.fixture(scope="session")
@pytest.mark.filterwarnings("ignore")
def pm_mod(ss_mod) -> pm.Model:
    with pm.Model(coords=ss_mod.coords) as pm_mod:
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)

        for var_name in ss_mod.observed_states:
            pm.Gamma(f"error_sigma_{var_name}", alpha=2, beta=100)

        with warnings.catch_warnings(action="ignore"):
            ss_mod.build_statespace_graph(np.full((100, 3), np.nan))

    return pm_mod


@pytest.fixture(scope="session")
def prior_idata(pm_mod, ss_mod) -> tuple[az.InferenceData, pd.DataFrame]:
    with warnings.catch_warnings(action="ignore"):
        with pm_mod:
            prior = pm.sample_prior_predictive(25)

        unconditional_prior = ss_mod.sample_unconditional_prior(
            prior, progressbar=False
        )

        prior["unconditional_prior"] = unconditional_prior
        fake_data = (
            unconditional_prior["prior_observed"]
            .sel(observed_state=["Y", "C", "L"], chain=0, draw=0)
            .to_dataframe()["prior_observed"]
            .unstack("observed_state")
        )
        fake_data.index = pd.RangeIndex(0, 100)

        with pm_mod:
            pm.set_data({"data": fake_data})
            ss_mod._fit_data = fake_data

        conditional_prior = ss_mod.sample_conditional_prior(prior, progressbar=False)
        prior["conditional_prior"] = conditional_prior

    return (prior, fake_data)


@pytest.mark.parametrize("kalman_output", ["predicted", "filtered", "smoothed"])
@pytest.mark.parametrize("vars_to_plot", [["Y"], ["Y", "C"], ["Y", "C", "L"]])
def test_plot_kalman_filter(
    ss_mod,
    prior_idata,
    kalman_output: Literal["predicted", "filtered", "smoothed"],
    vars_to_plot,
):
    idata, fake_data = prior_idata
    fig = plot_kalman_filter(
        idata["conditional_prior"],
        fake_data,
        kalman_output=kalman_output,
        group="prior",
        vars_to_plot=vars_to_plot,
    )

    assert len(fig.axes) == len(vars_to_plot)
    assert all(axis.get_title() in vars_to_plot for axis in fig.axes)


def test_plot_priors(ss_mod):
    fig = plot_priors(ss_mod)
    titles = [ax.get_title() for ax in fig.axes]
    assert len(titles) == len(ss_mod.shock_priors | ss_mod.param_priors)


if __name__ == "__main__":
    unittest.main()
