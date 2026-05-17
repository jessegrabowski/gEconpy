import warnings

from types import SimpleNamespace
from typing import Literal

import arviz_base as azb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytest
import xarray as xr

from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from gEconpy.model.simulate import impulse_response_function, simulate
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.statistics import (
    autocorrelation_matrix,
    check_bk_condition,
    eigenvalue_sensitivity,
    stationary_covariance_matrix,
)
from gEconpy.plotting import (
    plot_acf,
    plot_corner,
    plot_covariance_matrix,
    plot_eigenvalue_sensitivity,
    plot_eigenvalues,
    plot_estimated_matrix,
    plot_heatmap,
    plot_irf,
    plot_kalman_filter,
    plot_posterior_with_prior,
    plot_priors,
    plot_simulation,
    plot_solvability,
    plot_solvability_summary,
    plot_timeseries,
    prepare_gridspec_figure,
)
from tests._resources.cache_compiled_models import (
    load_and_cache_model,
    load_and_cache_statespace,
)

# These tests render to an in-memory buffer rather than a window: the
# MPLBACKEND=Agg environment variable, set under [tool.pytest.ini_options] in
# pyproject.toml, forces the non-interactive Agg backend for the whole suite.

# Steady-state solve kwargs shared by the fixtures — quiet and progressbar-free.
SS_KW = {"progressbar": False, "verbose": False}
LINEARIZE_KW = {"verbose": False, "steady_state_kwargs": SS_KW}


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test leaves open so they do not accumulate."""
    yield
    plt.close("all")


# ----------------------------------------------------------------------------
# Fixtures: each builds a plotting input once per session. Plotting tests then
# sweep argument combinations through these without recompiling models.
# ----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rbc_model():
    return load_and_cache_model("rbc_linearized.gcn")


@pytest.fixture(scope="session")
def one_block_model():
    return load_and_cache_model("one_block_1.gcn")


@pytest.fixture(scope="session")
def simulation_data(rbc_model):
    return simulate(
        rbc_model,
        simulation_length=100,
        n_simulations=1000,
        shock_std=0.1,
        solver="gensys",
        verbose=False,
        steady_state_kwargs=SS_KW,
    )


@pytest.fixture(scope="session")
def irf_setup():
    model = load_and_cache_model("full_nk.gcn")
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


@pytest.fixture(scope="session")
def cov_matrix(one_block_model):
    return stationary_covariance_matrix(
        one_block_model,
        shock_cov_matrix=np.eye(1) * 0.01,
        return_df=True,
        verbose=False,
        steady_state_kwargs=SS_KW,
    )


@pytest.fixture(scope="session")
def acf_data(one_block_model):
    return autocorrelation_matrix(
        one_block_model,
        shock_cov_matrix=np.eye(1) * 0.01,
        return_xr=True,
        verbose=False,
        steady_state_kwargs=SS_KW,
    )


@pytest.fixture(scope="session")
def sensitivity_data(one_block_model):
    # eigenvalue_sensitivity forwards its own `verbose` into model.steady_state, so
    # `verbose` must not also appear in steady_state_kwargs.
    return eigenvalue_sensitivity(one_block_model, verbose=False, steady_state_kwargs={"progressbar": False})


@pytest.fixture(scope="session")
def timeseries_data():
    """Build a plain DataFrame of time series with a datetime index."""
    rng = np.random.default_rng(0)
    index = pd.date_range("1950-01-01", periods=160, freq="QS")
    return pd.DataFrame(
        {name: rng.standard_normal(160).cumsum() for name in ("Y", "C", "K")},
        index=index,
    )


@pytest.fixture(scope="session")
def solvability_data():
    """Build a synthetic solvability table: parameter draws plus a ``failure_step`` column."""
    rng = np.random.default_rng(0)
    n = 120
    steps = np.array(
        [None, "steady_state", "perturbation", "blanchard-kahn", "deterministic_norm", "stochastic_norm"],
        dtype=object,
    )
    return pd.DataFrame(
        {
            "alpha": rng.uniform(0.2, 0.5, n),
            "beta": rng.uniform(0.90, 0.99, n),
            "rho": rng.uniform(0.50, 0.95, n),
            "failure_step": rng.choice(steps, size=n, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
        }
    )


@pytest.fixture(scope="session")
def fake_posterior_idata():
    """Minimal posterior DataTree: three scalar parameters and a shock matrix.

    Built directly from arrays so the arviz-dependent plotting functions can be
    exercised without running MCMC.
    """
    rng = np.random.default_rng(1234)
    n_chains, n_draws, k = 2, 200, 3
    posterior = {
        "alpha": rng.beta(2.0, 5.0, size=(n_chains, n_draws)),
        "beta": rng.normal(0.0, 1.0, size=(n_chains, n_draws)),
        "sigma": rng.gamma(2.0, 0.5, size=(n_chains, n_draws)),
        "state_chol_corr": rng.uniform(-1.0, 1.0, size=(n_chains, n_draws, k, k)),
    }
    return azb.from_dict(
        {"posterior": posterior},
        dims={"state_chol_corr": ["shock_dim_0", "shock_dim_1"]},
    )


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
def prior_idata(pm_mod, ss_mod) -> tuple[xr.DataTree, pd.DataFrame]:
    with warnings.catch_warnings(action="ignore"):
        with pm_mod:
            prior = pm.sample_prior_predictive(25)

        unconditional_prior = ss_mod.sample_unconditional_prior(prior, progressbar=False)

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


# ----------------------------------------------------------------------------
# prepare_gridspec_figure
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_cols, n_plots",
    [(3, 9), (2, 9), (4, 9), (1, 5), (5, 3)],
    ids=["square", "tall", "wide", "single-col", "single-row"],
)
def test_prepare_gridspec_figure_count(n_cols, n_plots):
    _gs, locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_plots)
    assert len(locs) == n_plots


def test_prepare_gridspec_figure_tall_last_loc():
    _gs, locs = prepare_gridspec_figure(n_cols=2, n_plots=9)
    assert locs[-1] == (slice(8, 10, None), slice(1, 3, None))


def test_prepare_gridspec_figure_wide_last_loc():
    _gs, locs = prepare_gridspec_figure(n_cols=4, n_plots=9)
    assert locs[-1] == (slice(4, 6, None), slice(3, 5, None))


# ----------------------------------------------------------------------------
# plot_timeseries
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, n_axes",
    [
        ({}, 3),
        ({"vars_to_plot": ["Y", "C"]}, 2),
        ({"fig_kwargs": {"figsize": (8, 6), "dpi": 80}, "color": "tab:green"}, 3),
    ],
    ids=["defaults", "subset", "fig_and_line_kwargs"],
)
def test_plot_timeseries(timeseries_data, kwargs, n_axes):
    fig = plot_timeseries(timeseries_data, **kwargs)
    assert len(fig.axes) == n_axes


# ----------------------------------------------------------------------------
# plot_simulation
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, n_axes",
    [
        ({"vars_to_plot": ["Y", "C", "K"]}, 3),
        ({"vars_to_plot": ["Y", "C"]}, 2),
        ({"vars_to_plot": ["Y", "C", "K"], "ci": 0.95}, 3),
        ({"vars_to_plot": ["Y", "C", "K"], "ci": 0.95, "n_cols": 2}, 3),
        ({"vars_to_plot": ["Y", "C"], "cmap": "YlGn", "fill_color": "tab:red"}, 2),
    ],
    ids=["three_vars", "two_vars", "ci", "ci_ncols", "cmap_fill"],
)
def test_plot_simulation(simulation_data, kwargs, n_axes):
    fig = plot_simulation(simulation_data, **kwargs)
    assert len(fig.axes) == n_axes


def test_plot_simulation_defaults_plots_all_variables(simulation_data, rbc_model):
    fig = plot_simulation(simulation_data)
    assert len(fig.axes) == len(rbc_model.variables)


def test_plot_simulation_bad_var_raises(simulation_data):
    with pytest.raises(ValueError, match="Invalid not found among model variables"):
        plot_simulation(simulation_data, vars_to_plot=["Y", "C", "Invalid"])


# ----------------------------------------------------------------------------
# plot_irf
# ----------------------------------------------------------------------------
def test_plot_irf_axis_and_line_counts(irf_setup):
    model, irf = irf_setup
    fig = plot_irf(irf, legend=True)
    assert len(fig.axes) == len(model.variables)
    assert len(fig.axes[0].get_lines()) == len(model.shocks)


@pytest.mark.parametrize("shocks_to_plot", ["epsilon_Y", ["epsilon_Y"]], ids=["str", "list"])
def test_plot_irf_one_shock_line_count(irf_setup, shocks_to_plot):
    model, irf = irf_setup
    fig = plot_irf(irf, shocks_to_plot=shocks_to_plot)
    assert len(fig.axes) == len(model.variables)
    assert len(fig.axes[0].get_lines()) == 1


def test_plot_irf_one_variable_axis_count(irf_setup):
    model, irf = irf_setup
    fig = plot_irf(irf, vars_to_plot="Y")
    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) == len(model.shocks)


def test_plot_irf_grid_and_cmap(irf_setup):
    _model, irf = irf_setup
    fig = plot_irf(irf, vars_to_plot=["Y", "C"], n_cols=1, cmap="viridis")
    assert len(fig.axes) == 2


def test_plot_irf_multiple_scenarios_list(irf_setup):
    model, irf = irf_setup
    fig = plot_irf([irf, irf], vars_to_plot=["Y", "C"])
    assert len(fig.axes) == 2
    # Both scenarios draw every shock on each panel, so lines double.
    assert len(fig.axes[0].get_lines()) == 2 * len(model.shocks)


def test_plot_irf_multiple_scenarios_dict(irf_setup):
    model, irf = irf_setup
    fig = plot_irf({"baseline": irf, "alternative": irf}, vars_to_plot=["Y"], legend=True)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) == 2 * len(model.shocks)


def test_plot_irf_bad_var_raises(irf_setup):
    _model, irf = irf_setup
    with pytest.raises(ValueError, match="variable 'Invalid' not found among available:"):
        plot_irf(irf, vars_to_plot=["Y", "C", "Invalid"])


def test_plot_irf_bad_shock_raises(irf_setup):
    _model, irf = irf_setup
    with pytest.raises(ValueError, match=r"shock 'Invalid' not found among available: "):
        plot_irf(irf, vars_to_plot=["Y", "C"], shocks_to_plot=["epsilon_Y", "Invalid"])


def test_plot_irf_legend_on_figure(irf_setup):
    _model, irf = irf_setup
    fig = plot_irf(irf, vars_to_plot=["Y", "C"], shocks_to_plot=["epsilon_Y"], legend=True)
    assert all(axis.get_legend() is None for axis in fig.axes)
    assert len(fig.figure.legends) == 1


# ----------------------------------------------------------------------------
# plot_solvability / plot_solvability_summary
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, n_axes",
    [({}, 9), ({"params_to_plot": ["alpha", "beta"]}, 4)],
    ids=["defaults", "subset"],
)
def test_plot_solvability(solvability_data, kwargs, n_axes):
    fig = plot_solvability(solvability_data, **kwargs)
    assert len(fig.axes) == n_axes


def test_plot_solvability_hides_upper_triangle(solvability_data):
    fig = plot_solvability(solvability_data)
    # Three parameters -> the strict upper triangle (3 panels) is hidden.
    assert sum(not ax.get_visible() for ax in fig.axes) == 3


def test_plot_solvability_summary(solvability_data):
    fig = plot_solvability_summary(solvability_data)
    n_categories = solvability_data["failure_step"].fillna("success").nunique()
    assert len(fig.axes) == 1
    # One stacked bar segment per failure-step category present in the data.
    assert len(fig.axes[0].patches) == n_categories


# ----------------------------------------------------------------------------
# plot_eigenvalues
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [{}, {"plot_circle": False}], ids=["defaults", "no_circle"])
def test_plot_eigenvalues(one_block_model, kwargs):
    fig = plot_eigenvalues(one_block_model, linearize_model_kwargs=LINEARIZE_KW, **kwargs)
    assert isinstance(fig, Figure)


def test_plot_eigenvalues_scatter_point_count(one_block_model):
    fig = plot_eigenvalues(one_block_model, linearize_model_kwargs=LINEARIZE_KW)
    scatter_points = fig.axes[0].findobj(PathCollection)[0].get_offsets().data
    data = check_bk_condition(one_block_model, return_value="dataframe", verbose=False, steady_state_kwargs=SS_KW)
    INF_CUTOFF = 1.5
    n_finite = (data["Modulus"] < INF_CUTOFF).sum()
    assert n_finite == scatter_points.shape[0]


def test_plot_eigenvalues_explicit_matrices_match_model(one_block_model):
    A, B, C, D = one_block_model.linearize_model(verbose=False, steady_state_kwargs=SS_KW)
    explicit = plot_eigenvalues(one_block_model, A=A, B=B, C=C, D=D)
    from_model = plot_eigenvalues(one_block_model, linearize_model_kwargs=LINEARIZE_KW)

    def n_points(fig):
        return fig.axes[0].findobj(PathCollection)[0].get_offsets().data.shape[0]

    # Supplying A, B, C, D must reproduce the eigenvalues found by linearizing internally.
    assert n_points(explicit) == n_points(from_model)


# ----------------------------------------------------------------------------
# plot_eigenvalue_sensitivity
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"params_to_plot": ["alpha", "rho"]},
        {"plot_circle": False, "n_cols": 2},
        {"perturbation": 0.05, "figsize": (8, 8), "dpi": 80},
    ],
    ids=["defaults", "subset", "no_circle_ncols", "perturbation"],
)
def test_plot_eigenvalue_sensitivity(one_block_model, sensitivity_data, kwargs):
    fig = plot_eigenvalue_sensitivity(one_block_model, sensitivity_data=sensitivity_data, **kwargs)
    assert isinstance(fig, Figure)


def test_plot_eigenvalue_sensitivity_subset_panel_count(one_block_model, sensitivity_data):
    fig = plot_eigenvalue_sensitivity(
        one_block_model, sensitivity_data=sensitivity_data, params_to_plot=["alpha", "rho"]
    )
    assert sum(ax.get_visible() for ax in fig.axes) == 2


# ----------------------------------------------------------------------------
# plot_covariance_matrix / plot_heatmap
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cmap": "viridis"},
        {"annotation_kwargs": {"threshold": 0.5, "fontsize": 5}},
        {"heatmap_kwargs": {"interpolation": "antialiased"}},
        {"figsize": (5, 5), "dpi": 110, "cbarlabel": "Cov"},
    ],
    ids=["defaults", "cmap", "annotation_kwargs", "heatmap_kwargs", "cbarlabel"],
)
def test_plot_covariance_matrix(cov_matrix, kwargs):
    fig = plot_covariance_matrix(cov_matrix, **kwargs)
    assert fig.findobj(AxesImage)


def test_plot_heatmap_returns_image_and_colorbar(cov_matrix):
    im, cbar = plot_heatmap(cov_matrix)
    assert isinstance(im, AxesImage)
    assert isinstance(cbar, Colorbar)


# ----------------------------------------------------------------------------
# plot_acf
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("vars_to_plot", [["C", "K", "A"], ["C"]], ids=["three", "one"])
def test_plot_acf_subset_titles(acf_data, vars_to_plot):
    fig = plot_acf(acf_data, vars_to_plot=vars_to_plot)
    assert [ax.get_title() for ax in fig.axes] == vars_to_plot


def test_plot_acf_defaults_plots_all_variables(acf_data, one_block_model):
    fig = plot_acf(acf_data)
    assert [ax.get_title() for ax in fig.axes] == [v.base_name for v in one_block_model.variables]


def test_plot_acf_bad_var_raises(acf_data):
    with pytest.raises(ValueError, match="Can not plot variable Invalid"):
        plot_acf(acf_data, vars_to_plot=["K", "C", "Invalid"])


# ----------------------------------------------------------------------------
# plot_kalman_filter
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("kalman_output", ["predicted", "filtered", "smoothed"])
@pytest.mark.parametrize("vars_to_plot", [["Y"], ["Y", "C"], ["Y", "C", "L"]])
def test_plot_kalman_filter(
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


def test_plot_kalman_filter_bad_output_raises(prior_idata):
    idata, fake_data = prior_idata
    with pytest.raises(ValueError, match='kalman_output must be one of "filtered", "predicted", "smoothed"'):
        plot_kalman_filter(idata["conditional_prior"], fake_data, kalman_output="bogus", group="prior")


@pytest.mark.parametrize("vars_to_plot", [["Y"], ["Y", "C", "L"]], ids=["one", "three"])
def test_plot_kalman_filter_observed(prior_idata, vars_to_plot):
    # observed=True switches to the observed_state coord and the *_observed variables.
    idata, fake_data = prior_idata
    fig = plot_kalman_filter(
        idata["conditional_prior"],
        fake_data,
        kalman_output="predicted",
        group="prior",
        vars_to_plot=vars_to_plot,
        observed=True,
    )
    assert len(fig.axes) == len(vars_to_plot)
    assert all(axis.get_title() in vars_to_plot for axis in fig.axes)


# ----------------------------------------------------------------------------
# plot_priors
# ----------------------------------------------------------------------------
def test_plot_priors_plots_all_priors(ss_mod):
    fig = plot_priors(ss_mod)
    titles = [ax.get_title() for ax in fig.axes]
    assert len(titles) == len(ss_mod.shock_priors | ss_mod.param_priors)


def test_plot_priors_marks_initial_values(ss_mod):
    def n_initial_value_marks(fig):
        return sum(
            1
            for ax in fig.axes
            for line in ax.get_lines()
            if line.get_linestyle() == "--" and len(set(np.ravel(line.get_xdata()))) == 1
        )

    marked = plot_priors(ss_mod, mark_initial_value=True)
    unmarked = plot_priors(ss_mod, mark_initial_value=False)
    assert n_initial_value_marks(marked) > n_initial_value_marks(unmarked)


# ----------------------------------------------------------------------------
# plot_corner
# ----------------------------------------------------------------------------
def test_plot_corner_grid_shape(prior_idata):
    idata, _ = prior_idata
    var_names = list(idata.prior.data_vars)[:3]
    fig = plot_corner(idata, group="prior", var_names=var_names)
    k = len(var_names)
    assert len(fig.axes) == k * k
    assert sum(not ax.get_visible() for ax in fig.axes) == k * (k - 1) // 2


def test_plot_corner_axis_labels(prior_idata):
    idata, _ = prior_idata
    var_names = list(idata.prior.data_vars)[:2]
    fig = plot_corner(idata, group="prior", var_names=var_names)
    # Off-diagonal lower-left axis is at row=1, col=0 -> index 2.
    ax = fig.axes[2]
    assert ax.get_xlabel() == var_names[0]
    assert ax.get_ylabel() == var_names[1]


def test_plot_corner_bad_var_raises(prior_idata):
    idata, _ = prior_idata
    valid = next(iter(idata.prior.data_vars))
    with pytest.raises(ValueError, match=r'Variable "bad" not found in idata\[prior\]'):
        plot_corner(idata, group="prior", var_names=[valid, "bad"])


def test_plot_corner_colorby_adds_scatter(prior_idata):
    idata, _ = prior_idata
    x, y, colorby = list(idata.prior.data_vars)[:3]
    fig = plot_corner(idata, group="prior", var_names=[x, y], colorby=colorby)
    ax = fig.axes[2]
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) >= 1


def test_plot_corner_missing_colorby_raises(prior_idata):
    idata, _ = prior_idata
    var_names = list(idata.prior.data_vars)[:2]
    with pytest.raises(ValueError, match=r'colorby "missing" not found in idata\[prior\]'):
        plot_corner(idata, group="prior", var_names=var_names, colorby="missing")


# ----------------------------------------------------------------------------
# plot_posterior_with_prior
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, n_var",
    [
        ({"var_names": ["alpha", "beta", "sigma"]}, 3),
        ({"var_names": ["alpha", "beta"], "n_cols": 1}, 2),
        ({"var_names": ["alpha"], "fig_kwargs": {"figsize": (6, 4), "dpi": 110}}, 1),
    ],
    ids=["three_vars", "n_cols", "fig_kwargs"],
)
def test_plot_posterior_with_prior(fake_posterior_idata, kwargs, n_var):
    fig = plot_posterior_with_prior(fake_posterior_idata, prior_dict={}, **kwargs)
    assert len(fig.axes) >= n_var


def test_plot_posterior_with_prior_overlays_prior(fake_posterior_idata):
    def orange_lines(fig):
        target = mcolors.to_rgba("tab:orange")
        return [line for ax in fig.axes for line in ax.get_lines() if mcolors.to_rgba(line.get_color()) == target]

    without_prior = plot_posterior_with_prior(fake_posterior_idata, var_names=["alpha", "beta"], prior_dict={})
    with_prior = plot_posterior_with_prior(
        fake_posterior_idata, var_names=["alpha", "beta"], prior_dict={"alpha": pz.Beta(2.0, 5.0)}
    )
    # The single prior entry adds exactly one orange density curve.
    assert len(orange_lines(without_prior)) == 0
    assert len(orange_lines(with_prior)) == 1


def test_plot_posterior_with_prior_marks_true_values(fake_posterior_idata):
    true_values = xr.Dataset({"alpha": xr.DataArray(0.3), "beta": xr.DataArray(-0.5)})
    fig = plot_posterior_with_prior(
        fake_posterior_idata,
        var_names=["alpha", "beta"],
        prior_dict={},
        true_values=true_values,
    )
    black = mcolors.to_rgba("k")
    marked_x = {
        round(float(np.ravel(line.get_xdata())[0]), 4)
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_linestyle() == "--"
        and mcolors.to_rgba(line.get_color()) == black
        and len(set(np.ravel(line.get_xdata()))) == 1
    }
    assert {0.3, -0.5} <= marked_x


def test_plot_posterior_with_prior_accepts_one_shot_iterable(fake_posterior_idata):
    # Regression guard for `var_names = list(var_names)`: a generator must
    # survive being measured with len() and iterated more than once.
    fig = plot_posterior_with_prior(
        fake_posterior_idata,
        var_names=(name for name in ["alpha", "beta"]),
        prior_dict={},
    )
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------------
# plot_estimated_matrix
# ----------------------------------------------------------------------------
def _matrix_dsge_stub(n_shocks):
    return SimpleNamespace(
        k_posdef=n_shocks,
        shocks=[SimpleNamespace(base_name=f"epsilon_{n}") for n in range(n_shocks)],
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"symmetrical": False}, {"subplot_kwargs": {"figsize": (6, 6), "dpi": 90}}],
    ids=["defaults", "not_symmetrical", "subplot_kwargs"],
)
def test_plot_estimated_matrix(fake_posterior_idata, kwargs):
    fig = plot_estimated_matrix(fake_posterior_idata, _matrix_dsge_stub(3), matrix_name="state_chol_corr", **kwargs)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 9


def test_plot_estimated_matrix_symmetrical_hides_upper_triangle(fake_posterior_idata):
    fig = plot_estimated_matrix(fake_posterior_idata, _matrix_dsge_stub(3), matrix_name="state_chol_corr")
    # Only the strict lower triangle (3 panels) stays visible.
    assert sum(ax.get_visible() for ax in fig.axes) == 3


def test_plot_estimated_matrix_single_shock(fake_posterior_idata):
    # Regression guard: k_posdef=1 makes plt.subplots return a bare Axes; the
    # function must use squeeze=False to keep ax[i, j] indexing valid.
    fig = plot_estimated_matrix(fake_posterior_idata, _matrix_dsge_stub(1), matrix_name="state_chol_corr")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
