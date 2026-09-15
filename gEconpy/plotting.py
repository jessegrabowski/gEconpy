import math
import warnings

from collections.abc import Mapping
from itertools import product
from typing import Any, Literal, NamedTuple, cast

import arviz_stats as azs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from arviz_plots import plot_dist
from matplotlib.colors import Colormap
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import Formatter, StrMethodFormatter
from scipy import stats
from xarray_einstats.linalg import diagonal as xr_diagonal

from gEconpy.model.model import Model
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.statistics import check_bk_condition, eigenvalue_sensitivity

_SOLVABILITY_COLORS = {
    "success": "tab:blue",
    "steady_state": "tab:red",
    "perturbation": "tab:orange",
    "blanchard-kahn": "tab:green",
    "deterministic_norm": "tab:purple",
    "stochastic_norm": "tab:pink",
}
_SOLVABILITY_STAGES = [
    "success",
    "steady_state",
    "perturbation",
    "blanchard-kahn",
    "deterministic_norm",
    "stochastic_norm",
]

# Columns that solvability_check appends to the parameter draws. They are diagnostics, so they are never plotted as
# parameters.
_SOLVABILITY_META_COLS = {"failure_step", "norm_deterministic", "norm_stochastic"}

_EIGENVALUE_PLOT_MODULUS_CUTOFF = 10

_SCENARIO_LINE_STYLES = ["-", "--", "-.", ":"]

_REFERENCE_DEFAULTS = {"facecolors": "none", "edgecolors": "k", "s": 60, "linewidths": 1.6, "zorder": 4}


def set_matplotlib_style() -> None:
    """
    Update the global matplotlib rcParams with the gEconpy plotting defaults.

    The defaults set a wide 14 by 4 inch figure at 144 dpi, dashed grid lines, no axis spines, and constrained
    layout.

    Examples
    --------
    Apply the style once at the top of a script or notebook so that every later figure picks it up:

    .. code-block:: python

        import matplotlib.pyplot as plt

        from gEconpy.plotting import set_matplotlib_style

        set_matplotlib_style()
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [1, 3, 2])
    """
    config = {
        "figure.figsize": (14, 4),
        "figure.dpi": 144,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    }

    plt.rcParams.update(config)


def prepare_gridspec_figure(
    n_cols: int, n_plots: int, figure: Figure | None = None
) -> tuple[GridSpec, list[tuple[slice, slice]]]:
    """
    Lay out a grid of subplots, centering the last row when the number of plots does not fill it.

    Parameters
    ----------
    n_cols : int
        Number of columns in the grid.
    n_plots : int
        Number of subplots in the grid.
    figure : Figure, optional
        Figure the grid belongs to. By default the grid is not attached to a figure.

    Returns
    -------
    gs : GridSpec
        Grid layout with two grid cells per subplot in each direction, so that a partial last row can be centered.
    plot_locs : list of tuple of slice
        Row and column slices into ``gs`` for each subplot, in row-major order.
    """
    n_full_rows, remainder = divmod(n_plots, n_cols)
    n_rows = math.ceil(n_plots / n_cols)

    gs = GridSpec(2 * n_rows, 2 * n_cols, figure=figure)
    plot_locs = [
        (slice(i * 2, (i + 1) * 2), slice(j * 2, (j + 1) * 2)) for i, j in product(range(n_full_rows), range(n_cols))
    ]

    if remainder > 0:
        last_row = slice((n_rows - 1) * 2, n_rows * 2)
        left_pad = n_cols - remainder
        for j in range(remainder):
            col_slice = slice(left_pad + j * 2, left_pad + (j + 1) * 2)
            plot_locs.append((last_row, col_slice))

    return gs, plot_locs


def set_axis_cmap(axis: plt.Axes, cmap: str | Colormap | None) -> None:
    """
    Set the color cycle of an axis from a colormap.

    Parameters
    ----------
    axis : matplotlib Axes
        Axis whose property cycle is set.
    cmap : str, Colormap, or None
        Name of a matplotlib colormap or a Colormap object, sampled at 20 evenly spaced points. If None, the axis
        reverts to the default color cycle.
    """
    cycler = None
    if cmap is not None:
        colormap = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        cycler = plt.cycler(color=colormap(np.linspace(0, 1, 20)))
    axis.set_prop_cycle(cycler)


def plot_timeseries(
    df: pd.DataFrame,
    vars_to_plot: list[str] | None = None,
    n_cols: int | None = None,
    fig_kwargs: dict | None = None,
    **line_kwargs,
) -> Figure:
    """
    Plot each column of a DataFrame of time series in its own panel.

    Parameters
    ----------
    df : DataFrame
        Data to plot. Each column is a variable and the index is a datetime index.
    vars_to_plot : list of str, optional
        Columns to plot. All columns are plotted by default.
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to the smaller of 4 and the number of variables plotted.
    fig_kwargs : dict, optional
        Keyword arguments forwarded to :func:`matplotlib.pyplot.figure`. Empty by default.
    **line_kwargs
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    figure : Figure
        Figure containing one panel per variable.

    Examples
    --------
    Plot three random walks on a quarterly index:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        from gEconpy.plotting import plot_timeseries

        rng = np.random.default_rng(0)
        index = pd.date_range("1960-01-01", periods=200, freq="QS")
        df = pd.DataFrame({name: rng.standard_normal(200).cumsum() for name in ["Y", "C", "I"]}, index=index)

        fig = plot_timeseries(df, n_cols=3, color="tab:red")
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    if n_cols is None:
        n_cols = min(4, len(df.columns))
    if vars_to_plot is None:
        vars_to_plot = df.columns

    figure = plt.figure(**fig_kwargs)
    gs, plot_locs = prepare_gridspec_figure(n_cols, len(vars_to_plot), figure=figure)

    for var, loc in zip(vars_to_plot, plot_locs, strict=False):
        axis = figure.add_subplot(gs[loc])
        axis.plot(df.index, df[var], **line_kwargs)

        axis.xaxis.set_major_locator(YearLocator(10))
        axis.xaxis.set_minor_locator(YearLocator(1))
        axis.xaxis.set_major_formatter(DateFormatter("%Y"))

        axis.set(title=var)

    return figure


def plot_simulation(
    simulation: xr.DataArray,
    vars_to_plot: list[str] | None = None,
    ci: float | None = None,
    n_cols: int | None = None,
    cmap: str | Colormap | None = None,
    fill_color: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 100,
) -> Figure:
    """
    Plot simulated trajectories of model variables, one panel per variable.

    Parameters
    ----------
    simulation : DataArray
        Output of :func:`~gEconpy.model.simulate.simulate`, with ``simulation``, ``time``, and ``variable``
        dimensions.
    vars_to_plot : list of str, optional
        Variables to plot. All variables in ``simulation`` are plotted by default.
    ci : float, optional
        Width of the credible band to draw around the mean trajectory, between 0 and 1. By default every simulated
        trajectory is drawn and no band is shown.
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to the smaller of 4 and the number of variables plotted.
    cmap : str or Colormap, optional
        Colormap used for the trajectory lines. Defaults to the matplotlib color cycle.
    fill_color : str, optional
        Color of the credible band. Defaults to matplotlib's default fill color.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (12, 8).
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 100.

    Returns
    -------
    fig : Figure
        Figure containing one panel per variable.

    Examples
    --------
    Simulate the RBC example model and draw a 90 percent band around the mean path of output and consumption:

    .. code-block:: python

        from gEconpy import model_from_gcn, simulate
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_simulation

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        simulation = simulate(model, n_simulations=200, simulation_length=40, shock_std=0.01, verbose=False)

        fig = plot_simulation(simulation, vars_to_plot=["Y", "C"], ci=0.9)
    """
    if vars_to_plot is None:
        vars_to_plot = simulation.coords["variable"].values.tolist()
    for variable in vars_to_plot:
        if variable not in simulation.coords["variable"]:
            raise ValueError(f"{variable} not found among model variables.")

    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots)

    for variable, plot_loc in zip(vars_to_plot, plot_locs, strict=True):
        axis = fig.add_subplot(gs[plot_loc])

        _plot_single_variable(
            simulation.sel(variable=variable),
            ci=ci,
            ax=axis,
            fill_color=fill_color,
            cmap=cmap,
        )

        axis.set(title=variable)
        _style_panel(axis)

    fig.tight_layout()
    return fig


def plot_irf(
    irf: xr.DataArray | list[xr.DataArray] | dict[str, xr.DataArray],
    vars_to_plot: str | list[str] | None = None,
    shocks_to_plot: str | list[str] | None = None,
    n_cols: int | None = None,
    legend: bool = False,
    cmap: str | Colormap | None = None,
    legend_kwargs: dict | None = None,
    figsize: tuple[int, int] = (14, 10),
    dpi: int = 100,
) -> Figure:
    """
    Plot impulse response functions, one panel per variable and one line per shock.

    Parameters
    ----------
    irf : DataArray, list of DataArray, or dict mapping str to DataArray
        Output of :func:`~gEconpy.model.simulate.impulse_response_function`, with ``time`` and ``variable``
        dimensions and, when computed shock by shock, a ``shock`` dimension. Pass a list or a dictionary of such
        arrays to overlay several scenarios. Scenarios are distinguished by line style, and dictionary keys are used
        as the scenario names in the legend.
    vars_to_plot : str or list of str, optional
        Variables to plot. All variables are plotted by default.
    shocks_to_plot : str or list of str, optional
        Shocks to plot. All shocks are plotted by default.
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to the smaller of 4 and the number of variables plotted.
    legend : bool, optional
        If True, add a figure-level legend naming the shocks. Defaults to False.
    cmap : str or Colormap, optional
        Colormap used for the shock lines. Defaults to the matplotlib color cycle.
    legend_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.figure.Figure.legend`. Defaults to a legend centered above
        the panels with up to four columns.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (14, 10).
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 100.

    Returns
    -------
    fig : Figure
        Figure containing one panel per variable.

    Examples
    --------
    Plot the response of output, consumption, and investment to a technology shock in the RBC example model:

    .. code-block:: python

        from gEconpy import impulse_response_function, model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_irf

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        irf = impulse_response_function(model, simulation_length=40, shock_size=0.01, verbose=False)

        fig = plot_irf(irf, vars_to_plot=["Y", "C", "I"], legend=True)

    Overlay two calibrations of the same model by passing a dictionary keyed by scenario name:

    .. code-block:: python

        from gEconpy import impulse_response_function, model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_irf

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        baseline = impulse_response_function(model, simulation_length=40, shock_size=0.01, verbose=False)
        persistent = impulse_response_function(model, simulation_length=40, shock_size=0.01, verbose=False, rho_A=0.99)

        fig = plot_irf({"baseline": baseline, "rho_A = 0.99": persistent}, vars_to_plot=["Y", "C", "I"])
    """
    irf_map = _irf_to_mapping(irf)
    vars_to_plot_resolved, shocks_to_plot_resolved = _resolve_vars_and_shocks(irf_map, vars_to_plot, shocks_to_plot)

    fig, gs, plot_locs, show_xticks = _prepare_irf_grid(
        figsize=figsize, dpi=dpi, n_plots=len(vars_to_plot_resolved), n_cols=n_cols
    )
    scenario_names = list(irf_map.keys())

    for idx, (variable, plot_loc) in enumerate(zip(vars_to_plot_resolved, plot_locs, strict=True)):
        axis = fig.add_subplot(gs[plot_loc])
        _plot_irf_panel(
            axis=axis,
            irf_map=irf_map,
            variable=variable,
            shocks_to_plot=shocks_to_plot_resolved,
            cmap=cmap,
            add_scenario_legend=(idx == 0),
            scenario_names=scenario_names,
            show_xticks=show_xticks[idx],
        )

    _add_shocks_legend(fig, shocks_to_plot_resolved, legend, legend_kwargs)
    return fig


def plot_solvability(
    data: pd.DataFrame,
    params_to_plot: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 100,
) -> Figure:
    """
    Pair-plot parameter draws, colored by the stage at which solving the model failed.

    Diagonal panels show kernel density estimates of successful draws in blue and failed draws in red. Off-diagonal
    panels scatter the draws, colored by failure stage.

    Parameters
    ----------
    data : DataFrame
        Output of :func:`~gEconpy.model.statistics.perturbation_diagnostics.solvability_check` or
        :func:`~gEconpy.model.statistics.perturbation_diagnostics.prior_solvability_check`. Must contain a
        ``failure_step`` column.
    params_to_plot : list of str, optional
        Parameter columns to include. All non-constant numeric columns are plotted by default.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to a square of 4 inches per parameter, capped at 20 inches.
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 100.

    Returns
    -------
    fig : Figure
        Figure containing the pair-plot grid.

    Examples
    --------
    Draw parameters from the priors declared in the RBC example model, record where solving fails, and plot the
    result for two parameters:

    .. code-block:: python

        from gEconpy import model_from_gcn, prior_solvability_check
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_solvability

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        results = prior_solvability_check(model, n_samples=50, seed=0, progressbar=False)

        fig = plot_solvability(results, params_to_plot=["alpha", "beta"])
    """
    plot_data, failure_step, params = _solv_prepare_data(data)
    params_use = _validate_params_to_plot(params_to_plot, params)
    n = len(params_use)

    if figsize is None:
        side = max(4.0, min(4.0 * n, 20.0))
        figsize = (side, side)

    fig, axes = plt.subplots(n, n, figsize=figsize, dpi=dpi, squeeze=False)

    for row, col in product(range(n), range(n)):
        ax = axes[row, col]
        if col > row:
            ax.set_visible(False)
            continue

        y_name = params_use[row]
        x_name = params_use[col]

        if row == col:
            _solv_plot_diagonal(ax, plot_data[x_name], plot_data["success"])
        else:
            _solv_plot_offdiag(ax, plot_data, failure_step, x_name, y_name)

    _solv_format_axes(axes, params_use)

    # Only off-diagonal panels carry scatter labels, so the legend is read from the first one below the diagonal.
    legend_ax = axes[1, 0] if n > 1 else axes[0, 0]
    handles, labels = legend_ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=min(len(handles), 4),
            fontsize=8,
            frameon=False,
        )

    fig.suptitle("Solvability by Parameter Values", y=0.97)
    return fig


def plot_solvability_summary(data: pd.DataFrame, figsize: tuple[float, float] = (8, 1.5), dpi: int = 144) -> Figure:
    """
    Draw a stacked horizontal bar showing the share of draws that succeeded or failed at each solving stage.

    Parameters
    ----------
    data : DataFrame
        Output of :func:`~gEconpy.model.statistics.perturbation_diagnostics.solvability_check` or
        :func:`~gEconpy.model.statistics.perturbation_diagnostics.prior_solvability_check`. Must contain a
        ``failure_step`` column.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to (8, 1.5).
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 144.

    Returns
    -------
    fig : Figure
        Figure containing the single bar.

    Examples
    --------
    Summarize how often prior draws from the RBC example model fail to solve:

    .. code-block:: python

        from gEconpy import model_from_gcn, prior_solvability_check
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_solvability_summary

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        results = prior_solvability_check(model, n_samples=50, seed=0, progressbar=False)

        fig = plot_solvability_summary(results)
    """
    counts = data["failure_step"].fillna("success").value_counts(normalize=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    left = 0.0
    for label in _SOLVABILITY_STAGES:
        frac = counts.get(label, 0.0)
        if frac == 0:
            continue
        color = _SOLVABILITY_COLORS.get(label, "tab:gray")
        ax.barh(0, frac, left=left, color=color, label=label.replace("_", " ").title(), height=0.6)
        left += frac

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(len(counts), 6), fontsize=8, frameon=False)
    ax.set_title("Solvability Summary")
    fig.tight_layout()
    return fig


def plot_eigenvalues(
    model: Model,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    C: np.ndarray | None = None,
    D: np.ndarray | None = None,
    linearize_model_kwargs: dict | None = None,
    fig: Figure | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    plot_circle: bool = True,
    **parameter_updates,
) -> Figure:
    """
    Scatter the generalized eigenvalues of the linearized model on the complex plane.

    Eigenvalues with modulus greater than 1 are drawn in red and the rest in blue. Eigenvalues with modulus greater
    than 10 are treated as infinite and left out, and the title reports how many were dropped.

    Parameters
    ----------
    model : Model
        Model to linearize.
    A : ndarray, optional
        Jacobian of the model equations with respect to variables at ``t-1``. When given, ``B``, ``C``, and ``D``
        must be given too. By default the model is linearized here.
    B : ndarray, optional
        Jacobian with respect to variables at ``t``. See ``A``.
    C : ndarray, optional
        Jacobian with respect to variables at ``t+1``. See ``A``.
    D : ndarray, optional
        Jacobian with respect to the exogenous shocks. See ``A``.
    linearize_model_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`~gEconpy.model.model.Model.linearize_model`. Ignored when the
        Jacobians are given. Empty by default.
    fig : Figure, optional
        Figure to draw on. Its first axis is used. A new figure is created by default.
    figsize : tuple of float, optional
        Size of the created figure in inches. Defaults to (5, 5).
    dpi : int, optional
        Resolution of the created figure in dots per inch. Defaults to 100.
    plot_circle : bool, optional
        If True, draw the unit circle. Defaults to True.
    **parameter_updates
        Parameter values at which to linearize the model. Ignored when the Jacobians are given.

    Returns
    -------
    fig : Figure
        Figure containing the eigenvalue scatter.

    Examples
    --------
    Plot the eigenvalues of the RBC example model at its calibrated parameter values:

    .. code-block:: python

        from gEconpy import model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_eigenvalues

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        fig = plot_eigenvalues(model, linearize_model_kwargs={"verbose": False})

    Linearize at a different value of one parameter by passing it as a keyword argument:

    .. code-block:: python

        from gEconpy import model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_eigenvalues

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        fig = plot_eigenvalues(model, linearize_model_kwargs={"verbose": False}, beta=0.97)
    """
    if figsize is None:
        figsize = (5, 5)
    if dpi is None:
        dpi = 100

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        ax = fig.axes[0]

    linearize_model_kwargs = {**(linearize_model_kwargs or {}), **parameter_updates}

    eigenvalues = cast(
        pd.DataFrame,
        check_bk_condition(
            model,
            A=A,
            B=B,
            C=C,
            D=D,
            return_value="dataframe",
            **linearize_model_kwargs,
        ),
    )

    n_infinity = (eigenvalues["Modulus"] > _EIGENVALUE_PLOT_MODULUS_CUTOFF).sum()
    eigenvalues = eigenvalues[eigenvalues.Modulus < _EIGENVALUE_PLOT_MODULUS_CUTOFF]

    if plot_circle:
        _draw_unit_circle(ax)

    ax.set_aspect("equal")
    colors = ["tab:red" if x > 1.0 else "tab:blue" for x in eigenvalues.Modulus]
    ax.scatter(eigenvalues.Real, eigenvalues.Imaginary, color=colors, s=50, lw=1, edgecolor="k")
    _style_panel(ax)
    ax.set_title(f"Eigenvalues of Model Solution\n{n_infinity} Eigenvalues with Infinity Modulus not shown.")
    return fig


def plot_eigenvalue_sensitivity(
    model: Model,
    sensitivity_data: xr.Dataset | None = None,
    params_to_plot: list[str] | None = None,
    perturbation: float = 0.01,
    filter_zeros: bool = True,
    filter_infinite: bool = True,
    zero_tol: float = 1e-6,
    inf_tol: float | None = None,
    min_arrow_frac: float = 0.10,
    n_cols: int | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    plot_circle: bool = True,
    **eigenvalue_sensitivity_kwargs,
) -> Figure:
    """
    Plot how each eigenvalue of the linearized model moves when a parameter is increased.

    Each panel shows the eigenvalues on the complex plane for one parameter, with an arrow from each eigenvalue to
    where it moves when the parameter is increased by the fraction ``perturbation``.

    Parameters
    ----------
    model : Model
        Model whose eigenvalues are plotted.
    sensitivity_data : Dataset, optional
        Output of :func:`~gEconpy.model.statistics.perturbation_diagnostics.eigenvalue_sensitivity`. By default it is
        computed here from ``model`` and ``eigenvalue_sensitivity_kwargs``.
    params_to_plot : list of str, optional
        Parameters to draw a panel for. All parameters are plotted by default.
    perturbation : float, optional
        Fractional increase in the parameter value that the arrows show. Defaults to 0.01, a 1 percent increase.
    filter_zeros : bool, optional
        If True, leave out eigenvalues with modulus below ``zero_tol``. Defaults to True.
    filter_infinite : bool, optional
        If True, leave out eigenvalues with modulus above ``inf_tol``. Defaults to True.
    zero_tol : float, optional
        Modulus below which an eigenvalue counts as zero. Defaults to 1e-6.
    inf_tol : float, optional
        Modulus above which an eigenvalue counts as infinite. Defaults to 10 times the largest modulus below 1e6,
        with a floor of 10, or to 1e6 when every modulus is at least 1e6.
    min_arrow_frac : float, optional
        Smallest gradient magnitude, as a fraction of the largest gradient magnitude, for which an arrow is drawn.
        Defaults to 0.10.
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to the smaller of 4 and the number of parameters plotted.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to 4 inches per panel in each direction.
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 144.
    plot_circle : bool, optional
        If True, draw the unit circle on each panel. Defaults to True.
    **eigenvalue_sensitivity_kwargs
        Keyword arguments forwarded to
        :func:`~gEconpy.model.statistics.perturbation_diagnostics.eigenvalue_sensitivity` when ``sensitivity_data``
        is not given, including parameter values to linearize at. Parameter values are also used to label the
        panels.

    Returns
    -------
    fig : Figure
        Figure containing one panel per parameter.

    Examples
    --------
    Show how the eigenvalues of the RBC example model respond to a 5 percent increase in the discount factor and the
    capital share:

    .. code-block:: python

        from gEconpy import model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_eigenvalue_sensitivity

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        fig = plot_eigenvalue_sensitivity(model, params_to_plot=["beta", "alpha"], perturbation=0.05, verbose=False)
    """
    dpi = dpi or 144

    if sensitivity_data is None:
        sensitivity_data = eigenvalue_sensitivity(model, **eigenvalue_sensitivity_kwargs)

    filtered_data, n_filtered = _filter_eigenvalues(sensitivity_data, filter_zeros, filter_infinite, zero_tol, inf_tol)
    re_plot = filtered_data.eigenvalues.sel(component="real").values
    im_plot = filtered_data.eigenvalues.sel(component="imaginary").values
    mod_plot = filtered_data.eigenvalues.sel(component="modulus").values

    all_params = list(sensitivity_data.coords["parameter"].values)
    params_to_plot = _validate_params_to_plot(params_to_plot, all_params)

    n_params = len(params_to_plot)
    if n_params == 0:
        raise ValueError("No parameters to plot. Pass at least one parameter name in params_to_plot.")

    n_cols = n_cols or min(4, n_params)
    figsize = figsize or (4 * n_cols, 4 * math.ceil(n_params / n_cols))

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs, plot_locs = prepare_gridspec_figure(n_cols, n_params, figure=fig)

    xlim, ylim = _compute_axis_limits(re_plot, im_plot)

    grad_re = filtered_data.gradients.sel(part="real").values
    grad_im = filtered_data.gradients.sel(part="imaginary").values

    model_param_names = {param.name for param in model.params}
    param_updates = {k: v for k, v in eigenvalue_sensitivity_kwargs.items() if k in model_param_names}
    param_dict = model.parameters(**param_updates)

    for idx, (param, plot_loc) in enumerate(zip(params_to_plot, plot_locs, strict=True)):
        ax = fig.add_subplot(gs[plot_loc])
        param_idx = all_params.index(param)
        param_value = float(param_dict.get(param, 1.0))

        _draw_eigenvalue_panel(
            ax=ax,
            re_plot=re_plot,
            im_plot=im_plot,
            mod_plot=mod_plot,
            d_re=grad_re[:, param_idx],
            d_im=grad_im[:, param_idx],
            param_name=param,
            param_value=param_value,
            perturbation=perturbation,
            xlim=xlim,
            ylim=ylim,
            plot_circle=plot_circle,
            show_legend=(idx == 0),
            min_arrow_frac=min_arrow_frac,
        )

    title_parts = [f"Eigenvalue Sensitivity ({perturbation:.1%} perturbation)"]
    if n_filtered > 0:
        title_parts.append(f"{n_filtered} zero/infinite eigenvalues not shown")
    fig.suptitle("\n".join(title_parts))

    return fig


def plot_covariance_matrix(
    data: pd.DataFrame,
    vars_to_plot: list[str] | None = None,
    cbarlabel: str = "Covariance",
    figsize: tuple[float, float] = (4, 4),
    dpi: int = 100,
    cbar_kw: dict | None = None,
    cmap: str = "YlGn",
    heatmap_kwargs: dict | None = None,
    annotation_kwargs: dict | None = None,
) -> Figure:
    """
    Draw an annotated heatmap of a covariance matrix.

    Parameters
    ----------
    data : DataFrame
        Square covariance matrix whose index and columns hold the same variable names.
    vars_to_plot : list of str, optional
        Variables to include. All variables are plotted by default.
    cbarlabel : str, optional
        Label of the colorbar. Defaults to "Covariance".
    figsize : tuple of float, optional
        Figure size in inches. Defaults to (4, 4).
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 100.
    cbar_kw : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.figure.Figure.colorbar`. Defaults to ``{"shrink": 0.5}``.
    cmap : str, optional
        Colormap of the heatmap. Defaults to "YlGn".
    heatmap_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.imshow`. Empty by default.
    annotation_kwargs : dict, optional
        Keyword arguments forwarded to :func:`annotate_heatmap`. Empty by default.

    Returns
    -------
    fig : Figure
        Figure containing the heatmap.

    Examples
    --------
    Plot the stationary covariance of output, consumption, and investment in the RBC example model:

    .. code-block:: python

        from gEconpy import model_from_gcn, stationary_covariance_matrix
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_covariance_matrix

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        covariance = stationary_covariance_matrix(model, shock_std=0.01, verbose=False)

        fig = plot_covariance_matrix(covariance, vars_to_plot=["Y", "C", "I"])
    """
    if vars_to_plot is None:
        vars_to_plot = data.columns

    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    if annotation_kwargs is None:
        annotation_kwargs = {}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im, _cbar = plot_heatmap(
        data.loc[vars_to_plot, vars_to_plot],
        ax=ax,
        cbar_kw=cbar_kw,
        cmap=cmap,
        cbarlabel=cbarlabel,
        **heatmap_kwargs,
    )
    annotate_heatmap(im, valfmt="{x:.2f}", **annotation_kwargs)

    fig.tight_layout()
    return fig


def plot_heatmap(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    cbar_kw: dict | None = None,
    cbarlabel: str | None = "",
    **kwargs,
):
    """
    Draw a labeled heatmap of a DataFrame with a colorbar.

    Parameters
    ----------
    data : DataFrame
        Data to plot. Row and column labels are used as tick labels.
    ax : matplotlib Axes, optional
        Axis to draw on. Defaults to the current axis.
    cbar_kw : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.figure.Figure.colorbar`. Defaults to ``{"shrink": 0.5}``.
    cbarlabel : str, optional
        Label of the colorbar. Defaults to an empty string.
    **kwargs
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    im : AxesImage
        The image created by ``imshow``.
    cbar : Colorbar
        The colorbar attached to the image.
    """
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {"shrink": 0.5}

    im = ax.imshow(data, **kwargs)

    n_rows, n_columns = data.shape

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(n_columns),
        xticklabels=data.columns,
        yticks=np.arange(n_rows),
        yticklabels=data.index,
    )
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_visible(False)

    # Minor ticks between the cells carry a white grid that separates the cells.
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data: np.ndarray | None = None,
    valfmt: str | Formatter = "{x:.2f}",
    textcolors: tuple[str, str] = ("black", "white"),
    threshold: float | None = None,
    **textkw,
) -> list[Text]:
    """
    Write the value of each cell of a heatmap on top of it.

    Parameters
    ----------
    im : AxesImage
        The heatmap image to annotate.
    data : array-like, optional
        Values to write, with the same shape as the image. Defaults to the image's own data.
    valfmt : str or matplotlib Formatter, optional
        Format of the annotations. A string is used as a ``str.format`` template with the value bound to ``x``, for
        example "$ {x:.2f}". Defaults to "{x:.2f}".
    textcolors : tuple of str, optional
        Text colors for values below and above ``threshold``. Defaults to ("black", "white").
    threshold : float, optional
        Value in data units that separates the two text colors. Defaults to the middle of the colormap.
    **textkw
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.text` for each label.

    Returns
    -------
    texts : list of matplotlib Text
        The labels added to the heatmap, in row-major order.
    """
    data = im.get_array() if data is None else np.asarray(data)

    threshold = im.norm(threshold) if threshold is not None else im.norm(data.max()) / 2.0

    text_kwargs = {"horizontalalignment": "center", "verticalalignment": "center", **textkw}

    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    texts = []
    for i, j in product(range(data.shape[0]), range(data.shape[1])):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        texts.append(im.axes.text(j, i, valfmt(data[i, j], None), **{**text_kwargs, "color": color}))

    return texts


def plot_acf(
    acorr: xr.DataArray | Mapping[str, xr.DataArray],
    vars_to_plot: list[str] | None = None,
    sample_dims: tuple[str, ...] = ("chain", "draw"),
    ci_probs: tuple[float, float] = (0.5, 0.94),
    reference: xr.DataArray | None = None,
    dodge: float = 0.2,
    figsize: tuple[int, int] | None = (14, 4),
    dpi: int | None = 100,
    n_cols: int | None = 4,
    mean_kwargs: dict | None = None,
    inner_hdi_kwargs: dict | None = None,
    outer_hdi_kwargs: dict | None = None,
    stem_kwargs: dict | None = None,
    reference_kwargs: dict | None = None,
) -> Figure:
    """
    Plot the autocorrelation function of each variable, one panel per variable.

    ``acorr`` is an autocorrelation tensor with a ``lag`` dimension and two variable dimensions, for example
    ``variable`` and ``variable_aux``. Only the diagonal, each variable's own autocorrelation, is drawn. When the
    tensor also carries posterior sample dimensions, each lag shows the posterior mean with two nested
    credible-interval sticks. Otherwise each lag is drawn as a single stem.

    Pass a mapping of label to tensor to overlay several models on the same axes. Their sticks are dodged along the
    lag axis, colored per model, and a legend of the labels is added.

    Parameters
    ----------
    acorr : DataArray or mapping of str to DataArray
        Autocorrelation tensor, or one tensor per model to overlay. Each tensor must have a ``lag`` dimension and
        exactly two variable dimensions, and may also carry the ``sample_dims``.
    vars_to_plot : list of str, optional
        Variables to plot. All variables in ``acorr``, or in its first entry when it is a mapping, are plotted by
        default.
    sample_dims : tuple of str, optional
        Dimensions treated as posterior samples. Their presence switches from stems to credible intervals.
        Defaults to ``("chain", "draw")``.
    ci_probs : tuple of float, optional
        Probabilities of the inner and outer credible intervals drawn at each lag. The inner interval is drawn with
        the thicker line. Ignored for tensors without sample dimensions. Defaults to ``(0.5, 0.94)``.
    reference : DataArray, optional
        A second autocorrelation to overlay as hollow markers, typically the empirical autocorrelation of observed
        data. Indexed by ``lag`` and one variable dimension. Only variables that also appear in ``acorr`` are drawn.
        No reference is drawn by default.
    dodge : float, optional
        Horizontal offset between the sticks of successive models when ``acorr`` is a mapping. Defaults to 0.2.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to (14, 4).
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 100.
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to 4, or the number of panels if that is smaller. None is
        treated as the default.
    mean_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.scatter` for the posterior-mean point, merged
        over the defaults ``{"s": 38, "zorder": 3}``. The per-model color is set automatically. Empty by default.
    inner_hdi_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.vlines` for the inner credible-interval stick,
        merged over the default ``{"lw": 3.0}``. Empty by default.
    outer_hdi_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.vlines` for the outer credible-interval stick,
        merged over the default ``{"lw": 1.0}``. Empty by default.
    stem_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.scatter` for the stem marker drawn when a tensor
        has no sample dimensions. Empty by default.
    reference_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.scatter` for the hollow reference markers,
        merged over the defaults ``{"facecolors": "none", "edgecolors": "k", "s": 60, "linewidths": 1.6,
        "zorder": 4}``. The legend entry for the reference series uses the same settings. Empty by default.

    Returns
    -------
    fig : Figure
        Figure containing one panel per variable.

    Examples
    --------
    Plot the model-implied autocorrelation of output, consumption, and investment in the RBC example model:

    .. code-block:: python

        from gEconpy import autocorrelation_matrix, model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_acf

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        acorr = autocorrelation_matrix(model, shock_std=0.01, n_lags=10, verbose=False)

        fig = plot_acf(acorr, vars_to_plot=["Y", "C", "I"], n_cols=3)

    Overlay two calibrations and compare both against a reference series, here the autocorrelation of an AR(1)
    process with coefficient 0.9:

    .. code-block:: python

        import numpy as np
        import xarray as xr

        from gEconpy import autocorrelation_matrix, model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_acf

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        baseline = autocorrelation_matrix(model, shock_std=0.01, n_lags=10, verbose=False)
        persistent = autocorrelation_matrix(model, shock_std=0.01, n_lags=10, verbose=False, rho_A=0.99)

        lags = np.arange(11)
        reference = xr.DataArray(
            np.stack([0.9**lags] * 3, axis=1),
            dims=["lag", "variable"],
            coords={"lag": lags, "variable": ["Y", "C", "I"]},
        )

        fig = plot_acf(
            {"baseline": baseline, "rho_A = 0.99": persistent},
            vars_to_plot=["Y", "C", "I"],
            reference=reference,
            n_cols=3,
        )
    """
    tensors = {None: acorr} if isinstance(acorr, xr.DataArray) else dict(acorr)
    models, var_dim = _collect_acf_diagonals(tensors, sample_dims, dodge)

    diagonal = next(iter(models.values())).diagonal
    all_variables = diagonal.coords[var_dim].values
    if vars_to_plot is None:
        vars_to_plot = all_variables
    else:
        missing = [var for var in vars_to_plot if var not in all_variables]
        if missing:
            raise ValueError(f"Cannot plot {missing}: not found in the provided autocorrelation tensor")

    n_plots = len(vars_to_plot)
    n_cols = min(4 if n_cols is None else n_cols, n_plots)
    fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
    gs, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_plots, figure=fig)
    lags = diagonal.coords["lag"].values

    ref_var_dim = next((dim for dim in reference.dims if dim != "lag"), None) if reference is not None else None

    for variable, plot_loc in zip(vars_to_plot, plot_locs, strict=True):
        axis = fig.add_subplot(gs[plot_loc])
        _draw_acf_panel(
            axis,
            variable,
            models=models,
            var_dim=var_dim,
            lags=lags,
            ci_probs=ci_probs,
            reference=reference,
            ref_var_dim=ref_var_dim,
            mean_kwargs=mean_kwargs,
            inner_hdi_kwargs=inner_hdi_kwargs,
            outer_hdi_kwargs=outer_hdi_kwargs,
            stem_kwargs=stem_kwargs,
            reference_kwargs=reference_kwargs,
        )

    _add_acf_legend(fig, models, reference is not None, reference_kwargs)
    return fig


def plot_corner(
    idata: Any,
    group: str = "posterior",
    var_names: list[str] | None = None,
    colorby: str | None = None,
    figure_kwargs: dict | None = None,
    hist_bins: int = 100,
    rug_bins: int = 20,
    rug_levels: int = 6,
    fontsize: int = 6,
    show_marginal_modes: bool = True,
    scatter_kwargs: dict | None = None,
) -> Figure:
    """
    Draw a corner plot of the joint distribution of a set of sampled variables.

    Each off-diagonal panel shows the two-dimensional density of a pair of variables, marginalizing over the rest.
    Diagonal panels show the one-dimensional histogram of each variable.

    Parameters
    ----------
    idata : arviz InferenceData or DataTree
        Samples with a group named ``group``.
    group : str, optional
        Group of ``idata`` to plot, for example "prior" or "posterior". Defaults to "posterior".
    var_names : list of str, optional
        Variables to plot. All variables in the group are plotted by default.
    colorby : str, optional
        Variable in the group whose value colors a scatter of the draws drawn over each off-diagonal panel. No
        scatter is drawn by default.
    figure_kwargs : dict, optional
        Keyword arguments forwarded to :func:`matplotlib.pyplot.subplots`. Empty by default.
    hist_bins : int, optional
        Number of histogram bins on the diagonal panels. Defaults to 100.
    rug_bins : int, optional
        Number of bins per axis of the two-dimensional histograms on the off-diagonal panels. Defaults to 20.
    rug_levels : int, optional
        Number of contour levels on the off-diagonal panels. Defaults to 6.
    fontsize : int, optional
        Font size of the axis labels and ticks. Defaults to 6.
    show_marginal_modes : bool, optional
        If True, mark the mode of each two-dimensional histogram with dashed lines. Defaults to True.
    scatter_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.scatter` for the colored draws. Ignored unless
        ``colorby`` is given. Defaults to ``{"zorder": 100, "cmap": "viridis", "s": 10, "alpha": 0.5}``.

    Returns
    -------
    fig : Figure
        Figure containing the panel grid.

    Examples
    --------
    Draw the joint prior of the RBC example model's parameters from a PyMC model built from the priors declared in
    the GCN file:

    .. code-block:: python

        import pymc as pm

        from gEconpy import statespace_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_corner

        ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
        with pm.Model(coords=ss_mod.coords):
            ss_mod.to_pymc()
            prior = pm.sample_prior_predictive(500, random_seed=0)

        fig = plot_corner(prior, group="prior", var_names=["alpha", "beta", "delta"], colorby="rho_A")
    """
    var_names, color_data, resolved_scatter_kwargs = _validate_and_prepare_corner_inputs(
        idata=idata,
        group=group,
        var_names=var_names,
        colorby=colorby,
        scatter_kwargs=scatter_kwargs,
    )
    k_params = len(var_names)

    fig, axes = plt.subplots(k_params, k_params, **(figure_kwargs or {}))
    flat_data = {name: idata[group][name].values.ravel() for name in var_names}

    for row, col in product(range(k_params), range(k_params)):
        ax = axes[row, col]
        if col > row:
            ax.set_visible(False)
            continue

        _format_axis_for_corner(ax, fontsize)
        x_name, y_name = var_names[col], var_names[row]

        if col == row:
            _plot_diagonal_hist(
                ax=ax,
                data=flat_data[x_name],
                bins=hist_bins,
                fontsize=fontsize,
                is_last_row=(row == k_params - 1),
            )
            continue

        _plot_offdiag_panel(
            ax=ax,
            x_name=x_name,
            y_name=y_name,
            x_data=flat_data[x_name],
            y_data=flat_data[y_name],
            rug_bins=rug_bins,
            rug_levels=rug_levels,
            show_marginal_modes=show_marginal_modes,
            fontsize=fontsize,
            draw_xlabel=(row == k_params - 1),
            draw_ylabel=(col == 0),
            color_data=color_data,
            scatter_kwargs=resolved_scatter_kwargs,
        )

    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(h_pad=0.0, w_pad=0.0, wspace=0.05, hspace=0.05)

    return fig


def plot_kalman_filter(
    idata: xr.DataTree,
    data: pd.DataFrame,
    kalman_output: Literal["predicted", "filtered", "smoothed"] = "predicted",
    group: Literal["prior", "posterior"] = "posterior",
    n_cols: int | None = None,
    vars_to_plot: list[str] | None = None,
    fig: Figure | None = None,
    figsize: tuple[int, int] = (14, 6),
    dpi: int = 144,
    observed: bool = False,
) -> Figure:
    """
    Plot the mean and 95 percent credible band of the Kalman filter output for each state.

    Parameters
    ----------
    idata : DataTree
        Conditional prior or posterior samples, as returned by
        :meth:`~gEconpy.model.statespace.DSGEStateSpace.sample_conditional_prior` or
        :meth:`~gEconpy.model.statespace.DSGEStateSpace.sample_conditional_posterior`.
    data : DataFrame
        Observed data. Columns that match a plotted state are drawn as a dashed black line.
    kalman_output : {"predicted", "filtered", "smoothed"}, optional
        Which Kalman filter output to plot. Defaults to "predicted".
    group : {"prior", "posterior"}, optional
        Which group of ``idata`` holds the samples. Defaults to "posterior".
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to the smaller of 4 and the number of states plotted.
    vars_to_plot : list of str, optional
        States to plot. All states are plotted by default.
    fig : Figure, optional
        Figure to draw on. A new figure is created by default.
    figsize : tuple of int, optional
        Size of the created figure in inches. Defaults to (14, 6).
    dpi : int, optional
        Resolution of the created figure in dots per inch. Defaults to 144.
    observed : bool, optional
        If True, plot the observed states from the ``*_observed`` variables of ``idata``. If False, plot the
        latent states. Defaults to False.

    Returns
    -------
    fig : Figure
        Figure containing one panel per state.

    Examples
    --------
    Generate data from the prior of the RBC example model, run the Kalman filter over it under the prior, and plot
    the smoothed latent states against the generating data:

    .. code-block:: python

        import pymc as pm

        from gEconpy import data_from_prior, statespace_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_kalman_filter

        ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
        ss_mod.configure(observed_states=["Y"], verbose=False)

        with pm.Model(coords=ss_mod.coords) as pm_mod:
            ss_mod.to_pymc()
            pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)

        true_params, data, prior_idata = data_from_prior(ss_mod, pm_mod, n_samples=50, random_seed=0)

        with pm_mod:
            ss_mod.build_statespace_graph(data)
        conditional_prior = ss_mod.sample_conditional_prior(prior_idata, progressbar=False)

        fig = plot_kalman_filter(
            conditional_prior, data, kalman_output="smoothed", group="prior", vars_to_plot=["Y", "C", "K"]
        )
    """
    if kalman_output.lower() not in ["filtered", "predicted", "smoothed"]:
        raise ValueError(f'kalman_output must be one of "filtered", "predicted", "smoothed". Found {kalman_output}.')

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")

    state_name = "observed_state" if observed else "state"
    output_name = f"{kalman_output}_{group}_observed" if observed else f"{kalman_output}_{group}"
    if vars_to_plot is None:
        vars_to_plot = idata.coords[state_name].values

    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots, figure=fig)
    time_idx = idata.coords["time"]

    output = idata[output_name].sel({state_name: vars_to_plot})
    means = output.mean(dim=["chain", "draw"])
    hdis = azs.hdi(output, prob=0.95, skipna=True)

    for variable, plot_loc in zip(vars_to_plot, plot_locs, strict=True):
        axis = fig.add_subplot(gs[plot_loc])

        axis.plot(time_idx, means.sel({state_name: variable}), color="tab:red")
        axis.fill_between(
            time_idx,
            *hdis.sel({state_name: variable}).values.T,
            color="tab:blue",
            alpha=0.5,
        )

        if variable in data.columns:
            axis.plot(time_idx, data[variable].values, color="k", ls="--")

        axis.set(title=variable, xlabel=None, ylabel="% Deviation from SS")
        axis.tick_params(axis="x", rotation=45)
        _style_panel(axis)

    return fig


def plot_priors(
    model: Model | DSGEStateSpace,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int = 144,
    n_cols: int = 6,
    mark_initial_value: bool = True,
) -> Figure:
    """
    Plot the prior density of each model parameter and shock hyperparameter.

    Parameters
    ----------
    model : Model or DSGEStateSpace
        Model whose priors are plotted.
    var_names : list of str, optional
        Parameters to plot. All parameters with priors are plotted by default.
    figsize : tuple of int, optional
        Figure size in inches. Defaults to 14 inches wide and 2 inches per row of panels.
    dpi : int, optional
        Figure resolution in dots per inch. Defaults to 144.
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to 6.
    mark_initial_value : bool, optional
        If True, draw a dashed vertical line at each parameter's calibrated value. Defaults to True.

    Returns
    -------
    fig : Figure
        Figure containing one panel per parameter.

    Examples
    --------
    Plot the priors declared in the RBC example model's GCN file:

    .. code-block:: python

        from gEconpy import model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_priors

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        fig = plot_priors(model, n_cols=3)
    """
    priors = dict(model.param_priors) | {
        shock.param_name_to_hyper_name[name]: hyper_prior
        for shock in model.shock_priors.values()
        for name, hyper_prior in shock.hyper_param_dict.items()
    }

    if var_names is not None:
        priors = {name: priors[name] for name in var_names}
    n_params = len(priors)

    if figsize is None:
        n_rows = math.ceil(n_params / n_cols)
        figsize = (14, 2 * n_rows)

    fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
    gs, locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_params, figure=fig)

    all_params = (
        (model.param_dict | model.hyper_param_dict) if isinstance(model, DSGEStateSpace) else model.parameters()
    )

    for (name, prior), loc in zip(priors.items(), locs, strict=True):
        axis = fig.add_subplot(gs[loc])
        with warnings.catch_warnings(action="ignore"):
            prior.plot_pdf(
                ax=axis,
                interval="hdi",
                legend="title",
                pointinterval=True,
                levels=[0.025, 0.975],
            )

        dist_text = axis.get_title()
        axis.set_title(f"{name}\n{dist_text}")
        value = all_params.get(name)

        if mark_initial_value and value is not None:
            axis.axvline(value, ls="--", c="k")

    return fig


def plot_posterior_with_prior(
    idata: Any,
    var_names: list[str],
    prior_dict: dict,
    true_values: xr.Dataset | None = None,
    n_cols: int = 5,
    fig_kwargs: dict | None = None,
    plot_posterior_kwargs: dict | None = None,
) -> Figure:
    """
    Plot the marginal posterior of each variable with its prior density overlaid.

    Parameters
    ----------
    idata : arviz InferenceData or DataTree
        Samples with a posterior group.
    var_names : list of str
        Variables to plot.
    prior_dict : dict mapping str to preliz distribution
        Prior of each variable, keyed by variable name. Variables absent from the dictionary are drawn without a
        prior.
    true_values : Dataset, optional
        Reference values to mark with a dashed vertical line, keyed by variable name. No lines are drawn by default.
    n_cols : int, optional
        Number of columns in the panel grid. Defaults to 5.
    fig_kwargs : dict, optional
        Keyword arguments used to create the figure. Defaults to 14 inches wide, 3 inches per row of panels, at
        144 dpi.
    plot_posterior_kwargs : dict, optional
        Keyword arguments forwarded to :func:`arviz_plots.plot_dist`. Empty by default.

    Returns
    -------
    fig : Figure
        Figure containing one panel per variable.

    Examples
    --------
    Compare posterior samples against their priors. The posterior here is random, since no estimation is run:

    .. code-block:: python

        import arviz_base as azb
        import numpy as np
        import preliz as pz
        import xarray as xr

        from gEconpy.plotting import plot_posterior_with_prior

        rng = np.random.default_rng(0)
        priors = {"alpha": pz.Beta(2.0, 5.0), "rho_A": pz.Beta(10.0, 2.0)}
        posterior = {
            "alpha": rng.beta(20.0, 40.0, size=(2, 500)),
            "rho_A": rng.beta(80.0, 10.0, size=(2, 500)),
        }
        idata = azb.from_dict({"posterior": posterior})
        true_values = xr.Dataset({"alpha": 0.35, "rho_A": 0.9})

        fig = plot_posterior_with_prior(idata, var_names=["alpha", "rho_A"], prior_dict=priors, true_values=true_values)
    """
    var_names = list(var_names)

    if fig_kwargs is None:
        n_rows = max(1, math.ceil(len(var_names) / n_cols))
        fig_kwargs = {"figsize": (14, n_rows * 3), "dpi": 144}
    if plot_posterior_kwargs is None:
        plot_posterior_kwargs = {}

    plot_collection = plot_dist(
        idata,
        var_names=var_names,
        kind="kde",
        ci_kind="hdi",
        ci_prob=0.94,
        point_estimate="mean",
        backend="matplotlib",
        figure_kwargs=fig_kwargs,
        cols=["__variable__"],
        col_wrap=n_cols,
        **plot_posterior_kwargs,
    )

    fig = plot_collection.viz["figure"].item()
    axes_ds = plot_collection.viz["plot"]

    for var_name in var_names:
        if var_name not in axes_ds:
            continue
        axis = axes_ds[var_name].item()
        if var_name in prior_dict:
            prior_dict[var_name].plot_pdf(ax=axis, legend=False, color="tab:orange")
        if true_values is not None and var_name in true_values:
            for true_value in np.ravel(true_values[var_name].values):
                axis.axvline(true_value, ls="--", c="k", lw=1.0)

    return fig


def plot_estimated_matrix(
    idata: Any,
    dsge_mod: DSGEStateSpace,
    matrix_name: str = "state_chol_corr",
    subplot_kwargs: dict | None = None,
    symmetrical: bool = True,
) -> Figure:
    """
    Plot the posterior mean and credible interval of each entry of an estimated shock matrix.

    Parameters
    ----------
    idata : arviz InferenceData or DataTree
        Samples with a posterior group holding ``matrix_name``.
    dsge_mod : DSGEStateSpace
        Model the matrix was estimated for. Its shock names label the panels and its number of shocks sets the grid
        size.
    matrix_name : str, optional
        Posterior variable to plot. Defaults to "state_chol_corr".
    subplot_kwargs : dict, optional
        Keyword arguments forwarded to :func:`matplotlib.pyplot.subplots`. Empty by default.
    symmetrical : bool, optional
        If True, hide the upper triangle and the diagonal. Defaults to True.

    Returns
    -------
    fig : Figure
        Figure containing one panel per matrix entry.

    Examples
    --------
    Plot the estimated shock correlation matrix of a model configured with a full shock covariance. The posterior
    here is random, since no estimation is run:

    .. code-block:: python

        import arviz_base as azb
        import numpy as np

        from gEconpy import statespace_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.plotting import plot_estimated_matrix

        ss_mod = statespace_from_gcn(get_example_gcn("New_Keynesian"), verbose=False)
        ss_mod.configure(observed_states=["Y", "pi", "r"], full_shock_covariance=True, verbose=False)

        rng = np.random.default_rng(0)
        n_shocks = ss_mod.k_posdef
        posterior = {"state_chol_corr": rng.uniform(-1.0, 1.0, size=(2, 500, n_shocks, n_shocks))}
        idata = azb.from_dict({"posterior": posterior}, dims={"state_chol_corr": ["shock", "shock_aux"]})

        fig = plot_estimated_matrix(idata, ss_mod, subplot_kwargs={"figsize": (8, 8)})
    """
    n_shocks = dsge_mod.k_posdef
    subplot_kwargs = subplot_kwargs or {}

    fig, axes = plt.subplots(n_shocks, n_shocks, squeeze=False, **subplot_kwargs)

    mu = idata.posterior[matrix_name].mean(dim=["chain", "draw"])
    hdi = azs.hdi(idata.posterior[matrix_name])
    axes[0, 0].set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))

    for i, j in product(range(n_shocks), range(n_shocks)):
        axis = axes[i, j]
        if i <= j and symmetrical:
            axis.set_visible(False)
            continue

        y_var = dsge_mod.shocks[i].base_name
        x_var = dsge_mod.shocks[j].base_name

        axis.scatter(mu.values[i, j], 0, s=10)
        axis.hlines(0, *hdi.values[i, j])
        axis.axvline(0, ls="--", c="k", lw=0.5)

        axis.set_ylabel(y_var.replace("epsilon_", "") if j == 0 else "", fontsize=6)
        axis.set_xlabel(x_var.replace("epsilon_", "") if i == (n_shocks - 1) else "", fontsize=6)

        axis.set_yticklabels([])
        axis.tick_params(axis="x", labelsize=6)

    return fig


def _style_panel(axis: plt.Axes, **grid_kwargs) -> None:
    axis.spines[:].set_visible(False)
    axis.grid(ls="--", lw=0.5, **grid_kwargs)


def _draw_unit_circle(axis: plt.Axes, **line_kwargs) -> None:
    theta = np.linspace(0, 2 * np.pi, 200)
    axis.plot(np.cos(theta), np.sin(theta), color="k", lw=1, **line_kwargs)


def _plot_single_variable(
    data: xr.DataArray,
    ax: plt.Axes,
    ci: float | None = None,
    cmap: str | Colormap | None = None,
    fill_color: str | None = "tab:blue",
    **line_kwargs,
) -> None:
    """
    Plot every trajectory of one variable, or its mean with a credible band when ``ci`` is given.

    Parameters
    ----------
    data : DataArray
        Trajectories of one variable, with a ``time`` dimension and either a ``simulation`` or a ``shock``
        dimension.
    ax : matplotlib Axes
        Axis to draw on.
    ci : float, optional
        Width of the credible band, between 0 and 1. By default every trajectory is drawn and no band is shown.
    cmap : str or Colormap, optional
        Colormap of the trajectory lines. Defaults to the matplotlib color cycle.
    fill_color : str, optional
        Color of the credible band. Defaults to "tab:blue".
    **line_kwargs
        Keyword arguments forwarded to the line plot.
    """
    set_axis_cmap(ax, cmap)

    if ci is None:
        hue = "shock" if "shock" in data.coords else None
        data.plot.line(x="time", ax=ax, add_legend=False, hue=hue, **line_kwargs)
        if hue is not None:
            lines = ax.get_lines()
            for line, shock in zip(lines, data.coords["shock"].values, strict=False):
                line.set_label(shock)
        return

    q_low, q_high = ((1 - ci) / 2), 1 - ((1 - ci) / 2)
    ci_bounds = data.quantile([q_low, q_high], dim=["simulation"])

    data.mean(dim="simulation").plot.line(x="time", ax=ax, add_legend=False, **line_kwargs)
    ci_bounds.plot.line(
        ax=ax,
        x="time",
        hue="quantile",
        ls="--",
        lw=0.5,
        color="k",
        add_legend=False,
    )
    ax.fill_between(
        ci_bounds.coords["time"].values,
        *ci_bounds.transpose("quantile", "time").values,
        color=fill_color,
        alpha=0.25,
    )


def _irf_to_mapping(
    irf: xr.DataArray | list[xr.DataArray] | dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    if isinstance(irf, xr.DataArray):
        return {"": irf}
    if isinstance(irf, list):
        return {f"Scenario {i}": scenario for i, scenario in enumerate(irf)}
    if isinstance(irf, dict):
        return irf
    raise TypeError(
        f"irf must be a DataArray, a list of DataArrays, or a dict mapping scenario names to DataArrays, but got "
        f"{type(irf)}."
    )


def _resolve_list_to_plot(
    value: str | list[str] | None,
    available: list[str],
    item_name: str,
) -> list[str]:
    if value is None:
        return available
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"Expected str or list for {item_name}, got {type(value)}")
    for v in value:
        if v not in available:
            raise ValueError(f"{item_name} '{v}' not found among available: {available}")
    return value


def _resolve_vars_and_shocks(
    irf_map: dict[str, xr.DataArray],
    vars_to_plot: str | list[str] | None,
    shocks_to_plot: str | list[str] | None,
) -> tuple[list[str], list[str] | None]:
    coords = next(iter(irf_map.values())).coords
    variables = coords["variable"].values.tolist()
    vars_resolved = _resolve_list_to_plot(vars_to_plot, variables, "variable")

    shocks_resolved: list[str] | None
    if "shock" in coords:
        shocks = coords["shock"].values.tolist()
        shocks_resolved = _resolve_list_to_plot(shocks_to_plot, shocks, "shock")
    else:
        shocks_resolved = None

    return vars_resolved, shocks_resolved


def _prepare_irf_grid(
    figsize: tuple[int, int], dpi: int, n_plots: int, n_cols: int | None
) -> tuple[Figure, GridSpec, list[tuple[slice, slice]], list[bool]]:
    """
    Create the figure and grid, and flag the panels that keep their x ticks.

    Returns
    -------
    fig : Figure
        The new figure.
    gs : GridSpec
        Grid layout of the panels.
    plot_locs : list of tuple of slice
        Row and column slices into ``gs`` for each panel.
    show_xticks : list of bool
        Whether each panel is in a bottom row and so keeps its x ticks. A partial last row leaves panels of the row
        above it without a panel underneath, so both rows count as bottom rows.
    """
    n_cols = min(4, n_plots) if n_cols is None else n_cols
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots, figure=fig)

    plot_row_idxs = [loc[0].stop // 2 - 1 for loc in plot_locs]
    plot_rows = sorted(set(plot_row_idxs))
    is_square = all(plot_row_idxs.count(i) == n_cols for i in plot_rows)
    last_row_idxs = [plot_rows[-1]] if is_square else plot_rows[-2:]
    show_xticks = [row_idx in last_row_idxs for row_idx in plot_row_idxs]

    return fig, gs, plot_locs, show_xticks


def _plot_irf_panel(
    axis: plt.Axes,
    irf_map: dict[str, xr.DataArray],
    variable: str,
    shocks_to_plot: list[str] | None,
    cmap: str | Colormap | None,
    add_scenario_legend: bool,
    scenario_names: list[str],
    show_xticks: bool,
) -> None:
    sel_dict: dict[str, Any] = {"variable": variable}
    if shocks_to_plot is not None:
        sel_dict["shock"] = shocks_to_plot

    for scenario_idx, irf_data in enumerate(irf_map.values()):
        _plot_single_variable(
            irf_data.sel(**sel_dict),
            ax=axis,
            cmap=cmap,
            ls=_scenario_line_style(scenario_idx),
        )

    if add_scenario_legend and len(scenario_names) > 1 and scenario_names[0] != "":
        # Scenarios are told apart by line style and shocks by color, so the scenario legend gets one black handle
        # per line style.
        handles = [Line2D([], [], color="k", ls=_scenario_line_style(idx)) for idx in range(len(scenario_names))]
        axis.legend(handles=handles, labels=scenario_names)

    axis.set(title=variable)
    if not show_xticks:
        axis.set(xticklabels=[], xlabel="")

    _style_panel(axis)


def _scenario_line_style(scenario_idx: int) -> str:
    return _SCENARIO_LINE_STYLES[scenario_idx % len(_SCENARIO_LINE_STYLES)]


def _add_shocks_legend(fig: Figure, shocks_to_plot: list[str] | None, legend: bool, legend_kwargs: dict | None) -> None:
    if not legend:
        return
    if legend_kwargs is None:
        n_shocks_to_plot = len(shocks_to_plot) if shocks_to_plot is not None else 1
        legend_kwargs = {
            "ncol": min(4, n_shocks_to_plot),
            "loc": "lower center",
            "bbox_to_anchor": (0.5, 1.0),
        }
    handles = fig.axes[0].get_lines()
    if shocks_to_plot is not None:
        # With several scenarios the first panel holds n_scenarios * n_shocks lines, and the first n_shocks of them
        # already carry every shock color.
        handles = handles[: len(shocks_to_plot)]
    fig.legend(handles=handles, labels=shocks_to_plot, **legend_kwargs)


def _solv_prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Split a solvability table into the parameter columns to plot, the failure stage, and the parameter names.

    Returns
    -------
    plot_data : DataFrame
        Non-constant parameter columns plus a boolean ``success`` column.
    failure_step : Series
        The ``failure_step`` column. NaN marks a successful draw.
    params : list of str
        Parameter column names available for plotting.
    """
    failure_step = data["failure_step"].copy()

    plot_data = data.drop(columns=[col for col in data.columns if col in _SOLVABILITY_META_COLS])

    constant_mask = plot_data.apply(pd.api.types.is_numeric_dtype) & (plot_data.var() < 1e-18)
    plot_data = plot_data.loc[:, ~constant_mask].copy()

    plot_data["success"] = failure_step.isna()
    params = [c for c in plot_data.columns if c != "success"]
    return plot_data, failure_step, params


def _solv_plot_diagonal(ax: plt.Axes, values: pd.Series, success: pd.Series) -> None:
    for mask, color in [(success, _SOLVABILITY_COLORS["success"]), (~success, "tab:red")]:
        subset = values[mask]
        if len(subset) < 2:
            continue
        lo, hi = subset.min(), subset.max()
        pad = max((hi - lo) * 0.1, abs(lo) * 0.05 + 1e-12)
        grid = np.linspace(lo - pad, hi + pad, 100)
        kde = stats.gaussian_kde(subset)
        density = kde.pdf(grid)
        ax.plot(grid, density, color=color)
        ax.fill_between(grid, density, 0, color=color, alpha=0.25)


def _solv_plot_offdiag(
    ax: plt.Axes,
    plot_data: pd.DataFrame,
    failure_step: pd.Series,
    x_name: str,
    y_name: str,
) -> None:
    success = plot_data["success"]
    ax.scatter(
        plot_data.loc[success, x_name],
        plot_data.loc[success, y_name],
        c=_SOLVABILITY_COLORS["success"],
        s=10,
        label="Success",
    )

    reasons = failure_step[~success]
    for reason in reasons.unique():
        mask = reasons == reason
        color = _SOLVABILITY_COLORS.get(reason, "tab:gray")
        ax.scatter(
            plot_data.loc[~success, x_name][mask],
            plot_data.loc[~success, y_name][mask],
            c=color,
            s=10,
            label=reason.replace("_", " ").title(),
        )


def _solv_format_axes(axes: np.ndarray, params_use: list[str]) -> None:
    n = len(params_use)
    for row, col in product(range(n), range(n)):
        ax = axes[row, col]
        if not ax.get_visible():
            continue
        if col == 0:
            ax.set_ylabel(params_use[row])
        if row == n - 1:
            ax.set_xlabel(params_use[col])
        _style_panel(ax)


def _filter_eigenvalues(
    sensitivity_data: xr.Dataset,
    filter_zeros: bool,
    filter_infinite: bool,
    zero_tol: float,
    inf_tol: float | None,
) -> tuple[xr.Dataset, int]:
    """
    Drop zero and infinite eigenvalues from a sensitivity dataset.

    Returns
    -------
    filtered_data : Dataset
        ``sensitivity_data`` restricted to the kept eigenvalues.
    n_filtered : int
        Number of eigenvalues dropped.
    """
    mod_vals = sensitivity_data.eigenvalues.sel(component="modulus").values

    if inf_tol is None:
        finite_mods = mod_vals[mod_vals < 1e6]
        inf_tol = max(10 * finite_mods.max(), 10.0) if len(finite_mods) > 0 else 1e6

    mask = np.ones(len(mod_vals), dtype=bool)
    if filter_zeros:
        mask &= mod_vals > zero_tol
    if filter_infinite:
        mask &= mod_vals < inf_tol

    return sensitivity_data.isel(eigenvalue=mask), int(len(mod_vals) - mask.sum())


def _validate_params_to_plot(
    params_to_plot: list[str] | None,
    all_params: list[str],
) -> list[str]:
    if params_to_plot is None:
        return all_params

    for param in params_to_plot:
        if param not in all_params:
            raise ValueError(f"Parameter '{param}' not found. Available: {all_params}")
    return params_to_plot


def _compute_axis_limits(re_vals: np.ndarray, im_vals: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute square axis limits centered on the eigenvalues, padded by 30 percent of their spread.

    Returns
    -------
    xlim : tuple of float
        Limits of the real axis.
    ylim : tuple of float
        Limits of the imaginary axis.
    """
    if len(re_vals) == 0:
        return (-1.5, 1.5), (-1.5, 1.5)

    re_min, re_max = re_vals.min(), re_vals.max()
    im_min, im_max = im_vals.min(), im_vals.max()

    re_range = max(re_max - re_min, 0.1)
    im_range = max(im_max - im_min, 0.1)
    pad_re = re_range * 0.3
    pad_im = im_range * 0.3

    xlim = (re_min - pad_re, re_max + pad_re)
    ylim = (im_min - pad_im, im_max + pad_im)

    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    half_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2

    xlim = (x_center - half_range, x_center + half_range)
    ylim = (y_center - half_range, y_center + half_range)

    return xlim, ylim


def _draw_eigenvalue_panel(
    ax: plt.Axes,
    re_plot: np.ndarray,
    im_plot: np.ndarray,
    mod_plot: np.ndarray,
    d_re: np.ndarray,
    d_im: np.ndarray,
    param_name: str,
    param_value: float,
    perturbation: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    plot_circle: bool,
    show_legend: bool,
    min_arrow_frac: float,
) -> None:
    """
    Draw one eigenvalue sensitivity panel: the eigenvalues, the unit circle, and an arrow per eigenvalue.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis to draw on.
    re_plot : ndarray
        Real parts of the eigenvalues.
    im_plot : ndarray
        Imaginary parts of the eigenvalues.
    mod_plot : ndarray
        Moduli of the eigenvalues.
    d_re : ndarray
        Derivative of each real part with respect to the parameter.
    d_im : ndarray
        Derivative of each imaginary part with respect to the parameter.
    param_name : str
        Parameter name, used in the title.
    param_value : float
        Current parameter value.
    perturbation : float
        Fractional increase in the parameter value that the arrows show.
    xlim : tuple of float
        Limits of the real axis.
    ylim : tuple of float
        Limits of the imaginary axis.
    plot_circle : bool
        If True, draw the unit circle.
    show_legend : bool
        If True, add a legend to this panel.
    min_arrow_frac : float
        Smallest gradient magnitude, as a fraction of the largest, for which an arrow is drawn.
    """
    if plot_circle:
        _draw_unit_circle(ax, zorder=1, label="Unit circle")

    stable_mask = mod_plot <= 1.0
    for mask, color, label in [
        (stable_mask, "tab:blue", "Stable (|λ|≤1)"),
        (~stable_mask, "tab:red", "Unstable (|λ|>1)"),
    ]:
        if mask.any():
            ax.scatter(re_plot[mask], im_plot[mask], c=color, s=40, lw=0.5, edgecolor="k", zorder=3, label=label)

    # The arrow is the first-order displacement d(lambda)/d(p) * delta_p for a delta_p of perturbation * p.
    delta_p = param_value * perturbation
    arrow_re = d_re * delta_p
    arrow_im = d_im * delta_p

    grad_mags = np.sqrt(d_re**2 + d_im**2)
    max_grad = grad_mags.max() if grad_mags.size > 0 and grad_mags.max() > 1e-12 else 1.0
    min_grad_threshold = max_grad * min_arrow_frac

    for i in np.flatnonzero(grad_mags > min_grad_threshold):
        ax.annotate(
            "",
            xy=(re_plot[i] + arrow_re[i], im_plot[i] + arrow_im[i]),
            xytext=(re_plot[i], im_plot[i]),
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 1},
            zorder=2,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.5, ls="--", zorder=0)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    perturbed_value = param_value * (1 + perturbation)
    ax.set_title(f"{param_name}: {param_value:.4g} → {perturbed_value:.4g}")
    _style_panel(ax, alpha=0.5)

    if show_legend:
        ax.legend(loc="best", fontsize="small", framealpha=0.9)


def _acf_hdi_bounds(series: xr.DataArray, prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-lag lower and upper bounds of the ``prob`` highest-density interval of ``series``."""
    hdi = azs.hdi(series, prob=prob)
    ci_dim = next(dim for dim in hdi.dims if dim != "lag")
    return hdi.isel({ci_dim: 0}).values, hdi.isel({ci_dim: 1}).values


def _plot_acf_intervals(
    axis: plt.Axes,
    series: xr.DataArray,
    lags: np.ndarray,
    sample_dims: list[str],
    ci_probs: tuple[float, float],
    color: str = "tab:blue",
    offset: float = 0.0,
    mean_kwargs: dict | None = None,
    inner_hdi_kwargs: dict | None = None,
    outer_hdi_kwargs: dict | None = None,
) -> None:
    """Draw a posterior-mean point plus two nested credible-interval sticks at each lag."""
    inner_prob, outer_prob = sorted(ci_probs)
    mean = series.mean(sample_dims).values
    outer_lo, outer_hi = _acf_hdi_bounds(series, outer_prob)
    inner_lo, inner_hi = _acf_hdi_bounds(series, inner_prob)

    x = lags + offset
    axis.vlines(x, outer_lo, outer_hi, **{"color": color, "lw": 1.0, **(outer_hdi_kwargs or {})})
    axis.vlines(x, inner_lo, inner_hi, **{"color": color, "lw": 3.0, **(inner_hdi_kwargs or {})})
    axis.scatter(x, mean, **{"color": color, "s": 38, "zorder": 3, **(mean_kwargs or {})})


class _AcfModel(NamedTuple):
    """One model's autocorrelation diagonal with the dodge offset and color it is drawn with."""

    diagonal: xr.DataArray
    sample_dims: list[str]
    offset: float
    color: str


def _collect_acf_diagonals(
    tensors: dict[Any, xr.DataArray], sample_dims: tuple[str, ...], dodge: float
) -> tuple[dict[Any, _AcfModel], str | None]:
    """
    Take the diagonal of each model's autocorrelation tensor and assign each model a dodge offset and a color.

    Returns
    -------
    models : dict mapping label to _AcfModel
        Each variable's own autocorrelation, per model, with the sample dimensions its tensor carries.
    var_dim : str or None
        Name of the variable dimension of the diagonals.
    """
    n_models = len(tensors)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * dodge

    models, var_dim = {}, None
    for (label, tensor), offset, color in zip(tensors.items(), offsets, _color_cycle(n_models), strict=True):
        present = [dim for dim in sample_dims if dim in tensor.dims]
        var_dims = [dim for dim in tensor.dims if dim != "lag" and dim not in present]
        if len(var_dims) != 2:
            raise ValueError(
                "Expected an autocorrelation tensor with a 'lag' dimension and two variable dimensions, but found "
                f"non-lag/sample dimensions {var_dims}."
            )
        models[label] = _AcfModel(xr_diagonal(tensor, dims=var_dims), present, float(offset), color)
        var_dim = var_dims[0]
    return models, var_dim


def _color_cycle(n_colors: int) -> list[str]:
    return [f"C{i}" for i in range(n_colors)]


def _draw_acf_panel(
    axis: plt.Axes,
    variable: str,
    models: dict[Any, _AcfModel],
    var_dim: str,
    lags: np.ndarray,
    ci_probs: tuple[float, float],
    reference: xr.DataArray | None,
    ref_var_dim: str | None,
    mean_kwargs: dict | None,
    inner_hdi_kwargs: dict | None,
    outer_hdi_kwargs: dict | None,
    stem_kwargs: dict | None,
    reference_kwargs: dict | None,
) -> None:
    """Draw one autocorrelation panel: each model's sticks or stems plus the optional hollow reference markers."""
    axis.axhline(0, color="k", lw=0.5)
    for model in models.values():
        if variable not in model.diagonal.coords[var_dim].values:
            continue
        series = model.diagonal.sel({var_dim: variable})
        if model.sample_dims:
            _plot_acf_intervals(
                axis,
                series,
                lags,
                model.sample_dims,
                ci_probs,
                color=model.color,
                offset=model.offset,
                mean_kwargs=mean_kwargs,
                inner_hdi_kwargs=inner_hdi_kwargs,
                outer_hdi_kwargs=outer_hdi_kwargs,
            )
        else:
            axis.scatter(lags + model.offset, series.values, **{"color": model.color, **(stem_kwargs or {})})
            axis.vlines(lags + model.offset, 0, series.values, color=model.color)

    if ref_var_dim is not None and variable in reference.coords[ref_var_dim].values:
        ref_series = reference.sel({ref_var_dim: variable})
        axis.scatter(
            reference.coords["lag"].values, ref_series.values, **{**_REFERENCE_DEFAULTS, **(reference_kwargs or {})}
        )

    _style_panel(axis)
    axis.set(title=str(variable))


def _add_acf_legend(
    fig: Figure, models: dict[Any, _AcfModel], has_reference: bool, reference_kwargs: dict | None = None
) -> None:
    handles, labels = [], []
    if len(models) > 1:
        handles += [Line2D([0], [0], color=model.color, lw=3) for model in models.values()]
        labels += [str(label) for label in models]
    if has_reference:
        ref = {**_REFERENCE_DEFAULTS, **(reference_kwargs or {})}
        handles.append(
            Line2D(
                [0],
                [0],
                ls="none",
                marker="o",
                mfc=ref["facecolors"],
                mec=ref["edgecolors"],
                mew=ref["linewidths"],
                ms=ref["s"] ** 0.5,
            )
        )
        labels.append("data")
    if handles and fig.axes:
        fig.axes[0].legend(handles, labels, fontsize=8)


def _validate_and_prepare_corner_inputs(
    idata: Any,
    group: str,
    var_names: list[str] | None,
    colorby: str | None,
    scatter_kwargs: dict | None,
) -> tuple[list[str], np.ndarray | None, dict | None]:
    """
    Check that the requested variables exist and resolve the optional color mapping.

    Returns
    -------
    var_names : list of str
        Variables to plot.
    color_data : ndarray or None
        Flattened values of ``colorby``, or None when no scatter is drawn.
    scatter_kwargs : dict or None
        Keyword arguments for the colored scatter, or None when no scatter is drawn.
    """
    if not hasattr(idata, group):
        raise ValueError(f"Argument idata should be an arviz idata object with a {group} group")

    vars_available = list(idata[group].data_vars)
    var_names = var_names or vars_available
    for v in var_names:
        if v not in vars_available:
            raise ValueError(f'Variable "{v}" not found in idata[{group}]')

    if colorby is None:
        return var_names, None, None

    if colorby not in vars_available:
        raise ValueError(f'colorby "{colorby}" not found in idata[{group}]')
    color_data = idata[group][colorby].values.ravel()
    resolved_scatter_kwargs = {"zorder": 100, "cmap": "viridis", "s": 10, "alpha": 0.5, **(scatter_kwargs or {})}

    return var_names, color_data, resolved_scatter_kwargs


def _format_axis_for_corner(ax: plt.Axes, fontsize: int) -> None:
    ax.ticklabel_format(axis="both", style="sci")
    ax.yaxis.major.formatter.set_powerlimits((-2, 2))
    ax.yaxis.offsetText.set_fontsize(fontsize)
    ax.xaxis.major.formatter.set_powerlimits((-2, 2))
    ax.xaxis.offsetText.set_fontsize(fontsize)


def _plot_diagonal_hist(ax: plt.Axes, data: np.ndarray, bins: int, fontsize: int, is_last_row: bool) -> None:
    ax.hist(data, bins=bins, histtype="step", density=True)
    ax.set_yticklabels([])
    ax.tick_params(axis="both", left=False, bottom=is_last_row, labelsize=fontsize)
    if not is_last_row:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", bottom=False)


def _plot_offdiag_panel(
    ax: plt.Axes,
    x_name: str,
    y_name: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    rug_bins: int,
    rug_levels: int,
    show_marginal_modes: bool,
    fontsize: int,
    draw_xlabel: bool,
    draw_ylabel: bool,
    color_data: np.ndarray | None,
    scatter_kwargs: dict | None,
) -> None:
    if color_data is not None and scatter_kwargs is not None:
        ax.scatter(x_data, y_data, c=color_data, **scatter_kwargs)

    # histogram2d takes y first so that the histogram's axes line up with the contour's.
    H, y_edges, x_edges = np.histogram2d(y_data, x_data, bins=rug_bins)

    iy, ix = np.unravel_index(np.argmax(H), H.shape)
    x_mode, y_mode = x_edges[ix], y_edges[iy]

    ax.contourf(x_edges[:-1], y_edges[:-1], H, cmap="Blues", levels=rug_levels)

    if show_marginal_modes:
        ax.axvline(x_mode, ls="--", lw=0.5, color="k")
        ax.axhline(y_mode, ls="--", lw=0.5, color="k")
        ax.scatter(x_mode, y_mode, color="k", marker="s", s=20)

    if draw_ylabel:
        ax.set_ylabel(y_name, fontsize=fontsize)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False)

    if draw_xlabel:
        ax.set_xlabel(x_name, fontsize=fontsize)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", bottom=False)

    ax.tick_params(axis="both", which="both", labelsize=fontsize)


__all__ = [
    "plot_acf",
    "plot_corner",
    "plot_covariance_matrix",
    "plot_eigenvalues",
    "plot_irf",
    "plot_kalman_filter",
    "plot_posterior_with_prior",
    "plot_simulation",
    "plot_solvability",
    "plot_solvability_summary",
    "plot_timeseries",
    "prepare_gridspec_figure",
    "set_matplotlib_style",
]
