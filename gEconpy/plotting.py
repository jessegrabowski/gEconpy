import warnings

from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Any, Literal, cast

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy import stats
from xarray_einstats.linalg import diagonal as xr_diagonal

if TYPE_CHECKING:
    from gEconpy.model.statespace import DSGEStateSpace


def prepare_gridspec_figure(
    n_cols: int, n_plots: int, figure: plt.Figure | None = None
) -> tuple[GridSpec, list]:
    """
     Prepare a figure with a grid of subplots. Centers the last row of plots if the number of plots is not square.

    Parameters
    ----------
     n_cols : int
         The number of columns in the grid.
     n_plots : int
         The number of subplots in the grid.
    figure : Figure, optional
        The figure object to use

    Returns
    -------
     GridSpec
         A matplotlib GridSpec object representing the layout of the grid.
    list of tuple(slice, slice)
         A list of tuples of slices representing the indices of the grid cells to be used for each subplot.
    """

    remainder = n_plots % n_cols
    has_remainder = remainder > 0
    n_rows = n_plots // n_cols + int(has_remainder)

    gs = GridSpec(2 * n_rows, 2 * n_cols, figure=figure)
    plot_locs = []

    for i in range(n_rows - int(has_remainder)):
        for j in range(n_cols):
            plot_locs.append((slice(i * 2, (i + 1) * 2), slice(j * 2, (j + 1) * 2)))

    if has_remainder:
        last_row = slice((n_rows - 1) * 2, n_rows * 2)
        left_pad = int(n_cols - remainder)
        for j in range(remainder):
            col_slice = slice(left_pad + j * 2, left_pad + (j + 1) * 2)
            plot_locs.append((last_row, col_slice))

    return gs, plot_locs


def set_axis_cmap(axis, cmap):
    cycler = None
    if cmap is not None:
        color = getattr(plt.cm, cmap)(np.linspace(0, 1, 20))
        cycler = plt.cycler(color=color)
    axis.set_prop_cycle(cycler)


def _plot_single_variable(
    data: xr.DataArray, ax, ci=None, cmap=None, fill_color="tab:blue", **line_kwargs
):
    """
    Plot the mean and optionally a confidence interval for a single variable.

    Parameters
    ----------
    data : xr.DataArray
        A DataFrame with one or more columns containing the data to plot.
    ax : Matplotlib Axes
        The Axes object to plot on.
    ci : float, optional
        The confidence interval to plot, between 0 and 1. If not provided, only the mean will be plotted.
    cmap : str or Colormap, optional
        The color map to use for the data.
    fill_color : str, optional
        The color to use to fill the confidence interval.
    line_kwargs: optional
        Additional keyword arguments to pass to the line plot.

    Returns
    -------
    None
    """
    set_axis_cmap(ax, cmap)

    if ci is None:
        hue = "shock" if "shock" in data.coords else None
        data.plot.line(x="time", ax=ax, add_legend=False, hue=hue, **line_kwargs)
        if hue is not None:
            lines = ax.get_lines()
            for line, shock in zip(lines, data.coords["shock"].values):
                line.set_label(shock)

    else:
        q_low, q_high = ((1 - ci) / 2), 1 - ((1 - ci) / 2)
        ci_bounds = data.quantile([q_low, q_high], dim=["simulation"])

        data.mean(dim="simulation").plot.line(
            x="time", ax=ax, add_legend=False, **line_kwargs
        )
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


def plot_simulation(
    simulation: xr.DataArray,
    vars_to_plot: list[str] | None = None,
    ci: float | None = None,
    n_cols: int | None = None,
    cmap: str | Colormap | None = None,
    fill_color: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot a simulation of multiple variables.

    Parameters
    ----------
    simulation : pd.DataFrame
        A DataFrame with one or more columns containing the data to plot. The columns should be the variables to plot
        and the index should be the time.
    vars_to_plot : list of str, optional
        A list of the variables to plot. If not provided, all variables in the simulation DataFrame will be plotted.
    ci : float, optional
        The confidence interval to plot, between 0 and 1. If not provided, only the mean will be plotted.
    n_cols : int, optional
        The number of columns of plots to show. If not provided, the minimum of (4, number of columns in df) will be
        used.
    cmap : str or Colormap, optional
        The color map to use for the data.
    fill_color : str, optional
        The color to use to fill the confidence interval.
    figsize : tuple of int
        The size of the figure in inches. Default is (12, 8).
    dpi : int
        The resolution of the figure in dots per inch. Default is 100.

    Returns
    -------
    Figure
        The Matplotlib Figure object containing the plots.
    """

    if vars_to_plot is None:
        vars_to_plot = simulation.coords["variable"].values.tolist()
    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots)

    for idx, variable in enumerate(vars_to_plot):
        if variable not in simulation.coords["variable"]:
            raise ValueError(f"{variable} not found among model variables.")
        axis = fig.add_subplot(gs[plot_locs[idx]])

        _plot_single_variable(
            simulation.sel(variable=variable),
            ci=ci,
            ax=axis,
            fill_color=fill_color,
            cmap=cmap,
        )

        axis.set(title=variable)
        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)

    fig.tight_layout()
    return fig


def plot_irf(
    irf: az.InferenceData | list[az.InferenceData] | dict[str, az.InferenceData],
    vars_to_plot: str | list[str] | None = None,
    shocks_to_plot: str | list[str] | None = None,
    n_cols: int | None = None,
    legend: bool = False,
    cmap: str | Colormap | None = None,
    legend_kwargs: dict | None = None,
    figsize: tuple[int, int] = (14, 10),
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot the impulse response functions for a set of variables.

    Parameters
    ----------
    irf : xr.DataArray, list of xr.DataArray, or dict of xr.DataArray
        A DataArray with the impulse response functions. The index should contain the variables to plot, and the columns
        should contain the shocks, with a multi-index for the period and shock type. When plotting multiple scenarios,
        provide a list of DataArrays or a dictionary with the scenario names as keys.
    group: str, optional
        The group from the InferenceData to plot. Must be one of "prior" or "posterior". Default is 'posterior'.
    vars_to_plot : list of str, optional
        A list of variables to plot. If not provided, all variables in the DataFrame will be plotted.
    shocks_to_plot : list of str, optional
        A list of shocks to plot. If not provided, all shocks in the DataFrame will be plotted.
    n_cols : int, optional
        The number of columns to use in the plot grid. If not provided, the number of columns will be determined
        automatically based on the number of variables to plot.
    legend : bool, optional
        Whether to show a legend with the shocks.
    cmap : str or Colormap, optional
        The color map to use for the impulse response functions.
    legend_kwargs : dict, optional
        Keyword arguments to pass to `matplotlib.figure.Figure.legend()`.
    figsize : tuple, optional
        The size of the figure in inches.
    dpi : int, optional
        The DPI of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    if not isinstance(vars_to_plot, str | list | None):
        raise ValueError(
            f"Expected strings or list of strings for parameter vars_to_plot, got {vars_to_plot} of "
            f"type {type(vars_to_plot)}"
        )

    if isinstance(irf, xr.DataArray):
        irf = {"": irf}
    elif isinstance(irf, list):
        irf = {f"Scenario {i}": irf[i] for i in range(len(irf))}

    coords = irf[next(iter(irf.keys()))].coords

    if vars_to_plot is None:
        vars_to_plot = coords["variable"].values.tolist()
    if isinstance(vars_to_plot, str):
        vars_to_plot = [vars_to_plot]

    for var in vars_to_plot:
        if var not in coords["variable"]:
            raise ValueError(f"{var} not found among simulated impulse responses.")

    if "shock" in coords:
        shock_list = coords["shock"].values.tolist()
    else:
        shock_list = None

    if shocks_to_plot is None:
        shocks_to_plot = shock_list
    if isinstance(shocks_to_plot, str):
        shocks_to_plot = [shocks_to_plot]

    for shock in shocks_to_plot:
        if shock not in shock_list:
            raise ValueError(
                f"{shock} not found among shocks used in impulse response data."
            )

    if not isinstance(shocks_to_plot, list):
        raise ValueError(
            f"Expected list for parameter shocks_to_plot, got {shocks_to_plot} "
            f"of type {type(shocks_to_plot)}"
        )

    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    markers = ["-", "--", "-.", ":"]
    scenario_names = list(irf.keys())

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots, figure=fig)

    plot_row_idxs = [x[0].stop // 2 - 1 for x in plot_locs]
    plot_rows = sorted(list(set(plot_row_idxs)))
    is_square = all([plot_row_idxs.count(i) == n_cols for i in plot_rows])
    last_row_idxs = [plot_rows[-1]] if is_square else plot_rows[-2:]

    for idx, variable in enumerate(vars_to_plot):
        loc = plot_locs[idx]
        row_idx = plot_row_idxs[idx]

        axis = fig.add_subplot(gs[loc])
        sel_dict = {"variable": variable}
        if shocks_to_plot is not None:
            sel_dict["shock"] = shocks_to_plot

        for scenario_idx, (scenario, irf_data) in enumerate(irf.items()):
            _plot_single_variable(
                irf_data.sel(**sel_dict),
                ax=axis,
                cmap=cmap,
                ls=markers[scenario_idx % 4],
            )

        if (idx == 0) and len(scenario_names) > 1 and scenario_names[0] != "":
            lines = axis.get_lines()
            axis.legend(handles=lines, labels=scenario_names)

        axis.set(title=variable)
        if row_idx not in last_row_idxs:
            axis.set(xticklabels=[], xlabel="")

        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)

    if legend:
        if legend_kwargs is None:
            n_shocks_to_plot = len(shocks_to_plot) if shocks_to_plot is not None else 1
            legend_kwargs = {
                "ncol": min(4, n_shocks_to_plot),
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 1.0),
            }
        handles = fig.axes[0].get_lines()
        fig.legend(handles=handles, labels=shocks_to_plot, **legend_kwargs)

    return fig


def plot_prior_solvability(
    data: pd.DataFrame,
    params_to_plot: list[str] | None = None,
):
    """
    Plot the results of sampling from the prior distributions of a GCN and attempting to fit a DSGE model.

    This function produces a grid of plots that show the distribution of parameter values where model fitting was successful
    or where it failed. Each plot on the grid shows the distribution of one parameter against another, with successful
    fits plotted in blue and failed fits plotted in red.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the results of sampling from the prior distributions and attempting to fit a model.
    n_samples : int, optional
        The number of samples to draw from the prior distributions.
    seed : int, optional
        The seed to use for the random number generator.
    params_to_plot : list of str, optional
        A list of parameter names to include in the plots. If not provided, all parameters will be plotted.

    Returns
    -------
    fig : Matplotlib Figure
        The Figure object containing the plots

    Notes
    -----
    - Parameters will be sampled from prior distributions defined in the GCN.
    - The following failure modes are considered:
        - Steady state: The steady state of the model could not be calculated.
        - Perturbation: The perturbation of the model failed.
        - Blanchard-Kahn: The Blanchard-Kahn condition was not satisfied.
        - Deterministic norm: Residuals of the deterministic part of the solution matrix were not zero.
        - Stochastic norm: Residuals of the stochastic part of the solution matrix were not zero.
    """

    plot_data = data.copy()
    failure_step = plot_data["failure_step"].copy()
    plot_data.drop(columns=["failure_step"], inplace=True)

    color_dict = {
        "steady_state": "tab:red",
        "perturbation": "tab:orange",
        "blanchard-kahn": "tab:green",
        "deterministic_norm": "tab:purple",
        "stochastic_norm": "tab:pink",
    }

    constant_cols = plot_data.var() < 1e-18

    plot_data = plot_data.loc[:, ~constant_cols].copy()
    params = plot_data.columns
    n_params = len(params) if params_to_plot is None else len(params_to_plot)

    plot_data["success"] = failure_step.isna()
    fig, axes = plt.subplots(n_params, n_params, figsize=(16, 16), dpi=100)

    if params_to_plot is not None:
        for param in params_to_plot:
            if param not in params:
                raise ValueError(
                    f'Cannot plot parameter "{param}", it was not found in the provided data.'
                )

    if params_to_plot is None:
        param_pairs = list(combinations_with_replacement(params, 2))
    else:
        param_pairs = list(combinations_with_replacement(params_to_plot, 2))

    plot_grid = np.arange(1, n_params**2 + 1).reshape((n_params, n_params))
    plot_grid[np.tril_indices(n_params, k=-1)] = 0

    plot_idxs = np.where(plot_grid)
    blank_idxs = np.where(plot_grid == 0)

    for col, row in zip(*blank_idxs):
        axes[row][col].set_visible(False)

    for col, row, pair in zip(*plot_idxs, param_pairs):
        param_1, param_2 = pair
        axis = axes[row][col]
        if param_1 == param_2:
            X_sorted = plot_data[param_1].sort_values()
            X_success = X_sorted[plot_data["success"]]
            X_failure = X_sorted[~plot_data["success"]]

            n_success = X_success.shape[0]
            n_failure = X_failure.shape[0]

            if n_success > 0:
                success_grid = np.linspace(
                    X_success.min() * 0.9, X_success.max() * 1.1, 100
                )
                d_success = stats.gaussian_kde(X_success)
                axis.plot(success_grid, d_success.pdf(success_grid), color="tab:blue")
                axis.fill_between(
                    x=success_grid,
                    y1=d_success.pdf(success_grid),
                    y2=0,
                    color="tab:blue",
                    alpha=0.25,
                )

            if n_failure > 0:
                failure_grid = np.linspace(
                    X_failure.min() * 0.9, X_failure.max() * 1.1, 100
                )

                d_failure = stats.gaussian_kde(X_failure)
                axis.plot(failure_grid, d_failure.pdf(failure_grid), color="tab:red")
                axis.fill_between(
                    x=failure_grid,
                    y1=d_failure.pdf(failure_grid),
                    y2=0,
                    color="tab:red",
                    alpha=0.25,
                )

        else:
            axis.scatter(
                plot_data.loc[plot_data.success, param_1],
                plot_data.loc[plot_data.success, param_2],
                c="tab:blue",
                s=10,
                label="Model Successfully Fit",
            )
            why_failed = failure_step[~plot_data.success]
            for reason in why_failed.unique():
                reason_mask = why_failed == reason
                axis.scatter(
                    plot_data.loc[~plot_data.success, param_1][reason_mask],
                    plot_data.loc[~plot_data.success, param_2][reason_mask],
                    c=color_dict[reason],
                    s=10,
                    label=f"{reason.title()} Failed",
                )

        if col == 0:
            axis.set_ylabel(param_2)
        if row == n_params - 1:
            axis.set_xlabel(param_1)

        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)

    axes[1][0].legend(
        loc="center",
        bbox_to_anchor=(0.5, 0.91),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Model Solution Results by Parameter Values", y=0.95)

    return fig


def plot_eigenvalues(
    model: Any,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    C: np.ndarray | None = None,
    D: np.ndarray | None = None,
    linearize_model_kwargs: dict | None = None,
    fig: plt.Figure | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    plot_circle: bool = True,
    **parameter_updates,
):
    """
    Plot the eigenvalues of the model solution, along with a unit circle. Eigenvalues with modulus greater than 1 are
    shown in red, while those with modulus less than 1 are shown in blue. Eigenvalues greater than 10 in modulus
    are not drawn.

    Parameters
    ----------
    model : gEconModel
        DSGE model object
    A : np.ndarray, optional
        Matrix of partial derivative, linearized around the steady state. Derivatives taken with respect to variables
        at t-1. If provided, all of A, B, C and D must be provided.
    B : np.ndarray, optional
        Matrix of partial derivative, linearized around the steady state. Derivatives taken with respect to variables
        at t. If provided, all of A, B, C and D must be provided.
    C : np.ndarray, optional
        Matrix of partial derivative, linearized around the steady state. Derivatives taken with respect to variables
        at t+1. If provided, all of A, B, C and D must be provided.
    D : np.ndarray, optional
        Matrix of partial derivative, linearized around the steady state. Derivatives taken with respect to exogenous
        shocks. If provided, all of A, B, C and D must be provided.
    linearize_model_kwargs: dict, optional
        Arguments passed to model.linearize_model. Ignored if A, B, C, D are provided.
    fig: Matplotlib Figure, optional
        The figure object to plot on. If not provided, a new figure will be created.
    figsize : tuple[float, float], optional
        The size of the figure to create.
    dpi : int, optional
        The resolution of the figure to create.
    plot_circle: bool, optional
        Whether to plot the unit circle. Default is True.
    parameter_updates
        A dictionary of parameter at which to linearize the model.

    Returns
    -------
    Matplotlib Figure
        The figure object containing the plot.
    """
    from gEconpy.model.model import check_bk_condition

    if figsize is None:
        figsize = (5, 5)
    if dpi is None:
        dpi = 100

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        ax = fig.axes[0]

    if linearize_model_kwargs is None:
        linearize_model_kwargs = {}

    data = cast(
        pd.DataFrame,
        check_bk_condition(
            model,
            A=A,
            B=B,
            C=C,
            D=D,
            verbose=False,
            return_value="dataframe",
            **linearize_model_kwargs,
        ),
    )

    n_infinity = (data["Modulus"] > 10).sum()
    data = data[data.Modulus < 10]

    if plot_circle:
        x_circle = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
        ax.plot(np.cos(x_circle), np.sin(x_circle), color="k", lw=1)

    ax.set_aspect("equal")
    colors = ["tab:red" if x > 1.0 else "tab:blue" for x in data.Modulus]
    ax.scatter(data.Real, data.Imaginary, color=colors, s=50, lw=1, edgecolor="k")
    [spine.set_visible(False) for spine in ax.spines.values()]
    ax.grid(ls="--", lw=0.5)
    ax.set_title(
        f"Eigenvalues of Model Solution\n{n_infinity} Eigenvalues with Infinity Modulus not shown."
    )
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
) -> plt.Figure:
    """
    Plots a heatmap of the covariance matrix of the input data.

    Parameters
    ----------
    data : pd.DataFrame
        A square DataFrame, representing a covariance matrix. The index and the columns should both have the same
        values.
    vars_to_plot : list of str, optional
        A list of strings containing the names of the variables to plot. If not provided, all variables in the input data
        will be plotted.
    cbarlabel : str, optional
        The label for the colorbar.
    figsize : tuple of float, optional
        The size of the figure to create, in inches.
    dpi : int, optional
        The dots per inch of the figure.
    cbar_kw : dict, optional
        A dictionary of keyword arguments to pass to the colorbar.
    cmap : str, optional
        The color map to use for the heatmap.
    heatmap_kwargs : dict, optional
        Keyword arguments forwarded to plt.imshow
    annotation_kwargs: dict, optional
        Keyword arguments forwarded to gEconpy.plotting.annotate_heatmap

    Returns
    -------
    matplotlib.figure.Figure
        A figure containing the heatmap.
    """

    if vars_to_plot is None:
        vars_to_plot = data.columns

    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    if annotation_kwargs is None:
        annotation_kwargs = {}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im, cbar = plot_heatmap(
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
    ax: Any | None = None,
    cbar_kw: dict | None = None,
    cbarlabel: str | None = "",
    **kwargs,
):
    """
    Create a heatmap from a pandas dataframe.

    Parameters
    ----------
    data: Dataframe
        A pandas dataframe to plat
    ax: matplotlib.axes.ax, Optional
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.
    cbar_kw: Dict, Optional
        A dictionary with arguments to `matplotlib.Figure.colorbar`.
    cbarlabel: str, Optional
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {"shrink": 0.5}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    n_rows, n_columns = data.shape

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set(
        xticks=np.arange(n_rows),
        xticklabels=data.columns,
        yticks=np.arange(n_columns),
        yticklabels=data.index,
    )

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, list | np.ndarray):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_acf(
    acorr: np.ndarray | xr.DataArray,
    vars_to_plot: list[str] | None = None,
    figsize: tuple[int, int] | None = (14, 4),
    dpi: int | None = 100,
    n_cols: int | None = 4,
) -> plt.Figure:
    """
    Plot the autocorrelation function for a set of variables.

    Parameters
    ----------
    acorr_matrix: DataArray
        Tensor of correlations.
    vars_to_plot: list of str, optional
        List of variables to plot. If not provided, all variables in `acorr_matrix` will be plotted.
    figsize: tuple, optional
        Figure size in inches.
    dpi: int, optional
        Figure resolution in dots per inch.
    n_cols: int, optional
        Number of columns in the subplot grid.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plots.
    """
    all_variables = acorr.coords["variable"].values
    if vars_to_plot is None:
        vars_to_plot = all_variables

    else:
        for var in vars_to_plot:
            if var not in all_variables:
                raise ValueError(
                    f"Can not plot variable {var}, it was not found in the provided covariance matrix"
                )

    n_plots = len(vars_to_plot)
    n_cols = min(n_cols, n_plots)

    fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
    gc, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_plots, figure=fig)

    acorr_matrix = xr_diagonal(acorr, dims=["variable", "variable_aux"]).sel(
        variable=vars_to_plot
    )
    x_values = acorr_matrix.coords["lag"]

    for variable, plot_loc in zip(vars_to_plot, plot_locs):
        axis = fig.add_subplot(gc[plot_loc])
        axis.scatter(x_values, acorr_matrix.sel(variable=variable).values)
        axis.vlines(x_values, 0, acorr_matrix.sel(variable=variable).values)

        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)
        axis.set(title=variable)

    return fig


def plot_corner(
    idata: Any,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] = (14, 14),
    dpi: int = 144,
    hist_bins: int = 200,
    rug_bins: int = 50,
    rug_levels: int = 6,
    fontsize: int = 8,
    show_marginal_modes: bool = True,
) -> None:
    """
    Produces a corner plot, also known as a scatterplot matrix, of the posterior distributions of a set of variables.
    Each panel of the plot shows the two-dimensional distribution of two of the variables, with the remaining variables
    marginalized out. The diagonal panels show the one-dimensional distribution of each variable.

    Parameters
    ----------
    idata : arviz.InferenceData
        An arviz idata object with a posterior group.
    var_names : list of str, optional
        A list of strings specifying the variables to plot. If not provided, all variables in `idata` will be plotted.
    figsize : tuple, optional
        The size of the figure in inches. Default is (14, 14).
    dpi : int, optional
        The resolution of the figure in dots per inch. Default is 144.
    hist_bins : int, optional
        The number of bins to use for the histograms on the diagonal panels. Default is 200.
    rug_bins : int, optional
        The number of bins to use for the histograms on the off-diagonal panels. Default is 50.
    rug_levels : int, optional
        The number of contour levels to use for the histograms on the off-diagonal panels. Default is 6.
    fontsize : int, optional
        The font size for the axis labels and ticks.
    show_marginal_modes : bool, optional
        Whether or not to show the modes of the marginal distributions. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plots.
    """

    if not hasattr(idata, "posterior"):
        raise ValueError(
            "Argument idata should be an arviz idata object with a posterior group"
        )

    var_names = var_names or list(idata.posterior.data_vars)
    k_params = len(var_names)

    fig, ax = plt.subplots(k_params, k_params, figsize=figsize, dpi=dpi)

    for i, axis in enumerate(fig.axes):
        row = i // k_params
        col = i % k_params

        axis.ticklabel_format(axis="both", style="sci")
        axis.yaxis.major.formatter.set_powerlimits((-2, 2))
        axis.yaxis.offsetText.set_fontsize(fontsize)
        axis.xaxis.major.formatter.set_powerlimits((-2, 2))
        axis.xaxis.offsetText.set_fontsize(fontsize)
        if col <= row:
            if col == row:
                v = var_names[col]
                axis.hist(
                    idata.posterior[v].values.ravel(),
                    bins=hist_bins,
                    histtype="step",
                    density=True,
                )
                axis.set_yticklabels([])
                axis.set_title(v, fontsize=fontsize)
                axis.tick_params(
                    axis="both",
                    left=False,
                    bottom=row == (k_params - 1),
                    labelsize=fontsize,
                )
                if row != (k_params - 1):
                    axis.set_xticklabels([])
                    axis.tick_params(axis="x", which="both", bottom=False)

            else:
                x = var_names[col]
                y = var_names[row]

                data_x = idata.posterior[x].values.ravel()
                data_y = idata.posterior[y].values.ravel()

                # x_hist, edges = np.histogram(data_x, bins=hist_bins)
                # x_mode = edges[np.argmax(x_hist)]
                #
                # y_hist, edges = np.histogram(data_y, bins=hist_bins)
                # y_mode = edges[np.argmax(y_hist)]

                H, y_edges, x_edges = np.histogram2d(data_y, data_x, bins=rug_bins)

                ymax_idx, xmax_idx = np.where(H == H.max())
                x_mode = x_edges[xmax_idx]
                y_mode = y_edges[ymax_idx]
                if len(x_mode) > 1:
                    x_mode = x_mode[0]
                if len(y_mode) > 1:
                    y_mode = y_mode[0]

                axis.contourf(
                    x_edges[1:], y_edges[1:], H, cmap="Blues", levels=rug_levels
                )

                if show_marginal_modes:
                    axis.axvline(x_mode, ls="--", lw=0.5, color="k")
                    axis.axhline(y_mode, ls="--", lw=0.5, color="k")
                    axis.scatter(x_mode, y_mode, color="k", marker="s", s=20)

                if col == 0:
                    axis.set_ylabel(y, fontsize=fontsize)
                else:
                    axis.set_yticklabels([])
                    axis.tick_params(axis="y", which="both", left=False)

                if row != (k_params - 1):
                    axis.set_xticklabels([])
                    axis.tick_params(axis="x", which="both", bottom=False)
                else:
                    axis.set_xlabel(x, fontsize=fontsize)

                axis.tick_params(axis="both", which="both", labelsize=fontsize)
        else:
            axis.set(xticks=[], yticks=[], xlabel="", ylabel="")
            axis.set_visible(False)

    fig.tight_layout(h_pad=0.1, w_pad=0.5)
    return fig


def plot_kalman_filter(
    idata: az.InferenceData,
    data: pd.DataFrame,
    kalman_output: Literal["predicted", "filtered", "smoothed"] = "predicted",
    group: Literal["prior", "posterior"] = "posterior",
    n_cols: int | None = None,
    vars_to_plot: list[str] | None = None,
    fig: Figure | None = None,
    figsize: tuple[int, int] = (14, 6),
    dpi: int = 144,
    observed=False,
):
    """
    Plot Kalman filter, prediction or smoothed series for variables in idata.

    Parameters
    ----------
    idata : xarray.Dataset
        Dataset with Kalman filter variables.
    data : pandas.DataFrame
        DataFrame with original time series data.
    kalman_output : str, optional
        String indicating whether to plot filtered, predicted, or smoothed series.
        Must be one of 'filtered', 'predicted', or 'smoothed'.
    group: str, optional
        idata group to plot. One of "prior" or "posterior". Default is 'posterior'.
    n_cols : int, optional
        Number of columns in the plot.
    vars_to_plot : list of str, optional
        List of variable names to plot.
    fig : matplotlib.figure.Figure, optional
        Matplotlib Figure object to plot on.
    figsize : tuple of int, optional
        Figure size in inches.
    dpi : int, optional
        Figure DPI.
    cmap : str, optional
        Colormap name.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib Figure object with the plot.
    """

    if kalman_output.lower() not in ["filtered", "predicted", "smoothed"]:
        raise ValueError(
            f'kalman_output must be one of "filtered", "predicted", "smoothed". Found {kalman_output}.'
        )

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")

    state_name = "state" if not observed else "observed_state"
    if vars_to_plot is None:
        vars_to_plot = idata.coords[state_name].values

    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols
    output_name = (
        f"{kalman_output}_{group}"
        if not observed
        else f"{kalman_output}_{group}_observed"
    )

    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots, figure=fig)
    time_idx = idata.coords["time"]

    means = (
        idata[output_name].sel(**{state_name: vars_to_plot}).mean(dim=["chain", "draw"])
    )
    hdis = az.hdi(
        idata[output_name].sel(**{state_name: vars_to_plot}), hdi_prob=0.95, skipna=True
    )[output_name]

    for idx, variable in enumerate(vars_to_plot):
        axis = fig.add_subplot(gs[plot_locs[idx]])

        axis.plot(time_idx, means.sel(**{state_name: variable}), color="tab:red")
        axis.fill_between(
            time_idx,
            *hdis.sel(**{state_name: variable}).values.T,
            color="tab:blue",
            alpha=0.5,
        )

        if variable in data.columns:
            axis.plot(time_idx, data[variable].values, color="k", ls="--")

        axis.set(title=variable, xlabel=None, ylabel="% Deviation from SS")
        axis.tick_params(axis="x", rotation=45)
        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)

    return fig


def plot_priors(
    statespace_model: "DSGEStateSpace",
    var_names: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int = 144,
    n_cols: int = 6,
    mark_initial_value: bool = True,
):
    pz_priors = statespace_model.param_priors
    hyper_priors = {}

    if statespace_model.shock_priors:
        hyper_priors = {
            shock.param_name_to_hyper_name[name]: hyper_prior
            for shock in statespace_model.shock_priors.values()
            for name, hyper_prior in shock.hyper_param_dict.items()
        }

    pz_priors = pz_priors | hyper_priors

    if var_names is None:
        var_names = pz_priors.keys()

    priors = {k: pz_priors[k] for k in var_names}
    n_params = len(priors)

    if figsize is None:
        n_rows = n_params // n_cols
        figsize = (14, 2 * n_rows)

    fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
    gs, locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_params, figure=fig)

    all_params = statespace_model.param_dict | statespace_model.hyper_param_dict

    for (name, prior), loc in zip(pz_priors.items(), locs):
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
        axis.set_title(name + "\n" + dist_text)
        value = all_params.get(name, None)

        if mark_initial_value and value:
            axis.axvline(value, ls="--", c="k")

    return fig


def plot_posterior_with_prior(
    idata,
    var_names,
    prior_dict,
    true_values=None,
    n_cols=5,
    fig_kwargs=None,
    plot_posterior_kwargs=None,
) -> plt.Figure:
    if true_values is None:
        ref_val = None
    else:
        ref_val = np.r_[
            *[true_values[name].values.ravel() for name in var_names]
        ].tolist()

    if fig_kwargs is None:
        n_rows = len(var_names) // n_cols
        fig_kwargs = {"figsize": (14, n_rows * 3), "dpi": 144, "layout": "constrained"}
    if plot_posterior_kwargs is None:
        plot_posterior_kwargs = {"textsize": 10}

    fig = plt.figure(**fig_kwargs)
    gs, locs = prepare_gridspec_figure(
        n_cols=n_cols, n_plots=len(var_names), figure=fig
    )
    [fig.add_subplot(gs[loc]) for loc in locs]

    axes = az.plot_posterior(
        idata,
        var_names=var_names,
        ref_val=ref_val,
        ax=np.array(fig.axes),
        **plot_posterior_kwargs,
    )

    for axis in axes.ravel():
        var_name, *coords = axis.get_title().split("\n")

        if var_name in prior_dict:
            prior_dict[var_name].plot_pdf(ax=axis, legend=False, color="tab:orange")

    return fig


__all__ = [
    "prepare_gridspec_figure",
    "plot_simulation",
    "plot_irf",
    "plot_prior_solvability",
    "plot_eigenvalues",
    "plot_covariance_matrix",
    "plot_acf",
    "plot_corner",
    "plot_kalman_filter",
    "plot_posterior_with_prior",
]
