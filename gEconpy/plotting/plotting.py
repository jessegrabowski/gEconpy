from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from scipy import stats


class ScalarFormatterForceFormat(ScalarFormatter):
    """
    A ScalarFormatter that forces a specific format for the tick labels.
    """

    def _set_format(self, vmin, vmax):
        """
        Set the format for the tick labels.

        Parameters
        ----------
        vmin : float
            The minimum value of the data.
        vmax : float
            The maximum value of the data.
        """
        self.format = "%1.1f"


def prepare_gridspec_figure(n_cols: int, n_plots: int) -> Tuple[GridSpec, List]:
    """
     Prepare a figure with a grid of subplots. Centers the last row of plots if the number of plots is not square.

     Parameters
     ----------
     n_cols : int
         The number of columns in the grid.
     n_plots : int
         The number of subplots in the grid.

     Returns
     -------
     GridSpec
         A matplotlib GridSpec object representing the layout of the grid.
    list of tuple(slice, slice)
         A list of tuples of slices representing the indices of the grid cells to be used for each subplot.
    """

    remainder = n_plots % n_cols
    has_remainder = remainder > 0
    n_rows = n_plots // n_cols + 1

    gs = GridSpec(2 * n_rows, 2 * n_cols)
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


def _plot_single_variable(data, ax, ci=None, cmap=None, fill_color="tab:blue"):
    """
    Plot the mean and optionally a confidence interval for a single variable.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with one or more columns containing the data to plot.
    ax : Matplotlib Axes
        The Axes object to plot on.
    ci : float, optional
        The confidence interval to plot, between 0 and 1. If not provided, only the mean will be plotted.
    cmap : str or Colormap, optional
        The color map to use for the data.
    fill_color : str, optional
        The color to use to fill the confidence interval.

    Returns
    -------
    None
    """

    if ci is None:
        data.plot(ax=ax, legend=False, cmap=cmap)

    else:
        q_low, q_high = ((1 - ci) / 2), 1 - ((1 - ci) / 2)
        ci_bounds = data.quantile([q_low, q_high], axis=1).T

        data.mean(axis=1).plot(ax=ax, legend=False, cmap=cmap)
        ci_bounds.plot(ax=ax, ls="--", lw=0.5, color="k", legend=False)
        ax.fill_between(
            ci_bounds.index,
            y1=ci_bounds.iloc[:, 0],
            y2=ci_bounds.iloc[:, 1],
            color=fill_color,
            alpha=0.25,
        )


def plot_simulation(
    simulation: pd.DataFrame,
    vars_to_plot: Optional[List[str]] = None,
    ci: Optional[float] = None,
    n_cols: Optional[int] = None,
    cmap: Optional[Union[str, Colormap]] = None,
    fill_color: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
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
        vars_to_plot = simulation.index
    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, variable in enumerate(vars_to_plot):
        axis = fig.add_subplot(gs[plot_locs[idx]])

        _plot_single_variable(
            simulation.loc[variable].unstack(1),
            ci=ci,
            ax=axis,
            cmap=cmap,
            fill_color=fill_color,
        )

        axis.set(title=variable)
        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)

    fig.tight_layout()
    return fig


def plot_irf(
    irf: pd.DataFrame,
    vars_to_plot: Optional[List[str]] = None,
    shocks_to_plot: Optional[List[str]] = None,
    n_cols: Optional[int] = None,
    legend: bool = False,
    cmap: Optional[Union[str, Colormap]] = None,
    legend_kwargs: Optional[Dict] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot the impulse response functions for a set of variables.

    Parameters
    ----------
    irf : pd.DataFrame
        A DataFrame with the impulse response functions. The index should contain the variables to plot, and the columns
        should contain the shocks, with a multi-index for the period and shock type.
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

    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, variable in enumerate(vars_to_plot):
        axis = fig.add_subplot(gs[plot_locs[idx]])

        _plot_single_variable(
            irf.loc[variable, pd.IndexSlice[:, shocks_to_plot]].unstack(1),
            ax=axis,
            cmap=cmap,
        )

        axis.set(title=variable)
        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)

    fig.tight_layout()

    if legend:
        if legend_kwargs is None:
            legend_kwargs = {
                "ncol": min(4, len(shocks_to_plot)),
                "loc": "center",
                "bbox_to_anchor": (0.5, 1.05),
                "bbox_transform": fig.transFigure,
            }

        fig.axes[0].legend(**legend_kwargs)

    return fig


def plot_prior_solvability(
    data: pd.DataFrame,
    n_samples: int = 1_000,
    seed: Optional[int] = None,
    plotting_subset: Optional[List[str]] = None,
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
    plotting_subset : List[str], optional
        A list of parameter names to include in the plots. If not provided, all parameters will be plotted.

    Returns
    -------
    fig : Matplotlib Figure
        The Figure object containing the plots

    Notes
    ----------
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
    n_params = len(params) if plotting_subset is None else len(plotting_subset)

    plot_data["success"] = failure_step.isna()
    fig, axes = plt.subplots(n_params, n_params, figsize=(16, 16), dpi=100)

    if plotting_subset is None:
        param_pairs = list(combinations_with_replacement(params, 2))
    else:
        param_pairs = list(combinations_with_replacement(plotting_subset, 2))

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


def plot_eigenvalues(model: Any, figsize: Tuple[float, float] = None, dpi: int = None):
    """
    Plot the eigenvalues of the model solution, along with a unit circle. Eigenvalues with modulus greater than 1 are
    shown in red, while those with modulus less than 1 are shown in blue. Eigenvalues greater than 10 in modulus
    are not drawn.

    Parameters
    ----------
    model : gEconModel
        The model to plot the eigenvalues of.
    figsize : Tuple[float, float], optional
        The size of the figure to create.
    dpi : int, optional
        The resolution of the figure to create.

    Returns
    -------
    Matplotlib Figure
        The figure object containing the plot.
    """

    if figsize is None:
        figsize = (5, 5)
    if dpi is None:
        dpi = 100

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    data = model.check_bk_condition(verbose=False)
    n_infinity = (data.Modulus > 10).sum()

    data = data[data.Modulus < 10]

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
    vars_to_plot: Optional[List[str]] = None,
    cbarlabel: str = "Covariance",
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 100,
    cbar_kw: Optional[Dict] = None,
    cmap: str = "YlGn",
    annotation_fontsize: int = 8,
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
    annotation_fontsize : int, optional
        The font size for the annotation in the heatmap cells.

    Returns
    -------
    matplotlib.figure.Figure
        A figure containing the heatmap.
    """

    if vars_to_plot is None:
        vars_to_plot = data.columns

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im, cbar = plot_heatmap(
        data.loc[vars_to_plot, vars_to_plot],
        ax=ax,
        cbar_kw=cbar_kw,
        cmap=cmap,
        cbarlabel=cbarlabel,
    )
    annotate_heatmap(im, valfmt="{x:.2f}", fontsize=annotation_fontsize)

    fig.tight_layout()
    return fig


def plot_heatmap(
    data: pd.DataFrame,
    ax: Optional[Any] = None,
    cbar_kw: Optional[dict] = None,
    cbarlabel: Optional[str] = "",
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

    if not cbar_kw:
        cbar_kw = {}

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

    if not isinstance(data, (list, np.ndarray)):
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
    acorr_matrix: pd.DataFrame,
    vars_to_plot: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = (14, 4),
    dpi: Optional[int] = 100,
    n_cols: Optional[int] = 4,
) -> plt.Figure:
    """
    Plot the autocorrelation function for a set of variables.

    Parameters
    ----------
    acorr_matrix: pandas.DataFrame
        Matrix of autocorrelation values. Rows represent variables and columns represent lags.
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

    if vars_to_plot is None:
        vars_to_plot = acorr_matrix.index

    n_plots = len(vars_to_plot)
    n_cols = min(n_cols, n_plots)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gc, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_plots)

    x_values = acorr_matrix.columns

    for variable, plot_loc in zip(vars_to_plot, plot_locs):
        axis = fig.add_subplot(gc[plot_loc])
        axis.scatter(x_values, acorr_matrix.loc[variable, :])
        axis.vlines(x_values, 0, acorr_matrix.loc[variable, :])

        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)
        axis.set(title=variable)

    fig.tight_layout()
    return fig


def plot_corner(
    idata: Any,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 14),
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
    var_names : List[str], optional
        A list of strings specifying the variables to plot. If not provided, all variables in `idata` will be plotted.
    figsize : Tuple[int, int], optional
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
    ----------
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
    plt.show()


def plot_kalman_filter(
    idata: xr.Dataset,
    data: pd.DataFrame,
    kalman_output: str = "predicted",
    n_cols: Optional[int] = None,
    vars_to_plot: Optional[List[str]] = None,
    fig: Optional[Figure] = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 144,
    cmap: Optional[str] = None,
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
    kalman_output = kalman_output.capitalize()

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    if vars_to_plot is None:
        vars_to_plot = idata.coords["variable"].values

    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots)
    time_idx = idata.coords["time"]
    time_slice = (
        slice(None, None, None)
        if kalman_output.lower() == "predicted"
        else slice(1, None, None)
    )

    for idx, variable in enumerate(vars_to_plot):
        axis = fig.add_subplot(gs[plot_locs[idx]])

        mu = idata[f"{kalman_output}_State"].dropna(dim="time").sel(variable=variable)

        q05, q50, q95 = mu.quantile([0.05, 0.5, 0.95], dim="sample")

        sigma = (
            idata[f"{kalman_output}_Cov"]
            .dropna(dim="time")
            .sel(variable=variable, variable2=variable)
        )
        s05, s95 = sigma.quantile([0.05, 0.95], dim="sample")

        top_ci = mu + 1.98 * np.sqrt(s05 + 1e-6)
        bot_ci = mu - 1.98 * np.sqrt(s95 + 1e-6)

        axis.plot(time_idx[time_slice], q50.values, color="tab:red")
        axis.fill_between(time_idx[time_slice], q05, q95, color="tab:blue", alpha=1)
        axis.fill_between(
            time_idx[time_slice],
            top_ci.max(dim=["sample"]),
            bot_ci.min(dim=["sample"]),
            color="0.5",
            alpha=0.5,
        )

        if variable in data.columns:
            data[variable].plot(ax=axis, color="k", ls="--", lw=2)

        axis.set(title=variable, xlabel=None, ylabel="% Deviation from SS")
        axis.tick_params(axis="x", rotation=45)

    fig.tight_layout()
