import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pandas as pd
import numpy as np
from scipy import stats

from gEcon.sampling.prior_utilities import prior_steady_state_check
from gEcon.classes.progress_bar import ProgressBar

from itertools import combinations_with_replacement


def prepare_gridspec_figure(n_cols, n_plots):
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
        for j in range(n_cols):
            col_slice = slice(left_pad + j * 2, left_pad + (j + 1) * 2)
            plot_locs.append((last_row, col_slice))

    return gs, plot_locs


def _plot_single_variable(data, ax, ci=None, cmap=None, fill_color='tab:blue'):
    if ci is None:
        data.plot(ax=ax, legend=False, cmap=cmap)

    else:
        q_low, q_high = ((1 - ci) / 2), 1 - ((1 - ci) / 2)
        ci_bounds = data.quantile([q_low, q_high], axis=1).T

        data.mean(axis=1).plot(ax=ax, legend=False, cmap=cmap)
        ci_bounds.plot(ax=ax, ls='--', lw=0.5, color='k', legend=False)
        ax.fill_between(ci_bounds.index,
                          y1=ci_bounds.iloc[:, 0],
                          y2=ci_bounds.iloc[:, 1],
                          color=fill_color,
                          alpha=0.25)


def plot_simulation(simulation, vars_to_plot=None, ci=None, n_cols=None, cmap=None, fill_color=None,
                    figsize=(12, 8),
                    dpi=100):
    if vars_to_plot is None:
        vars_to_plot = simulation.index
    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, variable in enumerate(vars_to_plot):
        axis = fig.add_subplot(gs[plot_locs[idx]])

        _plot_single_variable(simulation.loc[variable].unstack(1), ci=ci, ax=axis, cmap=cmap, fill_color=fill_color)

        axis.set(title=variable)
        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls='--', lw=0.5)

    fig.tight_layout()
    return fig


def plot_irf(irf, vars_to_plot=None, shocks_to_plot=None, n_cols=None,
             legend=False, cmap=None, legend_kwargs=None, figsize=(14, 10), dpi=100):
    if vars_to_plot is None:
        vars_to_plot = irf.index
    if shocks_to_plot is None:
        shocks_to_plot = irf.columns.get_level_values(1).unique()

    n_plots = len(vars_to_plot)
    n_cols = min(4, n_plots) if n_cols is None else n_cols

    gs, plot_locs = prepare_gridspec_figure(n_cols, n_plots)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, variable in enumerate(vars_to_plot):
        axis = fig.add_subplot(gs[plot_locs[idx]])

        _plot_single_variable(irf.loc[variable, pd.IndexSlice[:, shocks_to_plot]].unstack(1),
                              ax=axis, cmap=cmap)

        axis.set(title=variable)
        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls='--', lw=0.5)

    fig.tight_layout()

    if legend:
        if legend_kwargs is None:
            legend_kwargs = {'ncol': min(4, len(shocks_to_plot)),
                             'loc': 'center',
                             'bbox_to_anchor': (0.5, 1.05),
                             'bbox_transform': fig.transFigure}

        fig.axes[0].legend(**legend_kwargs)

    return fig


def plot_prior_steady_state_solvability(model, n_samples=1_000, seed=None, param_subset=None):
    data = prior_steady_state_check(model, n_samples, seed)
    data = data.loc[:, ~(data == data.loc[0, :]).all(axis=0)].copy()
    params = data.columns[:-1]
    n_params = len(params) if param_subset is None else len(param_subset)

    fig, axes = plt.subplots(n_params, n_params, figsize=(8, 8), dpi=100)

    if param_subset is None:
        param_pairs = list(combinations_with_replacement(params, 2))
    else:
        param_pairs = list(combinations_with_replacement(data.columns[param_subset], 2))

    plot_grid = np.arange(1, n_params ** 2 + 1).reshape((n_params, n_params))
    plot_grid[np.tril_indices(n_params, k=-1)] = 0

    plot_idxs = np.where(plot_grid)
    blank_idxs = np.where(plot_grid == 0)

    for col, row in zip(*blank_idxs):
        axes[row][col].set_visible(False)

    for col, row, pair in zip(*plot_idxs, param_pairs):
        param_1, param_2 = pair
        axis = axes[row][col]
        if param_1 == param_2:

            X_sorted = data[param_1].sort_values()
            X_success = X_sorted[data['success']]
            X_failure = X_sorted[~data['success']]

            success_grid = np.linspace(X_success.min() * 0.9, X_success.max() * 1.1, 100)
            failure_grid = np.linspace(X_failure.min() * 0.9, X_failure.max() * 1.1, 100)

            d_success = stats.kde.gaussian_kde(X_success)
            d_failure = stats.kde.gaussian_kde(X_failure)

            axis.plot(success_grid, d_success.pdf(success_grid), color='tab:blue')
            axis.plot(failure_grid, d_failure.pdf(failure_grid), color='tab:red')
            xmin, xmax = axis.get_xlim()
            axis.fill_between(x=success_grid,
                              y1=d_success.pdf(success_grid),
                              y2=0, color='tab:blue', alpha=0.25)

            axis.fill_between(x=failure_grid,
                              y1=d_failure.pdf(failure_grid),
                              y2=0, color='tab:red', alpha=0.25)

        else:
            axis.scatter(data.loc[data.success, param_1],
                         data.loc[data.success, param_2],
                         c='tab:blue',
                         s=10,
                         label='Steady State Successful')
            axis.scatter(data.loc[~data.success, param_1],
                         data.loc[~data.success, param_2],
                         c='tab:red',
                         s=10,
                         label='Steady State Failed')

        if col == 0:
            axis.set_ylabel(param_2)
        if row == n_params - 1:
            axis.set_xlabel(param_1)

        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls='--', lw=0.5)

    axes[1][0].legend(loc='center',
                      bbox_to_anchor=(0.5, 0.91),
                      bbox_transform=fig.transFigure,
                      ncol=2,
                      fontsize=8,
                      frameon=False)
    fig.suptitle('Steady State Solution Found by Parameter Values', y=0.95)
    plt.show()


def plot_eigenvalues(model, figsize=None, dpi=None):
    if figsize is None:
        figsize = (5, 5)
    if dpi is None:
        dpi = 100

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    data = model.check_bk_condition(verbose=False)
    n_infinity = (data.Modulus > 10).sum()

    data = data[data.Modulus < 10]

    x_circle = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

    ax.plot(np.cos(x_circle), np.sin(x_circle), color='k', lw=1)
    ax.set_aspect('equal')
    colors = ['tab:red' if x > 1.0 else 'tab:blue' for x in data.Modulus]
    ax.scatter(data.Real, data.Imaginary, color=colors, s=50, lw=1, edgecolor='k')
    [spine.set_visible(False) for spine in ax.spines.values()]
    ax.grid(ls='--', lw=0.5)
    ax.set_title(f'Eigenvalues of Model Solution\n{n_infinity} Eigenvalues with Infinity Modulus not shown.')
    plt.show()
