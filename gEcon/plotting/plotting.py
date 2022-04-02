import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd


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
        data.fill_between(ci_bounds.index,
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
             legend=False, cmap=None, legend_kwargs=None, figsize=(14,10), dpi=100):
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



