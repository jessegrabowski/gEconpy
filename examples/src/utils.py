import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  
from itertools import chain


def plot_dfs(dfs, plot_func, dfs_cov: pd.DataFrame=None, when_crisis: pd.Series=None, legend: list=None, no_cols: int=3, figsize: tuple=(5, 2), **kwargs):
    assert type(dfs) == list, "dfs refers to a list of dataframes, please specify as such"
    
    cols = list(set([*chain(*[[col for col in df.columns] for df in dfs])]))
    rows = int(np.ceil(len(cols) / no_cols))

    fig, ax = plt.subplots(rows, no_cols, figsize=(figsize[0]*no_cols, figsize[1]*rows))

    if legend is not None:
            assert len(legend) == len(dfs), "not sufficient legend labels supplied"

    for i_d, df in enumerate(dfs):

        if dfs_cov is not None:
            assert set(df.columns) == set(dfs_cov[i_d].columns), "df columns are not matched by data on confidence intervals"
            
        for i, col in enumerate(df):

            # get axis
            _r, _c = int(np.floor(i/no_cols)), int(i%no_cols)
            if len(ax.shape) == 1: 
                _ax = ax[_c]
            else:
                _ax = ax[_r, _c]

            # plot data series
            if legend is not None:
                plot_func(df[col], ax=_ax, label=legend[i_d], **kwargs)
                _ax.legend()
            else:
                plot_func(df[col], ax=_ax, **kwargs)
            _ax.set_title(col)

            # plot confidence intervals
            if dfs_cov is not None:
                lower, upper = df[col] - 1.96 * dfs_cov[i_d][col], df[col] + 1.96 * dfs_cov[i_d][col]
                _ax.fill_between(df[col].index, lower, upper, color='b', alpha=.1, label=f'conf int')

    # plot recessions
    if (when_crisis is not None):
        for _ax in fig.axes:
            _ax.fill_between(df.index, _ax.get_ylim()[0], _ax.get_ylim()[1], where=when_crisis, color='red', alpha=.2, label='recession', linewidth=0)

    fig.tight_layout()
    plt.show()
    return fig