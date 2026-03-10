from __future__ import annotations

import functools as ft

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from scipy import linalg

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.statistics.validation import _maybe_solve_model, _validate_shock_options

if TYPE_CHECKING:
    from gEconpy.model.model import Model


def build_Q_matrix(
    model_shocks: list[TimeAwareSymbol],
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
) -> np.array:
    """
    Take different options for user input and reconcile them into a covariance matrix.

    Exactly one or zero of shock_dict or shock_cov_matrix should be provided.

    Parameters
    ----------
    model_shocks: list of str
        List of model shock names, used to infer positions in the covariance matrix.
    shock_std_dict: dict, optional
        Dictionary of shock names and standard deviations to be used to build Q.
    shock_cov_matrix: array, optional
        An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks.
    shock_std: float or sequence of float, optional
        Standard deviation of all model shocks.

    Returns
    -------
    Q: ndarray
        Shock variance-covariance matrix.
    """
    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=model_shocks,
    )

    if shock_cov_matrix is not None:
        return shock_cov_matrix

    if shock_std_dict is not None:
        shock_names = [x.base_name for x in model_shocks]
        indices = [shock_names.index(x) for x in shock_std_dict]
        Q = np.zeros((len(model_shocks), len(model_shocks)))
        for i, (_key, value) in enumerate(shock_std_dict.items()):
            Q[indices[i], indices[i]] = value**2
        return Q

    return np.eye(len(model_shocks)) * shock_std**2


def stationary_covariance_matrix(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    return_df: bool = True,
    **solve_model_kwargs,
) -> np.ndarray | pd.DataFrame:
    """
    Compute the stationary covariance matrix of the solved system.

    Solution is found by solving the associated discrete Lyapunov equation.

    Parameters
    ----------
    model: Model
        DSGE Model associated with T and R.
    T: np.ndarray, optional
        Transition matrix.
    R: np.ndarray, optional
        Selection matrix.
    shock_std_dict: dict, optional
        Shock standard deviations.
    shock_cov_matrix: array, optional
        Shock covariance matrix.
    shock_std: float, optional
        Common shock standard deviation.
    return_df: bool
        If True, return a DataFrame.
    **solve_model_kwargs
        Forwarded to ``solve_model``.

    Returns
    -------
    Sigma : ndarray or DataFrame
    """
    shocks = model.shocks
    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=shocks,
    )

    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
    )

    RQRT = np.linalg.multi_dot([R, Q, R.T])
    Sigma = linalg.solve_discrete_lyapunov(T, RQRT)

    if return_df:
        variables = [x.base_name for x in model.variables]
        Sigma = pd.DataFrame(Sigma, index=variables, columns=variables)

    return Sigma


def _compute_autocovariance_matrix(T, Sigma, n_lags=5, correlation=True):
    """Compute the autocorrelation matrix for the given state-space model.

    Parameters
    ----------
    T: np.ndarray
        Transition matrix.
    Sigma: np.ndarray
        Stationary covariance matrix.
    n_lags : int, optional
        Number of lags.
    correlation: bool
        If True, normalize by standard deviations.

    Returns
    -------
    acov : ndarray
        Shape ``(n_lags, n_variables, n_variables)``.
    """
    n_vars = T.shape[0]
    auto_coors = np.empty((n_lags, n_vars, n_vars))
    std_vec = np.sqrt(np.diag(Sigma))

    normalization_factor = np.outer(std_vec, std_vec) if correlation else np.ones_like(Sigma)

    for i in range(n_lags):
        auto_coors[i] = np.linalg.matrix_power(T, i) @ Sigma / normalization_factor

    return auto_coors


def autocovariance_matrix(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    n_lags: int = 10,
    correlation=False,
    return_xr=True,
    **solve_model_kwargs,
):
    """
    Compute the model's autocovariance matrix using the stationary covariance matrix.

    Alternatively, the autocorrelation matrix can be returned by specifying ``correlation = True``.

    Parameters
    ----------
    model: Model
        DSGE Model associated with T and R.
    T: np.ndarray, optional
        Transition matrix.
    R: np.ndarray, optional
        Selection matrix.
    shock_std_dict: dict, optional
        Shock standard deviations.
    shock_cov_matrix: array, optional
        Shock covariance matrix.
    shock_std: float, optional
        Common shock standard deviation.
    n_lags: int
        Number of lags. Default is 10.
    correlation: bool
        If True, return autocorrelation instead of autocovariance.
    return_xr: bool
        If True, return a DataArray.
    **solve_model_kwargs
        Forwarded to ``solve_model``.

    Returns
    -------
    acorr_mat : DataArray or ndarray
    """
    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    Sigma = stationary_covariance_matrix(
        model,
        T=T,
        R=R,
        shock_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        return_df=False,
        **solve_model_kwargs,
    )
    result = _compute_autocovariance_matrix(T, Sigma, n_lags=n_lags, correlation=correlation)

    if return_xr:
        variables = [x.base_name for x in model.variables]
        result = xr.DataArray(
            result,
            dims=["lag", "variable", "variable_aux"],
            coords={
                "lag": range(n_lags),
                "variable": variables,
                "variable_aux": variables,
            },
        )

    return result


autocorrelation_matrix = ft.partial(autocovariance_matrix, correlation=True)
autocorrelation_matrix.__doc__ = autocovariance_matrix.__doc__
