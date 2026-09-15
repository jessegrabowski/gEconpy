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
) -> np.ndarray:
    """
    Build the shock covariance matrix from one of three user-facing specifications.

    Exactly one of ``shock_std_dict``, ``shock_cov_matrix``, and ``shock_std`` must be given.

    Parameters
    ----------
    model_shocks : list of TimeAwareSymbol
        Model shocks, in the order that fixes the rows and columns of the covariance matrix.
    shock_std_dict : dict of str to float, optional
        Standard deviation of every shock, keyed by shock name. Defaults to None.
    shock_cov_matrix : ndarray, optional
        Covariance matrix of shape ``(n_shocks, n_shocks)``, returned as is. Defaults to None.
    shock_std : float or sequence of float, optional
        Standard deviation shared by every shock, or one standard deviation per shock. Defaults to None.

    Returns
    -------
    Q : ndarray
        Shock covariance matrix of shape ``(n_shocks, n_shocks)``.

    Examples
    --------
    Build a diagonal covariance matrix from per-shock standard deviations. The row and column order follows
    ``model.shocks``:

    .. code-block:: python

        from gEconpy import build_Q_matrix, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        Q = build_Q_matrix(model.shocks, shock_std_dict={"epsilon_A": 0.01})
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
        Q = np.zeros((len(model_shocks), len(model_shocks)))
        for name, std in shock_std_dict.items():
            position = shock_names.index(name)
            Q[position, position] = std**2
        return Q

    std = np.broadcast_to(np.asarray(shock_std, dtype=float), (len(model_shocks),))
    return np.diag(std**2)


def stationary_covariance_matrix(
    model: "Model",
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    return_df: bool = True,
    **solve_model_kwargs,
) -> np.ndarray | pd.DataFrame:
    r"""
    Compute the stationary covariance matrix of the solved model.

    The stationary covariance :math:`\Sigma` solves the discrete Lyapunov equation

    .. math::

        \Sigma = T \Sigma T^\top + R Q R^\top

    where :math:`Q` is the shock covariance matrix built by :func:`build_Q_matrix`. Exactly one of ``shock_std_dict``,
    ``shock_cov_matrix``, and ``shock_std`` must be given.

    Parameters
    ----------
    model : Model
        DSGE model whose solution is T and R.
    T : ndarray, optional
        Transition matrix. Solved from ``model`` when None. Defaults to None.
    R : ndarray, optional
        Selection matrix. Solved from ``model`` when None. Defaults to None.
    shock_std_dict : dict of str to float, optional
        Standard deviation of every shock, keyed by shock name. Defaults to None.
    shock_cov_matrix : ndarray, optional
        Covariance matrix of shape ``(n_shocks, n_shocks)``. Defaults to None.
    shock_std : float or sequence of float, optional
        Standard deviation shared by every shock, or one standard deviation per shock. Defaults to None.
    return_df : bool, optional
        Return a DataFrame labeled with variable names. Defaults to True.
    **solve_model_kwargs
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model` when T and R are solved here.

    Returns
    -------
    Sigma : ndarray or DataFrame
        Stationary covariance matrix of shape ``(n_variables, n_variables)``.

    Examples
    --------
    Compute the unconditional variances of the model variables under a common shock standard deviation:

    .. code-block:: python

        import numpy as np

        from gEconpy import model_from_gcn, stationary_covariance_matrix
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        Sigma = stationary_covariance_matrix(model, shock_std=0.01, verbose=False)
        standard_deviations = np.sqrt(np.diag(Sigma))
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


def autocovariance_matrix(
    model: "Model",
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    n_lags: int = 10,
    correlation: bool = False,
    return_xr: bool = True,
    **solve_model_kwargs,
) -> xr.DataArray | np.ndarray:
    r"""
    Compute the autocovariance matrices of the solved model at lags 0 through ``n_lags - 1``.

    The autocovariance at lag :math:`k` is :math:`T^k \Sigma`, with :math:`\Sigma` the stationary covariance from
    :func:`stationary_covariance_matrix`. Exactly one of ``shock_std_dict``, ``shock_cov_matrix``, and ``shock_std``
    must be given.

    Parameters
    ----------
    model : Model
        DSGE model whose solution is T and R.
    T : ndarray, optional
        Transition matrix. Solved from ``model`` when None. Defaults to None.
    R : ndarray, optional
        Selection matrix. Solved from ``model`` when None. Defaults to None.
    shock_std_dict : dict of str to float, optional
        Standard deviation of every shock, keyed by shock name. Defaults to None.
    shock_cov_matrix : ndarray, optional
        Covariance matrix of shape ``(n_shocks, n_shocks)``. Defaults to None.
    shock_std : float or sequence of float, optional
        Standard deviation shared by every shock, or one standard deviation per shock. Defaults to None.
    n_lags : int, optional
        Number of lags, starting from lag 0. Defaults to 10.
    correlation : bool, optional
        Normalize each entry by the product of the two variables' standard deviations, giving autocorrelations.
        Defaults to False.
    return_xr : bool, optional
        Return a DataArray with ``lag``, ``variable``, and ``variable_aux`` coordinates. Defaults to True.
    **solve_model_kwargs
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model` when T and R are solved here.

    Returns
    -------
    acov : DataArray or ndarray
        Autocovariance matrices of shape ``(n_lags, n_variables, n_variables)``.

    Examples
    --------
    Compute the autocovariance between capital and output at each lag:

    .. code-block:: python

        from gEconpy import autocovariance_matrix, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        acov = autocovariance_matrix(model, shock_std=0.01, n_lags=5, verbose=False)
        capital_output = acov.sel(variable="K", variable_aux="Y")
    """
    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    Sigma = stationary_covariance_matrix(
        model,
        T=T,
        R=R,
        shock_std_dict=shock_std_dict,
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


def autocorrelation_matrix(
    model: "Model",
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    n_lags: int = 10,
    return_xr: bool = True,
    **solve_model_kwargs,
) -> xr.DataArray | np.ndarray:
    """
    Compute the autocorrelation matrices of the solved model at lags 0 through ``n_lags - 1``.

    Equivalent to :func:`autocovariance_matrix` with ``correlation=True``. Exactly one of ``shock_std_dict``,
    ``shock_cov_matrix``, and ``shock_std`` must be given.

    Parameters
    ----------
    model : Model
        DSGE model whose solution is T and R.
    T : ndarray, optional
        Transition matrix. Solved from ``model`` when None. Defaults to None.
    R : ndarray, optional
        Selection matrix. Solved from ``model`` when None. Defaults to None.
    shock_std_dict : dict of str to float, optional
        Standard deviation of every shock, keyed by shock name. Defaults to None.
    shock_cov_matrix : ndarray, optional
        Covariance matrix of shape ``(n_shocks, n_shocks)``. Defaults to None.
    shock_std : float or sequence of float, optional
        Standard deviation shared by every shock, or one standard deviation per shock. Defaults to None.
    n_lags : int, optional
        Number of lags, starting from lag 0. Defaults to 10.
    return_xr : bool, optional
        Return a DataArray with ``lag``, ``variable``, and ``variable_aux`` coordinates. Defaults to True.
    **solve_model_kwargs
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model` when T and R are solved here.

    Returns
    -------
    acorr : DataArray or ndarray
        Autocorrelation matrices of shape ``(n_lags, n_variables, n_variables)``.

    Examples
    --------
    Read off the persistence of output as its autocorrelation with itself at each lag:

    .. code-block:: python

        from gEconpy import autocorrelation_matrix, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        acorr = autocorrelation_matrix(model, shock_std=0.01, n_lags=5, verbose=False)
        output_persistence = acorr.sel(variable="Y", variable_aux="Y")
    """
    return autocovariance_matrix(
        model,
        T=T,
        R=R,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        n_lags=n_lags,
        correlation=True,
        return_xr=return_xr,
        **solve_model_kwargs,
    )


def _compute_autocovariance_matrix(
    T: np.ndarray, Sigma: np.ndarray, n_lags: int = 5, correlation: bool = True
) -> np.ndarray:
    n_vars = T.shape[0]
    autocovariances = np.empty((n_lags, n_vars, n_vars))
    std_vec = np.sqrt(np.diag(Sigma))

    normalization_factor = np.outer(std_vec, std_vec) if correlation else np.ones_like(Sigma)

    for i in range(n_lags):
        autocovariances[i] = np.linalg.matrix_power(T, i) @ Sigma / normalization_factor

    return autocovariances
