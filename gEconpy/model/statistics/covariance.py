from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from scipy import linalg

from gEconpy.classes.perturbation_solution import FirstOrderPerturbationSolution, SecondOrderPerturbationSolution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.pruning import compile_pruned_second_order_moments
from gEconpy.model.statistics.validation import _given_or_solved, _maybe_solve_model, _validate_shock_options

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
    solution: FirstOrderPerturbationSolution | None = None,
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

    A second-order solution gives the covariance of its pruned deviations, from the linear state-space form of
    :func:`~gEconpy.model.pruning.pruned_second_order_system`.

    Parameters
    ----------
    model : Model
        DSGE model whose solution is T and R.
    solution : FirstOrderPerturbationSolution or SecondOrderPerturbationSolution, optional
        Solution from :meth:`~gEconpy.model.model.Model.solve_model`. Solved from ``model`` when None. Defaults to
        None.
    T : ndarray, optional
        Transition matrix. Deprecated, pass ``solution`` instead. Defaults to None.
    R : ndarray, optional
        Selection matrix. Deprecated, pass ``solution`` instead. Defaults to None.
    shock_std_dict : dict of str to float, optional
        Standard deviation of every shock, keyed by shock name. Defaults to None.
    shock_cov_matrix : ndarray, optional
        Covariance matrix of shape ``(n_shocks, n_shocks)``. Defaults to None.
    shock_std : float or sequence of float, optional
        Standard deviation shared by every shock, or one standard deviation per shock. Defaults to None.
    return_df : bool, optional
        Return a DataFrame labeled with variable names. Defaults to True.
    **solve_model_kwargs
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model` when the model is solved here.

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

    Sigma = _model_autocovariances(
        model,
        solution,
        T,
        R,
        n_lags=1,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        **solve_model_kwargs,
    )[0]

    if return_df:
        variables = [x.base_name for x in model.variables]
        Sigma = pd.DataFrame(Sigma, index=variables, columns=variables)

    return Sigma


def stationary_mean(
    model: "Model",
    solution: FirstOrderPerturbationSolution | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    **solve_model_kwargs,
) -> xr.Dataset:
    r"""
    Compute the stationary mean of the solved model, as a deviation from the steady state and in levels.

    A first-order solution has a stationary mean of zero deviation, the steady state. A second-order solution's mean
    is that of its pruned deviations, which the risk correction and the variance of the shocks move away from the
    steady state. Exactly one of ``shock_std_dict``, ``shock_cov_matrix``, and ``shock_std`` must be given.

    A log-linearized variable's deviation is a log deviation, so its mean deviation is
    :math:`E[\log y] - \log y_{ss}`, and its ``level`` is ``steady_state * exp(deviation)``, the exponential of the
    mean log. That is not the mean of the level, which also depends on the variance. Every other variable's level is
    ``steady_state + deviation``.

    Parameters
    ----------
    model : Model
        DSGE model whose variables label the output.
    solution : FirstOrderPerturbationSolution or SecondOrderPerturbationSolution, optional
        Solution from :meth:`~gEconpy.model.model.Model.solve_model`. Solved from ``model`` when None. Defaults to
        None.
    shock_std_dict : dict of str to float, optional
        Standard deviation of every shock, keyed by shock name. Defaults to None.
    shock_cov_matrix : ndarray, optional
        Covariance matrix of shape ``(n_shocks, n_shocks)``. Defaults to None.
    shock_std : float or sequence of float, optional
        Standard deviation shared by every shock, or one standard deviation per shock. Defaults to None.
    **solve_model_kwargs
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model` when the model is solved here.

    Returns
    -------
    mean : Dataset
        Variables ``deviation`` and ``level`` over ``variable``, with the steady state and the log-linearization
        flags as coordinates.

    Examples
    --------
    Compute the stationary mean of a second-order solution as a deviation from the steady state:

    .. code-block:: python

        from gEconpy import model_from_gcn, stationary_mean
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        mean = stationary_mean(model, shock_std=0.01, order=2, verbose=False)
        mean_deviation = mean["deviation"]
    """
    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=model.shocks,
    )
    solution = _given_or_solved(model, solution, T=None, R=None, purpose="compute moments from", **solve_model_kwargs)

    if isinstance(solution, SecondOrderPerturbationSolution):
        deviation, _ = _pruned_moments(
            model,
            solution,
            n_lags=1,
            shock_std_dict=shock_std_dict,
            shock_cov_matrix=shock_cov_matrix,
            shock_std=shock_std,
        )
    else:
        deviation = np.zeros(len(solution.variable_names))

    return xr.Dataset(
        {"deviation": ("variable", deviation), "level": ("variable", solution.to_levels(deviation))},
        coords={
            "variable": solution.variable_names,
            "steady_state": ("variable", solution.steady_state),
            "log_linearized": ("variable", solution.log_linearized),
        },
    )


def autocovariance_matrix(
    model: "Model",
    solution: FirstOrderPerturbationSolution | None = None,
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
    must be given. A second-order solution gives the autocovariances of its pruned deviations.

    Parameters
    ----------
    model : Model
        DSGE model whose solution is T and R.
    solution : FirstOrderPerturbationSolution or SecondOrderPerturbationSolution, optional
        Solution from :meth:`~gEconpy.model.model.Model.solve_model`. Solved from ``model`` when None. Defaults to
        None.
    T : ndarray, optional
        Transition matrix. Deprecated, pass ``solution`` instead. Defaults to None.
    R : ndarray, optional
        Selection matrix. Deprecated, pass ``solution`` instead. Defaults to None.
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
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model` when the model is solved here.

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
    autocovariances = _model_autocovariances(
        model,
        solution,
        T,
        R,
        n_lags=n_lags,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        **solve_model_kwargs,
    )
    if correlation:
        autocovariances = _to_correlation(autocovariances)
    return _label_lags(model, autocovariances) if return_xr else autocovariances


def autocorrelation_matrix(
    model: "Model",
    solution: FirstOrderPerturbationSolution | None = None,
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
    ``shock_cov_matrix``, and ``shock_std`` must be given. A second-order solution gives the autocorrelations of its
    pruned deviations.

    Parameters
    ----------
    model : Model
        DSGE model whose solution is T and R.
    solution : FirstOrderPerturbationSolution or SecondOrderPerturbationSolution, optional
        Solution from :meth:`~gEconpy.model.model.Model.solve_model`. Solved from ``model`` when None. Defaults to
        None.
    T : ndarray, optional
        Transition matrix. Deprecated, pass ``solution`` instead. Defaults to None.
    R : ndarray, optional
        Selection matrix. Deprecated, pass ``solution`` instead. Defaults to None.
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
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model` when the model is solved here.

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
    autocorrelations = _to_correlation(
        _model_autocovariances(
            model,
            solution,
            T,
            R,
            n_lags=n_lags,
            shock_std_dict=shock_std_dict,
            shock_cov_matrix=shock_cov_matrix,
            shock_std=shock_std,
            **solve_model_kwargs,
        )
    )
    return _label_lags(model, autocorrelations) if return_xr else autocorrelations


def _stationary_covariance(
    T: np.ndarray,
    R: np.ndarray,
    shocks: list[TimeAwareSymbol],
    shock_std_dict: dict[str, float] | None,
    shock_cov_matrix: np.ndarray | None,
    shock_std: np.ndarray | list | float | None,
) -> np.ndarray:
    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
    )
    return linalg.solve_discrete_lyapunov(T, np.linalg.multi_dot([R, Q, R.T]))


def _model_autocovariances(
    model: "Model",
    solution: FirstOrderPerturbationSolution | np.ndarray | None,
    T: np.ndarray | None,
    R: np.ndarray | None,
    n_lags: int,
    shock_std_dict: dict[str, float] | None,
    shock_cov_matrix: np.ndarray | None,
    shock_std: np.ndarray | list | float | None,
    **solve_model_kwargs,
) -> np.ndarray:
    """Return the autocovariances of the solved model at lags 0 through ``n_lags - 1``, in model order."""
    solution = _given_or_solved(model, solution, T, R, purpose="compute moments from", **solve_model_kwargs)
    if isinstance(solution, SecondOrderPerturbationSolution):
        _, autocovariances = _pruned_moments(
            model,
            solution,
            n_lags=n_lags,
            shock_std_dict=shock_std_dict,
            shock_cov_matrix=shock_cov_matrix,
            shock_std=shock_std,
        )
        return autocovariances

    # The deprecation warning skips this frame and the public function to point at the user's line.
    T, R = _maybe_solve_model(model, solution, T, R, stacklevel=4, **solve_model_kwargs)
    Sigma = _stationary_covariance(
        T, R, shocks=model.shocks, shock_std_dict=shock_std_dict, shock_cov_matrix=shock_cov_matrix, shock_std=shock_std
    )
    return _compute_autocovariance_matrix(T, Sigma, n_lags=n_lags)


def _label_lags(model: "Model", autocovariances: np.ndarray) -> xr.DataArray:
    variables = [x.base_name for x in model.variables]
    return xr.DataArray(
        autocovariances,
        dims=["lag", "variable", "variable_aux"],
        coords={"lag": range(len(autocovariances)), "variable": variables, "variable_aux": variables},
    )


def _to_correlation(autocovariances: np.ndarray) -> np.ndarray:
    if len(autocovariances) == 0:
        return autocovariances
    standard_deviations = np.sqrt(np.diag(autocovariances[0]))
    return autocovariances / np.outer(standard_deviations, standard_deviations)


def _pruned_moments(
    model: "Model",
    solution: SecondOrderPerturbationSolution,
    n_lags: int,
    shock_std_dict: dict[str, float] | None,
    shock_cov_matrix: np.ndarray | None,
    shock_std: np.ndarray | list | float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the stationary mean and autocovariances of the pruned deviations, in model order."""
    state_transition = solution.ghx[solution.dr_order.lag_positions]
    if np.any(np.abs(np.linalg.eigvals(state_transition)) >= 1):
        raise ValueError("The first-order state transition is not stable, so the pruned moments do not exist.")

    shock_cov = build_Q_matrix(
        model_shocks=model.shocks, shock_std_dict=shock_std_dict, shock_cov_matrix=shock_cov_matrix, shock_std=shock_std
    )
    mean, autocovariances = compile_pruned_second_order_moments()(
        ghx=solution.ghx,
        ghu=solution.ghu,
        ghxx=solution.ghxx,
        ghxu=solution.ghxu,
        ghuu=solution.ghuu,
        ghs2_operator=solution.ghs2_operator,
        shock_cov=np.asarray(shock_cov, dtype=float),
        state_positions=solution.dr_order.lag_positions,
        n_lags=n_lags,
    )
    model_order = solution.dr_order.inv_var_order
    autocovariances = autocovariances.reshape(n_lags, mean.shape[0], mean.shape[0])
    return mean[model_order], autocovariances[:, model_order][:, :, model_order]


def _compute_autocovariance_matrix(T: np.ndarray, Sigma: np.ndarray, n_lags: int) -> np.ndarray:
    n_vars = T.shape[0]
    autocovariances = np.empty((n_lags, n_vars, n_vars))
    for i in range(n_lags):
        autocovariances[i] = np.linalg.matrix_power(T, i) @ Sigma
    return autocovariances
