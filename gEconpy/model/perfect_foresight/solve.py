from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import cache, partial

import numpy as np
import pandas as pd
import xarray as xr

from scipy import sparse
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import spsolve

from gEconpy.exceptions import NonFiniteInitialGuessError
from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight.assemble import assemble_stacked_jacobian
from gEconpy.model.perfect_foresight.compile import (
    PerfectForesightProblem,
    compile_perfect_foresight_problem,
)
from gEconpy.model.perfect_foresight.homotopy import _solve_by_homotopy
from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs
from gEconpy.rcparams import rcParams
from gEconpy.solvers.sparse_root import newton_armijo
from gEconpy.solvers.sparse_root.base import RootSolver
from gEconpy.solvers.sparse_root.direction import ChordDirection, NewtonDirection
from gEconpy.solvers.sparse_root.dogleg import SparseDogleg
from gEconpy.solvers.sparse_root.line_search import LineSearchSolver
from gEconpy.solvers.sparse_root.sparse_root import _run_sparse_root


def solve_perfect_foresight(
    model: Model,
    simulation_length: int,
    x0: np.ndarray | pd.DataFrame | dict[str, float] | None = None,
    initial_conditions: dict[str, float] | xr.DataArray | None = None,
    terminal_conditions: dict[str, float] | xr.DataArray | None = None,
    shocks: dict[str, np.ndarray] | None = None,
    param_paths: dict[str, float | Sequence[float] | np.ndarray] | None = None,
    compile_kwargs: dict | None = None,
    solver: RootSolver | None = None,
    steady_state_kwargs: dict | None = None,
    sparse_root_kwargs: dict | None = None,
    max_homotopy_solves: int = 100,
) -> tuple[pd.DataFrame, OptimizeResult]:
    """
    Solve the nonlinear model under perfect foresight over a finite horizon.

    The equations of every period are stacked into one system in the ``simulation_length * n_vars`` unknowns and
    solved with a sparse Newton method. Variables at ``t = -1`` and ``t = simulation_length`` are fixed boundary
    conditions, and any shock or parameter path is known to the agents in advance.

    Parameters
    ----------
    model : Model
        The DSGE model to solve.
    simulation_length : int
        Number of periods, written ``T`` below.
    x0 : DataFrame, ndarray, dict, or None, optional
        Initial guess for the Newton solver. A dict is a steady-state dictionary as returned by
        ``model.steady_state()``, tiled across every period and used in place of computing the steady state. It
        must hold a value for every model variable, and it also supplies both boundary conditions, so it is not
        the right input for a permanent change in a parameter. A DataFrame has variable names as columns and
        periods as rows, and must hold a column for every model variable. This is what :func:`make_piecewise_x0`
        returns. An ndarray of shape ``(T, n_vars)`` must already be in the model's variable order. Defaults to
        the steady state tiled across every period.
    initial_conditions : dict mapping str to float, or DataArray, optional
        Variable values at ``t = -1``, keyed by base name such as ``"K"`` or by steady-state name such as
        ``"K_ss"``. A DataArray has one dimension over variable names, such as one draw of
        :meth:`~gEconpy.model.statespace.DSGEStateSpace.deviations_to_levels` output. Only the values of
        :attr:`~gEconpy.model.model.Model.predetermined_variables` change the solution, but every value must be
        finite. Variables not listed take their steady-state values.
    terminal_conditions : dict mapping str to float, or DataArray, optional
        Variable values at ``t = T``, keyed like ``initial_conditions``. Variables not listed take their
        steady-state values.
    shocks : dict mapping str to ndarray, optional
        Shock paths over the horizon, keyed by shock name, each of length ``T``. Shocks not listed are zero.
    param_paths : dict mapping str to float, sequence of float, or ndarray, optional
        Overrides for the free parameters, keyed by parameter name. A scalar holds for every period and a list or
        array of length ``T`` varies over time. Parameters not listed keep their model values, and deterministic
        parameters are rederived from the free values of each period. When boundary conditions are not fully
        specified, the initial and terminal steady states are computed at the parameter values of ``t = 0`` and
        ``t = T - 1``. A steady-state term written into an equation, such as the inflation target of a Taylor
        rule, is an expression in the parameters, so under a parameter path it too takes the values of each
        period rather than staying fixed at the initial steady state.
    compile_kwargs : dict, optional
        Keyword arguments forwarded to :func:`pytensor.function` when compiling the model.
    solver : RootSolver, optional
        The root-finding solver. See :doc:`gEconpy.solvers.sparse_root </api/gEconpy.solvers.sparse_root>` for the
        available solvers. Defaults to Newton's method with Armijo backtracking. A solver whose linear solver or
        column ordering is None factorizes the stacked Jacobian in the column order of
        :attr:`~gEconpy.model.perfect_foresight.compile.PerfectForesightProblem.column_order`, on a copy, so the
        solver passed in is not changed.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state()``. A parameter value given here also holds in every
        period of the path, unless ``param_paths`` names the same parameter. ``verbose`` defaults to False.
    sparse_root_kwargs : dict, optional
        Keyword arguments forwarded to :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root`, such as
        ``tol``, ``maxiter``, or ``progressbar``.
    max_homotopy_solves : int, optional
        Number of solves a homotopy may spend when the solve from ``x0`` fails. The homotopy scales the shocks and
        each initial and terminal condition's deviation from its steady state by a weight that rises toward 1. Its
        first solve starts from a linear path between the initial and terminal steady states, and each later one
        from the last that succeeded. The same path replaces an ``x0`` at which the model is not finite. Parameter
        paths are not scaled, so a failure that a parameter path causes is not helped. While the homotopy is on, the
        solve from ``x0`` stops after 50 iterations and each homotopy solve after 20, or after ``maxiter`` if that is
        smaller. The homotopy's solves show no progress bar. Zero turns the homotopy off, and the solve from ``x0``
        then runs for up to ``maxiter`` iterations. Defaults to 100.

    Returns
    -------
    trajectory : DataFrame
        Solution path with periods as the index and variables as columns.
    result : OptimizeResult
        Convergence information from the last solve of the full problem. Each equation is divided by the largest entry
        of its Jacobian row at the initial guess before solving, so convergence is judged on these scaled residuals.
        ``fun`` and ``jac`` are reported unscaled, so an equation with large terms can show a residual above the
        tolerance by its own scale. ``nit``, ``nfev``, ``njev`` and ``nsolve`` count every solve, homotopy included,
        and ``homotopy_solves`` counts the homotopy's own.

    Examples
    --------
    With capital starting ten percent below its steady-state value, every other boundary condition defaults to the
    steady state:

    .. code-block:: python

        from gEconpy import model_from_gcn, solve_perfect_foresight
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        steady_state = model.steady_state(verbose=False, progressbar=False)

        trajectory, result = solve_perfect_foresight(
            model,
            simulation_length=100,
            initial_conditions={"K": 0.9 * steady_state["K_ss"]},
        )
        print(result.success, trajectory["K"].iloc[[0, -1]].to_numpy())

    A permanent, pre-announced rise in the depreciation rate moves the economy from the steady state under the old
    value to the steady state under the new one, so both boundary conditions and an initial guess that bridges them
    are supplied:

    .. code-block:: python

        import numpy as np

        from gEconpy import model_from_gcn, solve_perfect_foresight
        from gEconpy.data import get_example_gcn
        from gEconpy.model.perfect_foresight import make_piecewise_x0

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        initial_ss = model.steady_state(delta=0.02, verbose=False, progressbar=False)
        final_ss = model.steady_state(delta=0.03, verbose=False, progressbar=False)

        delta_path = np.where(np.arange(200) < 100, 0.02, 0.03)
        x0 = make_piecewise_x0(initial_ss, final_ss, simulation_length=200, transition_periods=50)

        trajectory, result = solve_perfect_foresight(
            model,
            simulation_length=200,
            x0=x0,
            initial_conditions=initial_ss,
            terminal_conditions=final_ss,
            param_paths={"delta": delta_path},
        )
        print(result.success, trajectory["K"].iloc[[0, -1]].to_numpy())
    """
    initial_conditions = _normalize_condition_keys(initial_conditions)
    terminal_conditions = _normalize_condition_keys(terminal_conditions)
    compile_kwargs = compile_kwargs or {}
    steady_state_kwargs = {"verbose": False, **(steady_state_kwargs or {})}
    sparse_root_kwargs = sparse_root_kwargs or {}

    problem = compile_perfect_foresight_problem(model, simulation_length, **compile_kwargs)
    var_names = problem.var_names
    var_set = set(var_names)

    param_names, deterministic_param_names = _settable_parameter_names(model)
    parameter_updates, steady_state_kwargs = _split_parameter_updates(steady_state_kwargs, param_names)
    param_paths = {**parameter_updates, **(param_paths or {})}

    # Validation runs before anything expensive, and before any name reaches the model: a typo in param_paths
    # otherwise surfaces as whatever model.steady_state makes of it.
    validate_perfect_foresight_inputs(
        initial_conditions=initial_conditions,
        terminal_conditions=terminal_conditions,
        shocks=shocks,
        param_paths=param_paths,
        var_names=var_names,
        shock_names=problem.shock_names,
        param_names=param_names,
        deterministic_param_names=deterministic_param_names,
        simulation_length=simulation_length,
    )

    init_param_kwargs, term_param_kwargs = _extract_boundary_param_kwargs(param_paths)
    same_boundary_params = term_param_kwargs == init_param_kwargs

    needs_initial_ss = not (var_set <= initial_conditions.keys())
    needs_terminal_ss = not (var_set <= terminal_conditions.keys())

    def compute_ss(**param_kwargs):
        return model.steady_state(**steady_state_kwargs, **param_kwargs)

    x0_vec = None
    if isinstance(x0, dict):
        init_ss_dict = x0
        term_ss_dict = x0
    elif x0 is None:
        init_ss_dict = compute_ss(**init_param_kwargs)
        term_ss_dict = init_ss_dict if same_boundary_params else compute_ss(**term_param_kwargs)
    else:
        x0_vec = _x0_to_vector(x0=x0, var_names=var_names, simulation_length=simulation_length)
        init_ss_dict = compute_ss(**init_param_kwargs) if needs_initial_ss else None
        term_ss_dict = None
        if needs_terminal_ss:
            reuse_initial = init_ss_dict is not None and same_boundary_params
            term_ss_dict = init_ss_dict if reuse_initial else compute_ss(**term_param_kwargs)

    if x0_vec is None:
        x0_vec = np.tile(_ss_dict_to_array(init_ss_dict, var_names), simulation_length)

    y_initial = _boundary_vector(
        conditions=initial_conditions,
        ss_dict=init_ss_dict if needs_initial_ss else None,
        var_names=var_names,
    )
    y_terminal = _boundary_vector(
        conditions=terminal_conditions,
        ss_dict=term_ss_dict if needs_terminal_ss else None,
        var_names=var_names,
    )

    shock_matrix = _build_shock_matrix(shocks=shocks, shock_names=problem.shock_names, T=simulation_length)
    param_matrix = _build_param_matrix(
        param_paths=param_paths,
        param_names=problem.param_names,
        model=model,
        T=simulation_length,
    )

    _check_all_finite(
        x0=x0_vec,
        initial_conditions=y_initial,
        terminal_conditions=y_terminal,
        shocks=shock_matrix,
        param_paths=param_matrix,
    )

    @cache
    def homotopy_bases() -> tuple[np.ndarray, np.ndarray]:
        """Return the boundary steady states the homotopy scales the boundary conditions around."""
        initial = init_ss_dict if init_ss_dict is not None else compute_ss(**init_param_kwargs)
        if term_ss_dict is not None:
            terminal = term_ss_dict
        else:
            terminal = initial if same_boundary_params else compute_ss(**term_param_kwargs)
        return _ss_dict_to_array(initial, var_names), _ss_dict_to_array(terminal, var_names)

    def weight_zero_guess() -> np.ndarray:
        initial_base, terminal_base = homotopy_bases()
        return np.linspace(initial_base, terminal_base, simulation_length).ravel()

    ordered_solver = _order_for_problem(solver if solver is not None else newton_armijo(), problem)

    def solve_at_weight(weight: float, x0: np.ndarray, maxiter: int) -> tuple[np.ndarray, OptimizeResult]:
        initial, terminal, weighted_shocks = y_initial, y_terminal, shock_matrix
        root_kwargs = {**sparse_root_kwargs, "maxiter": maxiter}
        if weight != 1.0:
            initial_base, terminal_base = homotopy_bases()
            initial = initial_base + weight * (y_initial - initial_base)
            terminal = terminal_base + weight * (y_terminal - terminal_base)
            weighted_shocks = weight * shock_matrix
            root_kwargs["progressbar"] = False
        result = _solve_stacked(
            problem=problem,
            x0=x0,
            y_initial=initial,
            y_terminal=terminal,
            shocks=weighted_shocks,
            params=param_matrix,
            solver=ordered_solver,
            sparse_root_kwargs=root_kwargs,
        )
        return result.x, result

    maxiter = sparse_root_kwargs.get("maxiter")
    if maxiter is None:
        maxiter = rcParams["sparse_root.maxiter"]

    outcome = _solve_by_homotopy(
        solve=solve_at_weight,
        warm_guess=x0_vec,
        base_guess=weight_zero_guess,
        max_homotopy_solves=max_homotopy_solves,
        maxiter=maxiter,
    )
    if outcome.full_weight_result is None:
        raise outcome.non_finite_guess
    result = outcome.full_weight_result
    if outcome.path is None and outcome.homotopy_solves:
        result.message = outcome.failure
    for field, count in outcome.work_counts.items():
        result[field] = count
    result.homotopy_solves = outcome.homotopy_solves

    trajectory = pd.DataFrame(result.x.reshape(simulation_length, problem.n_vars), columns=var_names)
    return trajectory, result


def make_piecewise_x0(
    initial_ss: dict[str, float],
    terminal_ss: dict[str, float],
    simulation_length: int,
    var_names: list[str] | None = None,
    transition_start: int | None = None,
    transition_periods: int = 1,
) -> pd.DataFrame:
    """
    Build a guess for :func:`~gEconpy.model.perfect_foresight.solve.solve_perfect_foresight` between two steady states.

    Parameters
    ----------
    initial_ss : dict mapping str to float
        Steady state of the initial regime. Keys may carry the ``_ss`` suffix.
    terminal_ss : dict mapping str to float
        Steady state of the terminal regime. Keys may carry the ``_ss`` suffix.
    simulation_length : int
        Total number of periods.
    var_names : list of str, optional
        Variable names, fixing the column order. Both dictionaries must hold a value for every name. Defaults to
        the union of the keys of both dictionaries with the ``_ss`` suffix stripped, sorted alphabetically.
    transition_start : int, optional
        Period at which the transition begins. Defaults to ``simulation_length // 2``.
    transition_periods : int, optional
        Number of periods over which to interpolate linearly between the two steady states. A value of 1 is a step
        change. Larger values give a smoother guess that helps convergence after large permanent shocks. Defaults
        to 1.

    Returns
    -------
    x0 : DataFrame
        Initial guess of shape ``(simulation_length, n_vars)`` with variable names as columns.
    """
    if var_names is None:
        var_names = _infer_var_names_from_ss(initial_ss, terminal_ss)

    if transition_start is None:
        transition_start = simulation_length // 2

    init_vals = _ss_dict_to_array(initial_ss, var_names)
    term_vals = _ss_dict_to_array(terminal_ss, var_names)

    x0 = np.empty((simulation_length, len(var_names)))
    transition_end = min(transition_start + transition_periods, simulation_length)

    x0[:transition_start] = init_vals
    x0[transition_end:] = term_vals

    n_transition = transition_end - transition_start
    if n_transition == 1:
        x0[transition_start] = term_vals
    elif n_transition > 1:
        weights = np.linspace(0, 1, n_transition)[:, None]
        x0[transition_start:transition_end] = (1 - weights) * init_vals + weights * term_vals

    return pd.DataFrame(x0, columns=var_names)


def _settable_parameter_names(model: Model) -> tuple[list[str], list[str]]:
    """Return the names of every free and deterministic parameter of ``model``, and the deterministic ones alone."""
    # A free parameter can reach the equations only through a deterministic one, so the compiled problem's parameter
    # list misses it.
    deterministic_names = [param.name for param in model.deterministic_params]
    return [param.name for param in model.params] + deterministic_names, deterministic_names


def _split_parameter_updates(steady_state_kwargs: dict, param_names: Sequence[str]) -> tuple[dict, dict]:
    """Separate the parameter values in ``steady_state_kwargs`` from the options for ``model.steady_state``."""
    parameter_updates = {name: value for name, value in steady_state_kwargs.items() if name in param_names}
    options = {name: value for name, value in steady_state_kwargs.items() if name not in parameter_updates}
    return parameter_updates, options


def _normalize_condition_keys(conditions: dict[str, float] | xr.DataArray | None) -> dict[str, float]:
    """
    Return conditions keyed by base name, from a dict or from a DataArray with one dimension over variable names.

    The ``_ss`` suffix is stripped from every key, so a steady-state dictionary can be passed directly.
    """
    if conditions is None:
        return {}
    if isinstance(conditions, xr.DataArray):
        if conditions.ndim != 1:
            raise ValueError(
                f"A DataArray of conditions needs one dimension, over variable names, but has {conditions.dims}. "
                "Select or reduce the others first, for example with .mean(('chain', 'draw'))."
            )
        (dim,) = conditions.dims
        conditions = dict(zip(conditions[dim].values.tolist(), conditions.values.tolist(), strict=True))
    return {key.removesuffix("_ss"): value for key, value in conditions.items()}


def _ss_dict_to_array(ss_dict: dict[str, float], var_names: Sequence[str]) -> np.ndarray:
    """Read steady-state values in ``var_names`` order, accepting suffixed or bare keys."""
    values = _normalize_condition_keys(ss_dict)
    missing = [name for name in var_names if name not in values]
    if missing:
        raise ValueError(f"Steady-state dictionary is missing values for: {', '.join(missing)}")

    return np.array([values[name] for name in var_names])


def _infer_var_names_from_ss(*ss_dicts: dict[str, float]) -> list[str]:
    names: set[str] = set()
    for ss_dict in ss_dicts:
        names.update(k.removesuffix("_ss") for k in ss_dict)
    return sorted(names)


def _check_all_finite(**arrays: np.ndarray) -> None:
    """
    Check that no input carries a NaN or an infinity, naming the offender.

    Raises
    ------
    ValueError
        If any array holds a non-finite value.
    """
    # Newton reports a non-finite input as a singular Jacobian several layers down, which says nothing about
    # which input was wrong.
    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")


def _extract_boundary_param_kwargs(
    param_paths: dict[str, float | Sequence[float] | np.ndarray] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Read the parameter values at ``t = 0`` and ``t = T - 1``, for computing the boundary steady states.

    A scalar gives the same value at both ends. A list or array of length ``T`` gives its first and last entries.
    """
    if not param_paths:
        return {}, {}

    initial_kwargs = {}
    terminal_kwargs = {}
    for name, value in param_paths.items():
        path = np.atleast_1d(np.asarray(value, dtype=float))
        initial_kwargs[name] = float(path[0])
        terminal_kwargs[name] = float(path[-1])

    return initial_kwargs, terminal_kwargs


def _x0_to_vector(x0: np.ndarray | pd.DataFrame, var_names: Sequence[str], simulation_length: int) -> np.ndarray:
    """Flatten a user-supplied trajectory guess into the ``(T * n_vars,)`` layout of the stacked system."""
    if isinstance(x0, pd.DataFrame):
        if len(x0) != simulation_length:
            raise ValueError(f"x0 DataFrame must have {simulation_length} rows, got {len(x0)}")

        # Reindexing a missing column fills it with NaN, which reaches Newton as a singular Jacobian rather than
        # as the input error it is.
        missing = [name for name in var_names if name not in x0.columns]
        if missing:
            raise ValueError(f"x0 DataFrame is missing columns for: {', '.join(missing)}")

        return x0.reindex(columns=var_names).to_numpy().ravel()

    expected_shape = (simulation_length, len(var_names))
    if x0.shape != expected_shape:
        raise ValueError(f"x0 array must have shape {expected_shape}, got {x0.shape}")
    return x0.ravel()


def _boundary_vector(
    conditions: dict[str, float],
    ss_dict: dict[str, float] | None,
    var_names: Sequence[str],
) -> np.ndarray:
    """Build a boundary vector from user conditions, filling any unlisted variable from ``ss_dict``."""
    if ss_dict is None:
        return np.array([conditions[name] for name in var_names])

    values = _ss_dict_to_array(ss_dict, var_names)
    for name, value in conditions.items():
        values[var_names.index(name)] = value
    return values


def _build_shock_matrix(
    shocks: dict[str, np.ndarray] | None,
    shock_names: Sequence[str],
    T: int,
) -> np.ndarray:
    """Arrange the user's shock paths into a ``(T, n_shocks)`` matrix, with zeros for shocks not given."""
    shock_matrix = np.zeros((T, len(shock_names)))

    if shocks:
        for name, path in shocks.items():
            shock_matrix[:, shock_names.index(name)] = path

    return shock_matrix


def _build_param_matrix(
    param_paths: Mapping[str, float | Sequence[float] | np.ndarray] | None,
    param_names: Sequence[str],
    model: Model,
    T: int,
) -> np.ndarray:
    """
    Arrange parameter values into a ``(T, n_params)`` matrix, overriding the model values with any given paths.

    Every row goes through ``model.parameters()``, so each period's deterministic parameters are derived from
    that period's free values.
    """
    # The boundary steady states are computed through the same resolver, so a period whose deterministic
    # parameters were not rederived would be solving a different model than its own boundary conditions.
    param_paths = param_paths or {}
    if all(np.ndim(value) == 0 for value in param_paths.values()):
        by_period = [model.parameters(**param_paths)] * T
    else:
        paths = {name: np.broadcast_to(np.asarray(value, dtype=float), (T,)) for name, value in param_paths.items()}
        by_period = [model.parameters(**{name: path[t] for name, path in paths.items()}) for t in range(T)]

    return np.array([[parameters[name] for name in param_names] for parameters in by_period])


def _period_inputs(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    problem: PerfectForesightProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lay the iterate out as the three arrays of lagged, current, and led values.

    Returns
    -------
    Y_tm1, Y_t, Y_tp1 : ndarray
        Values at ``t-1``, ``t``, and ``t+1``, each of shape ``(T, n_vars)``. The boundary vectors stand in for
        the lag of the first period and the lead of the last.
    """
    y_mat = y.reshape(problem.T, problem.n_vars)

    return np.vstack([y_initial, y_mat[:-1]]), y_mat, np.vstack([y_mat[1:], y_terminal])


def _compute_stacked_residuals_and_jacobian(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    problem: PerfectForesightProblem,
) -> tuple[np.ndarray, sparse.csc_matrix]:
    Y_tm1, Y_t, Y_tp1 = _period_inputs(y=y, y_initial=y_initial, y_terminal=y_terminal, problem=problem)
    residuals, jacobians = problem.f_resid_and_jac(Y_tm1, Y_t, Y_tp1, x, params)

    return residuals.ravel(), assemble_stacked_jacobian(period_jacobians=jacobians, layout=problem.jacobian_layout)


def _solve_stacked(
    problem: PerfectForesightProblem,
    x0: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    shocks: np.ndarray,
    params: np.ndarray,
    solver: RootSolver,
    sparse_root_kwargs: dict,
) -> OptimizeResult:
    """
    Solve the stacked system from validated inputs, with each equation scaled by its largest Jacobian entry at ``x0``.

    ``fun`` and ``jac`` on the result are unscaled.
    """
    residuals, jacobian = _compute_stacked_residuals_and_jacobian(x0, y_initial, y_terminal, shocks, params, problem)
    row_scale = _residual_row_scale(residuals=residuals, jacobian=jacobian, n_vars=problem.n_vars)

    # The solver's first evaluation is at x0, which the row scale has just paid for.
    first_evaluation = [(row_scale * residuals, _scale_rows_in_place(jacobian, row_scale))]

    def scaled_fun(y):
        if first_evaluation and np.array_equal(y, x0):
            return first_evaluation.pop()
        residuals, jacobian = _compute_stacked_residuals_and_jacobian(y, y_initial, y_terminal, shocks, params, problem)
        return row_scale * residuals, _scale_rows_in_place(jacobian, row_scale)

    # The residual-only function lets the line search evaluate the merit function at rejected trial points
    # without paying for a Jacobian there.
    def merit_fun(y, *_args):
        return _compute_stacked_residuals(
            y=y, y_initial=y_initial, y_terminal=y_terminal, x=shocks, params=params, problem=problem
        )

    # Trust-region solvers take no globalization strategy, so there is nowhere to put the merit function. A caller's
    # own merit function returns raw residuals, so it is scaled like everything else the line search compares.
    globalization = getattr(solver, "globalization", None)
    if globalization is not None:
        unscaled_merit_fun = globalization.merit_fun or merit_fun

        def scaled_merit_fun(y, *args):
            return row_scale * unscaled_merit_fun(y, *args)

        solver = replace(solver, globalization=replace(globalization, merit_fun=scaled_merit_fun))

    result = _run_sparse_root(scaled_fun, x0, solver=solver, **sparse_root_kwargs)
    if result.fun is not None:
        result.fun = result.fun / row_scale
        result.jac = _scale_rows_in_place(result.jac.copy(), 1 / row_scale)
    return result


def _order_for_problem(solver: RootSolver, problem: PerfectForesightProblem) -> RootSolver:
    """Return ``solver``, or a copy whose unset linear solver or column ordering suits ``problem``'s Jacobian."""
    if isinstance(solver, SparseDogleg) and solver.linear_solver is None:
        return replace(solver, linear_solver=_ordered_sparse_solver(problem.column_order))
    if not isinstance(solver, LineSearchSolver):
        return solver

    direction = solver.direction
    if isinstance(direction, NewtonDirection) and direction.linear_solver is None:
        ordered_direction = replace(direction, linear_solver=_ordered_sparse_solver(problem.column_order))
        return replace(solver, direction=ordered_direction)
    if isinstance(direction, ChordDirection) and direction.permc_spec is None:
        # ChordDirection takes an ordering name only, so a problem that needs its own column order gets COLAMD.
        permc_spec = "NATURAL" if problem.column_order is None else "COLAMD"
        return replace(solver, direction=replace(direction, permc_spec=permc_spec))
    return solver


def _ordered_sparse_solver(column_order: np.ndarray | None) -> Callable[[sparse.csc_matrix, np.ndarray], np.ndarray]:
    """Return a SuperLU solve that factorizes in ``column_order``, so no factorization recomputes an ordering."""
    if column_order is None:
        return partial(spsolve, permc_spec="NATURAL", use_umfpack=False)
    order = column_order

    def solve(A: sparse.csc_matrix, b: np.ndarray) -> np.ndarray:
        x = np.empty(A.shape[1])
        x[order] = spsolve(A[:, order], b, permc_spec="NATURAL", use_umfpack=False)
        return x

    return solve


def _residual_row_scale(residuals: np.ndarray, jacobian: sparse.csc_matrix, n_vars: int) -> np.ndarray:
    """Return the reciprocal of the largest Jacobian entry in each stacked equation."""
    non_finite_rows = np.union1d(np.flatnonzero(~np.isfinite(residuals)), jacobian.indices[~np.isfinite(jacobian.data)])
    if non_finite_rows.size:
        period, equation = divmod(int(non_finite_rows[0]), n_vars)
        raise NonFiniteInitialGuessError(equation=equation, period=period)

    row_max = abs(jacobian).max(axis=1).toarray().ravel()
    return 1 / np.where(row_max > 0, row_max, 1.0)


def _scale_rows_in_place(matrix: sparse.csc_matrix, scale: np.ndarray) -> sparse.csc_matrix:
    matrix.data *= scale[matrix.indices]
    return matrix


def _compute_stacked_residuals(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    problem: PerfectForesightProblem,
) -> np.ndarray:
    Y_tm1, Y_t, Y_tp1 = _period_inputs(y=y, y_initial=y_initial, y_terminal=y_terminal, problem=problem)
    (residuals,) = problem.f_resid_only(Y_tm1, Y_t, Y_tp1, x, params)

    return residuals.ravel()
