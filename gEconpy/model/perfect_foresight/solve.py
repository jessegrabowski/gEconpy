from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.optimize import OptimizeResult

from gEconpy.model.perfect_foresight.assemble import assemble_stacked_jacobian
from gEconpy.model.perfect_foresight.compile import (
    PerfectForesightProblem,
    compile_perfect_foresight_problem,
)
from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs
from gEconpy.solvers.sparse_root import NewtonArmijo, sparse_root
from gEconpy.solvers.sparse_root.base import RootSolver
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking

if TYPE_CHECKING:
    from gEconpy.model.model import Model


def _build_shock_matrix(
    shocks: dict[str, np.ndarray] | None,
    shock_names: list[str],
    T: int,
) -> np.ndarray:
    """Construct (T, n_shocks) shock matrix from user-provided paths.

    Parameters
    ----------
    shocks : dict of str to ndarray or None
        Shock paths provided by user. Keys are shock names, values are arrays.
    shock_names : list of str
        Ordered list of shock names from the problem.
    T : int
        Number of time periods.

    Returns
    -------
    ndarray
        Shock matrix of shape (T, n_shocks).
    """
    n_shocks = len(shock_names)
    if n_shocks == 0:
        return np.zeros((T, 0))

    shock_matrix = np.zeros((T, n_shocks))
    if shocks:
        for i, name in enumerate(shock_names):
            if name in shocks:
                shock_matrix[: len(shocks[name]), i] = shocks[name]
    return shock_matrix


def _build_param_matrix(
    param_paths: dict[str, float | np.ndarray] | None,
    param_names: list[str],
    param_defaults: dict[str, float],
    T: int,
) -> np.ndarray:
    """Construct (T, n_params) parameter matrix from defaults and user-specified paths.

    Parameters
    ----------
    param_paths : dict or None
        Parameter paths. Values are either a scalar (constant across all periods) or an array of length T
        (time-varying).
    param_names : list of str
        Ordered parameter names from the compiled problem.
    param_defaults : dict of str to float
        Default parameter values from the model.
    T : int
        Number of time periods.

    Returns
    -------
    ndarray
        Parameter matrix of shape (T, n_params).
    """
    defaults = np.array([param_defaults[name] for name in param_names])
    param_matrix = np.tile(defaults, (T, 1))

    if param_paths:
        for name, value in param_paths.items():
            idx = param_names.index(name)
            param_matrix[:, idx] = value

    return param_matrix


def _evaluate_periods(
    f,
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    n_vars: int,
    n_shocks: int,
    T: int,
):
    """Iterate over T periods, calling ``f`` with the appropriate boundary vectors.

    Yields
    ------
    t : int
        Period index.
    result : tuple
        Whatever ``f`` returns for that period (residuals, or residuals + Jacobian).
    """
    y_mat = y.reshape(T, n_vars)
    for t in range(T):
        y_tm1 = y_initial if t == 0 else y_mat[t - 1]
        y_t = y_mat[t]
        y_tp1 = y_terminal if t == T - 1 else y_mat[t + 1]

        if n_shocks > 0:
            yield t, f(y_tm1, y_t, y_tp1, x[t], *params[t])
        else:
            yield t, f(y_tm1, y_t, y_tp1, *params[t])


def _compute_stacked_residuals_and_jacobian(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    problem: PerfectForesightProblem,
) -> tuple[np.ndarray, sparse.csc_matrix]:
    """Evaluate the single-period function T times and assemble the stacked system."""
    n_eq = problem.n_eq
    T = problem.T

    residuals = np.zeros(T * n_eq)
    jacobians = [None] * T

    for t, (r, J) in _evaluate_periods(
        problem.f_resid_and_jac, y, y_initial, y_terminal, x, params, problem.n_vars, problem.n_shocks, T
    ):
        residuals[t * n_eq : (t + 1) * n_eq] = r
        jacobians[t] = J

    jac = assemble_stacked_jacobian(jacobians, problem.n_vars, n_eq, T)
    return residuals, jac


def _compute_stacked_residuals(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    problem: PerfectForesightProblem,
) -> np.ndarray:
    """Evaluate residuals only (no Jacobian) for cheap merit evaluation during line search."""
    n_eq = problem.n_eq
    T = problem.T

    residuals = np.zeros(T * n_eq)

    for t, (r,) in _evaluate_periods(
        problem.f_resid_only, y, y_initial, y_terminal, x, params, problem.n_vars, problem.n_shocks, T
    ):
        residuals[t * n_eq : (t + 1) * n_eq] = r

    return residuals


def _build_trajectory_dataframe(x: np.ndarray, var_names: list[str], T: int) -> pd.DataFrame:
    """Convert solution vector to a trajectory DataFrame indexed by time."""
    n_vars = len(var_names)
    data = x.reshape(T, n_vars)
    return pd.DataFrame(data, index=range(T), columns=var_names)


def _normalize_condition_keys(conditions: dict[str, float]) -> dict[str, float]:
    """Strip the ``_ss`` suffix from condition keys so that steady-state dictionaries can be used directly.

    Accepts both ``"K"`` and ``"K_ss"`` as keys; the returned dict always uses base names.
    """
    return {k.removesuffix("_ss"): v for k, v in conditions.items()}


def _ss_dict_to_array(ss_dict: dict[str, float], var_names: list[str]) -> np.ndarray:
    """Extract a 1-d array of steady-state values in ``var_names`` order from a dict with ``_ss``-suffixed keys."""
    return np.array([ss_dict.get(f"{name}_ss", ss_dict.get(name, 1.0)) for name in var_names])


def _infer_var_names_from_ss(*ss_dicts: dict[str, float]) -> list[str]:
    """Infer variable base names from one or more steady-state dictionaries.

    Takes the union of keys across all dicts, strips the ``_ss`` suffix, and returns a sorted list.
    """
    names: set[str] = set()
    for d in ss_dicts:
        names.update(k.removesuffix("_ss") for k in d)
    return sorted(names)


def make_piecewise_x0(
    initial_ss: dict[str, float],
    terminal_ss: dict[str, float],
    simulation_length: int,
    var_names: list[str] | None = None,
    transition_start: int | None = None,
    transition_periods: int = 1,
) -> pd.DataFrame:
    """Build an initial guess that transitions between two steady states.

    Parameters
    ----------
    initial_ss : dict
        Steady-state dictionary for the initial regime (keys may have ``_ss`` suffix).
    terminal_ss : dict
        Steady-state dictionary for the terminal regime.
    simulation_length : int
        Total number of periods.
    var_names : list of str, optional
        Variable names and their order. If not provided, names are inferred from the union of keys in
        ``initial_ss`` and ``terminal_ss`` (with ``_ss`` suffix stripped), sorted alphabetically.
    transition_start : int, optional
        Period index where the transition begins. Defaults to ``simulation_length // 2``.
    transition_periods : int, default 1
        Number of periods over which to linearly interpolate. Set to 1 for a step change; larger values produce
        a smoother initial guess that can improve convergence for large permanent shocks.

    Returns
    -------
    DataFrame
        Shape ``(simulation_length, n_vars)`` with variable names as columns.
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

    if transition_end > transition_start:
        n = transition_end - transition_start
        if n == 1:
            x0[transition_start] = term_vals
        else:
            weights = np.linspace(0, 1, n)[:, None]
            x0[transition_start:transition_end] = (1 - weights) * init_vals + weights * term_vals

    return pd.DataFrame(x0, columns=var_names)


def _extract_boundary_param_kwargs(
    param_paths: dict[str, float | np.ndarray] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract parameter values at t=0 and t=T-1 from param_paths for steady-state computation.

    Returns two dicts suitable for passing as ``**kwargs`` to ``model.steady_state()``.
    """
    if not param_paths:
        return {}, {}

    initial_kwargs = {}
    terminal_kwargs = {}
    for name, value in param_paths.items():
        if isinstance(value, np.ndarray):
            initial_kwargs[name] = float(value[0])
            terminal_kwargs[name] = float(value[-1])
        else:
            initial_kwargs[name] = float(value)
            terminal_kwargs[name] = float(value)

    return initial_kwargs, terminal_kwargs


def solve_perfect_foresight(
    model: "Model",
    simulation_length: int,
    x0: np.ndarray | pd.DataFrame | dict[str, float] | None = None,
    initial_conditions: dict[str, float] | None = None,
    terminal_conditions: dict[str, float] | None = None,
    shocks: dict[str, np.ndarray] | None = None,
    param_paths: dict[str, float | np.ndarray] | None = None,
    compile_kwargs: dict | None = None,
    solver: RootSolver | None = None,
    steady_state_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, OptimizeResult]:
    """Solve the model under perfect foresight.

    Parameters
    ----------
    model : Model
        The DSGE model to solve.
    simulation_length : int
        Number of time periods for the simulation.
    x0 : DataFrame, ndarray, dict, or None, optional
        Initial guess for the Newton solver. Accepts four forms:

        - ``None`` (default): compute the steady state and tile it across all periods.
        - ``dict``: a steady-state dictionary (as returned by ``model.steady_state()``). Used in place of
          calling ``model.steady_state()`` internally, and also used for default initial/terminal conditions.
        - ``DataFrame``: columns are variable names, rows are periods. Columns are reindexed to match the
          model's canonical variable order. This is the output format of :func:`make_piecewise_x0`.
        - ``ndarray`` of shape ``(simulation_length, n_vars)``: a full initial trajectory. Columns must be in
          the model's canonical variable order.
    initial_conditions : dict of str to float, optional
        Initial values for state variables at t=-1. Keys are variable base names (e.g., ``"K"``). Keys with an
        ``_ss`` suffix (e.g., ``"K_ss"``) are also accepted. If not provided, steady-state values are used.
    terminal_conditions : dict of str to float, optional
        Terminal values for state variables at t=T. Keys are variable base names (e.g., ``"K"``). Keys with an
        ``_ss`` suffix are also accepted. If not provided, steady-state values are used.
    shocks : dict of str to ndarray, optional
        Shock paths over the simulation horizon. Keys are shock base names, values are arrays of length T. If
        None, all shocks are set to zero.
    param_paths : dict of str to float or ndarray, optional
        Parameter paths over the simulation horizon, analogous to ``shocks``. Keys are parameter names. Values
        are either a scalar (constant across all periods) or an array of length T (time-varying). Parameters not
        listed use their current model values. When time-varying paths are provided and boundary conditions are
        not fully specified, the steady state for initial/terminal conditions is computed using the parameter
        values at t=0 and t=T-1 respectively.
    compile_kwargs : dict, optional
        Additional arguments passed to ``pytensor.function`` when compiling.
    solver : RootSolver, optional
        Root-finding solver instance. Default uses Newton's method with Armijo backtracking
        (``NewtonArmijo()``). See :mod:`~gEconpy.solvers.sparse_root.line_search` for available
        solvers, or pass a custom solver.
    steady_state_kwargs : dict, optional
        Additional arguments passed to ``model.steady_state()``. If ``'verbose'`` is not explicitly set, it
        defaults to ``False``.

    Returns
    -------
    trajectory : DataFrame
        Solution trajectory with time as index and variables as columns.
    result : OptimizeResult
        Optimization result with convergence info.

    Examples
    --------
    Temporary shock (transition from perturbed initial condition back to steady state):

    .. code-block:: python

        import gEconpy as ge

        model = ge.model_from_gcn("rbc.gcn")
        ss_dict = model.steady_state()

        # Simulate transition from 90% of steady state capital
        trajectory, result = solve_perfect_foresight(
            model,
            simulation_length=100,
            initial_conditions={"K": ss_dict["K_ss"] * 0.9},
        )

    Permanent shock (transition between two steady states):

    .. code-block:: python

        from gEconpy.model.perfect_foresight import make_piecewise_x0
        import numpy as np

        init_ss = model.steady_state(tax_rate=0.08, how="root")
        final_ss = model.steady_state(tax_rate=0.18, how="root")

        x0 = make_piecewise_x0(init_ss, final_ss, simulation_length=200, transition_periods=50)
        tax_path = np.where(np.arange(200) < 100, 0.08, 0.18)

        trajectory, result = solve_perfect_foresight(
            model,
            simulation_length=200,
            x0=x0,
            initial_conditions=init_ss,
            terminal_conditions=final_ss,
            param_paths={"tax_rate": tax_path},
        )
    """
    initial_conditions = _normalize_condition_keys(initial_conditions or {})
    terminal_conditions = _normalize_condition_keys(terminal_conditions or {})
    compile_kwargs = compile_kwargs or {}
    steady_state_kwargs = steady_state_kwargs or {}
    steady_state_kwargs.setdefault("verbose", False)

    problem = compile_perfect_foresight_problem(model, simulation_length, **compile_kwargs)
    var_names = problem.var_names
    n_vars = len(var_names)
    var_set = set(var_names)

    init_param_kwargs, term_param_kwargs = _extract_boundary_param_kwargs(param_paths)

    needs_initial_ss = not (var_set <= initial_conditions.keys())
    needs_terminal_ss = not (var_set <= terminal_conditions.keys())

    def _compute_ss(**extra_kwargs):
        return model.steady_state(**steady_state_kwargs, **extra_kwargs)

    if isinstance(x0, dict):
        init_ss_dict = x0
        term_ss_dict = x0
        x0_vec = np.tile(_ss_dict_to_array(init_ss_dict, var_names), simulation_length)
    elif isinstance(x0, pd.DataFrame):
        if len(x0) != simulation_length:
            raise ValueError(f"x0 DataFrame must have {simulation_length} rows, got {len(x0)}")
        x0_vec = x0.reindex(columns=var_names).to_numpy().ravel()
        init_ss_dict = _compute_ss(**init_param_kwargs) if needs_initial_ss else None
        if needs_terminal_ss:
            term_ss_dict = (
                init_ss_dict
                if init_ss_dict is not None and term_param_kwargs == init_param_kwargs
                else _compute_ss(**term_param_kwargs)
            )
        else:
            term_ss_dict = None
    elif isinstance(x0, np.ndarray):
        if x0.shape != (simulation_length, n_vars):
            raise ValueError(f"x0 array must have shape ({simulation_length}, {n_vars}), got {x0.shape}")
        x0_vec = x0.ravel()
        init_ss_dict = _compute_ss(**init_param_kwargs) if needs_initial_ss else None
        if needs_terminal_ss:
            term_ss_dict = (
                init_ss_dict
                if init_ss_dict is not None and term_param_kwargs == init_param_kwargs
                else _compute_ss(**term_param_kwargs)
            )
        else:
            term_ss_dict = None
    else:
        init_ss_dict = _compute_ss(**init_param_kwargs)
        term_ss_dict = _compute_ss(**term_param_kwargs) if term_param_kwargs != init_param_kwargs else init_ss_dict
        x0_vec = np.tile(_ss_dict_to_array(init_ss_dict, var_names), simulation_length)

    validate_perfect_foresight_inputs(
        initial_conditions,
        terminal_conditions,
        shocks,
        param_paths,
        var_names,
        problem.shock_names,
        problem.param_names,
        simulation_length,
    )

    if not needs_initial_ss:
        y_initial = np.array([initial_conditions[name] for name in var_names])
    else:
        y_initial = _ss_dict_to_array(init_ss_dict, var_names)
        for name, val in initial_conditions.items():
            idx = var_names.index(name)
            y_initial[idx] = val

    if not needs_terminal_ss:
        y_terminal = np.array([terminal_conditions[name] for name in var_names])
    else:
        y_terminal = _ss_dict_to_array(term_ss_dict, var_names)
        for name, val in terminal_conditions.items():
            idx = var_names.index(name)
            y_terminal[idx] = val

    shock_matrix = _build_shock_matrix(shocks, problem.shock_names, simulation_length)

    param_dict = model.parameters()
    param_matrix = _build_param_matrix(param_paths, problem.param_names, param_dict, simulation_length)

    def system(x, y_init, y_term, shock_mat, param_vals, prob):
        return _compute_stacked_residuals_and_jacobian(x, y_init, y_term, shock_mat, param_vals, prob)

    # Attach cheap residual-only merit function to the solver's globalization strategy
    # so line search avoids computing the Jacobian at rejected trial points.
    if problem.f_resid_only is not None:

        def merit_fun(x, *_args):
            return _compute_stacked_residuals(x, y_initial, y_terminal, shock_matrix, param_matrix, problem)

        if solver is None:
            solver = NewtonArmijo(globalization=ArmijoBacktracking(merit_fun=merit_fun))
        else:
            glob = getattr(solver, "globalization", None)
            if glob is not None and hasattr(glob, "merit_fun"):
                glob.merit_fun = merit_fun

    result = sparse_root(
        system, x0_vec, args=(y_initial, y_terminal, shock_matrix, param_matrix, problem), solver=solver
    )

    trajectory = _build_trajectory_dataframe(result.x, var_names, simulation_length)
    return trajectory, result
