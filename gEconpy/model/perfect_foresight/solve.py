from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.optimize import OptimizeResult

from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight.assemble import assemble_stacked_jacobian
from gEconpy.model.perfect_foresight.compile import (
    PerfectForesightProblem,
    compile_perfect_foresight_problem,
)
from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs
from gEconpy.solvers.sparse_root import NewtonArmijo, sparse_root
from gEconpy.solvers.sparse_root.base import RootSolver
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking


def solve_perfect_foresight(
    model: Model,
    simulation_length: int,
    x0: np.ndarray | pd.DataFrame | dict[str, float] | None = None,
    initial_conditions: dict[str, float] | None = None,
    terminal_conditions: dict[str, float] | None = None,
    shocks: dict[str, np.ndarray] | None = None,
    param_paths: dict[str, float | Sequence[float] | np.ndarray] | None = None,
    compile_kwargs: dict | None = None,
    solver: RootSolver | None = None,
    steady_state_kwargs: dict | None = None,
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
        must hold a value for every model variable. A
        DataFrame has variable names as columns and periods as rows, and its columns are reindexed to the model's
        variable order. This is what :func:`make_piecewise_x0` returns. An ndarray of shape ``(T, n_vars)`` must
        already be in the model's variable order. Defaults to the steady state tiled across every period.
    initial_conditions : dict mapping str to float, optional
        Variable values at ``t = -1``, keyed by base name such as ``"K"`` or by steady-state name such as
        ``"K_ss"``. Variables not listed take their steady-state values.
    terminal_conditions : dict mapping str to float, optional
        Variable values at ``t = T``, keyed like ``initial_conditions``. Variables not listed take their
        steady-state values.
    shocks : dict mapping str to ndarray, optional
        Shock paths over the horizon, keyed by shock name, each of length ``T``. Shocks not listed are zero.
    param_paths : dict mapping str to float, sequence of float, or ndarray, optional
        Parameter overrides, keyed by parameter name. A scalar holds for every period and a list or array of length
        ``T`` varies over time. Parameters not listed keep their model values. When boundary conditions are not fully
        specified, the initial and terminal steady states are computed at the parameter values of ``t = 0`` and
        ``t = T - 1``.
    compile_kwargs : dict, optional
        Keyword arguments forwarded to :func:`pytensor.function` when compiling the model.
    solver : RootSolver, optional
        The root-finding solver. See :doc:`gEconpy.solvers.sparse_root </api/gEconpy.solvers.sparse_root>` for the
        available solvers. Defaults to Newton's method with Armijo backtracking.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state()``. ``verbose`` defaults to False.

    Returns
    -------
    trajectory : DataFrame
        Solution path with periods as the index and variables as columns.
    result : OptimizeResult
        Convergence information from the solver.

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
    initial_conditions = _normalize_condition_keys(initial_conditions or {})
    terminal_conditions = _normalize_condition_keys(terminal_conditions or {})
    compile_kwargs = compile_kwargs or {}
    steady_state_kwargs = {"verbose": False, **(steady_state_kwargs or {})}

    problem = compile_perfect_foresight_problem(model, simulation_length, **compile_kwargs)
    var_names = problem.var_names
    var_set = set(var_names)

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
        x0_vec = _x0_to_vector(x0, var_names, simulation_length)
        init_ss_dict = compute_ss(**init_param_kwargs) if needs_initial_ss else None
        term_ss_dict = None
        if needs_terminal_ss:
            reuse_initial = init_ss_dict is not None and same_boundary_params
            term_ss_dict = init_ss_dict if reuse_initial else compute_ss(**term_param_kwargs)

    if x0_vec is None:
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

    y_initial = _boundary_vector(initial_conditions, init_ss_dict if needs_initial_ss else None, var_names)
    y_terminal = _boundary_vector(terminal_conditions, term_ss_dict if needs_terminal_ss else None, var_names)

    shock_matrix = _build_shock_matrix(shocks, problem.shock_names, simulation_length)
    param_matrix = _build_param_matrix(param_paths, problem.param_names, model.parameters(), simulation_length)

    # The residual-only function lets the line search evaluate the merit function at rejected trial points without
    # paying for a Jacobian there.
    if problem.f_resid_only is not None:

        def merit_fun(x, *_args):
            return _compute_stacked_residuals(x, y_initial, y_terminal, shock_matrix, param_matrix, problem)

        if solver is None:
            solver = NewtonArmijo(globalization=ArmijoBacktracking(merit_fun=merit_fun))
        else:
            globalization = getattr(solver, "globalization", None)
            if globalization is not None and hasattr(globalization, "merit_fun"):
                solver = replace(solver, globalization=replace(globalization, merit_fun=merit_fun))

    result = sparse_root(
        _compute_stacked_residuals_and_jacobian,
        x0_vec,
        args=(y_initial, y_terminal, shock_matrix, param_matrix, problem),
        solver=solver,
    )

    trajectory = pd.DataFrame(result.x.reshape(simulation_length, len(var_names)), columns=var_names)
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


def _normalize_condition_keys(conditions: dict[str, float]) -> dict[str, float]:
    """Strip the ``_ss`` suffix from condition keys so a steady-state dictionary can be passed directly."""
    return {k.removesuffix("_ss"): v for k, v in conditions.items()}


def _ss_dict_to_array(ss_dict: dict[str, float], var_names: list[str]) -> np.ndarray:
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
        path = np.asarray(value, dtype=float)
        initial_kwargs[name] = float(path.flat[0])
        terminal_kwargs[name] = float(path.flat[-1])

    return initial_kwargs, terminal_kwargs


def _x0_to_vector(x0: np.ndarray | pd.DataFrame, var_names: list[str], simulation_length: int) -> np.ndarray:
    """Flatten a user-supplied trajectory guess into the ``(T * n_vars,)`` layout of the stacked system."""
    if isinstance(x0, pd.DataFrame):
        if len(x0) != simulation_length:
            raise ValueError(f"x0 DataFrame must have {simulation_length} rows, got {len(x0)}")
        return x0.reindex(columns=var_names).to_numpy().ravel()

    expected_shape = (simulation_length, len(var_names))
    if x0.shape != expected_shape:
        raise ValueError(f"x0 array must have shape {expected_shape}, got {x0.shape}")
    return x0.ravel()


def _boundary_vector(
    conditions: dict[str, float],
    ss_dict: dict[str, float] | None,
    var_names: list[str],
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
    shock_names: list[str],
    T: int,
) -> np.ndarray:
    """Arrange the user's shock paths into a ``(T, n_shocks)`` matrix, with zeros for shocks not given."""
    shock_matrix = np.zeros((T, len(shock_names)))

    if shocks:
        for name, path in shocks.items():
            shock_matrix[:, shock_names.index(name)] = path

    return shock_matrix


def _build_param_matrix(
    param_paths: dict[str, float | Sequence[float] | np.ndarray] | None,
    param_names: list[str],
    param_defaults: dict[str, float],
    T: int,
) -> np.ndarray:
    """Arrange parameter values into a ``(T, n_params)`` matrix, overriding the defaults with any given paths."""
    defaults = np.array([param_defaults[name] for name in param_names])
    param_matrix = np.tile(defaults, (T, 1))

    if param_paths:
        for name, value in param_paths.items():
            param_matrix[:, param_names.index(name)] = value

    return param_matrix


def _evaluate_periods(
    f: Callable,
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    problem: PerfectForesightProblem,
) -> Iterator[list[np.ndarray]]:
    """Call ``f`` once per period, in order, with the neighboring periods and the boundary vectors at the ends."""
    T = problem.T
    y_mat = y.reshape(T, problem.n_vars)
    for t in range(T):
        y_tm1 = y_initial if t == 0 else y_mat[t - 1]
        y_t = y_mat[t]
        y_tp1 = y_terminal if t == T - 1 else y_mat[t + 1]

        if problem.n_shocks > 0:
            yield f(y_tm1, y_t, y_tp1, x[t], *params[t])
        else:
            yield f(y_tm1, y_t, y_tp1, *params[t])


def _compute_stacked_residuals_and_jacobian(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    problem: PerfectForesightProblem,
) -> tuple[np.ndarray, sparse.csc_matrix]:
    period_outputs = _evaluate_periods(problem.f_resid_and_jac, y, y_initial, y_terminal, x, params, problem)
    residuals, jacobians = zip(*period_outputs, strict=True)

    stacked_jacobian = assemble_stacked_jacobian(
        jacobians, problem.jacobian_sparsity, problem.n_vars, problem.n_eq, problem.T
    )
    return np.concatenate(residuals), stacked_jacobian


def _compute_stacked_residuals(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    problem: PerfectForesightProblem,
) -> np.ndarray:
    period_outputs = _evaluate_periods(problem.f_resid_only, y, y_initial, y_terminal, x, params, problem)
    return np.concatenate([residuals for (residuals,) in period_outputs])
