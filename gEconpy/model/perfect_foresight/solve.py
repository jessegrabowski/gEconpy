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
from gEconpy.solvers.sparse_newton import sparse_newton

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


def _compute_stacked_residuals_and_jacobian(
    y: np.ndarray,
    y_initial: np.ndarray,
    y_terminal: np.ndarray,
    x: np.ndarray,
    params: tuple,
    problem: PerfectForesightProblem,
) -> tuple[np.ndarray, sparse.csc_matrix]:
    """Evaluate the single-period function T times and assemble the stacked system."""
    n_vars = problem.n_vars
    n_eq = problem.n_eq
    n_shocks = problem.n_shocks
    T = problem.T

    y_mat = y.reshape(T, n_vars)
    residuals = np.zeros(T * n_eq)
    jacobians = []

    for t in range(T):
        # Get y_{t-1}, y_t, y_{t+1}
        y_tm1 = y_initial if t == 0 else y_mat[t - 1]
        y_t = y_mat[t]
        y_tp1 = y_terminal if t == T - 1 else y_mat[t + 1]

        # Get shocks for this period
        x_t = x[t] if n_shocks > 0 else np.array([])

        # Call the compiled function
        if n_shocks > 0:
            r, J = problem.f_resid_and_jac(y_tm1, y_t, y_tp1, x_t, *params)
        else:
            r, J = problem.f_resid_and_jac(y_tm1, y_t, y_tp1, *params)

        residuals[t * n_eq : (t + 1) * n_eq] = r
        jacobians.append(J)

    jac = assemble_stacked_jacobian(jacobians, n_vars, n_eq, T)
    return residuals, jac


def _build_trajectory_dataframe(x: np.ndarray, var_names: list[str], T: int) -> pd.DataFrame:
    """Convert solution vector to a trajectory DataFrame indexed by time."""
    n_vars = len(var_names)
    data = x.reshape(T, n_vars)
    return pd.DataFrame(data, index=range(T), columns=var_names)


def solve_perfect_foresight(
    model: "Model",
    simulation_length: int,
    initial_conditions: dict[str, float] | None = None,
    terminal_conditions: dict[str, float] | None = None,
    shocks: dict[str, np.ndarray] | None = None,
    compile_kwargs: dict | None = None,
    optimize_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, OptimizeResult]:
    """Solve the model under perfect foresight.

    Parameters
    ----------
    model : Model
        The DSGE model to solve.
    simulation_length : int
        Number of time periods for the simulation.
    initial_conditions : dict of str to float, optional
        Initial values for state variables at t=-1. Keys are variable base names.
        If not provided, steady state values are used.
    terminal_conditions : dict of str to float, optional
        Terminal values for state variables at t=T. Keys are variable base names.
        If not provided, steady state values are used.
    shocks : dict of str to ndarray, optional
        Shock paths over the simulation horizon. Keys are shock base names,
        values are arrays of length T. If None, all shocks are set to zero.
    compile_kwargs : dict, optional
        Additional arguments passed to pytensor.function when compiling.
    optimize_kwargs : dict, optional
        Additional arguments passed to sparse_newton when solving.

    Returns
    -------
    trajectory : DataFrame
        Solution trajectory with time as index and variables as columns.
    result : OptimizeResult
        Optimization result with convergence info.

    Examples
    --------
    .. code-block:: python

        from gEconpy.model import load_gcn

        model = load_gcn("rbc.gcn")
        ss_dict = model.steady_state()

        # Simulate transition from 90% of steady state capital
        trajectory, result = solve_perfect_foresight(
            model,
            simulation_length=100,
            initial_conditions={"K": ss_dict["K_ss"] * 0.9},
        )
        trajectory.plot()
    """
    initial_conditions = initial_conditions or {}
    terminal_conditions = terminal_conditions or {}
    compile_kwargs = compile_kwargs or {}
    optimize_kwargs = optimize_kwargs or {}

    problem = compile_perfect_foresight_problem(model, simulation_length, **compile_kwargs)
    ss_dict = model.steady_state(verbose=False)

    var_names = problem.var_names

    validate_perfect_foresight_inputs(
        initial_conditions,
        terminal_conditions,
        shocks,
        var_names,
        problem.shock_names,
        simulation_length,
    )

    x0 = np.tile([ss_dict.get(f"{name}_ss", 1.0) for name in var_names], simulation_length)

    y_initial = np.array([initial_conditions.get(name, ss_dict.get(f"{name}_ss", 1.0)) for name in var_names])
    y_terminal = np.array([terminal_conditions.get(name, ss_dict.get(f"{name}_ss", 1.0)) for name in var_names])

    shock_matrix = _build_shock_matrix(shocks, problem.shock_names, simulation_length)

    param_dict = model.parameters()
    params = tuple(param_dict[name] for name in problem.param_names)

    def system(x, y_init, y_term, shock_mat, param_vals, prob):
        return _compute_stacked_residuals_and_jacobian(x, y_init, y_term, shock_mat, param_vals, prob)

    result = sparse_newton(system, x0, args=(y_initial, y_terminal, shock_matrix, params, problem), **optimize_kwargs)

    trajectory = _build_trajectory_dataframe(result.x, var_names, simulation_length)
    return trajectory, result
