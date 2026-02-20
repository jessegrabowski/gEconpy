from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import sympy as sp

from pytensor.graph.replace import graph_replace
from scipy import sparse
from scipy.optimize import OptimizeResult
from sympytensor import as_tensor

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import make_cache_key
from gEconpy.pytensorf.sparse_jacobian import sparse_jacobian
from gEconpy.solvers.sparse_newton import sparse_newton

if TYPE_CHECKING:
    from gEconpy.model.model import Model


def _collect_time_aware_atoms(equations: list[sp.Expr]) -> set[TimeAwareSymbol]:
    """Collect all TimeAwareSymbol atoms from a list of equations."""
    return reduce(lambda a, b: a.union(b), (eq.atoms(TimeAwareSymbol) for eq in equations), set())


def _classify_variables_by_timing(
    equations: list[sp.Expr],
    shock_names: list[str],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """Classify variables by time index into (vars_tm1, vars_t, vars_tp1, shocks_t)."""
    all_atoms = _collect_time_aware_atoms(equations)

    shocks_t = sorted([x for x in all_atoms if x.base_name in shock_names], key=lambda x: x.base_name)
    vars_tm1 = sorted(
        [x for x in all_atoms if x.time_index == -1 and x.base_name not in shock_names], key=lambda x: x.base_name
    )
    vars_t = sorted(
        [x for x in all_atoms if x.time_index == 0 and x.base_name not in shock_names], key=lambda x: x.base_name
    )
    vars_tp1 = sorted(
        [x for x in all_atoms if x.time_index == 1 and x.base_name not in shock_names], key=lambda x: x.base_name
    )

    return vars_tm1, vars_t, vars_tp1, shocks_t


def _compile_single_period_function(
    model: "Model",
    **compile_kwargs,
) -> tuple[Callable, list[str], list[str], list[str]]:
    """Compile fused single-period residual and Jacobian function.

    Returns a pytensor function computing residuals and sparse Jacobian for one
    time period. Jacobian shape is (n_eq, 3*n_vars) with columns [y_{t-1}, y_t, y_{t+1}].
    """
    shock_names = [s.base_name for s in model.shocks]
    param_names = [p.name for p in model.params]

    vars_tm1, vars_t, vars_tp1, shocks_t = _classify_variables_by_timing(model.equations, shock_names)

    cache: dict = {}
    equations_pt = [as_tensor(eq, cache=cache) for eq in model.equations]

    def get_pt_var(sym: TimeAwareSymbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=TimeAwareSymbol)]

    def get_pt_param(sym: sp.Symbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=sp.Symbol)]

    [get_pt_var(x) for x in vars_tm1]
    vars_t_pt = [get_pt_var(x) for x in vars_t]
    [get_pt_var(x) for x in vars_tp1]
    shocks_pt = [get_pt_var(x) for x in shocks_t]
    params_pt = [get_pt_param(p) for p in model.params]

    n_vars = len(vars_t_pt)
    n_shocks = len(shocks_pt)
    var_names = [x.base_name for x in vars_t]

    tm1_by_name = {x.base_name: get_pt_var(x) for x in vars_tm1}
    tp1_by_name = {x.base_name: get_pt_var(x) for x in vars_tp1}

    # Build jacobian variables for all (timing, var) combinations
    # Use existing scalars where present, create dummies otherwise
    jac_vars_tm1 = []
    jac_vars_t = []
    jac_vars_tp1 = []

    for i, name in enumerate(var_names):
        jac_vars_tm1.append(tm1_by_name.get(name, pt.dscalar(f"{name}_tm1_dummy")))
        jac_vars_t.append(vars_t_pt[i])
        jac_vars_tp1.append(tp1_by_name.get(name, pt.dscalar(f"{name}_tp1_dummy")))

    jac_vars = jac_vars_tm1 + jac_vars_t + jac_vars_tp1
    jacobian = sparse_jacobian(equations_pt, jac_vars, return_sparse=False)
    residuals = pt.stack(equations_pt)

    # Create vector inputs and replace scalars with indexed elements
    y_tm1_vec = pt.dvector("y_tm1", shape=(n_vars,))
    y_t_vec = pt.dvector("y_t", shape=(n_vars,))
    y_tp1_vec = pt.dvector("y_tp1", shape=(n_vars,))
    x_t_vec = pt.dvector("x_t", shape=(n_shocks,)) if n_shocks > 0 else None

    vec_replacements = {}
    for i, _name in enumerate(var_names):
        vec_replacements[jac_vars_tm1[i]] = y_tm1_vec[i]
        vec_replacements[jac_vars_t[i]] = y_t_vec[i]
        vec_replacements[jac_vars_tp1[i]] = y_tp1_vec[i]

    for i, var in enumerate(shocks_pt):
        vec_replacements[var] = x_t_vec[i]

    residuals_vec = graph_replace(residuals, vec_replacements, strict=False)
    jacobian_vec = graph_replace(jacobian, vec_replacements, strict=False)

    func_inputs = [y_tm1_vec, y_t_vec, y_tp1_vec]
    if x_t_vec is not None:
        func_inputs.append(x_t_vec)
    func_inputs.extend(params_pt)

    if "on_unused_input" not in compile_kwargs:
        compile_kwargs["on_unused_input"] = "ignore"

    f = pytensor.function(func_inputs, [residuals_vec, jacobian_vec], **compile_kwargs)

    return f, var_names, [x.base_name for x in shocks_t], param_names


@dataclass
class PerfectForesightProblem:
    """Compiled perfect foresight problem."""

    f_resid_and_jac: Callable
    var_names: list[str]
    shock_names: list[str]
    param_names: list[str]
    T: int

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def n_shocks(self) -> int:
        return len(self.shock_names)

    @property
    def n_eq(self) -> int:
        return len(self.var_names)


def _assemble_stacked_jacobian(
    period_jacobians: list[np.ndarray],
    n_vars: int,
    n_eq: int,
    T: int,
) -> sparse.csc_matrix:
    """Assemble block-tridiagonal Jacobian from period-wise Jacobians.

    Period Jacobian columns are ordered [y_{t-1}, y_t, y_{t+1}].
    """
    rows = []
    cols = []
    data = []

    for t in range(T):
        J_period = period_jacobians[t]
        row_offset = t * n_eq

        J_tm1 = J_period[:, :n_vars]
        J_t = J_period[:, n_vars : 2 * n_vars]
        J_tp1 = J_period[:, 2 * n_vars : 3 * n_vars]

        # J^{-1} block (skip at t=0, initial condition is fixed)
        if t > 0:
            col_offset = (t - 1) * n_vars
            r, c = np.nonzero(J_tm1)
            rows.extend(row_offset + r)
            cols.extend(col_offset + c)
            data.extend(J_tm1[r, c])

        # J^{0} block
        col_offset = t * n_vars
        r, c = np.nonzero(J_t)
        rows.extend(row_offset + r)
        cols.extend(col_offset + c)
        data.extend(J_t[r, c])

        # J^{+1} block (skip at t=T-1, terminal condition is fixed)
        if t < T - 1:
            col_offset = (t + 1) * n_vars
            r, c = np.nonzero(J_tp1)
            rows.extend(row_offset + r)
            cols.extend(col_offset + c)
            data.extend(J_tp1[r, c])

    return sparse.csc_matrix((data, (rows, cols)), shape=(T * n_eq, T * n_vars))


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

    jac = _assemble_stacked_jacobian(jacobians, n_vars, n_eq, T)
    return residuals, jac


def compile_perfect_foresight_problem(
    model: "Model",
    T: int,
    **compile_kwargs,
) -> PerfectForesightProblem:
    """Compile the single-period dynamic function for perfect foresight simulation."""
    f_resid_and_jac, var_names, shock_names, param_names = _compile_single_period_function(model, **compile_kwargs)
    return PerfectForesightProblem(
        f_resid_and_jac=f_resid_and_jac,
        var_names=var_names,
        shock_names=shock_names,
        param_names=param_names,
        T=T,
    )


def _validate_perfect_foresight_inputs(
    initial_conditions: dict[str, float],
    terminal_conditions: dict[str, float],
    shocks: dict[str, np.ndarray] | None,
    var_names: list[str],
    shock_names: list[str],
    T: int,
) -> None:
    """Validate inputs to solve_perfect_foresight."""
    var_set = set(var_names)
    shock_set = set(shock_names)

    invalid_initial = set(initial_conditions.keys()) - var_set
    if invalid_initial:
        raise ValueError(f"Unknown variables in initial_conditions: {invalid_initial}. Valid: {var_names}")

    invalid_terminal = set(terminal_conditions.keys()) - var_set
    if invalid_terminal:
        raise ValueError(f"Unknown variables in terminal_conditions: {invalid_terminal}. Valid: {var_names}")

    if shocks:
        invalid_shocks = set(shocks.keys()) - shock_set
        if invalid_shocks:
            raise ValueError(f"Unknown shocks: {invalid_shocks}. Valid: {shock_names}")

        for name, values in shocks.items():
            if len(values) != T:
                raise ValueError(f"Shock '{name}' has length {len(values)}, expected {T}")


def solve_perfect_foresight(
    model: "Model",
    T: int,
    initial_conditions: dict[str, float] | None = None,
    terminal_conditions: dict[str, float] | None = None,
    shocks: dict[str, np.ndarray] | None = None,
    compile_kwargs: dict | None = None,
    tol: float = 1e-10,
    maxiter: int = 100,
) -> tuple[pd.DataFrame, OptimizeResult]:
    """Solve the model under perfect foresight using Newton iteration.

    Parameters
    ----------
    model : Model
        The DSGE model to solve.
    T : int
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
    tol : float, default 1e-10
        Convergence tolerance for the Newton solver.
    maxiter : int, default 100
        Maximum number of Newton iterations.

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
            T=100,
            initial_conditions={"K": ss_dict["K_ss"] * 0.9},
        )
        trajectory.plot()
    """
    initial_conditions = initial_conditions or {}
    terminal_conditions = terminal_conditions or {}
    compile_kwargs = compile_kwargs or {}

    problem = compile_perfect_foresight_problem(model, T, **compile_kwargs)
    ss_dict = model.steady_state(verbose=False)

    var_names = problem.var_names

    _validate_perfect_foresight_inputs(
        initial_conditions,
        terminal_conditions,
        shocks,
        var_names,
        problem.shock_names,
        T,
    )

    x0 = np.tile([ss_dict.get(f"{name}_ss", 1.0) for name in var_names], T)

    y_initial = np.array([initial_conditions.get(name, ss_dict.get(f"{name}_ss", 1.0)) for name in var_names])
    y_terminal = np.array([terminal_conditions.get(name, ss_dict.get(f"{name}_ss", 1.0)) for name in var_names])

    n_shocks = problem.n_shocks
    if n_shocks > 0:
        shock_matrix = np.zeros((T, n_shocks))
        if shocks:
            for i, name in enumerate(problem.shock_names):
                if name in shocks:
                    shock_matrix[: len(shocks[name]), i] = shocks[name]
    else:
        shock_matrix = np.zeros((T, 0))

    param_dict = model.parameters()
    params = tuple(param_dict[name] for name in problem.param_names)

    def system(x, y_init, y_term, shock_mat, param_vals, prob):
        return _compute_stacked_residuals_and_jacobian(x, y_init, y_term, shock_mat, param_vals, prob)

    result = sparse_newton(
        system,
        x0,
        args=(y_initial, y_terminal, shock_matrix, params, problem),
        tol=tol,
        maxiter=maxiter,
    )

    trajectory = _build_trajectory_dataframe(result.x, var_names, T)
    return trajectory, result


def _build_trajectory_dataframe(x: np.ndarray, var_names: list[str], T: int) -> pd.DataFrame:
    """Convert solution vector to a trajectory DataFrame indexed by time."""
    n_vars = len(var_names)
    data = x.reshape(T, n_vars)
    return pd.DataFrame(data, index=range(T), columns=var_names)
