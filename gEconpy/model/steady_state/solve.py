import logging

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt
import sympy as sp

from better_optimize import minimize, root
from pytensor.tensor import TensorVariable
from scipy.optimize import OptimizeResult

from gEconpy.classes.containers import SteadyStateResults
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import compile_for_scipy, compile_packed
from gEconpy.model.steady_state.pytensor_engine import (
    ERROR_FUNCTIONS,
    SteadyStateSystem,
    _residual_scale,
    build_minimize_graphs,
    build_root_graphs,
    infer_variable_bounds,
    pt_error_from_resid,
    steady_state_transforms,
    transform_steady_state_system,
)
from gEconpy.rcparams import rcParams

_log = logging.getLogger(__name__)

BOX_BOUND_METHODS = ("trust-constr", "L-BFGS-B", "powell")

GRADIENT_REQUIRED_METHODS = ("trust-ncg", "trust-krylov", "trust-exact", "dogleg", "newton-cg")


def print_steady_state(ss_dict: SteadyStateResults) -> None:
    """
    Log a table of steady-state values, listing calibrated parameters after the variables.

    Parameters
    ----------
    ss_dict : SteadyStateResults
        Steady-state values to print. The table is prefixed with a warning when the results are not flagged as
        successful.

    Examples
    --------
    Solve the packaged RBC model for its steady state and print the result:

    .. code-block:: python

        import gEconpy as ge
        from gEconpy.data import get_example_gcn

        model = ge.model_from_gcn(get_example_gcn("RBC"), verbose=False)
        steady_state = model.steady_state(verbose=False, progressbar=False)
        ge.print_steady_state(steady_state)
    """
    output = []
    if not ss_dict.success:
        output.append("Values come from the latest solver iteration but are NOT a valid steady state.")

    name_width = max(len(name) for name in ss_dict) + 5

    calibrated_outputs = []
    for key, value in ss_dict.to_sympy().items():
        line = f"{key.name:{name_width}}{value:>10.3f}"
        if isinstance(key, TimeAwareSymbol):
            output.append(line)
        else:
            calibrated_outputs.append(line)

    if calibrated_outputs:
        output.append("\n")
        output.extend(calibrated_outputs)

    _log.info("\n".join(output))


def _compile_residual_and_scale(equations: list[TensorVariable], mode: str | None) -> Callable:
    """Compile one function returning the stacked residuals and the stacked scales of ``equations``."""
    return compile_for_scipy(
        [pt.stack(equations), pt.stack([_residual_scale(equation) for equation in equations])], mode=mode
    )


def _steady_state_tolerance(scale: np.ndarray) -> np.ndarray:
    """Return the largest residual an equation whose terms sum to ``scale`` in magnitude may have at a steady state."""
    return rcParams["steady_state.tol"] + rcParams["steady_state.rtol"] * scale


def _residual_within_tolerance(residual: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Whether each residual is finite and zero up to the steady-state tolerance at its equation's scale."""
    return np.isfinite(residual) & (np.abs(residual) <= _steady_state_tolerance(scale))


@dataclass(frozen=True)
class MinimizeSolver:
    """Compiled pieces of one minimization: the fused objective, the Hessian-vector product, and the unit maps."""

    objective: Callable
    hessp: Callable | None
    arg_names: list[str]
    to_unconstrained: Callable | None
    to_constrained: Callable | None


def _stack(rows: list[TensorVariable]) -> TensorVariable:
    return pt.stack(rows) if rows else pt.zeros(0)


def _positional(names: list[str], values: Mapping[str, float]) -> tuple[float, ...]:
    """Order ``values`` by ``names``, the argument order of a function built by ``compile_packed``."""
    return tuple(values[name] for name in names)


def _initialize_x0(optimizer_kwargs: dict, variables: list[sp.Symbol], jitter_x0: bool) -> np.ndarray:
    n_variables = len(variables)
    user_x0 = optimizer_kwargs.pop("x0", None)

    if user_x0 is None:
        x0 = np.full(n_variables, 0.8)
        is_negative = np.array([x.assumptions0.get("negative", False) for x in variables], dtype=bool)
        x0[is_negative] = -x0[is_negative]
    else:
        x0 = np.array(user_x0, dtype=float)

    if jitter_x0:
        rng = np.random.default_rng()
        x0 += rng.normal(scale=1e-4, size=n_variables)

    return x0


def compile_check_rows(system: SteadyStateSystem, mode: str | None = None) -> tuple[Callable, list[str]]:
    """
    Compile ``f(x, *args)`` returning the check rows' residuals and scales.

    Check rows depend on no unknown, so ``x`` is empty and the values hold before and after a solve.

    Parameters
    ----------
    system : gEconpy.model.steady_state.pytensor_engine.SteadyStateSystem
        The system whose check rows are evaluated.
    mode : str, optional
        PyTensor compilation mode. Default is None, which uses the PyTensor default mode.

    Returns
    -------
    f_checks : callable
        Compiled function returning the check residuals and their scales.
    arg_names : list of str
        Name of each trailing argument, in positional order.
    """
    outputs = [_stack(system.checks), _stack([_residual_scale(row) for row in system.checks])]
    f_checks, args = compile_packed(outputs, [], mode=mode)
    return f_checks, [arg.name for arg in args]


def compile_diagnostics(
    system: SteadyStateSystem, gradient_error_func: ERROR_FUNCTIONS | None, mode: str | None = None
) -> tuple[Callable, list[str]]:
    """
    Compile ``f(x, *args)`` returning what a verdict on the solve at ``x`` needs.

    Parameters
    ----------
    system : gEconpy.model.steady_state.pytensor_engine.SteadyStateSystem
        The solved system.
    gradient_error_func : str or None
        Error metric whose gradient over the solved rows is reported, see
        :func:`~gEconpy.model.steady_state.pytensor_engine.pt_error_from_resid`. None reports zeros.
    mode : str, optional
        PyTensor compilation mode. Default is None, which uses the PyTensor default mode.

    Returns
    -------
    diagnostics : callable
        Compiled function returning every steady-state value, the solved rows' residuals, and the error gradient.
    arg_names : list of str
        Name of each trailing argument, in positional order.
    """
    residual = _stack(system.residuals)
    if gradient_error_func is None or not system.unknowns:
        error_gradient = pt.zeros(len(system.unknowns))
    else:
        error = pt_error_from_resid(residual, gradient_error_func)
        error_gradient = pt.stack(pt.grad(error, system.unknowns, disconnected_inputs="ignore"))
    outputs = [pt.stack(system.steady_state), residual, error_gradient]
    diagnostics, args = compile_packed(outputs, system.unknowns, mode=mode)
    return diagnostics, [arg.name for arg in args]


def compile_root_objective(
    system: SteadyStateSystem, use_jac: bool, mode: str | None = None
) -> tuple[Callable, list[str]]:
    """
    Compile the objective of a root solve: the residuals fused with their Jacobian, or the residuals alone.

    Parameters
    ----------
    system : gEconpy.model.steady_state.pytensor_engine.SteadyStateSystem
        The system to solve.
    use_jac : bool
        Fuse the Jacobian of the residuals into the objective.
    mode : str, optional
        PyTensor compilation mode. Default is None, which uses the PyTensor default mode.

    Returns
    -------
    objective : callable
        Compiled function ``f(x, *args)``.
    arg_names : list of str
        Name of each trailing argument, in positional order.
    """
    residual, jacobian = build_root_graphs(system.residuals, system.unknowns, use_jac=use_jac)
    outputs = residual if jacobian is None else [residual, jacobian]
    objective, args = compile_packed(outputs, system.unknowns, mode=mode)
    return objective, [arg.name for arg in args]


def compile_minimize_solver(
    system: SteadyStateSystem,
    unknown_symbols: list[sp.Symbol],
    transform_bounds: dict[str, tuple[float, float]] | None,
    declared_bounds: dict[str, tuple[float | None, float | None]],
    error_func: ERROR_FUNCTIONS,
    use_jac: bool,
    use_hess: bool,
    use_hessp: bool,
    mode: str | None = None,
) -> MinimizeSolver:
    """
    Compile the fused error objective, and the Hessian-vector product when requested, with shared arguments.

    Parameters
    ----------
    system : gEconpy.model.steady_state.pytensor_engine.SteadyStateSystem
        The system to solve.
    unknown_symbols : list of sympy Symbol
        The symbol of each unknown, in the order of ``system.unknowns``.
    transform_bounds : dict mapping str to tuple of (float, float), or None
        Caller bounds for reparametrizing the unknowns onto the real line, so both functions take the unconstrained
        point. None leaves the unknowns in their own units for a box-constrained method.
    declared_bounds : dict mapping str to tuple of (float or None, float or None)
        Bounds declared in the GCN ``symbols`` block, used by the reparametrization.
    error_func : str
        Error metric minimized over the residuals, see
        :func:`~gEconpy.model.steady_state.pytensor_engine.pt_error_from_resid`.
    use_jac : bool
        Fuse the gradient into the objective.
    use_hess : bool
        Fuse the Hessian into the objective.
    use_hessp : bool
        Compile the Hessian-vector product.
    mode : str, optional
        PyTensor compilation mode. Default is None, which uses the PyTensor default mode.

    Returns
    -------
    solver : gEconpy.model.steady_state.solve.MinimizeSolver
        The compiled objective and the maps between constrained and unconstrained points.
    """
    equations, solve_nodes = system.residuals, system.unknowns
    to_unconstrained = to_constrained = None
    if transform_bounds is not None:
        transforms = steady_state_transforms(unknown_symbols, declared_bounds, transform_bounds)
        equations, solve_nodes, to_unconstrained, to_constrained = transform_steady_state_system(
            equations, solve_nodes, transforms
        )

    error, grad, hess, hessp, direction = build_minimize_graphs(
        equations,
        solve_nodes,
        error_func=error_func,
        use_jac=use_jac,
        use_hess=use_hess,
        use_hessp=use_hessp,
    )
    fused = [graph for graph in (error, grad, hess) if graph is not None]
    objective, args = compile_packed(fused if len(fused) > 1 else error, solve_nodes, mode=mode)

    hessp_function = None
    if hessp is not None:
        hessp_function, _ = compile_packed(hessp, solve_nodes, leading_inputs=(direction,), args=args, mode=mode)

    return MinimizeSolver(
        objective=objective,
        hessp=hessp_function,
        arg_names=[arg.name for arg in args],
        to_unconstrained=to_unconstrained,
        to_constrained=to_constrained,
    )


def solve_root(
    objective: Callable,
    args: tuple[float, ...],
    unknown_symbols: list[sp.Symbol],
    progressbar: bool,
    optimizer_kwargs: dict,
    jitter_x0: bool,
) -> tuple[np.ndarray, OptimizeResult]:
    """
    Run ``better_optimize.root`` on a compiled root objective.

    Parameters
    ----------
    objective : callable
        Objective from :func:`compile_root_objective`.
    args : tuple of float
        Values of the objective's trailing arguments, in positional order.
    unknown_symbols : list of sympy Symbol
        The symbol of each unknown, which sets the sign of the default initial point.
    progressbar : bool
        Display a progress bar while solving.
    optimizer_kwargs : dict
        Options for the root finder, consumed in place. ``method`` defaults to ``'hybr'``, ``maxiter`` to 5000, and
        ``x0`` to 0.8 in each unknown's sign.
    jitter_x0 : bool
        Add ``N(0, 1e-4)`` noise to the initial point.

    Returns
    -------
    x : ndarray
        The final point.
    result : OptimizeResult
        The optimizer's result.
    """
    maxiter = optimizer_kwargs.pop("maxiter", 5000)
    method = optimizer_kwargs.pop("method", "hybr")
    options = optimizer_kwargs.setdefault("options", {})
    options["maxfev" if method in ("hybr", "df-sane") else "maxiter"] = maxiter
    x0 = _initialize_x0(optimizer_kwargs, unknown_symbols, jitter_x0)

    with np.errstate(all="ignore"):
        res = root(f=objective, x0=x0, args=args, method=method, progressbar=progressbar, **optimizer_kwargs)
    return res.x, res


def prepare_minimize(
    optimizer_kwargs: dict, prefer_transform: bool, use_jac: bool, use_hess: bool, use_hessp: bool
) -> tuple[str, bool, bool]:
    """
    Choose the minimize method and set its iteration options, consuming ``method`` and ``maxiter`` in place.

    Parameters
    ----------
    optimizer_kwargs : dict
        Options for the minimizer. ``method`` defaults to ``'trust-ncg'`` and ``maxiter`` to 5000.
    prefer_transform : bool
        Enforce bounds by reparametrization even for a bounds-capable method.
    use_jac : bool
        A gradient is available.
    use_hess : bool
        The full Hessian is requested.
    use_hessp : bool
        The Hessian-vector product is requested.

    Returns
    -------
    method : str
        The method to run.
    use_box_bounds : bool
        Whether bounds go to scipy as box constraints, as opposed to a reparametrization.
    use_hess : bool
        Whether to use the full Hessian, which yields to the Hessian-vector product when both are requested.
    """
    # Box-constrained solvers crawl an ill-conditioned interior-point central path on DSGE steady states, so they
    # run only when the caller names one and leaves ``prefer_transform`` off. Otherwise each bounded variable is
    # reparametrized onto the real line (x = transform.backward(y)) and the unconstrained problem is solved, which
    # also lets a bounds-capable method such as L-BFGS-B run on the transformed space when ``prefer_transform``
    # is set.
    requested_method = optimizer_kwargs.pop("method", None)
    use_box_bounds = (not prefer_transform) and requested_method in BOX_BOUND_METHODS
    method = requested_method if use_box_bounds else (requested_method or "trust-ncg")

    if not (use_jac or use_hess or use_hessp) and method.lower() in GRADIENT_REQUIRED_METHODS:
        raise ValueError(
            f"Method {method!r} requires a gradient, but use_jac, use_hess, and use_hessp are all False. Pass "
            f"use_jac=True, or choose a gradient-free method such as 'nelder-mead' or 'powell' via "
            f"optimizer_kwargs={{'method': ...}}."
        )

    maxiter = optimizer_kwargs.pop("maxiter", 5000)
    options = optimizer_kwargs.setdefault("options", {})
    options["maxiter"] = maxiter
    if method == "L-BFGS-B":
        options["maxfun"] = maxiter

    if use_hess and use_hessp:
        _log.warning("Both use_hess and use_hessp are set to True. use_hessp will be used.")
        use_hess = False

    return method, use_box_bounds, use_hess


def solve_minimize(
    solver: MinimizeSolver,
    args: tuple[float, ...],
    unknown_symbols: list[sp.Symbol],
    method: str,
    bounds: dict[str, tuple[float | None, float | None]],
    use_box_bounds: bool,
    progressbar: bool,
    optimizer_kwargs: dict,
    jitter_x0: bool,
) -> tuple[np.ndarray, OptimizeResult]:
    """
    Run ``better_optimize.minimize`` on a compiled solver and map the result back to the unknowns' units.

    Parameters
    ----------
    solver : gEconpy.model.steady_state.solve.MinimizeSolver
        Solver from :func:`compile_minimize_solver`.
    args : tuple of float
        Values of the solver's trailing arguments, in positional order.
    unknown_symbols : list of sympy Symbol
        The symbol of each unknown, which sets the sign of the default initial point and its box bounds.
    method : str
        The method from :func:`prepare_minimize`.
    bounds : dict mapping str to tuple of (float or None, float or None)
        Caller bounds keyed by steady-state name, applied as box constraints when ``use_box_bounds`` is True.
    use_box_bounds : bool
        Pass bounds to scipy as box constraints, completing them from each unknown's sign assumptions.
    progressbar : bool
        Display a progress bar while solving.
    optimizer_kwargs : dict
        Remaining options for the minimizer, consumed in place. ``tol`` defaults to 1e-30 and ``x0`` to 0.8 in each
        unknown's sign.
    jitter_x0 : bool
        Add ``N(0, 1e-4)`` noise to the initial point.

    Returns
    -------
    x : ndarray
        The final point, in the unknowns' own units.
    result : OptimizeResult
        The optimizer's result.
    """
    x0 = _initialize_x0(optimizer_kwargs, unknown_symbols, jitter_x0)
    tol = optimizer_kwargs.pop("tol", 1e-30)

    bounds_arg = None
    if use_box_bounds:
        box_bounds = {x.name: infer_variable_bounds(x) for x in unknown_symbols} | bounds
        bounds_arg = [box_bounds[x.name] for x in unknown_symbols]

    res = minimize(
        f=solver.objective,
        x0=x0 if solver.to_unconstrained is None else solver.to_unconstrained(x0),
        args=args,
        hessp=solver.hessp,
        method=method,
        bounds=bounds_arg,
        tol=tol,
        progressbar=progressbar,
        **optimizer_kwargs,
    )
    x_star = res.x if solver.to_constrained is None else solver.to_constrained(res.x)
    return x_star, res
