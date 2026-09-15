import logging

from typing import Literal

import pytensor.tensor as pt
import sympy as sp

from pytensor.gradient import hessian_vector_product
from pytensor.gradient import jacobian as pt_jacobian
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor import TensorVariable

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import (
    compile_function,
    dictionary_return_wrapper,
    make_cache_key,
    make_return_dict_and_update_cache,
    sympy_to_pytensor,
)
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.utilities import eq_to_ss, flatten_substitution_dict, safe_to_ss

_log = logging.getLogger(__name__)

ERROR_FUNCTIONS = Literal["squared", "mean_squared", "abs", "l2-norm"]


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


def make_steady_state_shock_dict(shocks: list[TimeAwareSymbol]) -> SymbolDictionary:
    """
    Build a substitution dictionary setting every shock to zero in the steady state.

    Parameters
    ----------
    shocks : list of TimeAwareSymbol
        Model shocks.

    Returns
    -------
    shock_dict : SymbolDictionary
        Mapping from each steady-state shock symbol to 0.0.
    """
    return SymbolDictionary.fromkeys(shocks, 0.0).to_ss()


def system_to_steady_state(system: list[sp.Expr], shocks: list[TimeAwareSymbol]) -> list[sp.Expr]:
    """
    Rewrite a dynamic system of equations in steady-state form.

    Parameters
    ----------
    system : list of sympy expression
        Model equations in time-indexed variables.
    shocks : list of TimeAwareSymbol
        Model shocks, which are set to zero.

    Returns
    -------
    system : list of sympy expression
        Simplified steady-state equations.
    """
    shock_dict = make_steady_state_shock_dict(shocks)
    return [eq_to_ss(eq).subs(shock_dict).simplify() for eq in system]


def simplify_provided_ss_equations(
    ss_solution_dict: SymbolDictionary, variables: list[TimeAwareSymbol]
) -> SymbolDictionary:
    """
    Substitute intermediate definitions out of the user-provided steady-state equations.

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        Steady-state equations from the ``STEADY_STATE`` block.
    variables : list of TimeAwareSymbol
        Model variables.

    Returns
    -------
    ss_solution_dict : SymbolDictionary
        Steady-state equations keyed by model variables only.
    """
    if not ss_solution_dict:
        return SymbolDictionary()

    ss_variables = [variable.to_ss() for variable in variables]
    ss_dict_sympy = ss_solution_dict.to_sympy()

    intermediates = {key: value for key, value in ss_dict_sympy.items() if key not in ss_variables}
    if not intermediates:
        return ss_solution_dict

    simplified = SymbolDictionary({key: value for key, value in ss_dict_sympy.items() if key in ss_variables})
    flat_intermediates = flatten_substitution_dict(intermediates)
    for variable, eq in simplified.items():
        if hasattr(eq, "subs"):
            simplified[variable] = eq.subs(flat_intermediates)

    return simplified


def propagate_steady_state_through_identities(
    ss_solution_dict: SymbolDictionary,
    steady_state_equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    max_iterations: int = 100,
) -> SymbolDictionary:
    """
    Extend the user-provided steady state by solving equations with a single remaining unknown.

    Each pass substitutes the known values into every equation and solves those left with exactly one unknown. Passes
    repeat until one makes no progress. A solution is accepted only if it is unique and simple: no conditionals, no
    unevaluated integrals or derivatives, no complex numbers, and limited nesting depth.

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        User-provided steady-state values.
    steady_state_equations : list of sympy expression
        Model equations in steady-state residual form, each equal to zero.
    variables : list of TimeAwareSymbol
        Model variables.
    max_iterations : int, optional
        Maximum number of passes over the equation system. Default is 100.

    Returns
    -------
    ss_solution_dict : SymbolDictionary
        The provided values plus every value that could be inferred.
    """
    ss_variables = {variable.to_ss() for variable in variables}
    known = ss_solution_dict.to_sympy().copy() if ss_solution_dict else {}

    for _ in range(max_iterations):
        progress = False

        for eq in steady_state_equations:
            if not isinstance(eq, sp.Basic):
                continue

            unknowns = (eq.free_symbols & ss_variables) - set(known.keys())
            if len(unknowns) != 1:
                continue

            unknown = next(iter(unknowns))
            solution = _try_solve_for_unknown(eq, unknown, known, ss_variables)
            if solution is None:
                continue

            known[unknown] = float(solution) if solution.is_number else solution
            progress = True

        if not progress:
            break

    return SymbolDictionary(known)


def compile_known_ss(
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol | sp.Symbol],
    parameters: list[sp.Symbol],
    cache: dict | None,
    return_symbolic: bool = False,
    stack_return: bool | None = None,
    **kwargs,
):
    """
    Compile a function returning the analytic steady-state values as a function of the parameters.

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        Known steady-state values, keyed by variable.
    variables : list of TimeAwareSymbol or sympy Symbol
        Model variables, which fix the order of the outputs.
    parameters : list of sympy Symbol
        Model parameters, which become the inputs of the compiled function.
    cache : dict or None
        Sympytensor cache mapping cache keys to PyTensor variables, or None for a new empty cache.
    return_symbolic : bool, optional
        Return a dictionary of PyTensor graphs instead of a compiled function. Default is False.
    stack_return : bool, optional
        Stack the outputs into a single array. Default is None, meaning the opposite of ``return_symbolic``.
    **kwargs
        Forwarded to :func:`~gEconpy.model.compile.compile_function`.

    Returns
    -------
    f_ss : callable, dict, or None
        Function mapping parameter values to steady-state values, or a dictionary from each variable's PyTensor node
        to its steady-state graph when ``return_symbolic`` is True. None when ``ss_solution_dict`` is empty.
    cache : dict
        The cache, extended with every variable created during conversion.
    """
    cache = {} if cache is None else cache
    if not ss_solution_dict:
        return None, cache

    ss_solution_dict = ss_solution_dict.to_sympy()
    ss_variables = [safe_to_ss(variable) for variable in variables]
    ordered_solutions = {
        variable: ss_solution_dict[variable] for variable in ss_variables if variable in ss_solution_dict
    }

    if stack_return is None:
        stack_return = not return_symbolic

    f_ss, cache = compile_function(
        parameters,
        list(ordered_solutions.values()),
        cache=cache,
        stack_return=stack_return,
        return_symbolic=return_symbolic,
        **kwargs,
    )
    if return_symbolic:
        return make_return_dict_and_update_cache(ss_variables, f_ss, cache, TimeAwareSymbol)

    return dictionary_return_wrapper(f_ss, list(ordered_solutions.keys())), cache


def pt_error_from_resid(
    resid: TensorVariable,
    func: ERROR_FUNCTIONS = "squared",
) -> TensorVariable:
    """
    Reduce a residual vector to a scalar error.

    Parameters
    ----------
    resid : TensorVariable
        Stacked residual vector of shape ``(n_eq,)``.
    func : str, optional
        Error metric. One of ``'squared'``, ``'mean_squared'``, ``'abs'``, or ``'l2-norm'``. Default is
        ``'squared'``.

    Returns
    -------
    error : TensorVariable
        Scalar error graph.
    """
    if func == "squared":
        return (resid**2).sum()
    if func == "mean_squared":
        return (resid**2).mean()
    if func == "abs":
        return pt.abs(resid).sum()
    if func == "l2-norm":
        return pt.sqrt((resid**2).sum())
    raise NotImplementedError(f"Error function {func} not implemented, must be one of {ERROR_FUNCTIONS}")


def build_root_graphs(
    equations: list[TensorVariable],
    ss_input_nodes: list[TensorVariable],
    use_jac: bool = True,
) -> tuple[TensorVariable, TensorVariable | None]:
    """
    Build the residual and Jacobian graphs that :func:`scipy.optimize.root` needs.

    Parameters
    ----------
    equations : list of TensorVariable
        Scalar equation graphs, each equal to zero at the steady state.
    ss_input_nodes : list of TensorVariable
        Scalar input node for each steady-state variable.
    use_jac : bool, optional
        Build the Jacobian graph. Default is True.

    Returns
    -------
    resid : TensorVariable
        Stacked residual vector of shape ``(n_eq,)``.
    jac : TensorVariable or None
        Jacobian of shape ``(n_eq, n_var)``, or None when ``use_jac`` is False.
    """
    resid = pt.stack(equations) if equations else pt.zeros(0)
    if not use_jac:
        return resid, None

    if equations and ss_input_nodes:
        jac = pt.stack(pt_jacobian(resid, ss_input_nodes), axis=1)
    else:
        jac = pt.zeros((len(equations), len(ss_input_nodes)))

    return resid, jac


def build_minimize_graphs(
    equations: list[TensorVariable],
    ss_input_nodes: list[TensorVariable],
    error_func: ERROR_FUNCTIONS = "squared",
    use_jac: bool = True,
    use_hess: bool = False,
    use_hessp: bool = True,
) -> tuple[TensorVariable, TensorVariable | None, TensorVariable | None, TensorVariable | None, TensorVariable | None]:
    """
    Build the error and derivative graphs that :func:`scipy.optimize.minimize` needs.

    Parameters
    ----------
    equations : list of TensorVariable
        Scalar equation graphs, each equal to zero at the steady state.
    ss_input_nodes : list of TensorVariable
        Scalar input node for each steady-state variable.
    error_func : str, optional
        Error metric, see :func:`pt_error_from_resid`. Default is ``'squared'``.
    use_jac : bool, optional
        Build the gradient graph. Default is True.
    use_hess : bool, optional
        Build the full Hessian graph. Default is False.
    use_hessp : bool, optional
        Build the Hessian-vector product graph. Default is True.

    Returns
    -------
    error : TensorVariable
        Scalar error.
    grad : TensorVariable or None
        Gradient of shape ``(n_var,)``, or None when no derivative graph is requested.
    hess : TensorVariable or None
        Hessian of shape ``(n_var, n_var)``, or None when ``use_hess`` is False.
    hessp_out : TensorVariable or None
        Hessian-vector product of shape ``(n_var,)``, or None when ``use_hessp`` is False.
    hessp_p : TensorVariable or None
        Direction vector input of the Hessian-vector product, or None when ``use_hessp`` is False.
    """
    resid = pt.stack(equations) if equations else pt.zeros(0)
    error = pt_error_from_resid(resid, error_func)

    grad = None
    if use_jac or use_hess or use_hessp:
        grad = pt.stack(pt.grad(error, ss_input_nodes))

    hess = None
    if use_hess:
        hess = pt.stack(pt_jacobian(grad, ss_input_nodes), axis=1)

    hessp_out = None
    hessp_p = None
    if use_hessp:
        hessp_p = pt.dvector("hess_eval_point")
        directions = [hessp_p[i] for i in range(len(ss_input_nodes))]
        hessp_out = pt.stack(hessian_vector_product(error, ss_input_nodes, directions))

    return error, grad, hess, hessp_out, hessp_p


def _ss_residual_to_pytensor(
    steady_state_equations: list[sp.Expr],
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol],
    param_dict: SymbolDictionary,
    deterministic_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    cache: dict | None = None,
) -> tuple[list[TensorVariable], dict]:
    """
    Convert the steady-state residual system from sympy to PyTensor graphs.

    This is the only place the steady-state solvers touch sympy. Every derivative they need is built from the returned
    equations by PyTensor autodiff in :func:`build_root_graphs` and :func:`build_minimize_graphs`. Callers recover
    the parameter and variable nodes from the returned cache with :func:`~gEconpy.model.compile.make_cache_key`.

    Parameters
    ----------
    steady_state_equations : list of sympy expression
        Steady-state equations in residual form, each equal to zero.
    ss_solution_dict : SymbolDictionary
        Analytically known steady-state values, from the ``STEADY_STATE`` block and identity propagation.
    variables : list of TimeAwareSymbol
        Model variables, without calibrated parameter symbols.
    param_dict : SymbolDictionary
        Free parameter names and default values.
    deterministic_dict : SymbolDictionary
        Deterministic parameters defined as functions of the free parameters.
    calib_dict : SymbolDictionary
        Calibrating equations mapping each calibrated parameter to the steady-state expression that pins it.
    cache : dict, optional
        Sympytensor cache to extend, so that PyTensor nodes are shared across graph-building calls. Default is a new
        empty cache.

    Returns
    -------
    equations : list of TensorVariable
        Scalar equation graphs, each equal to zero at the steady state. Equations that no longer contain an unknown
        after substituting the known steady-state values are dropped.
    cache : dict
        The cache, extended with every parameter, deterministic parameter, steady-state variable, and known
        steady-state graph created during conversion.
    """
    if cache is None:
        cache = {}

    compile_param_dict_func(param_dict, deterministic_dict, cache=cache, return_symbolic=True)

    calib_symbols = calib_dict.to_sympy()
    full_equations = steady_state_equations + list(calib_symbols.values())

    parameters = [param for param in (param_dict | deterministic_dict).to_sympy() if param not in calib_symbols]
    ss_variables = [safe_to_ss(variable) for variable in [*variables, *calib_symbols.keys()]]

    _input_pt, resid_pt, cache = sympy_to_pytensor(ss_variables + parameters, full_equations, cache)

    if ss_solution_dict:
        resid_pt, cache = _substitute_and_filter(resid_pt, ss_solution_dict, ss_variables, parameters, cache)

    return resid_pt, cache


def _substitute_and_filter(
    equations: list[TensorVariable],
    ss_solution_dict: SymbolDictionary,
    ss_variables: list[sp.Symbol],
    parameters: list[sp.Symbol],
    cache: dict,
) -> tuple[list[TensorVariable], dict]:
    """
    Replace known steady-state variables in each equation and drop equations with no remaining unknowns.

    Dropping the equations that depend on parameters alone keeps the system square when some variables are
    analytically known.

    Parameters
    ----------
    equations : list of TensorVariable
        Scalar equation graphs, each equal to zero at the steady state.
    ss_solution_dict : SymbolDictionary
        Analytically known steady-state values.
    ss_variables : list of sympy Symbol
        All steady-state variable symbols.
    parameters : list of sympy Symbol
        Free parameter symbols.
    cache : dict
        Sympytensor cache, extended in place with the graphs of the known values.

    Returns
    -------
    filtered_equations : list of TensorVariable
        Equations that still contain at least one unknown after substitution.
    cache : dict
        The extended cache.
    """
    ss_variable_set = set(ss_variables)
    known = {
        safe_to_ss(symbol): expression
        for symbol, expression in ss_solution_dict.to_sympy().items()
        if safe_to_ss(symbol) in ss_variable_set
    }
    if not known:
        return equations, cache

    _, known_pt, cache = sympy_to_pytensor(parameters, list(known.values()), cache)

    replacements = {}
    for symbol, expression_pt in zip(known.keys(), known_pt, strict=True):
        cache_key = make_cache_key(symbol.name, type(symbol))
        if cache_key in cache:
            replacements[cache[cache_key]] = expression_pt

    unknown_nodes = set()
    for variable in ss_variables:
        cache_key = make_cache_key(variable.name, type(variable))
        if cache_key in cache and cache[cache_key] not in replacements:
            unknown_nodes.add(cache[cache_key])

    substituted = graph_replace(equations, replacements, strict=False)
    filtered = [eq for eq in substituted if any(inp in unknown_nodes for inp in explicit_graph_inputs(eq))]

    return filtered, cache


def _try_solve_for_unknown(
    eq: sp.Expr,
    unknown: sp.Symbol,
    known_values: dict[sp.Symbol, sp.Expr],
    ss_variables: set[sp.Symbol],
) -> sp.Expr | None:
    """Solve ``eq`` for ``unknown`` after substituting the known values, or return None if no simple unique solution."""
    eq_substituted = eq.subs(known_values)

    remaining_ss_vars = eq_substituted.free_symbols & ss_variables
    if remaining_ss_vars != {unknown}:
        return None

    try:
        solutions = sp.solve(eq_substituted, unknown, dict=False)
    except (NotImplementedError, ValueError, TypeError):
        return None

    if len(solutions) != 1:
        return None

    solution = solutions[0]
    return solution if _solution_is_simple(solution) else None


def _solution_is_simple(expr: sp.Expr, max_nesting_depth: int = 5) -> bool:
    """Reject solutions with conditionals, unevaluated calculus, complex numbers, or deep nesting."""
    if expr.has(sp.Piecewise, sp.Integral, sp.Derivative, sp.I):
        return False

    def depth(node: sp.Basic, current: int = 0) -> int:
        return current if not node.args else max(depth(arg, current + 1) for arg in node.args)

    return depth(expr) <= max_nesting_depth
