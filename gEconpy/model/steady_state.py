import logging

from typing import Literal

import pytensor.tensor as pt
import sympy as sp

from pytensor.gradient import hessian_vector_product
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
from gEconpy.pytensorf.sparse_jacobian import sparse_jacobian
from gEconpy.utilities import eq_to_ss, safe_to_ss, substitute_repeatedly

_log = logging.getLogger(__name__)

ERROR_FUNCTIONS = Literal["squared", "mean_squared", "abs", "l2-norm"]


def make_steady_state_shock_dict(shocks: list[TimeAwareSymbol]) -> SymbolDictionary:
    return SymbolDictionary.fromkeys(shocks, 0.0).to_ss()


def make_steady_state_variables(variables: list[TimeAwareSymbol]) -> list[sp.Symbol]:
    return [x.to_ss() for x in variables]


def system_to_steady_state(system: list[sp.Expr], shocks: list[TimeAwareSymbol]) -> list[sp.Expr]:
    shock_dict = make_steady_state_shock_dict(shocks)
    return [eq_to_ss(eq).subs(shock_dict).simplify() for eq in system]


def pt_error_from_resid(
    resid: TensorVariable,
    func: ERROR_FUNCTIONS = "squared",
) -> TensorVariable:
    """
    Build a pytensor scalar error graph from a stacked residual vector.

    Parameters
    ----------
    resid : TensorVariable
        Stacked residual vector of shape ``(n_eq,)``.
    func : str, optional
        Error metric. One of ``'squared'``, ``'mean_squared'``, ``'abs'``, or ``'l2-norm'``. Default is ``'squared'``.

    Returns
    -------
    error : TensorVariable
        Scalar error graph node.
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


def _ss_residual_to_pytensor(
    steady_state_equations: list[sp.Expr],
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol],
    param_dict: SymbolDictionary,
    deterministic_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    cache: dict | None = None,
) -> tuple[list[TensorVariable], dict]:
    """Convert the steady-state residual system from sympy to a pytensor computation graph.

    This is the single sympy-to-pytensor bridge for steady-state solving. The returned equation list is the only
    thing derived from sympy; all downstream derivatives (Jacobian, gradient, Hessian, Hessian-vector product) are
    built from it via pytensor autodiff in :func:`build_root_graphs` and :func:`build_minimize_graphs`.

    The returned cache contains every sympy-to-pytensor mapping created during conversion, including parameter
    nodes, deterministic parameter expressions, steady-state variable nodes, and known steady-state solution
    subgraphs. Callers recover specific nodes by looking up ``(name, cls, ...)`` keys in the cache.

    Parameters
    ----------
    steady_state_equations : list of sp.Expr
        Steady-state equations in residual form (each expression equals zero).
    ss_solution_dict : SymbolDictionary
        Analytically known steady-state solutions from the ``STEADY_STATE`` block and identity propagation.
    variables : list of TimeAwareSymbol
        Model variables (without calibrated parameter symbols).
    param_dict : SymbolDictionary
        Free parameter names and default values.
    deterministic_dict : SymbolDictionary
        Deterministic parameters defined as functions of free parameters.
    calib_dict : SymbolDictionary
        Calibration equations mapping calibrated parameter symbols to the steady-state expressions that pin them.
    cache : dict, optional
        Existing sympytensor cache to extend. If None, a fresh cache is created. Passing an existing cache ensures
        that pytensor nodes are shared across multiple graph-building calls.

    Returns
    -------
    equations : list of TensorVariable
        Individual scalar equation graphs, each equal to zero at the steady state.
    cache : dict
        Sympytensor cache mapping ``(name, assumptions, ...)`` tuples to pytensor nodes. Contains all parameter,
        deterministic, steady-state variable, and known-solution nodes created during conversion.
    """
    if cache is None:
        cache = {}

    compile_param_dict_func(param_dict, deterministic_dict, cache=cache, return_symbolic=True)

    calib_eqs = list(calib_dict.to_sympy().values())
    full_equations = steady_state_equations + calib_eqs

    parameters = list((param_dict | deterministic_dict).to_sympy().keys())
    parameters = [x for x in parameters if x not in calib_dict.to_sympy()]

    full_variables = list(variables) + list(calib_dict.to_sympy().keys())
    ss_variables = [x.to_ss() if hasattr(x, "to_ss") else x for x in full_variables]

    input_symbols = ss_variables + parameters
    _input_pt, resid_pt, cache = sympy_to_pytensor(input_symbols, full_equations, cache)

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

    After substitution, any equation that depends only on parameters (not on any unknown SS variable) is removed.
    This keeps the system square when some variables are analytically known.

    Parameters
    ----------
    equations : list of TensorVariable
        Individual scalar equation graphs, each equal to zero at the steady state.
    ss_solution_dict : SymbolDictionary
        Analytically known steady-state solutions.
    ss_variables : list of sp.Symbol
        All steady-state variable symbols.
    parameters : list of sp.Symbol
        Free parameter symbols.
    cache : dict
        Sympytensor cache. **Mutated in place**.

    Returns
    -------
    filtered_equations : list of TensorVariable
        Equations that still contain at least one unknown SS variable after substitution.
    cache : dict
        Updated cache.
    """
    ss_dict_sympy = ss_solution_dict.to_sympy()
    known = {safe_to_ss(k): v for k, v in ss_dict_sympy.items() if safe_to_ss(k) in ss_variables}

    if not known:
        return equations, cache

    known_symbols = list(known.keys())
    known_exprs = list(known.values())

    _, known_pt, cache = sympy_to_pytensor(parameters, known_exprs, cache)

    replacements = {}
    known_node_ids = set()
    for sym, expr_pt in zip(known_symbols, known_pt, strict=True):
        cache_key = make_cache_key(sym.name, type(sym))
        if cache_key in cache:
            old_node = cache[cache_key]
            replacements[old_node] = expr_pt
            known_node_ids.add(id(old_node))

    unknown_node_ids: set[int] = set()
    for var in ss_variables:
        cache_key = make_cache_key(var.name, type(var))
        if cache_key in cache and id(cache[cache_key]) not in known_node_ids:
            unknown_node_ids.add(id(cache[cache_key]))

    substituted = graph_replace(equations, replacements, strict=False)

    filtered = []
    for eq in substituted:
        eq_inputs = explicit_graph_inputs(eq)
        has_unknown = any(id(inp) in unknown_node_ids for inp in eq_inputs)
        if has_unknown:
            filtered.append(eq)

    return filtered, cache


def build_root_graphs(
    equations: list[TensorVariable],
    ss_input_nodes: list[TensorVariable],
    use_jac: bool = True,
) -> tuple[TensorVariable, TensorVariable | None]:
    """
    Build the pytensor graphs needed by ``scipy.optimize.root``.

    The equations are stacked into a residual vector internally. The Jacobian is computed via :func:`sparse_jacobian`,
    which exploits the sparsity pattern of the steady-state system.

    Parameters
    ----------
    equations : list of TensorVariable
        Individual scalar equation graphs, each equal to zero at the steady state.
    ss_input_nodes : list of TensorVariable
        Scalar input nodes for each steady-state variable.
    use_jac : bool, optional
        Whether to build the Jacobian graph. Default is True.

    Returns
    -------
    resid : TensorVariable
        Stacked residual vector of shape ``(n_eq,)``.
    jac : TensorVariable or None
        Jacobian of shape ``(n_eq, n_var)``, or None if ``use_jac`` is False.
    """
    resid = pt.stack(equations) if equations else pt.zeros(0)
    jac = None
    if use_jac:
        jac = sparse_jacobian(equations, ss_input_nodes)

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
    Build the pytensor graphs needed by ``scipy.optimize.minimize``.

    The equations are stacked into a residual vector and reduced to a scalar error. Only the requested derivative
    graphs are constructed.

    Parameters
    ----------
    equations : list of TensorVariable
        Individual scalar equation graphs, each equal to zero at the steady state.
    ss_input_nodes : list of TensorVariable
        Scalar input nodes for each steady-state variable.
    error_func : str, optional
        Error metric. Default is ``'squared'``.
    use_jac : bool, optional
        Whether to build the gradient graph. Default is True.
    use_hess : bool, optional
        Whether to build the full Hessian graph. Default is False.
    use_hessp : bool, optional
        Whether to build the Hessian-vector product graph. Default is True.

    Returns
    -------
    error : TensorVariable
        Scalar error.
    grad : TensorVariable or None
        Gradient of shape ``(n_var,)``, or None if ``use_jac`` is False.
    hess : TensorVariable or None
        Hessian of shape ``(n_var, n_var)``, or None if ``use_hess`` is False.
    hessp_out : TensorVariable or None
        Hessian-vector product output of shape ``(n_var,)``, or None if ``use_hessp`` is False.
    hessp_p : TensorVariable or None
        Direction vector input for the Hessian-vector product, or None if ``use_hessp`` is False.
    """
    resid = pt.stack(equations) if equations else pt.zeros(0)
    error = pt_error_from_resid(resid, error_func)

    grad = None
    if use_jac or use_hess or use_hessp:
        grad_components = pt.grad(error, ss_input_nodes)
        grad = pt.stack(grad_components)

    hess = None
    if use_hess:
        hess = sparse_jacobian(grad_components, ss_input_nodes)

    hessp_out = None
    hessp_p = None
    if use_hessp:
        hessp_p = pt.dvector("hess_eval_point")
        hessp_out = pt.stack(
            hessian_vector_product(error, ss_input_nodes, [hessp_p[i] for i in range(len(ss_input_nodes))])
        )

    return error, grad, hess, hessp_out, hessp_p


def compile_known_ss(
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol | sp.Symbol],
    parameters: list[sp.Symbol],
    cache: dict,
    return_symbolic: bool = False,
    stack_return: bool | None = None,
    **kwargs,
):
    def to_ss(x):
        if isinstance(x, TimeAwareSymbol):
            return x.to_ss()
        return x

    cache = {} if cache is None else cache
    if not ss_solution_dict:
        return None, cache

    ss_solution_dict = ss_solution_dict.to_sympy()
    ss_variables = [to_ss(x) for x in variables]

    sorted_solution_dict = {to_ss(k): ss_solution_dict[to_ss(k)] for k in ss_variables if k in ss_solution_dict}

    output_vars, output_exprs = (
        list(sorted_solution_dict.keys()),
        list(sorted_solution_dict.values()),
    )
    if stack_return is None:
        stack_return = bool(not return_symbolic)

    f_ss, cache = compile_function(
        parameters,
        output_exprs,
        cache=cache,
        stack_return=stack_return,
        return_symbolic=return_symbolic,
        **kwargs,
    )
    if return_symbolic:
        return make_return_dict_and_update_cache(ss_variables, f_ss, cache, TimeAwareSymbol)

    return dictionary_return_wrapper(f_ss, output_vars), cache


def print_steady_state(ss_dict: SteadyStateResults):
    output = []
    if not ss_dict.success:
        output.append("Values come from the latest solver iteration but are NOT a valid steady state.")

    max_var_name = max(len(x) for x in list(ss_dict.keys())) + 5

    calibrated_outputs = []
    for key, value in ss_dict.to_sympy().items():
        if isinstance(key, TimeAwareSymbol):
            output.append(f"{key.name:{max_var_name}}{value:>10.3f}")
        else:
            calibrated_outputs.append(f"{key.name:{max_var_name}}{value:>10.3f}")

    if len(calibrated_outputs) > 0:
        output.append("\n")
        output.extend(calibrated_outputs)

    _log.info("\n".join(output))


def simplify_provided_ss_equations(
    ss_solution_dict: SymbolDictionary, variables: list[TimeAwareSymbol]
) -> SymbolDictionary:
    """
    Substitute intermediate variables out of user-provided steady-state equations.

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        User-provided steady-state equations from the STEADY_STATE block.
    variables : list of TimeAwareSymbol
        Model variables.

    Returns
    -------
    SymbolDictionary
        Steady-state equations containing only model variables.
    """
    if not ss_solution_dict:
        return SymbolDictionary()

    ss_variables = [x.to_ss() for x in variables]
    ss_dict_sympy = ss_solution_dict.to_sympy()

    extra_equations = SymbolDictionary({k: v for k, v in ss_dict_sympy.items() if k not in ss_variables})
    if not extra_equations:
        return ss_solution_dict

    simplified = SymbolDictionary({k: v for k, v in ss_dict_sympy.items() if k in ss_variables})
    for var, eq in simplified.items():
        if hasattr(eq, "subs"):
            simplified[var] = substitute_repeatedly(eq, extra_equations)

    return simplified


def _solution_is_simple(expr: sp.Expr, max_nesting_depth: int = 5) -> bool:
    """Reject solutions with conditionals, unevaluated calculus, or excessive nesting."""
    if expr.has(sp.Piecewise, sp.Integral, sp.Derivative, sp.I):
        return False

    def depth(e: sp.Basic, current: int = 0) -> int:
        """Recursively compute the maximum depth of the Sympy expression tree."""
        return current if not e.args else max(depth(arg, current + 1) for arg in e.args)

    return depth(expr) <= max_nesting_depth


def _try_solve_for_unknown(
    eq: sp.Expr,
    unknown: sp.Symbol,
    known_values: dict[sp.Symbol, sp.Expr],
    ss_variables: set[sp.Symbol],
) -> sp.Expr | None:
    """
    Attempt to solve a single equation for one unknown after substituting known values.

    Returns the solution only if it is unique and passes simplicity checks.
    Returns None if the equation cannot be solved or the solution is too complex.
    """
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


def propagate_steady_state_through_identities(
    ss_solution_dict: SymbolDictionary,
    steady_state_equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    max_iterations: int = 100,
) -> SymbolDictionary:
    """
    Extend user-provided steady-state values by solving simple single-unknown equations.

    Iterates over the equation system, solving any equation that has exactly one unknown
    after substituting currently-known values. Repeats until no further progress is made.

    The solver is conservative: it only accepts unique solutions that pass simplicity
    checks (no conditionals, no complex numbers, limited nesting depth).

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        User-provided steady-state values.
    steady_state_equations : list of sp.Expr
        Model equations in steady-state residual form (each expression equals zero).
    variables : list of TimeAwareSymbol
        Model variables.
    max_iterations : int, default 100
        Maximum number of passes over the equation system.

    Returns
    -------
    SymbolDictionary
        Original values plus any additional values that could be inferred.
    """
    # Use the actual SS symbols from variables to preserve assumptions
    ss_variables = {x.to_ss() for x in variables}

    # Start with a copy of the input, preserving symbol identity
    result = ss_solution_dict.to_sympy().copy() if ss_solution_dict else {}

    for _ in range(max_iterations):
        progress = False

        for eq in steady_state_equations:
            if not isinstance(eq, sp.Basic):
                continue

            unknowns = (eq.free_symbols & ss_variables) - set(result.keys())
            if len(unknowns) != 1:
                continue

            unknown = next(iter(unknowns))
            solution = _try_solve_for_unknown(eq, unknown, result, ss_variables)

            if solution is not None:
                result[unknown] = float(solution) if solution.is_number else solution
                progress = True

        if not progress:
            break

    return SymbolDictionary(result)
