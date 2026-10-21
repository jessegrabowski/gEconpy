import logging

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pytensor.tensor as pt
import sympy as sp

from pytensor.gradient import hessian_vector_product
from pytensor.gradient import jacobian as pt_jacobian
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor import TensorVariable
from sympytensor.pytensor import PytensorPrinter

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import (
    dictionary_return_wrapper,
    make_cache_key,
    make_return_dict_and_update_cache,
    output_to_tensor,
    sympy_to_pytensor,
)
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.pytensorf.compile import compile_pytensor_function
from gEconpy.utilities import (
    _rebuild_unevaluated,
    _xreplace_unevaluated,
    eq_to_ss,
    flatten_substitution_dict,
    safe_to_ss,
    substitute_unevaluated,
    topological_order,
)

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


def resubstitute_steady_state(relationships: Sequence[sp.Eq]) -> list[sp.Eq]:
    """
    Rewrite each steady-state relationship in terms of the other steady-state variables, for printing.

    A relationship the model holds has every other steady-state value inlined into it, so ``K_ss`` repeats the
    whole expression for ``r_ss``. Wherever a subexpression is exactly another relationship's right side, it is
    replaced by that relationship's variable. Matching is structural and top-down, so the largest match wins and
    no new names are introduced. A right side that is a single number or symbol is never a replacement target,
    because a constant such as ``A_ss = 1`` would otherwise rename every literal one.

    The trees are rebuilt unevaluated: evaluating an inlined steady-state tree costs time exponential in its
    depth.

    Parameters
    ----------
    relationships : sequence of Eq
        Steady-state variables equal to their solutions, as ``Model.steady_state_relationships`` holds them.

    Returns
    -------
    relationships : list of Eq
        The same relationships, in the same order, with every other variable's solution folded back in.
    """
    targets = {eq.rhs: eq.lhs for eq in relationships if not (eq.rhs.is_Number or eq.rhs.is_Symbol)}
    cache: dict[sp.Basic, sp.Basic] = {}

    resubstituted = []
    for eq in relationships:
        others = {rhs: lhs for rhs, lhs in targets.items() if lhs != eq.lhs}
        rhs = _drop_identity_terms(_xreplace_unevaluated(eq.rhs, others), cache)
        resubstituted.append(sp.Eq(eq.lhs, rhs, evaluate=False))
    return resubstituted


def _drop_identity_terms(expr: sp.Basic, cache: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """
    Remove the ones and zeros substitution leaves in an unevaluated tree, and print an integral float as an integer.

    A one is a factor of a product or the base of a power, and a zero is a term of a sum.

    The cache is keyed on subtrees, because an inlined tree shares them and a plain recursion revisits each share.
    """
    if expr in cache:
        return cache[expr]

    if isinstance(expr, sp.Float) and float(expr).is_integer():
        cleaned = sp.Integer(int(expr))
    elif expr.is_Atom or not expr.args:
        cleaned = expr
    else:
        args = [_drop_identity_terms(arg, cache) for arg in expr.args]
        identity = {sp.Mul: sp.Integer(1), sp.Add: sp.Integer(0)}.get(type(expr))
        if identity is not None:
            args = [arg for arg in args if arg != identity] or [identity]
        if isinstance(expr, sp.Pow) and args[0] == 1:
            cleaned = sp.Integer(1)
        elif identity is not None and len(args) == 1:
            cleaned = args[0]
        else:
            cleaned = _rebuild_unevaluated(expr, args)

    cache[expr] = cleaned
    return cleaned


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
    # One steady-state equation per dynamic equation, in the same order, so an equation id indexes both lists.
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
    flat_intermediates = flatten_substitution_dict(intermediates, evaluate=False)
    for variable, eq in simplified.items():
        if isinstance(eq, sp.Basic):
            simplified[variable] = substitute_unevaluated(eq, flat_intermediates)

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


def known_steady_state_graphs(
    known_values: dict[sp.Symbol, sp.Expr | float],
    ss_definitions: dict[sp.Expr, sp.Expr] | None,
    cache: dict,
) -> list[TensorVariable]:
    """
    Convert analytic steady-state values to PyTensor graphs that share the steady state block's subexpressions.

    With ``ss_definitions``, each entry of the block is converted once, in dependency order, and the values reuse
    those graphs. A value that is a plain number, or whose symbol has no entry in the block, is converted directly.

    Parameters
    ----------
    known_values : dict mapping sympy symbol to sympy expression or float
        Known steady-state values, keyed by steady-state symbol, each a function of the parameters only.
    ss_definitions : dict mapping sympy symbol to sympy expression, or None
        Entries of the steady state block as written, whose values may reference other entries. None converts each
        value of ``known_values`` alone.
    cache : dict
        Sympytensor cache, extended with a node for every symbol the values reference. The block's entries are bound
        to their graphs only in a copy, so elsewhere a steady-state symbol keeps mapping to its input node.

    Returns
    -------
    graphs : list of TensorVariable
        One graph per entry of ``known_values``, in the same order.
    """
    if not ss_definitions:
        printer = PytensorPrinter(cache=cache, settings={})
        return [output_to_tensor(value, cache, printer=printer) for value in known_values.values()]

    referenced = {
        symbol
        for value in (*known_values.values(), *ss_definitions.values())
        if isinstance(value, sp.Basic)
        for symbol in value.free_symbols
    }
    for symbol in referenced - set(ss_definitions):
        output_to_tensor(symbol, cache)

    # Each binding replaces a cache entry, so these conversions cannot share a printer.
    local_cache = dict(cache)
    definition_graphs = {}
    for key in topological_order(ss_definitions):
        graph = output_to_tensor(ss_definitions[key], local_cache)
        local_cache[make_cache_key(key.name, type(key))] = graph
        definition_graphs[key.name] = graph

    def from_definitions(symbol: sp.Symbol, value: sp.Expr | float) -> bool:
        # A plain number may be a user-fixed value that overrides the block, and is as cheap to convert either way.
        is_expression = isinstance(value, sp.Basic) and not value.is_number
        return is_expression and symbol.name in definition_graphs

    printer = PytensorPrinter(cache=local_cache, settings={})
    return [
        definition_graphs[symbol.name]
        if from_definitions(symbol, value)
        else output_to_tensor(value, local_cache, printer=printer)
        for symbol, value in known_values.items()
    ]


def compile_known_ss(
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol | sp.Symbol],
    parameters: list[sp.Symbol],
    cache: dict | None,
    return_symbolic: bool = False,
    stack_return: bool | None = None,
    ss_definitions: dict[sp.Expr, sp.Expr] | None = None,
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
    ss_definitions : dict mapping sympy symbol to sympy expression, optional
        Entries of the steady state block as written. When given, the outputs share the block's subexpressions; see
        :func:`known_steady_state_graphs`. Default is None, which converts each value of ``ss_solution_dict`` alone.
    **kwargs
        Forwarded to :func:`~gEconpy.pytensorf.compile.compile_pytensor_function`.

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

    input_pt, _, cache = sympy_to_pytensor(parameters, [], cache)
    output_pt = known_steady_state_graphs(ordered_solutions, ss_definitions, cache)
    if stack_return:
        output_pt = pt.stack(output_pt)

    if return_symbolic:
        return make_return_dict_and_update_cache(list(ordered_solutions.keys()), output_pt, cache, TimeAwareSymbol)

    f_ss = compile_pytensor_function(input_pt, output_pt, **kwargs)
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
    variables: list[TimeAwareSymbol],
    param_dict: SymbolDictionary,
    deterministic_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    cache: dict | None = None,
) -> tuple[list[TensorVariable], dict]:
    """
    Convert the steady-state residual system from sympy to PyTensor graphs.

    Every derivative the steady-state solvers need is built from the returned equations by PyTensor autodiff in
    :func:`build_root_graphs` and :func:`build_minimize_graphs`. Callers recover the parameter and variable nodes
    from the returned cache with :func:`~gEconpy.model.compile.make_cache_key`. Known steady-state values are applied
    to the returned graphs by :func:`build_steady_state_system`.

    Parameters
    ----------
    steady_state_equations : list of sympy expression
        Steady-state equations in residual form, each equal to zero.
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
        Scalar equation graphs, one per steady-state and calibrating equation, each equal to zero at the steady state.
    cache : dict
        The cache, extended with every parameter, deterministic parameter, and steady-state variable created during
        conversion.
    """
    if cache is None:
        cache = {}

    compile_param_dict_func(param_dict, deterministic_dict, cache=cache, return_symbolic=True)

    calib_symbols = calib_dict.to_sympy()
    full_equations = steady_state_equations + list(calib_symbols.values())

    parameters = [param for param in (param_dict | deterministic_dict).to_sympy() if param not in calib_symbols]
    ss_variables = [safe_to_ss(variable) for variable in [*variables, *calib_symbols.keys()]]

    _input_pt, resid_pt, cache = sympy_to_pytensor(ss_variables + parameters, full_equations, cache)
    return resid_pt, cache


def known_steady_state_replacements(
    ss_solution_dict: SymbolDictionary,
    ss_variables: list[sp.Symbol],
    parameters: list[sp.Symbol],
    cache: dict,
    ss_definitions: dict[sp.Expr, sp.Expr] | None = None,
) -> dict[TensorVariable, TensorVariable]:
    """
    Map each analytically known steady-state node to its graph in the parameters.

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        Analytically known steady-state values.
    ss_variables : list of sympy Symbol
        All steady-state variable symbols. Known values for other symbols are ignored.
    parameters : list of sympy Symbol
        Parameter symbols the known values may reference.
    cache : dict
        Sympytensor cache, extended in place with the parameter nodes the known values need.
    ss_definitions : dict mapping sympy symbol to sympy expression, optional
        Entries of the steady state block as written; see :func:`known_steady_state_graphs`. Default is None.

    Returns
    -------
    replacements : dict mapping TensorVariable to TensorVariable
        Steady-state input node to the graph of its known value. Only nodes already in the cache are included.
    """
    ss_variable_set = set(ss_variables)
    known = {
        safe_to_ss(symbol): expression
        for symbol, expression in ss_solution_dict.to_sympy().items()
        if safe_to_ss(symbol) in ss_variable_set
    }
    if not known:
        return {}

    _, _, cache = sympy_to_pytensor(parameters, [], cache)
    known_pt = known_steady_state_graphs(known, ss_definitions, cache)

    replacements = {}
    for symbol, expression_pt in zip(known.keys(), known_pt, strict=True):
        cache_key = make_cache_key(symbol.name, type(symbol))
        if cache_key in cache:
            replacements[cache[cache_key]] = expression_pt
    return replacements


@dataclass(frozen=True)
class SteadyStateSystem:
    """
    The steady-state residual system with known values substituted and fixed values left as inputs, as graphs only.

    Attributes
    ----------
    unknowns : list of TensorVariable
        Steady-state nodes left to solve for, in model order.
    residuals : list of TensorVariable
        Residual rows that depend on at least one unknown.
    checks : list of TensorVariable
        Residual rows that depend on no unknown. They are zero at a steady state but play no part in the solve.
    check_indices : list of int
        Position of each check row in the full residual system.
    steady_state : list of TensorVariable
        One graph per steady-state node, in model order: the known graph, the fixed node itself, or the unknown.
    """

    unknowns: list[TensorVariable]
    residuals: list[TensorVariable]
    checks: list[TensorVariable]
    check_indices: list[int]
    steady_state: list[TensorVariable]


def build_steady_state_system(
    equations: Sequence[TensorVariable],
    ss_nodes: Sequence[TensorVariable],
    known: dict[TensorVariable, TensorVariable],
    fixed: Sequence[TensorVariable] = (),
) -> SteadyStateSystem:
    """
    Split the steady-state residual system into rows to solve and rows to check.

    Known nodes are replaced by their graphs. Fixed nodes stay in the graph as inputs, so their values are supplied
    at call time like parameters. A fixed node takes precedence over a known graph for the same node. Every other
    steady-state node is an unknown.

    Parameters
    ----------
    equations : list of TensorVariable
        The full residual system, one scalar graph per equation, over every steady-state node.
    ss_nodes : list of TensorVariable
        Every steady-state node, in model order.
    known : dict mapping TensorVariable to TensorVariable
        Steady-state node to the graph of its analytically known value.
    fixed : list of TensorVariable, optional
        Steady-state nodes whose values the caller supplies. Default is none.

    Returns
    -------
    system : SteadyStateSystem
        The split system.
    """
    fixed_set = set(fixed)
    replacements = {node: graph for node, graph in known.items() if node not in fixed_set}
    unknowns = [node for node in ss_nodes if node not in replacements and node not in fixed_set]
    unknown_set = set(unknowns)

    replaced = graph_replace(list(equations), replacements, strict=False) if replacements else list(equations)
    depends_on_unknown = [any(node in unknown_set for node in explicit_graph_inputs(row)) for row in replaced]

    return SteadyStateSystem(
        unknowns=unknowns,
        residuals=[row for row, solve in zip(replaced, depends_on_unknown, strict=True) if solve],
        checks=[row for row, solve in zip(replaced, depends_on_unknown, strict=True) if not solve],
        check_indices=[index for index, solve in enumerate(depends_on_unknown) if not solve],
        steady_state=[replacements.get(node, node) for node in ss_nodes],
    )


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
