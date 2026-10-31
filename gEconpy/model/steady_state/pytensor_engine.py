from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
import sympy as sp

from pymc.distributions.transforms import Interval, Transform, log, logodds
from pytensor.gradient import hessian_vector_product
from pytensor.gradient import jacobian as pt_jacobian
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor import TensorVariable
from pytensor.tensor.elemwise import Elemwise
from sympytensor.pytensor import PytensorPrinter

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import (
    dictionary_return_wrapper,
    make_cache_key,
    make_return_dict_and_update_cache,
    output_to_tensor,
    sympy_to_pytensor,
)
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.steady_state.sympy_engine import SteadyStateSolve
from gEconpy.parser.ast import SteadyStateSolverCall, variable_key
from gEconpy.pytensorf.compile import compile_pytensor_function
from gEconpy.utilities import (
    safe_to_ss,
    topological_order,
)

ERROR_FUNCTIONS = Literal["squared", "mean_squared", "abs", "l2-norm"]

# Root methods that step along the expression's derivatives, and how many derivatives each one needs.
_DERIVATIVE_ORDER = {"newton": 1, "halley": 2}


def _residual_scale(equation: TensorVariable) -> TensorVariable:
    """Sum of the absolute values of an equation's additive terms, the size its rounding error grows with."""
    node = equation.owner
    if node is None or not (isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, ps.Add)):
        return pt.abs(equation)
    return pt.stack([pt.abs(term) for term in node.inputs]).sum()


def infer_variable_bounds(variable: TimeAwareSymbol | sp.Symbol) -> tuple[float | None, float | None]:
    """
    Read a variable's sign assumptions and return a bound that keeps it strictly signed.

    Parameters
    ----------
    variable : TimeAwareSymbol or Symbol
        Variable whose ``assumptions0`` are inspected.

    Returns
    -------
    lower : float or None
        Lower bound, 1e-8 for a positive variable and None otherwise.
    upper : float or None
        Upper bound, -1e-8 for a negative variable and None otherwise.
    """
    assumptions = variable.assumptions0
    lower = 1e-8 if assumptions.get("positive", False) else None
    upper = -1e-8 if assumptions.get("negative", False) else None

    return lower, upper


def infer_variable_transform(
    variable: TimeAwareSymbol | sp.Symbol,
    user_bound: tuple[float | None, float | None] | None = None,
) -> Transform | None:
    """
    Pick a bijection from the real line onto a variable's feasible region for unconstrained solving.

    The feasible region comes from ``user_bound`` when given, then from the variable's GCN-declared sympy
    assumptions, and is otherwise the whole real line. The returned transform's ``backward`` maps an unconstrained
    real input onto the region: ``log`` for a positive variable, ``logodds`` for the unit interval, and ``Interval``
    for general bounds.

    Parameters
    ----------
    variable : TimeAwareSymbol or Symbol
        The steady-state variable. Its ``assumptions0`` supply the fallback region.
    user_bound : tuple of float or None, optional
        Explicit ``(lower, upper)`` bound. Either side may be None for a one-sided bound. Takes precedence over
        assumptions. Default None.

    Returns
    -------
    transform : Transform or None
        The bijection, or None for an unconstrained variable.
    """
    if user_bound is not None and user_bound != (None, None):
        lower, upper = user_bound
        return Interval(lower, upper)

    assumptions = variable.assumptions0
    if assumptions.get("unit_interval"):
        return logodds
    if assumptions.get("positive"):
        return log
    if assumptions.get("negative"):
        return Interval(None, 0.0)
    return None


def steady_state_transforms(
    variables: Sequence[TimeAwareSymbol | sp.Symbol],
    declared_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    user_bounds: dict[str, tuple[float, float]] | None = None,
) -> list[Transform | None]:
    """
    Choose one transform per steady-state variable, preferring the most specific support available.

    A bound passed by the caller wins, then the bound the GCN ``symbols`` block declares for that symbol, then the
    variable's own sympy assumptions through :func:`infer_variable_transform`.

    Parameters
    ----------
    variables : list of TimeAwareSymbol or Symbol
        The steady-state variables, in solve order.
    declared_bounds : dict mapping str to tuple of (float or None, float or None), optional
        Bounds declared in the ``symbols`` block, keyed by symbol name. Default None.
    user_bounds : dict mapping str to tuple of (float, float), optional
        Explicit bounds keyed by the steady-state symbol's name. Default None.

    Returns
    -------
    transforms : list of Transform or None
        One entry per variable, None where the variable is unconstrained.
    """
    declared_bounds = {} if declared_bounds is None else declared_bounds
    user_bounds = {} if user_bounds is None else user_bounds

    def bound_for(variable: TimeAwareSymbol | sp.Symbol) -> tuple[float | None, float | None] | None:
        if variable.name in user_bounds:
            return user_bounds[variable.name]
        key = variable_key(variable.base_name) if isinstance(variable, TimeAwareSymbol) else variable.name
        return declared_bounds.get(key)

    return [infer_variable_transform(variable, bound_for(variable)) for variable in variables]


def transform_steady_state_system(
    equations: Sequence[TensorVariable],
    ss_nodes: Sequence[TensorVariable],
    transforms: Sequence[Transform | None],
) -> tuple[
    list[TensorVariable], list[TensorVariable], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]
]:
    """
    Reparametrize a steady-state system onto unconstrained variables.

    For each steady-state node ``x`` with transform ``t``, substitute ``x -> t.backward(y)`` (identity for an
    unconstrained variable), so the returned equations are functions of fresh unconstrained inputs ``y``.

    Parameters
    ----------
    equations : list of TensorVariable
        Scalar equation graphs, each zero at the steady state.
    ss_nodes : list of TensorVariable
        Scalar input node for each steady-state variable.
    transforms : list of Transform or None
        One transform per variable, as returned by :func:`infer_variable_transform`.

    Returns
    -------
    transformed_equations : list of TensorVariable
        The equations rewritten in terms of the unconstrained inputs.
    y_nodes : list of TensorVariable
        The unconstrained input nodes, one per variable.
    to_unconstrained : callable
        Map a constrained point ``x`` to its unconstrained image ``y``.
    to_constrained : callable
        Map an unconstrained point ``y`` back to the constrained ``x``.
    """
    y_nodes, x_nodes, backward_exprs, forward_exprs, replace = [], [], [], [], {}
    for ss_node, transform in zip(ss_nodes, transforms, strict=True):
        y, x = ss_node.type(), ss_node.type()
        y.name = x.name = ss_node.name
        y_nodes.append(y)
        x_nodes.append(x)

        x_of_y = y if transform is None else transform.backward(y)
        backward_exprs.append(x_of_y)
        forward_exprs.append(x if transform is None else transform.forward(x))
        replace[ss_node] = x_of_y

    transformed_equations = graph_replace(list(equations), replace, strict=False)
    f_to_x = pytensor.function(y_nodes, backward_exprs, on_unused_input="ignore")
    f_to_y = pytensor.function(x_nodes, forward_exprs, on_unused_input="ignore")

    def to_constrained(y: np.ndarray) -> np.ndarray:
        return np.asarray(f_to_x(*np.asarray(y, dtype=float)), dtype=float)

    def to_unconstrained(x: np.ndarray) -> np.ndarray:
        return np.asarray(f_to_y(*np.asarray(x, dtype=float)), dtype=float)

    return transformed_equations, y_nodes, to_unconstrained, to_constrained


def known_steady_state_graphs(
    known_values: Mapping[sp.Symbol, sp.Expr | float | SteadyStateSolve],
    ss_definitions: dict[sp.Expr, sp.Expr] | None,
    cache: dict,
) -> list[TensorVariable]:
    """
    Convert analytic steady-state values to PyTensor graphs that share the steady state block's subexpressions.

    With ``ss_definitions``, each entry of the block is converted once, in dependency order, and the values reuse
    those graphs. A value that is a plain number, or whose symbol has no entry in the block, is converted directly.
    A :class:`~gEconpy.model.steady_state.sympy_engine.SteadyStateSolve` value is solved for numerically, and its
    solution replaces the variable in every other graph.

    Parameters
    ----------
    known_values : dict mapping sympy symbol to sympy expression, float, or SteadyStateSolve
        Known steady-state values, keyed by steady-state symbol. An expression may reference the variables that
        ``SteadyStateSolve`` values solve for, and no other steady-state variable.
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
    solves = {symbol: value for symbol, value in known_values.items() if isinstance(value, SteadyStateSolve)}
    if solves:
        return _solve_known_steady_state(known_values, solves, ss_definitions, cache)

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


def _solve_known_steady_state(
    known_values: Mapping[sp.Symbol, sp.Expr | float | SteadyStateSolve],
    solves: dict[sp.Symbol, SteadyStateSolve],
    ss_definitions: dict[sp.Expr, sp.Expr] | None,
    cache: dict,
) -> list[TensorVariable]:
    # A call's variable has no entry in the block, so converting its expression under the variable's own key gives the
    # residual or objective as a graph of the block's entries, with the variable left as an input node.
    expressions = {
        symbol: solves[symbol].expression if symbol in solves else value for symbol, value in known_values.items()
    }
    graphs = dict(zip(expressions, known_steady_state_graphs(expressions, ss_definitions, cache), strict=True))

    variables = {symbol: cache[make_cache_key(symbol.name, type(symbol))] for symbol in solves}

    # Each call is solved after the calls whose variables its expression reads, with their solutions substituted, so
    # every solution is a graph of the parameters alone.
    solutions = {}
    for symbol in _solve_order(graphs, variables):
        expression = graph_replace(graphs[symbol], solutions, strict=False) if solutions else graphs[symbol]
        solutions[variables[symbol]] = _solve_steady_state_call(expression, variables[symbol], solves[symbol].call)
        graphs[symbol] = solutions[variables[symbol]]

    return graph_replace(list(graphs.values()), solutions, strict=False)


def _solve_order(
    graphs: dict[sp.Symbol, TensorVariable], variables: dict[sp.Symbol, TensorVariable]
) -> list[sp.Symbol]:
    """Order the solved symbols so that each follows every other solved variable its expression reads."""
    reads = {}
    for symbol in variables:
        inputs = set(explicit_graph_inputs(graphs[symbol]))
        reads[symbol] = {other for other, node in variables.items() if other != symbol and node in inputs}
    order: list[sp.Symbol] = []
    while reads:
        ready = [symbol for symbol, needed in reads.items() if needed <= set(order)]
        if not ready:
            raise ValueError(f"The solver calls for {sorted(map(str, reads))} depend on each other")
        order.extend(ready)
        for symbol in ready:
            del reads[symbol]
    return order


def _solve_steady_state_call(
    expression: TensorVariable, variable: TensorVariable, call: SteadyStateSolverCall
) -> TensorVariable:
    """Return the graph of the value of ``variable`` that solves ``call``, as a function of the parameters."""
    if call.func_name == "root":
        order = _DERIVATIVE_ORDER.get(call.method, 0)
        solution, _ = pt.optimize.root_scalar(
            expression, variable, method=call.method, jac=order >= 1, hess=order >= 2, bracket=call.bounds
        )
    else:
        solution, _ = pt.optimize.minimize_scalar(
            expression, variable, method=call.method, bracket=call.bracket, bounds=call.bounds
        )

    # The Op's first input is its starting point. scipy ignores it for the bracketing and scalar-minimize methods,
    # which start from their interval, so any value in that interval serves.
    interval = call.bounds or call.bracket
    start = call.x0 if call.x0 is not None else sum(interval) / 2 if interval else 1.0
    return graph_replace(solution, {variable: pt.constant(start, dtype=variable.dtype)}, strict=False)


def compile_known_ss(
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol | sp.Symbol],
    parameters: list[sp.Symbol],
    cache: dict | None,
    return_symbolic: bool = False,
    stack_return: bool | None = None,
    ss_definitions: dict[sp.Expr, sp.Expr] | None = None,
    ss_solver_calls: Mapping[sp.Symbol, SteadyStateSolve] | None = None,
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
    ss_solver_calls : dict mapping sympy symbol to SteadyStateSolve, optional
        Steady-state variables the steady state block solves for numerically. Each one is an output alongside the
        analytic values. Default is None.
    **kwargs
        Forwarded to :func:`~gEconpy.pytensorf.compile.compile_pytensor_function`.

    Returns
    -------
    f_ss : callable, dict, or None
        Function mapping parameter values to steady-state values, or a dictionary from each variable's PyTensor node
        to its steady-state graph when ``return_symbolic`` is True. None when there are no known values and no calls.
    cache : dict
        The cache, extended with every variable created during conversion.
    """
    cache = {} if cache is None else cache
    known = ss_solution_dict.to_sympy() | dict(ss_solver_calls or {})
    if not known:
        return None, cache

    ss_variables = [safe_to_ss(variable) for variable in variables]
    ordered_solutions = {variable: known[variable] for variable in ss_variables if variable in known}

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
    ss_solver_calls: Mapping[sp.Symbol, SteadyStateSolve] | None = None,
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
    ss_solver_calls : dict mapping sympy symbol to SteadyStateSolve, optional
        Steady-state variables the steady state block solves for numerically. Default is None.

    Returns
    -------
    replacements : dict mapping TensorVariable to TensorVariable
        Steady-state input node to the graph of its known value. Only nodes already in the cache are included.
    """
    ss_variable_set = set(ss_variables)
    known = {
        safe_to_ss(symbol): value
        for symbol, value in (ss_solution_dict.to_sympy() | dict(ss_solver_calls or {})).items()
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
