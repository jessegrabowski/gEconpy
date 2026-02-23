import logging

from typing import Literal, cast

import sympy as sp

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import (
    BACKENDS,
    compile_function,
    dictionary_return_wrapper,
    make_return_dict_and_update_cache,
)
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.utilities import eq_to_ss, substitute_repeatedly

_log = logging.getLogger(__name__)

ERROR_FUNCTIONS = Literal["squared", "mean_squared", "abs", "l2-norm"]


def make_steady_state_shock_dict(shocks):
    return SymbolDictionary.fromkeys(shocks, 0.0).to_ss()


def make_steady_state_variables(variables):
    return [x.to_ss() for x in variables]


def system_to_steady_state(system, shocks):
    shock_dict = make_steady_state_shock_dict(shocks)
    return [eq_to_ss(eq).subs(shock_dict).simplify() for eq in system]


def faster_simplify(x: sp.Expr, var_list: list[TimeAwareSymbol]) -> sp.Expr:  # noqa: ARG001
    """Simplify a sympy expressing, skipping heavier algorithms."""
    # return sp.powsimp(sp.powdenest(x, force=True), force=True)
    return x


def steady_state_error_function(steady_state, variables: list[sp.Symbol], func: ERROR_FUNCTIONS = "squared") -> sp.Expr:
    ss_vars = [x.to_ss() if isinstance(x, TimeAwareSymbol) else x for x in variables]

    if func == "squared":
        error = sum([faster_simplify(eq**2, ss_vars) for eq in steady_state])
    elif func == "mean_squared":
        error = sum([faster_simplify(eq**2, ss_vars) for eq in steady_state]) / len(steady_state)
    elif func == "abs":
        error = sum([faster_simplify(sp.Abs(eq), ss_vars) for eq in steady_state])
    elif func == "l2-norm":
        error = sp.sqrt(sum([faster_simplify(eq**2, ss_vars) for eq in steady_state]))
    else:
        raise NotImplementedError(f"Error function {func} not implemented, must be one of {ERROR_FUNCTIONS}")

    return error


def compile_ss_resid_and_sq_err(
    steady_state: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    parameters: list[sp.Symbol],
    ss_error: sp.Expr,
    backend: BACKENDS,
    cache: dict,
    return_symbolic: bool,
    **kwargs,
):
    cache = {} if cache is None else cache
    ss_variables = [x.to_ss() if hasattr(x, "to_ss") else x for x in variables]
    resid_jac = sp.Matrix([[faster_simplify(eq.diff(x), ss_variables) for x in ss_variables] for eq in steady_state])

    f_ss_resid, cache = compile_function(
        ss_variables + parameters,
        steady_state,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        stack_return=True,
        pop_return=False,
        **kwargs,
    )

    f_ss_jac, cache = compile_function(
        ss_variables + parameters,
        resid_jac,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        # for pytensor, the return is a single object; don't stack into a (1,n,n) array
        stack_return=backend == "numpy",
        pop_return=True,
        **kwargs,
    )

    error_grad = [faster_simplify(ss_error.diff(x), ss_variables) for x in ss_variables]
    error_hess = sp.Matrix([[faster_simplify(eq.diff(x), ss_variables) for eq in error_grad] for x in ss_variables])

    n = len(ss_variables)
    p = sp.IndexedBase("hess_eval_point", shape=n)
    hessp_loss = cast(sp.Expr, sum([error_grad[i] * p[i] for i in range(n)]))
    hessp = [faster_simplify(hessp_loss.diff(x), ss_variables) for x in ss_variables]

    f_ss_error, cache = compile_function(
        ss_variables + parameters,
        [ss_error],
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        pop_return=True,
        stack_return=False,
        **kwargs,
    )

    f_ss_grad, cache = compile_function(
        ss_variables + parameters,
        error_grad,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        stack_return=True,
        pop_return=False,
        **kwargs,
    )

    f_ss_hess, cache = compile_function(
        ss_variables + parameters,
        error_hess,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        # error_hess is a list of one element; don't stack into a (1,n,n) array
        stack_return=backend != "pytensor",
        pop_return=True,
        **kwargs,
    )

    f_ss_hessp, cache = compile_function(
        [p, *ss_variables, *parameters],
        hessp,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        stack_return=True,
        pop_return=False,
        **kwargs,
    )

    return (f_ss_resid, f_ss_jac), (f_ss_error, f_ss_grad, f_ss_hess, f_ss_hessp), cache


def compile_known_ss(
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol | sp.Symbol],
    parameters: list[sp.Symbol],
    backend: BACKENDS,
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
        backend=backend,
        cache=cache,
        stack_return=stack_return,
        return_symbolic=return_symbolic,
        **kwargs,
    )
    if return_symbolic and backend == "pytensor":
        return make_return_dict_and_update_cache(ss_variables, f_ss, cache, TimeAwareSymbol)

    return dictionary_return_wrapper(f_ss, output_vars), cache


def compile_model_ss_functions(
    steady_state_equations,
    ss_solution_dict,
    variables,
    param_dict,
    deterministic_dict,
    calib_dict,
    error_func: ERROR_FUNCTIONS = "squared",
    backend: BACKENDS = "numpy",
    return_symbolic: bool = False,
    **kwargs,
):
    cache = {}
    f_params, cache = compile_param_dict_func(
        param_dict,
        deterministic_dict,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
    )

    calib_eqs = list(calib_dict.to_sympy().values())
    steady_state_equations = steady_state_equations + calib_eqs

    parameters = list((param_dict | deterministic_dict).to_sympy().keys())
    parameters = [x for x in parameters if x not in calib_dict.to_sympy()]

    variables = variables + list(calib_dict.to_sympy().keys())
    ss_error = steady_state_error_function(steady_state_equations, variables, error_func)

    f_ss, cache = compile_known_ss(
        ss_solution_dict,
        variables,
        parameters,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        **kwargs,
    )

    (f_ss_resid, f_ss_jac), (f_ss_error, f_ss_grad, f_ss_hess, f_ss_hessp), cache = compile_ss_resid_and_sq_err(
        steady_state_equations,
        variables,
        parameters,
        ss_error,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        **kwargs,
    )

    return (
        f_params,
        f_ss,
        (f_ss_resid, f_ss_jac),
        (f_ss_error, f_ss_grad, f_ss_hess, f_ss_hessp),
    ), cache


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
