import logging

from collections.abc import Callable
from typing import Any

import numpy as np
import sympy as sp

from scipy.optimize import OptimizeResult

from gEconpy.classes.containers import (
    SteadyStateResults,
    string_keys_to_sympy,
)
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

_log = logging.getLogger(__name__)


def flatten_list(items: Any, result_list: list[Any] | None = None) -> list[Any]:
    """
    Flatten an arbitrarily nested list into a single flat list.

    Parameters
    ----------
    items : list or object
        Nested list to flatten. A non-list input is treated as a single element.
    result_list : list, optional
        List to append results to. A new list is created by default.

    Returns
    -------
    result_list : list
        Flat list of the non-list leaves of ``items``.
    """
    if result_list is None:
        result_list = []

    if not isinstance(items, list):
        result_list.append(items)
        return result_list

    for item in items:
        if isinstance(item, list):
            result_list = flatten_list(item, result_list)
        else:
            result_list.append(item)
    return result_list


def set_equality_equals_zero(eq: sp.Expr) -> sp.Expr:
    """
    Rewrite a sympy equality as an expression equal to zero.

    Parameters
    ----------
    eq : sympy expression
        Expression to rewrite. Non-equalities are returned unchanged.

    Returns
    -------
    eq : sympy expression
        ``eq.rhs - eq.lhs`` if ``eq`` is an equality, otherwise ``eq``.
    """
    if not isinstance(eq, sp.Eq):
        return eq

    return eq.rhs - eq.lhs


def eq_to_ss(eq: sp.Expr, shocks: list[TimeAwareSymbol] | None = None) -> sp.Expr:
    """
    Replace every time-aware symbol in an equation by its steady-state counterpart.

    Parameters
    ----------
    eq : sympy expression
        Equation to convert.
    shocks : list of TimeAwareSymbol, optional
        Shocks whose steady-state values are set to zero. No shocks are zeroed by default.

    Returns
    -------
    eq : sympy expression
        Equation written entirely in steady-state symbols.
    """
    shock_subs = {} if shocks is None else {x.to_ss(): 0.0 for x in shocks}
    to_ss_subs = {x: x.to_ss() for x in eq.atoms(TimeAwareSymbol)}

    return eq.subs(to_ss_subs).subs(shock_subs)


def safe_to_ss(x: sp.Symbol) -> sp.Symbol:
    """
    Convert ``x`` to steady-state if it is TimeAware, or return it unchanged otherwise.

    Parameters
    ----------
    x : sympy Symbol
        Symbol to convert.

    Returns
    -------
    x : sympy Symbol
        Steady-state symbol if ``x`` is a TimeAwareSymbol, otherwise ``x`` itself.
    """
    if isinstance(x, TimeAwareSymbol):
        return x.to_ss()
    return x


def expand_subs_for_all_times(sub_dict: dict[TimeAwareSymbol, TimeAwareSymbol]) -> dict[TimeAwareSymbol, sp.Expr]:
    """
    Expand a substitution dictionary to cover time indices t-1, t, t+1, and the steady state.

    Parameters
    ----------
    sub_dict : dict mapping TimeAwareSymbol to TimeAwareSymbol
        Substitutions defined at a single time index.

    Returns
    -------
    result : dict mapping TimeAwareSymbol to TimeAwareSymbol
        Substitutions repeated at every time index. Non-time-aware values are reused unchanged.
    """
    result = {}
    for lhs, rhs in sub_dict.items():
        for t in [-1, 0, 1, "ss"]:
            result[lhs.set_t(t)] = rhs.set_t(t) if isinstance(rhs, TimeAwareSymbol) else rhs

    return result


def step_equation_forward(eq: sp.Expr) -> sp.Expr:
    """
    Advance the time index of every time-aware symbol in an equation by one period.

    Parameters
    ----------
    eq : sympy expression
        Equation to step forward.

    Returns
    -------
    eq : sympy expression
        Equation with all time indices incremented.
    """
    to_step = [variable for variable in set(eq.atoms()) if hasattr(variable, "step_forward")]

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=True):
        eq = eq.subs({variable: variable.step_forward()})

    return eq


def step_equation_backward(eq: sp.Expr) -> sp.Expr:
    """
    Move the time index of every time-aware symbol in an equation back by one period.

    Parameters
    ----------
    eq : sympy expression
        Equation to step backward.

    Returns
    -------
    eq : sympy expression
        Equation with all time indices decremented.
    """
    to_step = [variable for variable in set(eq.atoms()) if hasattr(variable, "step_backward")]

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=False):
        eq = eq.subs({variable: variable.step_backward()})

    return eq


def diff_through_time(eq: sp.Expr, dx: TimeAwareSymbol, discount_factor: sp.Expr | int = 1) -> sp.Expr:
    r"""
    Differentiate an equation with respect to a time-aware symbol, summing across time shifts.

    Compute

    .. math::

        \sum_{k=0}^{K} \beta^k \frac{\partial}{\partial x} \mathrm{step}^k(\mathrm{eq})

    where each step shifts every time-aware symbol's time index forward by one. The number of steps :math:`K` is the
    spread between the time index of ``dx`` and the earliest time index at which its base symbol appears in ``eq``.
    Stepping forward further can only increase every time index, so once the earliest occurrence has shifted past
    ``dx`` no additional contribution is possible.

    Parameters
    ----------
    eq : sympy expression
        Equation to differentiate, typically a Lagrangian.
    dx : TimeAwareSymbol
        Variable to differentiate with respect to.
    discount_factor : sympy expression or int, optional
        Multiplicative discount factor applied at each forward step. Defaults to 1, which applies no discounting.

    Returns
    -------
    total : sympy expression
        Sum of discounted derivatives across all relevant time shifts.
    """
    times_in_eq = {a.time_index for a in eq.atoms(TimeAwareSymbol) if a.base_name == dx.base_name}
    if not times_in_eq:
        return sp.S.Zero
    n_iters = max(0, dx.time_index - min(times_in_eq))

    total = sp.S.Zero
    for _ in range(n_iters + 1):
        total += eq.diff(dx)
        eq = step_equation_forward(eq) * discount_factor
        discount_factor = step_equation_forward(discount_factor)
    return total


def substitute_all_equations(
    eqs: list[sp.Expr] | dict[Any, Any], *sub_dicts: dict[Any, Any]
) -> list[sp.Expr] | dict[Any, Any]:
    """
    Apply one or more substitution dictionaries to a collection of equations.

    Parameters
    ----------
    eqs : list of sympy expression or dict
        Equations to substitute into. Dictionary values that are plain numbers are left unchanged.
    *sub_dicts : dict
        Substitution dictionaries, merged left to right before use. Keys may be strings or sympy symbols.

    Returns
    -------
    eqs : list of sympy expression or dict
        Equations after substitution, in the same container type as the input.
    """
    sub_dict = string_keys_to_sympy(merge_dictionaries(*sub_dicts))

    if isinstance(eqs, list):
        return [eq.subs(sub_dict) for eq in eqs]
    return {key: value if isinstance(value, int | float) else value.subs(sub_dict) for key, value in eqs.items()}


def is_variable(x: object) -> bool:
    return isinstance(x, TimeAwareSymbol)


def is_number(x: str) -> bool:
    """
    Check if string x is a numeric string (int or float).

    Parameters
    ----------
    x : str
        Value to check. Non-string inputs return False, even if they are numeric types.

    Returns
    -------
    is_number : bool
        True if ``x`` is a string that parses as a float.
    """
    if not isinstance(x, str):
        return False

    s = x.strip()
    if not s:
        return False
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def unpack_keys_and_values(d: dict[Any, Any]) -> tuple[list[Any], list[Any]]:
    """
    Split a dictionary into parallel lists of keys and values.

    Parameters
    ----------
    d : dict
        Dictionary to unpack.

    Returns
    -------
    keys : list
        Dictionary keys, in insertion order.
    values : list
        Dictionary values, in the same order as ``keys``.
    """
    keys = list(d.keys())
    values = list(d.values())

    return keys, values


def merge_dictionaries(*dicts: dict[Any, Any]) -> dict[Any, Any]:
    """
    Merge dictionaries into a single dictionary.

    Parameters
    ----------
    *dicts : dict
        Dictionaries to merge. Later dictionaries overwrite keys set by earlier ones.

    Returns
    -------
    result : dict
        Merged dictionary.
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def make_all_var_time_combos(var_list: list[TimeAwareSymbol]) -> list[TimeAwareSymbol]:
    """
    List every variable at time indices t-1, t, t+1, and the steady state.

    Parameters
    ----------
    var_list : list of TimeAwareSymbol
        Variables to expand.

    Returns
    -------
    result : list of TimeAwareSymbol
        Each input variable repeated once per time index.
    """
    result = []
    for x in var_list:
        result.extend([x.set_t(-1), x.set_t(0), x.set_t(1), x.set_t("ss")])

    return result


def postprocess_optimizer_res(
    res: OptimizeResult,
    res_dict: SteadyStateResults,
    f_resid: Callable[..., np.ndarray],
    f_grad: Callable[..., np.ndarray],
    tol: float = 1e-6,
    verbose: bool = True,
) -> SteadyStateResults:
    """
    Check an optimizer result against residual and gradient tolerances and report the outcome.

    Success requires the numeric check to pass. The optimizer's own flag does not decide the outcome in either
    direction: an optimizer that reports failure at a point satisfying the tolerances still counts as a success, and
    one that reports convergence at a point violating them counts as a failure.

    Parameters
    ----------
    res : OptimizeResult
        Result returned by the optimizer.
    res_dict : SteadyStateResults
        Steady-state values found by the optimizer, used to evaluate the residuals and the jacobian.
    f_resid : callable
        Function returning the system residuals given the steady-state values as keyword arguments.
    f_grad : callable
        Function returning the system jacobian given the steady-state values as keyword arguments.
    tol : float, optional
        Threshold the sum of squared residuals, maximum absolute error, gradient L2 norm, and maximum absolute
        gradient must each fall below. Defaults to 1e-6.
    verbose : bool, optional
        If True, log a summary of the solution diagnostics. Defaults to True.

    Returns
    -------
    res_dict : SteadyStateResults
        The input results, with ``success`` updated.
    """
    success = res.success

    f_x = np.asarray(f_resid(**res_dict)).ravel()
    df_dx = f_grad(**res_dict)

    sse = (f_x**2).sum()
    max_abs_error = np.max(np.abs(f_x))
    grad_norm = np.linalg.norm(df_dx, ord=2)
    abs_max_grad = np.max(np.abs(df_dx))

    numeric_success = all(condition < tol for condition in [sse, max_abs_error, grad_norm, abs_max_grad])

    if numeric_success and not success:
        word = " IS "
    elif not numeric_success:
        word = " NOT "
    else:
        word = " "

    line_1 = f"Steady state{word}found"
    if numeric_success and not success:
        line_1 += (
            ", although optimizer returned success = False.\n"
            "This can be ignored, but to silence this message, try reducing the solver-specific tolerance, "
            "or use a different solution algorithm."
        )
    elif not numeric_success and success:
        line_1 += (
            ", although optimizer returned success = True.\n"
            "The optimizer stopped at a point that does not satisfy the residual tolerance. Try a different "
            "solution algorithm, a tighter solver-specific tolerance, or a better initial point."
        )

    msg = (
        f"{line_1}\n"
        f"{'-' * 80}\n"
        f"{'Optimizer message':<30}{res.message}\n"
        f"{'Sum of squared residuals':<30}{sse}\n"
        f"{'Maximum absolute error':<30}{max_abs_error}\n"
        f"{'Gradient L2-norm at solution':<30}{grad_norm}\n"
        f"{'Max abs gradient at solution':<30}{abs_max_grad}"
    )

    if verbose:
        _log.info(msg)
    res_dict.success = numeric_success
    return res_dict


def get_name(x: str | sp.Symbol, base_name: bool = False) -> str | None:
    """
    Return the name of a string, TimeAwareSymbol, or sympy Symbol.

    Parameters
    ----------
    x : str or sympy Symbol
        Object whose name is returned. A string is returned as is.
    base_name : bool, optional
        If True, return the base name of a TimeAwareSymbol, without its time suffix. Defaults to False.

    Returns
    -------
    name : str or None
        The name of the object, or None when ``x`` is none of the supported types.
    """
    if isinstance(x, str):
        return x

    if isinstance(x, TimeAwareSymbol):
        return x.safe_name if not base_name else x.base_name

    if isinstance(x, sp.Symbol):
        return x.name
    return None


def flatten_substitution_dict(
    sub_dict: dict[sp.Expr, sp.Expr],
) -> dict[sp.Expr, sp.Expr]:
    """
    Resolve a substitution dictionary so each value references no other keys.

    Walk the dependency graph of ``sub_dict`` in topological order, substituting each value through its resolved
    predecessors exactly once. Afterwards a single ``expr.subs(flat_dict)`` sweep converges, so callers need not
    iterate substitutions to a fixed point when values chain through each other.

    Parameters
    ----------
    sub_dict : dict mapping sympy symbol to sympy expression
        Substitutions, possibly with values that reference other keys.

    Returns
    -------
    flat_dict : dict mapping sympy symbol to sympy expression
        Same keys as the input, with values fully resolved against each other.
    """
    keys = set(sub_dict)
    deps: dict[sp.Expr, set[sp.Expr]] = {}
    for k, v in sub_dict.items():
        if isinstance(v, sp.Basic):
            deps[k] = (v.free_symbols & keys) - {k}
        else:
            deps[k] = set()

    flat: dict[sp.Expr, sp.Expr] = {}
    visiting: set[sp.Expr] = set()

    def resolve(key: sp.Expr) -> sp.Expr:
        if key in flat:
            return flat[key]
        if key in visiting:
            raise ValueError(
                f"Substitution dictionary has a cycle involving {key}. Rewrite the definitions so no value "
                "depends on itself through other keys."
            )
        visiting.add(key)

        value = sub_dict[key]
        for dep in deps[key]:
            resolve(dep)
        if deps[key] and isinstance(value, sp.Basic):
            value = value.subs({d: flat[d] for d in deps[key]})

        visiting.discard(key)
        flat[key] = value
        return value

    for key in sub_dict:
        resolve(key)

    return flat
