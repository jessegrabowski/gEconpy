import logging

from collections.abc import Callable

import numpy as np
import sympy as sp

from scipy.optimize import OptimizeResult

from gEconpy.classes.containers import (
    SteadyStateResults,
    string_keys_to_sympy,
)
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

_log = logging.getLogger(__name__)


def flatten_list(items, result_list=None):
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


def set_equality_equals_zero(eq):
    """
    Rewrite a sympy equality as an expression equal to zero.

    Parameters
    ----------
    eq : sp.Expr
        Expression to rewrite. Non-equalities are returned unchanged.

    Returns
    -------
    eq : sp.Expr
        ``eq.rhs - eq.lhs`` if ``eq`` is an equality, otherwise ``eq``.
    """
    if not isinstance(eq, sp.Eq):
        return eq

    return eq.rhs - eq.lhs


def eq_to_ss(eq: sp.Expr, shocks: list[TimeAwareSymbol] | None = None):
    """
    Replace every time-aware symbol in an equation by its steady-state counterpart.

    Parameters
    ----------
    eq : sp.Expr
        Equation to convert.
    shocks : list of TimeAwareSymbol, optional
        Shocks whose steady-state values are set to zero. No shocks are zeroed by default.

    Returns
    -------
    eq : sp.Expr
        Equation written entirely in steady-state symbols.
    """
    shock_subs = {} if shocks is None else {x.to_ss(): 0.0 for x in shocks}

    var_list = [x for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
    to_ss_subs = dict(zip(var_list, [x.to_ss() for x in var_list], strict=False))

    return eq.subs(to_ss_subs).subs(shock_subs)


def safe_to_ss(x: sp.Symbol):
    """
    Convert ``x`` to steady-state if it is TimeAware, or return it unchanged otherwise.

    Parameters
    ----------
    x : sp.Symbol
        Symbol to convert.

    Returns
    -------
    x : sp.Symbol
        Steady-state symbol if ``x`` is a TimeAwareSymbol, otherwise ``x`` itself.
    """
    if isinstance(x, TimeAwareSymbol):
        return x.to_ss()
    return x


def expand_subs_for_all_times(sub_dict: dict[TimeAwareSymbol, TimeAwareSymbol]):
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


def step_equation_forward(eq):
    """
    Advance the time index of every time-aware symbol in an equation by one period.

    Parameters
    ----------
    eq : sp.Expr
        Equation to step forward.

    Returns
    -------
    eq : sp.Expr
        Equation with all time indices incremented.
    """
    to_step = [variable for variable in set(eq.atoms()) if hasattr(variable, "step_forward")]

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=True):
        eq = eq.subs({variable: variable.step_forward()})

    return eq


def step_equation_backward(eq):
    """
    Move the time index of every time-aware symbol in an equation back by one period.

    Parameters
    ----------
    eq : sp.Expr
        Equation to step backward.

    Returns
    -------
    eq : sp.Expr
        Equation with all time indices decremented.
    """
    to_step = [variable for variable in set(eq.atoms()) if hasattr(variable, "step_backward")]

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=False):
        eq = eq.subs({variable: variable.step_backward()})

    return eq


def diff_through_time(eq, dx, discount_factor=1):
    r"""Differentiate an equation with respect to a time-aware symbol, summing across time shifts.

    Computes :math:`\sum_{k=0}^{K} \beta^k \cdot \frac{\partial}{\partial dx} \mathrm{step}^k(\mathrm{eq})` where each
    step shifts every TimeAwareSymbol's time index forward by one. The number of iterations ``K`` is determined by the
    spread between ``dx``'s time index and the earliest appearance of ``dx``'s base symbol in ``eq``: stepping forward
    further can only increase the time indices of all instances, so once the leftmost has shifted past ``dx``, no
    additional contribution is possible.

    Parameters
    ----------
    eq : sympy.Expr
        Equation (typically a Lagrangian) to differentiate.
    dx : TimeAwareSymbol
        Variable to differentiate with respect to.
    discount_factor : sympy.Expr or int, optional
        Multiplicative discount factor applied at each forward step. Default 1 (no discounting).

    Returns
    -------
    total : sympy.Expr
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


def substitute_all_equations(eqs, *sub_dicts):
    """
    Apply one or more substitution dictionaries to a collection of equations.

    Parameters
    ----------
    eqs : list of sp.Expr or dict
        Equations to substitute into. Dictionary values that are plain numbers are left unchanged.
    *sub_dicts : dict
        Substitution dictionaries, merged left to right before use. Keys may be strings or sympy symbols.

    Returns
    -------
    eqs : list of sp.Expr or dict
        Equations after substitution, in the same container type as the input.
    """
    if len(sub_dicts) > 1:
        merged_dict = merge_dictionaries(*sub_dicts)
        sub_dict = string_keys_to_sympy(merged_dict)
    else:
        sub_dict = string_keys_to_sympy(sub_dicts[0])

    if isinstance(eqs, list):
        return [eq.subs(sub_dict) for eq in eqs]
    result = {}
    for key in eqs:
        result[key] = eqs[key] if isinstance(eqs[key], int | float) else eqs[key].subs(sub_dict)
    return result


def is_variable(x):
    """Return True if ``x`` is a TimeAwareSymbol."""
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


def unpack_keys_and_values(d):
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


def merge_dictionaries(*dicts):
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
    if not isinstance(dicts, list | tuple):
        return dicts

    result = {}
    for d in dicts:
        result.update(d)
    return result


def make_all_var_time_combos(var_list):
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
    f_jac: Callable[..., np.ndarray],
    tol: float = 1e-6,
    verbose: bool = True,
) -> SteadyStateResults:
    """
    Check an optimizer result against residual and gradient tolerances and report the outcome.

    The optimizer sometimes reports failure at a point that satisfies the tolerances, so success is granted if either
    the optimizer or the numeric check accepts the solution.

    Parameters
    ----------
    res : OptimizeResult
        Result returned by the optimizer.
    res_dict : SteadyStateResults
        Steady-state values found by the optimizer, used to evaluate the residuals and the jacobian.
    f_resid : callable
        Function returning the system residuals given the steady-state values as keyword arguments.
    f_jac : callable
        Function returning the system jacobian given the steady-state values as keyword arguments.
    tol : float, optional
        Threshold the sum of squared residuals, maximum absolute error, gradient L2 norm, and maximum absolute
        gradient must each fall below. Default 1e-6.
    verbose : bool, optional
        If True, log a summary of the solution diagnostics. Default True.

    Returns
    -------
    res_dict : SteadyStateResults
        The input results, with ``success`` updated.
    """
    success = res.success

    f_x = np.r_[[x.ravel() for x in f_resid(**res_dict)]]
    df_dx = f_jac(**res_dict)

    sse = (f_x**2).sum()
    max_abs_error = np.max(np.abs(f_x))
    grad_norm = np.linalg.norm(df_dx, ord=2)
    abs_max_grad = np.max(np.abs(df_dx))

    # Sometimes the optimizer is way too strict and returns success of False even if the point is pretty clearly
    # minimum.
    numeric_success = all(condition < tol for condition in [sse, max_abs_error, grad_norm, abs_max_grad])

    if numeric_success and not success:
        word = " IS "
    elif not numeric_success and not success:
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

    msg = (
        f"{line_1}\n"
        f"{'-' * 80}\n"
        f"{'Optimizer message':<30}{res.message}\n"
        f"{'Sum of squared residuals':<30}{sse}\n"
        f"{'Maximum absoluate error':<30}{max_abs_error}\n"
        f"{'Gradient L2-norm at solution':<30}{grad_norm}\n"
        f"{'Max abs gradient at solution':<30}{abs_max_grad}"
    )

    if verbose:
        _log.info(msg)
    res_dict.success = success | numeric_success
    return res_dict


def get_name(x: str | sp.Symbol, base_name=False) -> str:
    """
    Return the name of a string, TimeAwareSymbol, or sp.Symbol object.

    Parameters
    ----------
    x : str, or sp.Symbol
        The object whose name is to be returned. If str, x is directly returned.
    base_name : bool
        If True, return TimeAwareSymbol base name (the name without any time suffix)

    Returns
    -------
    name : str
        The name of the object.
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

    Walks the dependency DAG of ``sub_dict`` in topological order, substituting each
    RHS through its already-resolved predecessors exactly once. After the pass, any
    ``expr.subs(flat_dict)`` call converges in a single sweep, which is much faster than
    iterating substitutions to a fixed point when the dict has many cross-references
    (e.g. a STEADY_STATE block whose hints chain through each other).

    Parameters
    ----------
    sub_dict : dict mapping sympy symbol to sympy expression
        Substitutions, possibly with values that reference other keys.

    Returns
    -------
    flat_dict : dict mapping sympy symbol to sympy expression
        Same keys as the input, with values fully resolved against each other.

    Raises
    ------
    ValueError
        If the dependency graph contains a cycle.
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
            raise ValueError(f"Cycle detected in substitution dictionary involving {key}")
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
