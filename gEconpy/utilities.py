import logging

from collections.abc import Callable
from copy import copy
from typing import Any

import numpy as np
import sympy as sp

from gEconpy.classes.containers import SymbolDictionary, string_keys_to_sympy
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

_log = logging.getLogger(__name__)


def flatten_list(items, result_list=None):
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
    if not isinstance(eq, sp.Eq):
        return eq

    return eq.rhs - eq.lhs


def eq_to_ss(eq: sp.Expr, shocks: list[TimeAwareSymbol] | None = None):
    if shocks is None:
        shock_subs = {}
    else:
        shock_subs = {x.to_ss(): 0.0 for x in shocks}

    var_list = [x for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
    to_ss_subs = dict(zip(var_list, [x.to_ss() for x in var_list]))

    return eq.subs(to_ss_subs).subs(shock_subs)


def expand_subs_for_all_times(sub_dict: dict[TimeAwareSymbol, TimeAwareSymbol]):
    result = {}
    for lhs, rhs in sub_dict.items():
        for t in [-1, 0, 1, "ss"]:
            result[lhs.set_t(t)] = (
                rhs.set_t(t) if isinstance(rhs, TimeAwareSymbol) else rhs
            )

    return result


def step_equation_forward(eq):
    to_step = []

    for variable in set(eq.atoms()):
        if hasattr(variable, "step_forward"):
            if variable.time_index != "ss":
                to_step.append(variable)

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=True):
        eq = eq.subs({variable: variable.step_forward()})

    return eq


def step_equation_backward(eq):
    to_step = []

    for variable in set(eq.atoms()):
        if hasattr(variable, "step_forward"):
            to_step.append(variable)

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=False):
        eq = eq.subs({variable: variable.step_backward()})

    return eq


def diff_through_time(eq, dx, discount_factor=1):
    total_dydx = 0
    next_dydx = 1

    while next_dydx != 0:
        next_dydx = eq.diff(dx)
        eq = step_equation_forward(eq) * discount_factor
        total_dydx += next_dydx

    return total_dydx


def substitute_all_equations(eqs, *sub_dicts):
    if len(sub_dicts) > 1:
        merged_dict = merge_dictionaries(*sub_dicts)
        sub_dict = string_keys_to_sympy(merged_dict)
    else:
        sub_dict = string_keys_to_sympy(sub_dicts[0])

    if isinstance(eqs, list):
        return [eq.subs(sub_dict) for eq in eqs]
    else:
        result = {}
        for key in eqs:
            result[key] = (
                eqs[key]
                if isinstance(eqs[key], int | float)
                else eqs[key].subs(sub_dict)
            )
        return result


def is_variable(x):
    return isinstance(x, TimeAwareSymbol)


def is_number(x: str):
    """
    Parameters
    ----------
    x: str
        string to test

    Returns
    -------
    is_number: bool
        Flag indicating whether this is a number

    A small extension to the .isnumeric() string built-in method, to allow float values with "." to pass.
    """
    if isinstance(x, float | int):
        return True
    elif isinstance(x, str):
        return all([c in set("0123456789.") for c in x])
    else:
        return False


def sequential(x: Any, funcs: list[Callable]) -> Any:
    """
    Parameters
    ----------
    x: Any
        A value to operate on
    funcs: list
        A list of functions to sequentially apply

    Returns
    -------
    x: Any

    Given a list of functions f, g, h, compute h(g(f(x)))
    """

    result = copy(x)
    for func in funcs:
        result = func(result)
    return result


def unpack_keys_and_values(d):
    keys = list(d.keys())
    values = list(d.values())

    return keys, values


def reduce_system_via_substitution(system, sub_dict):
    reduced_system = [eq.subs(sub_dict) for eq in system]
    return [eq for eq in reduced_system if eq != 0]


def merge_dictionaries(*dicts):
    if not isinstance(dicts, list | tuple):
        return dicts

    result = {}
    for d in dicts:
        result.update(d)
    return result


def recursively_self_substitute_dict(sub_dict, max_iter=5):
    eqs = list(sub_dict.values())
    for i in range(max_iter):
        new_eqs = substitute_all_equations(eqs, sub_dict)
        sub_dict = substitute_all_equations(sub_dict, sub_dict)
        no_changes = all([new_eq == old_eq for new_eq, old_eq in zip(new_eqs, eqs)])
        eqs = new_eqs
        if no_changes:
            break
    return {var: eq for var, eq in zip(sub_dict.keys(), eqs)}


def make_all_var_time_combos(var_list):
    result = []
    for x in var_list:
        result.extend([x.set_t(-1), x.set_t(0), x.set_t(1), x.set_t("ss")])

    return result


def get_shock_std_priors_from_hyperpriors(shocks, priors, out_keys="parent"):
    """
    Extract a single key, value pair from the model hyper_priors.

    Parameters
    ----------
    shocks: list of sympy Symbols
        Model shocks
    priors: dict of key, tuple
        Model hyper-priors. Key is symbol, values are (parent symbol, parameter type, distribution)
    out_keys: str
        One of "param" or "parent". Determines what will be the keys on the returned dictionary. If parent,
        the key will be the parent symbol. This is useful for putting sigmas in the right place of the
        covariance matrix. If param, it maintains the parameter name as the key and discards the parent and type
        information.

    Returns
    -------
    shock_std_dict: dict of str, distribution
        Dictionary of model shock standard deviations
    """

    if out_keys not in ["parent", "param"]:
        raise ValueError(
            f'out_keys must be one of "parent" or "param", found {out_keys}'
        )

    shock_std_dict = SymbolDictionary()
    for k, (parent, param, d) in priors.items():
        if parent in shocks and param in ["scale", "sd"]:
            if out_keys == "parent":
                shock_std_dict[parent] = d
            else:
                shock_std_dict[k] = d

    return shock_std_dict


def split_random_variables(param_dict, shock_names, obs_names):
    """
    Split a dictionary of parameters into dictionaries of shocks, observables, and other variables.

    Parameters
    ----------
    param_dict : dict
        A dictionary of parameters and their values.
    shock_names : list of str
        A list of the names of shock variables.
    obs_names : list of str
        A list of the names of observable variables.

    Returns
    -------
    out_param_dict: dict
        Dictionary mapping parameter names to values
    shock_dict: dict
        Dictionary mapping shock names to values
    obs_dict: dict
        Dictionary mapping names of observed variables to observation noise values
    """

    out_param_dict = SymbolDictionary()
    shock_dict = SymbolDictionary()
    obs_dict = SymbolDictionary()

    for k, v in param_dict.items():
        if k in shock_names:
            shock_dict[k] = v
        elif k in obs_names:
            obs_dict[k] = v
        else:
            out_param_dict[k] = v

    return out_param_dict, shock_dict, obs_dict


def postprocess_optimizer_res(
    res,
    res_dict,
    f_resid,
    f_jac,
    tol,
    verbose=True,
) -> tuple[SymbolDictionary, bool]:
    success = res.success

    f_x = np.r_[[x.ravel() for x in f_resid(**res_dict)]]
    df_dx = f_jac(**res_dict)

    sse = (f_x**2).sum()
    max_abs_error = np.max(np.abs(f_x))
    grad_norm = np.linalg.norm(df_dx, ord=2)
    abs_max_grad = np.max(np.abs(df_dx))

    # Sometimes the optimizer is way too strict and returns success of False even if the point is pretty clearly
    # minimum.
    numeric_success = all(
        condition < tol for condition in [sse, max_abs_error, grad_norm, abs_max_grad]
    )

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
        f'{line_1}\n'
        f'{"-"*80}\n'
        f"{'Optimizer message':<30}{res.message}\n"
        f"{'Sum of squared residuals':<30}{sse}\n"
        f"{'Maximum absoluate error':<30}{max_abs_error}\n"
        f"{'Gradient L2-norm at solution':<30}{grad_norm}\n"
        f"{'Max abs gradient at solution':<30}{abs_max_grad}"
    )

    if verbose:
        _log.info(msg)

    return res_dict, success | numeric_success


def get_name(x: str | sp.Symbol) -> str:
    """
    Return the name of a string, TimeAwareSymbol, or sp.Symbol object.

    Parameters
    ----------
    x : str, or sp.Symbol
        The object whose name is to be returned. If str, x is directly returned.

    Returns
    -------
    name: str
        The name of the object.
    """

    if isinstance(x, str):
        return x

    elif isinstance(x, TimeAwareSymbol):
        return x.safe_name

    elif isinstance(x, sp.Symbol):
        return x.name


def substitute_repeatedly(
    expr: sp.Expr, sub_dict: dict[sp.Expr, sp.Expr], max_subs: int = 10
) -> sp.Expr:
    """
    Repeatedly call ``expr = expr.sub(sub_dict)``. Useful when substitutions in ``sub_dict`` themselves require
    substitution.

    Parameters
    ----------
    expr: sp.Expr
        Expression to substitute into
    sub_dict: dict of sp.Expr, sp.Expr
        Dictionary of substitutions
    max_subs: int
        Maximum number of substitutions to make. If the number of substitutions exceeds this number, the function
        will return the expression as is.

    Returns
    -------
    substituted_expr: sp.Expr
        The expression with all substitutions made.
    """
    for i in range(max_subs):
        new_expr = expr.subs(sub_dict)
        if not any([new_expr.has(x) for x in sub_dict.keys()]):
            return new_expr
        expr = new_expr

    return expr


def simplify_matrix(A: sp.MutableMatrix):
    """
    Call ``sp.simplify`` on all cells of a matrix.

    Parameters
    ----------
    A: sp.MutableMatrix
        Matrix to simplify

    Returns
    -------
    A: sp.MutableMatrix
        Simplified matrix
    """

    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] = sp.simplify(A[i, j])

    return A
