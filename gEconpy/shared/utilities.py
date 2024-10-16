from copy import copy
from enum import EnumMeta
from typing import Any
from collections.abc import Callable

import numba as nb
import numpy as np
import sympy as sp

from gEconpy.classes.containers import SymbolDictionary, string_keys_to_sympy
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol


class IterEnum(EnumMeta):
    def __init__(self, *args, **kwargs):
        self.__idx = 0
        super().__init__(*args, **kwargs)

    def __contains__(self, item):
        return item in {v.value for v in self.__members__.values()}

    def __len__(self):
        return len(self.__members__)

    def __iter__(self):
        return self

    def __next__(self):
        self.__idx += 1
        try:
            return list(self.__members__)[self.__idx - 1]
        except IndexError:
            self.__idx = 0
            raise StopIteration


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


def eq_to_ss(eq):
    var_list = [x for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
    sub_dict = dict(zip(var_list, [x.to_ss() for x in var_list]))
    return eq.subs(sub_dict)


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
                if isinstance(eqs[key], (int, float))
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

    return all([c in set("0123456789.") for c in x])


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
    if not isinstance(dicts, (list, tuple)):
        return dicts

    result = {}
    for d in dicts:
        result.update(d)
    return result


def make_all_var_time_combos(var_list):
    result = []
    for x in var_list:
        result.extend([x.set_t(-1), x.set_t(0), x.set_t(1), x.set_t("ss")])

    return result


def build_Q_matrix(
    model_shocks: list[str],
    shock_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std_priors: dict[str, Any] | None = None,
    default_value: float | None = 0.01,
) -> np.array:
    """
    Take different options for user input and reconcile them into a covariance matrix. Exactly one or zero of shock_dict
    or shock_cov_matrix should be provided. Then, proceed according to the following logic:

    - If `shock_cov_matrix` is provided, it is Q. Return it.
    - If `shock_dict` is provided, insert these into a diagonal matrix at locations according to `model_shocks`.

    For values missing from `shock_dict`, or if neither `shock_dict` nor `shock_cov_matrix` are provided:

    - Fill missing values using the mean of the prior defined in `shock_priors`
    - If no prior is set, fill the value with `default_value`.

    Note that the only way to get off-diagonal elements is to explicitly pass the entire covariance matrix.

    Parameters
    ----------
    model_shocks: list of str
        List of model shock names, used to infer positions in the covariance matrix
    shock_dict: dict of str, float
        Dictionary of shock names and standard deviations to be used to build Q
    shock_cov_matrix: ndarray
        The shock covariance matrix. If provided, check that it is positive semi-definite, then return it.
    shock_std_priors: dict of str, frozendist
        Dictionary of model priors over shock standard deviations
    default_value: float
        A default value of fall back on if no other information is available about a shock's standard deviation

    Raises
    ---------
    LinalgError
        If the provided Q is not positive semi-definite
    ValueError
        If both model_shocks and shock_dict are provided

    Returns
    -------
    Q: ndarray
        Shock variance-covariance matrix
    """
    n = len(model_shocks)
    if shock_dict is not None and shock_cov_matrix is not None:
        raise ValueError("Both shock_dict and shock_cov_matrix cannot be provided.")

    if shock_dict is None:
        shock_dict = {}

    if shock_cov_matrix is not None:
        if not all([x == n for x in shock_cov_matrix.shape]):
            raise ValueError(
                f"Provided covariance matrix has shape {shock_cov_matrix.shape}, expected ({n}, {n})"
            )
        try:
            # check that the result is PSD
            np.linalg.cholesky(shock_cov_matrix)
            return shock_cov_matrix
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("The provided Q is not positive semi-definite.")

    Q = np.eye(len(model_shocks)) * default_value

    if shock_dict is not None:
        for i, shock in enumerate(model_shocks):
            if shock in shock_dict:
                Q[i, i] = shock_dict[shock]

    if shock_std_priors is not None:
        for i, shock in enumerate(model_shocks):
            if shock not in shock_dict and shock in shock_std_priors:
                Q[i, i] = shock_std_priors[shock].mean()

    return Q


@nb.njit(cache=True)
def compute_autocorrelation_matrix(A, sigma, n_lags=5):
    """Compute the autocorrelation matrix for the given state-space model.

    Parameters
    ----------
    A : ndarray
        An array of shape (n_endog, n_endog, n_lags) representing the transition matrix of the
        state-space system.
    sigma : ndarray
        An array of shape (n_endog, n_endog) representing the variance-covariance matrix of the errors of
        the transition equation.
    n_lags : int, optional
        The number of lags for which to compute the autocorrelation matrix.

    Returns
    -------
    acov : ndarray
        An array of shape (n_endog, n_lags) representing the autocorrelation matrix of the state-space process.
    """

    acov = np.zeros((A.shape[0], n_lags))
    acov_factor = np.eye(A.shape[0])
    for i in range(n_lags):
        cov = acov_factor @ sigma
        acov[:, i] = np.diag(cov) / np.diag(sigma)
        acov_factor = A @ acov_factor

    return acov


def get_shock_std_priors_from_hyperpriors(shocks, priors, out_keys="parent"):
    """
    Extract a single key, value pair from the model hyper_priors.

    Parameters
    -------
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
    param_dict : Dict[str, float]
        A dictionary of parameters and their values.
    shock_names : List[str]
        A list of the names of shock variables.
    obs_names : List[str]
        A list of the names of observable variables.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        A tuple containing three dictionaries: the first has parameters, the second has
        all shock variances parameters, and the third has observation noise variances.
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
