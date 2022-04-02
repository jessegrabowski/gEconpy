import sympy as sp
from gEcon.classes.time_aware_symbol import TimeAwareSymbol
from enum import EnumMeta
from typing import Dict, Union, Any, List, Callable
from functools import wraps
from copy import copy

import numpy as np

VariableType = Union[TimeAwareSymbol, sp.Symbol]


class ListEnum(EnumMeta):
    def __init__(self, *args, **kwargs):
        self.__idx = 0
        super().__init__(*args, **kwargs)

    def __contains__(self, item):
        return item in set(v.value for v in self.__members__.values())

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


def set_equality_equals_zero(eq):
    if not isinstance(eq, sp.Eq):
        return eq

    return eq.rhs - eq.lhs


def eq_to_ss(eq):
    var_list = [x for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
    sub_dict = dict(zip(var_list, [x.to_ss() for x in var_list]))
    return eq.subs(sub_dict)


def expand_subs_for_all_times(sub_dict: Dict[TimeAwareSymbol, TimeAwareSymbol]):
    result = {}
    for lhs, rhs in sub_dict.items():
        for t in [-1, 0, 1, 'ss']:
            result[lhs.set_t(t)] = rhs.set_t(t) if isinstance(rhs, TimeAwareSymbol) else rhs

    return result


def step_equation_forward(eq):
    to_step = []

    for variable in set(eq.atoms()):
        if hasattr(variable, 'step_forward'):
            if variable.time_index != 'ss':
                to_step.append(variable)

    for variable in sorted(to_step, key=lambda x: x.time_index, reverse=True):
        eq = eq.subs({variable: variable.step_forward()})

    return eq


def step_equation_backward(eq):
    to_step = []

    for variable in set(eq.atoms()):
        if hasattr(variable, 'step_forward'):
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

    return all([c in set('0123456789.') for c in x])


def sequential(x: Any, funcs: List[Callable]) -> Any:
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


def sympy_number_values_to_floats(d: Dict[VariableType, Any]):
    for var, value in d.items():
        if isinstance(value, sp.core.Number):
            d[var] = float(value)
    return d


def symbol_to_string(symbol: Union[str, VariableType]):
    if isinstance(symbol, str):
        return symbol
    else:
        return symbol.safe_name if isinstance(symbol, TimeAwareSymbol) else symbol.name


def sympy_keys_to_strings(d):
    result = {}
    for key in d.keys():
        result[symbol_to_string(key)] = d[key]

    return result


def sort_dictionary(d):
    result = {}
    sorted_keys = sorted(list(d.keys()))
    for key in sorted_keys:
        result[key] = d[key]

    return result


SAFE_STRING_TO_INDEX_DICT = dict(ss='ss', tp1=1, tm1=1, t=0)


def safe_string_to_sympy(s):
    *name, time_index_str = s.split('_')
    if time_index_str not in [str(x) for x in SAFE_STRING_TO_INDEX_DICT.keys()]:
        name.append(time_index_str)
        return sp.Symbol('_'.join(name))
    name = '_'.join(name)
    time_index = SAFE_STRING_TO_INDEX_DICT[time_index_str]
    symbol = TimeAwareSymbol(name, time_index)

    return symbol


def select_keys(d, keys):
    result = {}
    for key in keys:
        result[key] = d[key]
    return result


def string_keys_to_sympy(d):
    result = {}
    for key, value in d.items():
        if '_' not in key:
            result[sp.Symbol(key)] = value
            continue
        new_key = safe_string_to_sympy(key)
        result[new_key] = value

    return result


def reduce_system_via_substitution(system, sub_dict):
    reduced_system = [eq.subs(sub_dict) for eq in system]
    return [eq for eq in reduced_system if eq != 0]


def merge_dicts(d1, d2):
    result = {}
    for key in d1:
        result[key] = d1[key]
    for key in d2:
        result[key] = d2[key]
    return result
