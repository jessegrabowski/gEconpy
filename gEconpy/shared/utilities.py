from collections import defaultdict
from copy import copy
from enum import EnumMeta
from typing import Any, Callable, Dict, List, Union

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

VariableType = Union[TimeAwareSymbol, sp.Symbol]


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


def float_values_to_sympy_float(d: Dict[VariableType, Any]):
    for var, value in d.items():
        if isinstance(value, (float, int)):
            d[var] = sp.Float(value)

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


def sort_sympy_dict(d):
    result = {}
    sorted_keys = sorted(
        list(d.keys()),
        key=lambda x: x.base_name if isinstance(x, TimeAwareSymbol) else x.name,
    )
    for key in sorted_keys:
        result[key] = d[key]

    return result


SAFE_STRING_TO_INDEX_DICT = dict(ss="ss", tp1=1, tm1=-1, t=0)


def safe_string_to_sympy(s, assumptions=None):
    if isinstance(s, sp.Symbol):
        return s

    assumptions = assumptions or {}

    *name, time_index_str = s.split("_")
    if time_index_str not in [str(x) for x in SAFE_STRING_TO_INDEX_DICT.keys()]:
        name.append(time_index_str)
        return sp.Symbol("_".join(name), **assumptions)
    name = "_".join(name)
    time_index = SAFE_STRING_TO_INDEX_DICT[time_index_str]
    symbol = TimeAwareSymbol(name, time_index)

    return symbol


def select_keys(d, keys):
    result = {}
    for key in keys:
        result[key] = d[key]
    return result


def string_keys_to_sympy(d, assumptions=None):
    result = {}
    assumptions = assumptions or defaultdict(lambda: {})
    for key, value in d.items():
        if isinstance(key, sp.Symbol):
            result[key] = value
            continue

        if "_" not in key:
            result[sp.Symbol(key, **assumptions[key])] = value
            continue
        new_key = safe_string_to_sympy(key, assumptions[key])
        result[new_key] = value

    return result


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


def merge_functions(funcs, *args, **kwargs):
    def combined_function(*args, **kwargs):
        output = {}

        for f in funcs:
            output.update(f(*args, **kwargs))

        return output

    return combined_function


def find_exp_args(eq):
    if eq is None:
        return None
    comp_tree = list(sp.postorder_traversal(eq))
    for arg in comp_tree:
        if arg.func == sp.exp:
            return arg


def find_log_args(eq):
    if eq is None:
        return None

    comp_tree = list(sp.postorder_traversal(eq))
    for arg in comp_tree:
        if arg.func == sp.Mul:
            if isinstance(arg.args[0], sp.Symbol) and arg.args[1].func == sp.log:
                return arg


def is_log_transform_candidate(eq):
    inside_exp = sequential(eq, [find_exp_args, find_log_args])
    return inside_exp is not None


def log_transform_exp_shock(eq):
    out = (-sp.log(-eq.args[0]) + sp.log(eq.args[1])).simplify(inverse=True)
    return out


def expand_sub_dict_for_all_times(sub_dict):
    result = {}
    for k, v in sub_dict.items():
        result[k] = v
        result[step_equation_forward(k)] = step_equation_forward(v)
        result[step_equation_backward(k)] = step_equation_backward(v)
        result[eq_to_ss(k)] = eq_to_ss(v)

    return result


def make_all_var_time_combos(var_list):
    result = []
    for x in var_list:
        result.extend([x.set_t(-1), x.set_t(0), x.set_t(1), x.set_t("ss")])

    return result
