from functools import wraps
from typing import Callable, Literal

import numba as nb
import pytensor
import sympy as sp

from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.numba_tools.utilities import numba_lambdify

Backends = Literal["numpy", "numba", "pytensor"]


def dictionary_return_wrapper(f: Callable, outputs: list[sp.Symbol]) -> Callable:
    @wraps(f)
    def inner(*args, **kwargs):
        values = f(*args, **kwargs)
        return SymbolDictionary(zip(outputs, values))

    return inner


def numba_input_wrapper(f: Callable, inputs: list[str]) -> Callable:
    @wraps(f)
    @nb.njit
    def inner(**kwargs):
        input = [kwargs[k] for k in inputs]
        return f(input)

    return inner


def compile_param_dict_func(
    param_dict: SymbolDictionary, deterministic_dict: SymbolDictionary, backend: Backends = "numpy"
):
    if backend == "numpy":
        return compile_to_numpy(param_dict, deterministic_dict)
    elif backend == "numba":
        return compile_to_numba(param_dict, deterministic_dict)
    elif backend == "pytensor":
        return compile_to_pytensor_function(param_dict, deterministic_dict)
    else:
        raise NotImplementedError()


def compile_to_numpy(param_dict: SymbolDictionary, deterministic_dict: SymbolDictionary):
    inputs = list(param_dict.to_sympy().keys())
    output_params = inputs + list(deterministic_dict.to_sympy().keys())
    output_exprs = inputs + list(deterministic_dict.values())
    f = sp.lambdify(inputs, output_exprs)
    return dictionary_return_wrapper(f, output_params)


def compile_to_numba(param_dict: SymbolDictionary, deterministic_dict: SymbolDictionary):
    inputs = list(param_dict.to_sympy().keys())
    output_params = inputs + list(deterministic_dict.to_sympy().keys())
    output_exprs = inputs + list(deterministic_dict.values())
    f = numba_lambdify(inputs, output_exprs)
    return dictionary_return_wrapper(
        numba_input_wrapper(f, param_dict.to_string().keys()), output_params
    )


def compile_to_pytensor_function(
    param_dict: SymbolDictionary, deterministic_dict: SymbolDictionary
):
    inputs = list(param_dict.to_sympy().keys())
    output_params = inputs + list(deterministic_dict.to_sympy().keys())
    output_exprs = inputs + list(deterministic_dict.values())

    cache = {}
    input_pt = [as_tensor(x, cache) for x in inputs]
    output_pt = [as_tensor(x, cache) for x in output_exprs]

    f = pytensor.function(input_pt, output_pt, on_unused_input="ignore", mode="FAST_COMPILE")
    f = dictionary_return_wrapper(f, output_params)
    return f
