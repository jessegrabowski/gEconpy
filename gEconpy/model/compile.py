from functools import wraps
from typing import Literal
from collections.abc import Callable

import numpy as np
import pytensor
import sympy as sp

from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.numba_tools.utilities import numba_lambdify

BACKENDS = Literal["numpy", "numba", "pytensor"]


def sp_to_pt_from_cache(symbol_list: list[sp.Symbol], cache: dict) -> SymbolDictionary:
    """
    Look up a list of symbols in a Sympy PytensorPrinter cache and return a SymbolDictionary mapping each symbol
    to its corresponding tensor variable on the compute graph.

    Parameters
    ----------
    symbol_list: list[sp.Symbol]
        List of sympy symbols to look up in the cache

    cache: dict
        Dictionary created by SympyTensor during printing.

    Returns
    -------
    sp_to_pt: SymbolDictionary
        Mapping from sympy symbols to their pytensor Variables
    """

    sp_to_pt = {}
    cached_names = [x[0] for x in cache.keys()]
    cached_tensors = list(cache.values())
    for symbol in symbol_list:
        if symbol.name in cached_names:
            idx = cached_names.index(symbol.name)
            sp_to_pt[symbol] = cached_tensors[idx]
        else:
            raise ValueError(f"{symbol} not found in the provided cache")

    return SymbolDictionary(sp_to_pt)


def output_to_tensor(x, cache):
    if isinstance(x, (int, float)):
        return pytensor.tensor.constant(x, dtype=pytensor.config.floatX)
    return as_tensor(x, cache)


def dictionary_return_wrapper(f: Callable, outputs: list[sp.Symbol]) -> Callable:
    @wraps(f)
    def inner(*args, **kwargs):
        values = f(*args, **kwargs)
        return SymbolDictionary(zip(outputs, values)).to_string()

    return inner


def stack_return_wrapper(f: Callable) -> Callable:
    @wraps(f)
    def inner(*args, **kwargs):
        values = f(*args, **kwargs)
        return np.stack(values)

    return inner


def pop_return_wrapper(f: Callable) -> Callable:
    @wraps(f)
    def inner(*args, **kwargs):
        values = f(*args, **kwargs)
        return values[0]

    return inner


def _configue_pytensor_kwargs(kwargs: dict) -> dict:
    if "on_unused_input" not in kwargs:
        kwargs["on_unused_input"] = "ignore"
    return kwargs


def compile_function(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr] | sp.MutableDenseMatrix,
    backend: BACKENDS,
    cache: dict | None = None,
    stack_return: bool = False,
    pop_return: bool = False,
    return_symbolic: bool = False,
    **kwargs,
) -> tuple[Callable, dict]:
    """
    Dispatch compilation of a sympy function to one of three possible backends: numpy, numba, or pytensor.

    Parameters
    ----------
    inputs: list[sp.Symbol]
        The inputs to the function.

    outputs: list[Union[sp.Symbol, sp.Expr]]
        The outputs of the function.

    backend: str, one of "numpy", "numba", "pytensor"
        The backend to use for the compiled function.

    cache: dict, optional
        A dictionary mapping from pytensor symbols to sympy expressions. Used to prevent duplicate mappings from
        sympy symbol to pytensor symbol from being created. Default is a empty dictionary, implying no other functions
        have been compiled yet.

        Ignored if backend is not "pytensor".

    stack_return: bool, optional
        If True, the function will return a single numpy array with all outputs. Otherwise it will return a tuple of
        numpy arrays. Default is False.

    pop_return: bool, optional
        If True, the function will return only the 0th element of the output. Used to remove the list wrapper around
        scalar outputs. Default is False.

    return_symbolic: bool, default True
        If True, when mode="pytensor", the will return a symbolic pytensor computation graph instead of a compiled
        function. Ignored when mode is not "pytensor".

    Returns
    -------
    f: Callable
        A python function that computes the outputs from the inputs.

    cache: dict
        A dictionary mapping from sympy symbols to pytensor symbols.
    """
    if backend == "numpy":
        f, cache = compile_to_numpy(
            inputs, outputs, cache, stack_return, pop_return, **kwargs
        )
    elif backend == "numba":
        f, cache = compile_to_numba(
            inputs, outputs, cache, stack_return, pop_return, **kwargs
        )
    elif backend == "pytensor":
        f, cache = compile_to_pytensor_function(
            inputs, outputs, cache, stack_return, pop_return, return_symbolic, **kwargs
        )
    else:
        raise NotImplementedError(
            f"backend {backend} not implemented. Must be one of {BACKENDS}."
        )

    return f, cache


def compile_to_numpy(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr] | sp.MutableDenseMatrix,
    cache: dict,
    stack_return: bool,
    pop_return: bool,
    **kwargs,
):
    f = sp.lambdify(inputs, outputs)
    if stack_return:
        f = stack_return_wrapper(f)
    if pop_return:
        f = pop_return_wrapper(f)
    return f, cache


def compile_to_numba(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr],
    cache: dict,
    stack_return: bool,
    pop_return: bool,
    **kwargs,
):
    f = numba_lambdify(inputs, outputs)
    if stack_return:
        f = stack_return_wrapper(f)
    if pop_return:
        f = pop_return_wrapper(f)
    return f, cache


def compile_to_pytensor_function(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr],
    cache: dict,
    stack_return: bool,
    pop_return: bool,
    return_symbolic: bool,
    **kwargs,
):
    kwargs = _configue_pytensor_kwargs(kwargs)
    cache = {} if cache is None else cache

    outputs = [outputs] if not isinstance(outputs, list) else outputs
    input_pt = [as_tensor(x, cache) for x in inputs]
    output_pt = [output_to_tensor(x, cache) for x in outputs]
    if stack_return:
        output_pt = pytensor.tensor.stack(output_pt)
    if pop_return:
        output_pt = output_pt[0] if len(output_pt) == 1 else output_pt

    if return_symbolic:
        return output_pt, cache

    f = pytensor.function(input_pt, output_pt, **kwargs)
    return f, cache
