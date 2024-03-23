from functools import wraps
from typing import Callable, Literal, Optional, Union

import numba as nb
import numpy as np
import pytensor
import sympy as sp

from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.numba_tools.utilities import numba_lambdify

BACKENDS = Literal["numpy", "numba", "pytensor"]


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


def numba_input_wrapper(f: Callable, inputs: list[str]) -> Callable:
    @wraps(f)
    @nb.njit
    def inner(**kwargs):
        input = [kwargs[k] for k in inputs]
        return f(input)

    return inner


def _configue_pytensor_kwargs(kwargs: dict) -> dict:
    if "on_unused_input" not in kwargs:
        kwargs["on_unused_input"] = "ignore"
    return kwargs


def compile_function(
    inputs: list[sp.Symbol],
    outputs: Union[list[Union[sp.Symbol, sp.Expr]], sp.MutableDenseMatrix],
    backend: BACKENDS,
    cache: Optional[dict] = None,
    stack_return: bool = False,
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

    Returns
    -------
    f: Callable
        A python function that computes the outputs from the inputs.

    cache: dict
        A dictionary mapping from sympy symbols to pytensor symbols.
    """
    if backend == "numpy":
        f, cache = compile_to_numpy(inputs, outputs, cache, stack_return, **kwargs)
    elif backend == "numba":
        f, cache = compile_to_numba(inputs, outputs, cache, stack_return, **kwargs)
    elif backend == "pytensor":
        f, cache = compile_to_pytensor_function(inputs, outputs, cache, stack_return, **kwargs)
    else:
        raise NotImplementedError(f"backend {backend} not implemented. Must be one of {BACKENDS}.")

    return f, cache


def compile_to_numpy(
    inputs: list[sp.Symbol],
    outputs: Union[list[Union[sp.Symbol, sp.Expr]], sp.MutableDenseMatrix],
    cache: dict,
    stack_return: bool,
    **kwargs,
):
    f = sp.lambdify(inputs, outputs)
    if stack_return:
        f = stack_return_wrapper(f)
    return f, cache


def compile_to_numba(
    inputs: list[sp.Symbol],
    outputs: list[Union[sp.Symbol, sp.Expr]],
    cache: dict,
    stack_return: bool,
    **kwargs,
):
    f = numba_input_wrapper(numba_lambdify(inputs, outputs), [x.name for x in inputs])
    if stack_return:
        f = stack_return_wrapper(f)
    return f, cache


def compile_to_pytensor_function(
    inputs: list[sp.Symbol],
    outputs: list[Union[sp.Symbol, sp.Expr]],
    cache: dict,
    stack_return: bool,
    **kwargs,
):
    kwargs = _configue_pytensor_kwargs(kwargs)
    cache = {} if cache is None else cache

    outputs = [outputs] if not isinstance(outputs, list) else outputs
    input_pt = [as_tensor(x, cache) for x in inputs]
    output_pt = [output_to_tensor(x, cache) for x in outputs]
    output_pt = output_pt[0] if len(output_pt) == 1 else output_pt
    if stack_return:
        output_pt = pytensor.tensor.stack(output_pt)

    f = pytensor.function(input_pt, output_pt, **kwargs)
    return f, cache
