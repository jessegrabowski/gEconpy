from functools import wraps
from typing import Callable, Literal, Optional, Union

import numba as nb
import pytensor
import sympy as sp

from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.numba_tools.utilities import numba_lambdify

BACKENDS = Literal["numpy", "numba", "pytensor"]


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


def _configue_pytensor_kwargs(kwargs: dict) -> dict:
    if "on_unused_input" not in kwargs:
        kwargs["on_unused_input"] = "ignore"
    return kwargs


def compile_function(
    inputs: list[sp.Symbol],
    outputs: list[Union[sp.Symbol, sp.Expr]],
    backend: BACKENDS,
    cache: Optional[dict] = None,
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

    Returns
    -------
    f: Callable
        A python function that computes the outputs from the inputs.

    cache: dict
        A dictionary mapping from sympy symbols to pytensor symbols.
    """
    if backend == "numpy":
        f, cache = compile_to_numpy(inputs, outputs, cache, **kwargs)
    elif backend == "numba":
        f, cache = compile_to_numba(inputs, outputs, cache, **kwargs)
    elif backend == "pytensor":
        f, cache = compile_to_pytensor_function(inputs, outputs, cache, **kwargs)
    else:
        raise NotImplementedError(f"backend {backend} not implemented. Must be one of {BACKENDS}.")

    return f, cache


def compile_to_numpy(
    inputs: list[sp.Symbol], outputs: list[Union[sp.Symbol, sp.Expr]], cache: dict, **kwargs
):
    f = sp.lambdify(inputs, outputs)
    return f, cache


def compile_to_numba(
    inputs: list[sp.Symbol], outputs: list[Union[sp.Symbol, sp.Expr]], cache: dict, **kwargs
):
    f = numba_input_wrapper(numba_lambdify(inputs, outputs), [x.name for x in inputs])
    return f, cache


def compile_to_pytensor_function(
    inputs: list[sp.Symbol], outputs: list[Union[sp.Symbol, sp.Expr]], cache: dict, **kwargs
):
    kwargs = _configue_pytensor_kwargs(kwargs)
    cache = {} if cache is None else cache

    outputs = [outputs] if not isinstance(outputs, list) else outputs
    input_pt = [as_tensor(x, cache) for x in inputs]
    output_pt = [as_tensor(x, cache) for x in outputs]

    f = pytensor.function(input_pt, output_pt, **kwargs)
    return f, cache
