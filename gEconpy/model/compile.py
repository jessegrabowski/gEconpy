from collections.abc import Callable
from functools import wraps
from typing import Literal

import numpy as np
import pytensor
import sympy as sp

from pytensor.tensor import TensorVariable
from sympytensor import as_tensor

from gEconpy.classes.containers import SteadyStateResults
from gEconpy.numbaf.utilities import numba_lambdify

BACKENDS = Literal["numpy", "numba", "pytensor"]


def output_to_tensor(x, cache):
    if isinstance(x, int | float | sp.Float | sp.Integer):
        return pytensor.tensor.constant(x, dtype=pytensor.config.floatX)

    return as_tensor(x, cache)


def dictionary_return_wrapper(f: Callable, outputs: list[sp.Symbol]) -> Callable:
    """
    Wrap a function that returns a numpy array to instead return a SymbolDictionary object, with keys
    corresponding to the output symbols.


    Parameters
    ----------
    f: Callable
        The function to wrap
    outputs: list[sp.Symbol]
        The output symbols of the function, in the same order as the outputs of the function.

    Returns
    -------
    inner: Callable
        The wrapped function
    """

    @wraps(f)
    def inner(*args, **kwargs):
        values = f(*args, **kwargs)
        return SteadyStateResults(zip(outputs, values)).to_string()

    return inner


def stack_return_wrapper(f: Callable) -> Callable:
    """
    Wrap a function that returns a list of numpy arrays to instead return a single numpy array with all outputs stacked
    into a single flat array.

    This is useful when working with the output of :func:`sympy.lambdify`, which returns one numpy array per equation
    in the outputs. Scipy optimize routines, on the other hand, expect a single numpy array with all outputs stacked.

    Parameters
    ----------
    f: Callable
        The function to wrap

    Returns
    -------
    inner: Callable
        The wrapped function
    """

    @wraps(f)
    def inner(*args, **kwargs):
        values = f(*args, **kwargs)
        if not isinstance(values, list):
            # Special case for single output functions, for example a partially declared steady state
            # with only one equation
            values = [values]
        return np.stack(values)

    return inner


def pop_return_wrapper(f: Callable) -> Callable:
    """
    Wrap a function that (potentially) returns a list of numpy arrays to instead return the 0th element of the output.

    When the output of a function created by :func:`sympy.lambdify` is a single value, it is still returned as a list of
    one element. This wrapper removes the list wrapper around scalar outputs.

    Parameters
    ----------
    f: Callable
        The function to wrap

    Returns
    -------
    inner: Callable
        The wrapped function
    """

    @wraps(f)
    def inner(*args, **kwargs):
        values = np.array(f(*args, **kwargs))
        if values.ndim == 0:
            return values.item(0)
        else:
            return values[0]

    return inner


def array_return_wrapper(f: Callable) -> Callable:
    """
    Wrap a function to convert the output to a numpy array.

    When working with compiled JAX functions, the output will be a JAX array. This wrapper converts the JAX array to a
    numpy array, which is expected by e.g. scipy.optimize routines.

    Parameters
    ----------
    f: Callable
        The function to wrap

    Returns
    -------
    inner: Callable
        The wrapped function
    """

    @wraps(f)
    def inner(*args, **kwargs):
        return np.array(f(*args, **kwargs))

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
    """
    Convert a sympy function to a numpy function using :func:`sympy.lambdify`.

    Parameters
    ----------
    inputs: list[sp.Symbol]
        The inputs to the function.
    outputs: list[Union[sp.Symbol, sp.Expr]]
        The outputs of the function.
    cache: dict
        Mapping between sympy variables and pytensor variables. Ignored by this function; included for compatibility
        with other compile functions.
    stack_return: bool
        If True, the function will return a single numpy array with all outputs. Otherwise it will return a list
        of numpy arrays.
    pop_return: bool
        If True, the function will return only the 0th element of the output. Used to remove the list wrapper around
        single-output functions.
    kwargs: dict
        Ignored, included for compatibility with other compile functions.

    Returns
    -------
    f: Callable
        The compiled function.
    cache: dict
        Pytensor caching information.
    """

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
    """
    Convert a sympy function to a numba njit function using :func:`numba_lambdify`.

    Parameters
    ----------
    inputs: list[sp.Symbol]
        The inputs to the function.
    outputs: list[Union[sp.Symbol, sp.Expr]]
        The outputs of the function.
    cache: dict
        Mapping between sympy variables and pytensor variables. Ignored by this function; included for compatibility
        with other compile functions.
    stack_return: bool
        If True, the function will return a single numpy array with all outputs. Otherwise it will return a list
        of numpy arrays.
    pop_return: bool
        If True, the function will return only the 0th element of the output. Used to remove the list wrapper around
        single-output functions.
    kwargs: dict
        Ignored, included for compatibility with other compile functions

    Returns
    -------
    f: Callable
        The compiled function.
    cache: dict
        Pytensor caching information.
    """
    f = numba_lambdify(inputs, outputs, stack_outputs=stack_return)
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
    """
    Convert a sympy function to a pytensor function using :func:`pytensor.function`.

    Parameters
    ----------
    inputs: list[sp.Symbol]
        The inputs to the function.
    outputs: list[Union[sp.Symbol, sp.Expr]]
        The outputs of the function.
    cache: dict
        Dictionary mapping sympy symbols to pytensor symbols. Used to maintain namespace scope between different
        compiled functions.
    stack_return: bool
        If True, the function will return a single numpy array with all outputs. Otherwise it will return a list
        of numpy arrays.
    pop_return: bool
        If True, the function will return only the 0th element of the output. Used to remove the list wrapper around
        single-output functions.
    return_symbolic: bool
        If True, the function will return a symbolic pytensor computation graph instead of a compiled function.
    kwargs: dict
        Additional keyword arguments to pass to :func:`pytensor.function`. Ignored if return_symbolic is True.

    Returns
    -------
    f: Callable
        The compiled function.
    cache: dict
        Pytensor caching information.
    """
    kwargs = _configue_pytensor_kwargs(kwargs)
    cache = {} if cache is None else cache

    outputs = [outputs] if not isinstance(outputs, list) else outputs
    input_pt = [as_tensor(x, cache) for x in inputs]
    output_pt = [output_to_tensor(x, cache) for x in outputs]

    original_shape = [x.type.shape for x in output_pt]

    if stack_return:
        output_pt = pytensor.tensor.stack(output_pt)
    if pop_return:
        output_pt = (
            output_pt[0]
            if (isinstance(output_pt, list) and len(output_pt) == 1)
            else output_pt
        )

    if return_symbolic:
        return output_pt, cache

    f = pytensor.function(input_pt, output_pt, **kwargs)

    # If pytensor is in JAX mode, compiled functions will JAX array objects rather than numpy arrays
    # Add a wrapper to convert the JAX array to a numpy array
    if kwargs.get("mode", None) == "JAX":
        f = array_return_wrapper(f)

    # Pytensor never returns a scalar float (it will return a 0d array in this case), so we need to wrap the function
    # in this case
    if len(original_shape) == 1 and original_shape[0] == () and pop_return:
        f = pop_return_wrapper(f)

    return f, cache


def make_cache_key(name: str, cls: sp.Symbol) -> tuple:
    """
    Create a cache key for a sympy symbol.

    Used by sympytensor to map sympy symbols to pytensor symbols without creating duplicate mappings.

    Parameters
    ----------
    name: str
        The name of the sympy symbol.
    cls: sp.Symbol
        Type of sympy symbol (e.g. sp.Symbol, sp.Idx, TimeAwareSymbol, etc)

    Returns
    -------
    key: tuple
        A tuple containing information about the sympy symbol.
    """
    return name, cls, (), "floatX", ()


def make_return_dict_and_update_cache(
    input_symbols: list[sp.Symbol],
    output_tensors: TensorVariable | list[TensorVariable],
    cache: dict[tuple, TensorVariable],
    cls: sp.Symbol | None = None,
) -> tuple[dict, dict]:
    """
    Create a dictionary mapping from input symbols to output tensors, and update the cache.

    Parameters
    ----------
    input_symbols: list[sp.Symbol]
        Symbolic inputs to the function being compiled
    output_tensors: TensorVariable | list[TensorVariable]
        Output tensors of the function being compiled
    cache: dict
        Dictionary mapping sympy symbols to pytensor symbols, used to maintain a consistent namespace between
        compiled functions
    cls: sp.Symbol, optional
        The type of the sympy symbol (e.g. sp.Symbol, sp.Idx, sp.IndexedBase, TimeAwareSymbol, etc). Default is sp.Symbol

    Returns
    -------
    out_dict: dict
        Dictionary mapping from input symbols to output tensors
    cache: dict
        Updated cache dictionary
    """

    if cls is None:
        cls = sp.Symbol
    out_dict = {}
    for symbol, value in zip(input_symbols, output_tensors):
        cache_key = make_cache_key(symbol.name, cls)

        if cache_key in cache:
            pt_symbol = cache[cache_key]
        else:
            pt_symbol = pytensor.tensor.scalar(name=symbol.name, dtype="floatX")
            cache[cache_key] = pt_symbol

        out_dict[pt_symbol] = value

    return out_dict, cache
