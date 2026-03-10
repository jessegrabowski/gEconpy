from collections.abc import Callable
from functools import wraps

import numpy as np
import pytensor
import sympy as sp

from pytensor import tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor import TensorVariable
from sympytensor import as_tensor

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.pytensorf.compile import compile_pytensor_function


def output_to_tensor(x, cache):
    if isinstance(x, int | float | sp.Float | sp.Integer):
        return pytensor.tensor.constant(x, dtype=pytensor.config.floatX)

    return as_tensor(x, cache)


def dictionary_return_wrapper(f: Callable, outputs: list[sp.Symbol]) -> Callable:
    """
    Wrap a function that returns a numpy array to instead return a SymbolDictionary.

    The dictionary returned has keys corresponding to the output symbols.

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
        return SteadyStateResults(zip(outputs, values, strict=False)).to_string()

    return inner


def compile_function(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr] | sp.MutableDenseMatrix,
    cache: dict | None = None,
    stack_return: bool = False,
    pop_return: bool = False,
    return_symbolic: bool = False,
    **kwargs,
) -> tuple[Callable, dict]:
    """
    Compile a sympy function to a pytensor function.

    Parameters
    ----------
    inputs: list[sp.Symbol]
        The inputs to the function.

    outputs: list[Union[sp.Symbol, sp.Expr]]
        The outputs of the function.


    cache: dict, optional
        A dictionary mapping from pytensor symbols to sympy expressions. Used to prevent duplicate mappings from
        sympy symbol to pytensor symbol from being created. Default is an empty dictionary.

    stack_return: bool, optional
        If True, the function will return a single numpy array with all outputs stacked. Default is False.

    pop_return: bool, optional
        If True, the function will return only the 0th element of the output. Default is False.

    return_symbolic: bool, default False
        If True, return a symbolic pytensor computation graph instead of a compiled function.

    Returns
    -------
    f: Callable
        A python function that computes the outputs from the inputs.

    cache: dict
        A dictionary mapping from sympy symbols to pytensor symbols.
    """
    f, cache = compile_to_pytensor_function(inputs, outputs, cache, stack_return, pop_return, return_symbolic, **kwargs)

    return f, cache


def sympy_to_pytensor(
    inputs: list[sp.Symbol],
    outputs: list[sp.Symbol | sp.Expr] | sp.MutableDenseMatrix,
    cache: dict | None = None,
) -> tuple[list[TensorVariable], list[TensorVariable], dict]:
    """
    Convert sympy expressions to pytensor graph nodes.

    This is the sympytensor bridge: it takes sympy symbols and expressions and returns the corresponding pytensor
    input and output nodes, along with the updated cache that maintains the mapping between sympy and pytensor symbols.

    Parameters
    ----------
    inputs : list of sp.Symbol
        Sympy input symbols.
    outputs : list of sp.Symbol, sp.Expr, or sp.MutableDenseMatrix
        Sympy output expressions.
    cache : dict, optional
        Dictionary mapping sympytensor cache keys to pytensor variables. Used to maintain a consistent namespace
        across multiple conversions. Default is an empty dictionary.

    Returns
    -------
    input_nodes : list of TensorVariable
        Pytensor input nodes corresponding to the sympy inputs.
    output_nodes : list of TensorVariable
        Pytensor output nodes corresponding to the sympy outputs.
    cache : dict
        Updated cache dictionary.
    """
    cache = {} if cache is None else cache
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    input_nodes = [as_tensor(x, cache) for x in inputs]
    output_nodes = [output_to_tensor(x, cache) for x in outputs]
    return input_nodes, output_nodes, cache


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
        Additional keyword arguments passed to ``pytensor.function`` via
        :func:`~gEconpy.pytensorf.compile.compile_pytensor_function`.
        Ignored if return_symbolic is True.

    Returns
    -------
    f: Callable
        The compiled function.
    cache: dict
        Pytensor caching information.
    """
    input_pt, output_pt, cache = sympy_to_pytensor(inputs, outputs, cache)

    if stack_return:
        output_pt = pytensor.tensor.stack(output_pt)
    if pop_return:
        output_pt = output_pt[0] if (isinstance(output_pt, list) and len(output_pt) == 1) else output_pt

    if return_symbolic:
        return output_pt, cache

    f = compile_pytensor_function(input_pt, output_pt, **kwargs)

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
        The type of the sympy symbol (e.g. sp.Symbol, sp.Idx, sp.IndexedBase, TimeAwareSymbol, etc). Default is
        sp.Symbol

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
    for symbol, value in zip(input_symbols, output_tensors, strict=False):
        cache_key = make_cache_key(symbol.name, cls)

        if cache_key in cache:
            pt_symbol = cache[cache_key]
        else:
            pt_symbol = pytensor.tensor.scalar(name=symbol.name, dtype="floatX")
            cache[cache_key] = pt_symbol

        out_dict[pt_symbol] = value

    return out_dict, cache


def compile_for_scipy(
    outputs: TensorVariable | list[TensorVariable],
    mode: str | None = None,
) -> Callable:
    """Compile a pytensor graph into a callable that accepts all inputs as keyword arguments.

    Unused kwargs are silently ignored. Intended for one-off evaluations (e.g. postprocessing at the solution
    point). For hot-loop scipy calls, use ``pack_and_compile`` instead.
    """
    inputs = list(explicit_graph_inputs(outputs))
    f = compile_pytensor_function(inputs, outputs, mode=mode, on_unused_input="ignore")
    accepted_names = frozenset(inp.name for inp in f.input_storage)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **{k: v for k, v in kwargs.items() if k in accepted_names})

    wrapper._inner = f
    return wrapper


def pack_and_compile(
    outputs: TensorVariable | list[TensorVariable],
    ss_nodes: list[TensorVariable],
    param_dict: SymbolDictionary | None = None,
    mode: str | None = None,
) -> Callable:
    """Replace SS variable scalars with ``x_flat[i]`` indexing and compile.

    Only SS nodes that are actual graph inputs are replaced. If ``param_dict`` is provided, parameter values
    are frozen into the closure and the returned function has signature ``f(x_flat)``; otherwise it is
    ``f(x_flat, *param_args)``.

    Parameters
    ----------
    outputs : TensorVariable or list of TensorVariable
        Graph outputs to compile.
    ss_nodes : list of TensorVariable
        All scalar SS variable nodes, defining the positional layout of ``x_flat``.
    param_dict : SymbolDictionary, optional
        Parameter values to freeze into the closure.
    mode : str, optional
        Pytensor compilation mode.

    Returns
    -------
    callable
        Compiled function.
    """
    graph_inputs = set(explicit_graph_inputs(outputs))
    active_nodes = [n for n in ss_nodes if n in graph_inputs]

    if not active_nodes:
        inner = compile_pytensor_function(
            list(explicit_graph_inputs(outputs)), outputs, mode=mode, on_unused_input="ignore"
        )
        accepted = frozenset(inp.name for inp in inner.input_storage)
        if param_dict is not None:
            frozen_params = {k: v for k, v in param_dict.items() if k in accepted}

            def f(_x_flat: np.ndarray) -> np.ndarray:
                return inner(**frozen_params)
        else:

            def f(_x_flat: np.ndarray, *args) -> np.ndarray:
                return inner(*args)

        return f

    x_flat = pt.dvector("x_flat")
    replacements = {node: x_flat[i] for i, node in enumerate(active_nodes)}
    new_outputs = graph_replace(outputs, replacements, strict=False)

    active_set = set(active_nodes)
    param_inputs = [v for v in explicit_graph_inputs(outputs) if v not in active_set]
    all_inputs = [x_flat, *list(param_inputs)]
    inner = compile_pytensor_function(all_inputs, new_outputs, mode=mode, on_unused_input="ignore")

    need_slice = len(active_nodes) != len(ss_nodes)
    active_indices = np.array([i for i, n in enumerate(ss_nodes) if n in graph_inputs]) if need_slice else None

    if param_dict is not None:
        param_names = [v.name for v in param_inputs]
        frozen_args = tuple(float(param_dict[n]) for n in param_names)

        def f(x_flat: np.ndarray) -> np.ndarray:
            x = x_flat[active_indices] if need_slice else x_flat
            return inner(x, *frozen_args)
    else:

        def f(x_flat: np.ndarray, *args) -> np.ndarray:
            x = x_flat[active_indices] if need_slice else x_flat
            return inner(x, *args)

    return f
