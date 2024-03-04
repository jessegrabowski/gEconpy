from typing import Callable, Optional

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.model.compile import (
    BACKENDS,
    compile_function,
    dictionary_return_wrapper,
)


def compile_param_dict_func(
    param_dict: SymbolDictionary,
    deterministic_dict: SymbolDictionary,
    backend: BACKENDS = "numpy",
    cache: Optional[dict] = None,
) -> tuple[Callable, dict]:
    """
    Compile a function to compute model parameters from given "free" parameters.

    Most model parameters are provided by the user as fixed values. We denote these are "free" parameters. Others are
    functions of the free parameters, and need to be dynamically recomputed each time the free parameters change.

    Parameters
    ----------
    param_dict: SymbolDictionary
        A dictionary of free parameters.
    deterministic_dict: SymbolDictionary
        A dictionary of deterministic parameters, with the keys being the parameters and the values being the
        expressions to compute them.
    backend: str, one of "numpy", "numba", "pytensor"
        The backend to use for the compiled function.
    cache: dict, optional
        A dictionary mapping from pytensor symbols to sympy expressions. Used to prevent duplicate mappings from
        sympy symbol to pytensor symbol from being created. Default is a empty dictionary, implying no other functions
        have been compiled yet.

    Returns
    -------
    f: Callable
        A function that takes the free parameters as keyword arguments and returns a dictionary of the computed
        parameters.
    cache: dict
        A dictionary mapping from sympy symbols to pytensor symbols.
    """
    cache = {} if cache is None else cache

    inputs = list(param_dict.to_sympy().keys())
    output_params = inputs + list(deterministic_dict.to_sympy().keys())
    output_exprs = inputs + list(deterministic_dict.values())

    f, cache = compile_function(inputs, output_exprs, backend=backend, cache=cache)

    return dictionary_return_wrapper(f, output_params), cache
