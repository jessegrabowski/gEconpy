from collections.abc import Callable

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.model.compile import (
    compile_function,
    dictionary_return_wrapper,
    make_return_dict_and_update_cache,
)


def compile_param_dict_func(
    param_dict: SymbolDictionary,
    deterministic_dict: SymbolDictionary,
    cache: dict | None = None,
    return_symbolic: bool = False,
    mode: str | None = None,
) -> tuple[Callable, dict]:
    """
    Compile a function that computes every model parameter from the free parameters.

    Free parameters are the ones the user supplies as fixed values. Deterministic parameters are functions of the
    free parameters and are recomputed each time the free parameters change.

    Parameters
    ----------
    param_dict : SymbolDictionary
        Free parameters and their default values.
    deterministic_dict : SymbolDictionary
        Deterministic parameters, keyed by parameter symbol with the expression that computes each one as value.
    cache : dict, optional
        Sympytensor cache mapping sympy symbols to pytensor variables, shared so that every graph built from the
        same model maps a symbol to one pytensor variable. Default None starts an empty cache.
    return_symbolic : bool, optional
        If True, return the symbolic pytensor graph of the parameter values instead of a compiled function. Default
        False.
    mode : str, optional
        Pytensor compilation mode, such as ``'JAX'`` or ``'FAST_COMPILE'``, forwarded to the compiler. Default None
        uses the pytensor default.

    Returns
    -------
    f : callable
        Function taking the free parameters as keyword arguments and returning a dictionary of every parameter
        value. With ``return_symbolic`` this is instead a dictionary mapping pytensor output variables to their
        graphs.
    cache : dict
        The sympytensor cache after compilation.
    """
    cache = {} if cache is None else cache

    inputs = list(param_dict.to_sympy().keys())
    output_params = inputs + list(deterministic_dict.to_sympy().keys())
    output_exprs = inputs + list(deterministic_dict.values_to_float().values())

    f, cache = compile_function(
        inputs,
        output_exprs,
        cache=cache,
        return_symbolic=return_symbolic,
        pop_return=False,
        stack_return=not return_symbolic,
        mode=mode,
    )

    if return_symbolic:
        return make_return_dict_and_update_cache(output_params, f, cache)

    return dictionary_return_wrapper(f, output_params), cache
