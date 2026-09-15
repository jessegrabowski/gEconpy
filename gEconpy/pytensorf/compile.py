from functools import lru_cache

import pytensor

from pytensor.compile.executor import Function
from pytensor.compile.mode import Mode
from pytensor.graph.rewriting import rewrite_graph
from pytensor.tensor.variable import TensorVariable


def rewrite_pregrad(graph):
    """Run the canonicalize and stabilize rewrites on a graph before gradient propagation.

    Parameters
    ----------
    graph : TensorVariable or list of TensorVariable
        Graph to rewrite.

    Returns
    -------
    rewritten : TensorVariable or list of TensorVariable
        The rewritten graph, matching the structure of ``graph``.
    """
    return rewrite_graph(graph, include=("canonicalize", "stabilize"))


@lru_cache(maxsize=128)
def _compile_cached(
    inputs: tuple[TensorVariable, ...],
    outputs: TensorVariable | tuple[TensorVariable, ...],
    mode: str | Mode | None = None,
    updates: tuple[tuple[TensorVariable, TensorVariable], ...] | None = None,
    givens: tuple[tuple[TensorVariable, TensorVariable], ...] | None = None,
    accept_inplace: bool = False,
    name: str | None = None,
    rebuild_strict: bool = True,
    allow_input_downcast: bool | None = None,
    on_unused_input: str | None = "ignore",
    trust_input: bool = False,
) -> Function:
    """Compile pytensor function with caching."""
    if isinstance(outputs, tuple):
        outputs = list(outputs)

    outputs = rewrite_pregrad(outputs)

    return pytensor.function(
        list(inputs),
        outputs,
        mode=mode,
        updates=dict(updates) if updates is not None else None,
        givens=dict(givens) if givens is not None else None,
        accept_inplace=accept_inplace,
        name=name,
        rebuild_strict=rebuild_strict,
        allow_input_downcast=allow_input_downcast,
        on_unused_input=on_unused_input,
        trust_input=trust_input,
    )


def compile_pytensor_function(
    inputs: list[TensorVariable],
    outputs: TensorVariable | list[TensorVariable],
    mode: str | Mode | None = None,
    updates: dict[TensorVariable, TensorVariable] | list[tuple[TensorVariable, TensorVariable]] | None = None,
    givens: dict[TensorVariable, TensorVariable] | list[tuple[TensorVariable, TensorVariable]] | None = None,
    accept_inplace: bool = False,
    name: str | None = None,
    rebuild_strict: bool = True,
    allow_input_downcast: bool | None = None,
    on_unused_input: str | None = "ignore",
    trust_input: bool = False,
) -> Function:
    """Compile a pytensor graph to a callable function.

    Wraps ``pytensor.function``, adding pre-gradient rewrites via
    :func:`~gEconpy.pytensorf.compile.rewrite_pregrad` and caching keyed on the identity of the graph nodes.
    Parameters mirror ``pytensor.function``, except that ``on_unused_input`` defaults to ``"ignore"``.

    Parameters
    ----------
    inputs : list of TensorVariable
        Input nodes for the compiled function.
    outputs : TensorVariable or list of TensorVariable
        Output nodes for the compiled function.
    mode : str or Mode, optional
        Pytensor compilation mode, such as ``"FAST_COMPILE"``, ``"FAST_RUN"``, ``"NUMBA"`` or ``"JAX"``.
        Defaults to None, which uses pytensor's own default mode.
    updates : dict or list of tuples, optional
        Expressions for shared variable updates.
    givens : dict or list of tuples, optional
        Substitutions to apply before compiling.
    accept_inplace : bool, optional
        If True, accept a graph containing in-place operations. Defaults to False.
    name : str, optional
        Name for the compiled function, used in debugging output. Defaults to None.
    rebuild_strict : bool, optional
        If True, require the inputs to match the graph exactly. Defaults to True.
    allow_input_downcast : bool, optional
        If True, allow numeric inputs to be silently downcast. Defaults to None.
    on_unused_input : str, optional
        What to do when an input is unused. Defaults to ``"ignore"``.
    trust_input : bool, optional
        If True, skip input validation at call time. Defaults to False.

    Returns
    -------
    f : callable
        Compiled pytensor function.
    """
    if isinstance(outputs, list):
        outputs = tuple(outputs)

    # Convert mutable containers to hashable tuples for caching
    frozen_updates = tuple(updates.items() if isinstance(updates, dict) else updates) if updates is not None else None
    frozen_givens = tuple(givens.items() if isinstance(givens, dict) else givens) if givens is not None else None

    return _compile_cached(
        tuple(inputs),
        outputs,
        mode=mode,
        updates=frozen_updates,
        givens=frozen_givens,
        accept_inplace=accept_inplace,
        name=name,
        rebuild_strict=rebuild_strict,
        allow_input_downcast=allow_input_downcast,
        on_unused_input=on_unused_input,
        trust_input=trust_input,
    )


def clear_compile_cache() -> None:
    """Release all cached compiled functions."""
    _compile_cached.cache_clear()


def compile_cache_info():
    """Report the state of the compiled function cache.

    Returns
    -------
    info : tuple
        Named tuple of cache statistics, with fields ``hits``, ``misses``, ``maxsize`` and ``currsize``.
    """
    return _compile_cached.cache_info()
