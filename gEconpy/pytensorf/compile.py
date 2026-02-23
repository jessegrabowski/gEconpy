from functools import lru_cache

import pytensor

from pymc.pytensorf import rewrite_pregrad
from pytensor.compile.function.types import Function
from pytensor.compile.mode import Mode
from pytensor.compile.profiling import ProfileStats
from pytensor.tensor.variable import TensorVariable


@lru_cache(maxsize=128)
def _compile_cached(
    inputs: tuple[TensorVariable, ...],
    outputs: TensorVariable | tuple[TensorVariable, ...],
    mode: str | Mode | None = None,
    updates: tuple[tuple[TensorVariable, TensorVariable], ...] | None = None,
    givens: tuple[tuple[TensorVariable, TensorVariable], ...] | None = None,
    no_default_updates: bool = False,
    accept_inplace: bool = False,
    name: str | None = None,
    rebuild_strict: bool = True,
    allow_input_downcast: bool | None = None,
    profile: bool | ProfileStats | None = None,
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
        no_default_updates=no_default_updates,
        accept_inplace=accept_inplace,
        name=name,
        rebuild_strict=rebuild_strict,
        allow_input_downcast=allow_input_downcast,
        profile=profile,
        on_unused_input=on_unused_input,
        trust_input=trust_input,
    )


def compile_pytensor_function(
    inputs: list[TensorVariable],
    outputs: TensorVariable | list[TensorVariable],
    mode: str | Mode | None = None,
    updates: dict[TensorVariable, TensorVariable] | list[tuple[TensorVariable, TensorVariable]] | None = None,
    givens: dict[TensorVariable, TensorVariable] | list[tuple[TensorVariable, TensorVariable]] | None = None,
    no_default_updates: bool = False,
    accept_inplace: bool = False,
    name: str | None = None,
    rebuild_strict: bool = True,
    allow_input_downcast: bool | None = None,
    profile: bool | ProfileStats | None = None,
    on_unused_input: str | None = "ignore",
    trust_input: bool = False,
) -> Function:
    """Compile a pytensor graph to a callable function.

    Wraps ``pytensor.function`` with:

    - Pre-grad rewrites (canonicalize + stabilize) via ``rewrite_pregrad``
    - Function caching via ``functools.cache`` (identity-based on graph nodes)
    - ``on_unused_input="ignore"`` by default

    All parameters mirror ``pytensor.function`` except for the changed defaults
    noted above.

    Parameters
    ----------
    inputs : list of TensorVariable
        Input nodes for the compiled function.
    outputs : TensorVariable or list of TensorVariable
        Output nodes for the compiled function.
    mode : str or Mode, optional
        Pytensor compilation mode. Common values: ``"FAST_COMPILE"``
        (Python, no C), ``"FAST_RUN"`` (C compilation), ``"JAX"``
        (JAX backend). Default is ``None`` (pytensor's default).
    updates : dict or list of tuples, optional
        Expressions for shared variable updates.
    givens : dict or list of tuples, optional
        Substitutions to apply before compiling.
    no_default_updates : bool
        If True, do not perform default updates on shared variables.
    accept_inplace : bool
        If True, accept graph with in-place operations.
    name : str, optional
        Name for the compiled function (for debugging).
    rebuild_strict : bool
        If True, require inputs to match graph exactly.
    allow_input_downcast : bool, optional
        If True, allow numeric inputs to be silently downcast.
    profile : bool or ProfileStats, optional
        If True, enable profiling.
    on_unused_input : str, optional
        What to do if an input is unused. Default is ``"ignore"``.
    trust_input : bool
        If True, skip input validation at call time. Default is ``False``.

    Returns
    -------
    f : pytensor.function
        Compiled callable.
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
        no_default_updates=no_default_updates,
        accept_inplace=accept_inplace,
        name=name,
        rebuild_strict=rebuild_strict,
        allow_input_downcast=allow_input_downcast,
        profile=profile,
        on_unused_input=on_unused_input,
        trust_input=trust_input,
    )


def clear_compile_cache() -> None:
    """Release all cached compiled functions."""
    _compile_cached.cache_clear()


def compile_cache_info():
    """Return cache statistics (hits, misses, maxsize, currsize)."""
    return _compile_cached.cache_info()
