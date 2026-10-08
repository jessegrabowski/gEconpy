from functools import _CacheInfo, lru_cache

import pytensor

from pytensor.compile.executor import Function
from pytensor.compile.mode import Mode
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import graph_inputs
from pytensor.graph.utils import MissingInputError
from pytensor.tensor.variable import TensorVariable


def rewrite_pregrad(graph: TensorVariable | list[TensorVariable]) -> TensorVariable | list[TensorVariable]:
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

    Wraps :func:`pytensor.function`, adding pre-gradient rewrites via
    :func:`~gEconpy.pytensorf.compile.rewrite_pregrad` and caching keyed on the structure of the graph.
    Parameters mirror :func:`pytensor.function`, except that ``on_unused_input`` defaults to ``"ignore"``.

    Parameters
    ----------
    inputs : list of TensorVariable
        Input nodes for the compiled function.
    outputs : TensorVariable or list of TensorVariable
        Output nodes for the compiled function.
    mode : str or Mode, optional
        Pytensor compilation mode, such as ``"FAST_COMPILE"``, ``"FAST_RUN"``, ``"NUMBA"`` or ``"JAX"``. Defaults to
        ``None``, which uses pytensor's own default mode.
    updates : dict or list of tuples, optional
        Expressions for shared variable updates. Defaults to ``None``.
    givens : dict or list of tuples, optional
        Substitutions to apply before compiling. Defaults to ``None``.
    accept_inplace : bool, optional
        Whether to accept a graph containing in-place operations. Defaults to ``False``.
    name : str, optional
        Name for the compiled function, used in debugging output. Defaults to ``None``.
    rebuild_strict : bool, optional
        Whether to require the inputs to match the graph exactly. Defaults to ``True``.
    allow_input_downcast : bool, optional
        Whether to allow numeric inputs to be silently downcast. Defaults to ``None``, pytensor's own default.
    on_unused_input : str, optional
        What to do when an input is unused. Defaults to ``"ignore"``.
    trust_input : bool, optional
        Whether to skip input validation at call time. Defaults to ``False``.

    Returns
    -------
    f : callable
        Compiled pytensor function.
    """
    frozen_updates = _freeze_pairs(updates)
    frozen_givens = _freeze_pairs(givens)
    try:
        graph = _GraphKey(tuple(inputs), outputs, frozen_givens)
    except MissingInputError:
        # pytensor.function tolerates an input list FunctionGraph rejects. Those still compile, just uncached.
        return _compile(
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

    return _compile_cached(
        graph,
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


def compile_cache_info() -> _CacheInfo:
    """Report the state of the compiled function cache.

    Returns
    -------
    info : CacheInfo
        Named tuple of cache statistics, with fields ``hits``, ``misses``, ``maxsize`` and ``currsize``.
    """
    return _compile_cached.cache_info()


def _freeze_pairs(
    pairs: dict[TensorVariable, TensorVariable] | list[tuple[TensorVariable, TensorVariable]] | None,
) -> tuple[tuple[TensorVariable, TensorVariable], ...] | None:
    """Return substitutions as a hashable tuple, so the same pairs given as a dict and as a list share a key."""
    if pairs is None:
        return None
    return tuple(pairs.items() if isinstance(pairs, dict) else pairs)


class _GraphKey:
    """
    Carry a graph while comparing by its structure, so the compile cache can key on one argument.

    Two keys are equal when their graphs are structurally identical and they agree on the input names, on
    whether the caller asked for one output or a sequence, and on which shared variables the graph reads.
    """

    __slots__ = ("_identity", "inputs", "outputs")

    def __init__(
        self,
        inputs: tuple[TensorVariable, ...],
        outputs: TensorVariable | list[TensorVariable] | tuple[TensorVariable, ...],
        givens: tuple[tuple[TensorVariable, TensorVariable], ...] | None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        is_sequence = isinstance(outputs, list | tuple)
        output_list = list(outputs) if is_sequence else [outputs]
        # Neither a substituted variable nor a shared one is listed as an input, so the frozen graph has to
        # name them both or it cannot be built at all.
        substituted = tuple(source for source, _ in givens or ())
        shared = tuple(root for root in graph_inputs(output_list) if isinstance(root, SharedVariable))
        frozen = FunctionGraph(list(inputs) + list(substituted) + list(shared), output_list, clone=True).freeze()
        # Freezing compares a shared variable by its type, so two holding different values look identical. The
        # compiled function reads one specific variable's storage, so they are compared by identity instead.
        self._identity = (frozen, tuple(variable.name for variable in inputs), is_sequence, shared)

    def __hash__(self) -> int:
        return hash(self._identity)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _GraphKey) and self._identity == other._identity


@lru_cache(maxsize=128)
def _compile_cached(
    graph: "_GraphKey",
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
    return _compile(
        graph.inputs,
        graph.outputs,
        mode=mode,
        updates=updates,
        givens=givens,
        accept_inplace=accept_inplace,
        name=name,
        rebuild_strict=rebuild_strict,
        allow_input_downcast=allow_input_downcast,
        on_unused_input=on_unused_input,
        trust_input=trust_input,
    )


def _compile(
    inputs: tuple[TensorVariable, ...],
    outputs: TensorVariable | list[TensorVariable] | tuple[TensorVariable, ...],
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
    return pytensor.function(
        list(inputs),
        rewrite_pregrad(list(outputs) if isinstance(outputs, tuple) else outputs),
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
