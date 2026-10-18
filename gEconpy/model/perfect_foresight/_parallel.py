import multiprocessing
import platform
import warnings

from collections.abc import Callable, Mapping, Sequence
from multiprocessing.context import BaseContext
from typing import Any

import cloudpickle
import numpy as np

from pytensor.compile.mode import Mode, get_mode
from pytensor.link.jax.linker import JAXLinker

# Solver work counts reported per solve, summed by the callers over every solve a unit of work took.
_WORK_COUNT_FIELDS = ("nit", "nfev", "njev", "nsolve")

_WORKER_STATE: dict = {}


def _map_in_workers(
    function: Callable[..., Any],
    items: Sequence[Mapping[str, Any]],
    workers: int,
    context: BaseContext,
) -> list[Any]:
    """Call ``function(**item)`` for every item in a pool of ``workers`` processes, returning the results in order."""
    # A forked worker inherits the function through the initializer, unpickled. Under any other start method the child
    # reads initializer arguments only after importing gEconpy, so passing the function there would start the workers
    # one at a time, and it travels pickled with each block instead. cloudpickle also carries a solver class defined
    # in the user's script.
    if context.get_start_method() == "fork":
        payload, initargs = None, (function,)
    else:
        payload, initargs = cloudpickle.dumps(function, protocol=-1), (None,)

    blocks = [list(block) for block in np.array_split(np.arange(len(items)), workers)]
    with context.Pool(workers, initializer=_initialize_worker, initargs=initargs) as pool:
        mapped_blocks = pool.starmap(_map_block, [(payload, [items[index] for index in block]) for block in blocks])
    return [result for block in mapped_blocks for result in block]


def _initialize_worker(function: Callable[..., Any] | None) -> None:
    _WORKER_STATE["function"] = function


def _map_block(payload: bytes | None, block: list[Mapping[str, Any]]) -> list[Any]:
    function = _WORKER_STATE["function"] if payload is None else cloudpickle.loads(payload)
    return [function(**item) for item in block]


def _multiprocessing_context(mp_ctx: str | BaseContext | None, mode: str | Mode | None) -> BaseContext:
    """Resolve ``mp_ctx`` to a multiprocessing context by the rules :func:`pymc.sample` uses."""
    user_specified = mp_ctx is not None
    if mp_ctx is None and platform.system() == "Darwin":
        mp_ctx = "fork" if platform.processor() == "arm" else "forkserver"
    if mp_ctx is None or isinstance(mp_ctx, str):
        mp_ctx = multiprocessing.get_context(mp_ctx)

    jax_mode = mode is not None and isinstance(get_mode(mode).linker, JAXLinker)
    if not jax_mode or mp_ctx.get_start_method() != "fork":
        return mp_ctx
    if user_specified:
        warnings.warn(
            "Using a JAX compile mode with multiprocessing start method 'fork' is unsafe and may deadlock. Consider "
            "passing mp_ctx='forkserver' or mp_ctx='spawn'.",
            UserWarning,
            stacklevel=3,
        )
        return mp_ctx
    return multiprocessing.get_context(
        "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    )
