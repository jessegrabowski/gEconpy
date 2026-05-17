import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block.basic import Block

# Ordered list of registered block subclasses. Order matters: the first matching detector wins, so place strictly
# simpler forms before more general ones (e.g. Cobb-Douglas before CES). Subclasses register themselves via
# :func:`register_block` at their own module's import time, so this list is empty until those modules are imported.
_REGISTRY: list[type[Block]] = []


def register_block(cls: type[Block]) -> type[Block]:
    """Register a :class:`Block` subclass for dispatch.

    Intended for use as a decorator on subclass definitions. Subclasses are appended in registration order, so callers
    who care about precedence should ensure the simpler form is imported (and thus registered) first.

    Parameters
    ----------
    cls : type
        The Block subclass to register.

    Returns
    -------
    cls : type
        The class, unchanged. Returned to support decorator usage.
    """
    if not (isinstance(cls, type) and issubclass(cls, Block)):
        raise TypeError(f"register_block expects a Block subclass, got {cls!r}")
    if cls in _REGISTRY:
        return cls
    _REGISTRY.append(cls)
    return cls


def dispatch_block(
    name: str,
    definitions: dict[int, sp.Eq] | None = None,
    controls: list[TimeAwareSymbol] | None = None,
    objective: dict[int, sp.Eq] | None = None,
    constraints: dict[int, sp.Eq] | None = None,
    identities: dict[int, sp.Eq] | None = None,
    calibration: dict[int, sp.Eq] | None = None,
    shocks: list[TimeAwareSymbol] | None = None,
    multipliers: dict[int, TimeAwareSymbol | None] | None = None,
    equation_flags: dict[int, dict[str, bool]] | None = None,
    source: str | None = None,
    symbol_locations: dict | None = None,
    ss_solution_dict=None,
) -> Block:
    """Construct a :class:`Block` or one of its specialized subclasses.

    Walks the registry in order. The first subclass whose ``detect`` returns True is constructed with the same kwargs.
    If none match, returns a general :class:`Block`. Detection is conservative: false positives (silent miss of
    user-added terms) are bugs; false negatives (no dispatch, slow path) are acceptable.

    Parameters mirror :class:`Block.__init__` exactly so the dispatcher is a drop-in replacement for ``Block(...)`` at
    the parser construction site.
    """
    kwargs = {
        "name": name,
        "definitions": definitions,
        "controls": controls,
        "objective": objective,
        "constraints": constraints,
        "identities": identities,
        "calibration": calibration,
        "shocks": shocks,
        "multipliers": multipliers,
        "equation_flags": equation_flags,
        "source": source,
        "symbol_locations": symbol_locations,
        "ss_solution_dict": ss_solution_dict,
    }
    for cls in _REGISTRY:
        if cls.detect(constraints=constraints, objective=objective, identities=identities):
            return cls(**kwargs)
    return Block(**kwargs)
