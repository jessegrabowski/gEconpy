import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block.basic import Block

# Ordered list of registered block subclasses. Order matters: the first matching detector wins, so place strictly
# simpler forms before more general ones (e.g. Cobb-Douglas before CES). Subclasses register themselves via
# :func:`register_block` at their own module's import time, so this list is empty until those modules are imported.
_REGISTRY: list[type[Block]] = []


def register_block(cls: type[Block]) -> type[Block]:
    """
    Register a :class:`~gEconpy.model.block.basic.Block` subclass for dispatch.

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
    """
    Construct a :class:`~gEconpy.model.block.basic.Block` or one of its specialized subclasses.

    Walk the registry in order and construct the first subclass whose ``detect`` returns True, passing it the same
    keyword arguments. If none match, return a general :class:`~gEconpy.model.block.basic.Block`. Detection is
    conservative, so a missed dispatch costs only speed while a false positive is a bug.

    Parameters mirror :class:`~gEconpy.model.block.basic.Block` exactly, so the dispatcher is a drop-in replacement
    for ``Block(...)`` at the parser construction site.

    Parameters
    ----------
    name : str
        The name of the block.
    definitions : dict mapping int to sp.Eq, optional
        Definition equations, indexed by equation number.
    controls : list of TimeAwareSymbol, optional
        Control variables.
    objective : dict mapping int to sp.Eq, optional
        The objective equation, indexed by equation number.
    constraints : dict mapping int to sp.Eq, optional
        Constraint equations, indexed by equation number.
    identities : dict mapping int to sp.Eq, optional
        Identity equations, indexed by equation number.
    calibration : dict mapping int to sp.Eq, optional
        Calibration equations, indexed by equation number.
    shocks : list of TimeAwareSymbol, optional
        Shock variables.
    multipliers : dict mapping int to TimeAwareSymbol, optional
        Mapping from constraint index to the Lagrange multiplier on that constraint.
    equation_flags : dict mapping int to dict, optional
        Mapping from equation index to that equation's flag dictionary.
    source : str, optional
        The source code of the GCN file, used for rich error reporting.
    symbol_locations : dict, optional
        Mapping from symbol name to its ParseLocation in the source, used for rich error reporting during
        validation.
    ss_solution_dict : SymbolDictionary, optional
        Analytically known steady-state solutions. Used to resolve calibration expressions that reference
        steady-state variables.

    Returns
    -------
    block : Block
        The constructed block, of the most specific registered subclass that matches.
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
