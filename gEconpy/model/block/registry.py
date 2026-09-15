import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block.basic import Block
from gEconpy.parser.errors import ParseLocation

# Registration order is dispatch precedence: the first subclass whose ``detect`` matches wins. Subclasses register
# themselves when their module is imported, so the import order in ``gEconpy/model/block/__init__.py`` sets the
# precedence, and this list is empty until that package has loaded.
_REGISTRY: list[type[Block]] = []


def register_block(cls: type[Block]) -> type[Block]:
    """
    Register a :class:`~gEconpy.model.block.basic.Block` subclass for dispatch.

    Use as a decorator on the subclass definition. Registering a class twice is a no-op.

    Parameters
    ----------
    cls : type of Block
        The subclass to register. It must define a ``detect`` classmethod.

    Returns
    -------
    cls : type of Block
        The class, unchanged.
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
    symbol_locations: dict[str, ParseLocation] | None = None,
    ss_solution_dict: SymbolDictionary | None = None,
) -> Block:
    """
    Construct the most specific registered :class:`~gEconpy.model.block.basic.Block` subclass for a parsed block.

    Walk the registry in order and construct the first subclass whose ``detect`` returns True. Construct a general
    :class:`~gEconpy.model.block.basic.Block` when none matches. Detection is conservative: a missed dispatch only
    costs compile time, while a false positive would silently drop terms from the user's equations.

    The parameters are those of :class:`~gEconpy.model.block.basic.Block`, so this function is a drop-in replacement
    for ``Block(...)`` at the parser's construction site.

    Parameters
    ----------
    name : str
        The name of the block.
    definitions : dict mapping int to sympy.Eq, optional
        Definition equations, keyed by equation number.
    controls : list of TimeAwareSymbol, optional
        Control variables.
    objective : dict mapping int to sympy.Eq, optional
        The objective equation, keyed by equation number.
    constraints : dict mapping int to sympy.Eq, optional
        Constraint equations, keyed by equation number.
    identities : dict mapping int to sympy.Eq, optional
        Identity equations, keyed by equation number.
    calibration : dict mapping int to sympy.Eq, optional
        Calibration equations, keyed by equation number.
    shocks : list of TimeAwareSymbol, optional
        Shock variables.
    multipliers : dict mapping int to TimeAwareSymbol or None, optional
        The Lagrange multiplier on each constraint, keyed by constraint index.
    equation_flags : dict mapping int to dict, optional
        The flag dictionary of each equation, keyed by equation number.
    source : str, optional
        The source code of the GCN file, for error reporting.
    symbol_locations : dict mapping str to ParseLocation, optional
        The location of each symbol in ``source``, for error reporting.
    ss_solution_dict : SymbolDictionary, optional
        Analytically known steady-state solutions, used to resolve calibration expressions that reference
        steady-state variables.

    Returns
    -------
    block : Block
        The constructed block.
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
