import contextlib
import re
import warnings

from collections import defaultdict
from typing import Any

import pyparsing as pp

from gEconpy.classes.time_aware_symbol import DEFAULT_ASSUMPTIONS
from gEconpy.exceptions import DeprecatedAssumptionsBlockWarning
from gEconpy.parser.ast import SymbolDeclaration, Variable, assumptions_implied_by_bounds, variable_key
from gEconpy.parser.constants import (
    GCN_ASSUMPTIONS,
    KNOWN_ASSUMPTIONS,
    KNOWN_SYMBOL_FIELDS,
    SIGN_ASSUMPTION_INTERVALS,
    SYMBOL_METADATA_FIELDS,
)
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure
from gEconpy.parser.grammar.statements import VARIABLE_LIST, VARIABLE_REF
from gEconpy.parser.grammar.tokens import (
    COMMA,
    COMMENT,
    EQUALS,
    EQUALS_LITERAL,
    IDENTIFIER,
    KW_ASSUMPTIONS,
    KW_FALSE,
    KW_NONE,
    KW_OPTIONS,
    KW_SYMBOLS,
    KW_TRUE,
    KW_TRYREDUCE,
    LBRACE,
    NUMBER,
    RBRACE,
    SEMI,
    STRING,
)
from gEconpy.parser.suggestions import suggest_assumption, suggest_symbol_field


def parse_options(text: str) -> dict[str, bool | str]:
    """
    Parse the ``options`` block of a GCN file.

    Parameters
    ----------
    text : str
        GCN text to scan for an options block.

    Returns
    -------
    options : dict mapping str to bool or str
        The declared options. Empty if the text has no options block.
    """
    return _first_match(OPTIONS_BLOCK, text, default={})


def parse_tryreduce(text: str) -> list[str]:
    """
    Parse the ``tryreduce`` block of a GCN file.

    Parameters
    ----------
    text : str
        GCN text to scan for a tryreduce block.

    Returns
    -------
    variables : list of str
        Names of the variables to try to eliminate. Empty if the text has no tryreduce block.
    """
    return _first_match(TRYREDUCE_BLOCK, text, default=[])


def parse_assumptions(text: str) -> dict[str, dict[str, bool]]:
    """
    Parse the ``assumptions`` block of a GCN file.

    Parameters
    ----------
    text : str
        GCN text to scan for an assumptions block.

    Returns
    -------
    assumptions : dict mapping str to dict
        Sympy assumptions for each named variable. If the text has no assumptions block, returns a default
        dictionary that supplies the package defaults for any name.
    """
    return _first_match(ASSUMPTIONS_BLOCK, text, default=defaultdict(DEFAULT_ASSUMPTIONS.copy))


def parse_symbols(text: str) -> dict[str, SymbolDeclaration]:
    """
    Parse the ``symbols`` block of a GCN file.

    Parameters
    ----------
    text : str
        GCN text to scan for a symbols block.

    Returns
    -------
    symbols : dict mapping str to SymbolDeclaration
        One declaration per declared symbol. Empty if the text has no symbols block.
    """
    return _first_match(SYMBOLS_BLOCK, text, default={})


def extract_special_block_content(text: str, block_name: str) -> str | None:
    """
    Extract the raw text of a special block, braces and trailing semicolon included.

    Parameters
    ----------
    text : str
        GCN text to search.
    block_name : str
        Name of the block to extract, for example ``options``. Matching ignores case.

    Returns
    -------
    content : str or None
        The matched text, or None if the block is not present.
    """
    match = _special_block_pattern(block_name).search(text)
    if match is None:
        return None
    return match.group(0)


def remove_special_block(text: str, block_name: str) -> str:
    """
    Delete every occurrence of a special block from GCN text.

    Parameters
    ----------
    text : str
        GCN text to edit.
    block_name : str
        Name of the block to remove, for example ``options``. Matching ignores case.

    Returns
    -------
    text : str
        The text with the block removed.
    """
    return _special_block_pattern(block_name).sub("", text)


def _first_match[DefaultT](block: pp.ParserElement, text: str, default: DefaultT) -> Any | DefaultT:
    with contextlib.suppress(pp.ParseException):
        for result, _start, _end in block.scan_string(text):
            return result[0]
    return default


def _special_block_pattern(block_name: str) -> re.Pattern[str]:
    return re.compile(rf"{block_name}\s*\{{.*?\}};", re.DOTALL | re.IGNORECASE)


OPTION_KEY = pp.Combine(IDENTIFIER + pp.ZeroOrMore(pp.White(" ") + IDENTIFIER))
OPTION_VALUE = (
    KW_TRUE.copy().set_parse_action(lambda _: True) | KW_FALSE.copy().set_parse_action(lambda _: False) | IDENTIFIER
)
OPTION_ENTRY = pp.Group(OPTION_KEY("key") - EQUALS - OPTION_VALUE("value") - SEMI)
OPTIONS_BLOCK = KW_OPTIONS.suppress() - LBRACE - pp.ZeroOrMore(OPTION_ENTRY)("entries") - RBRACE - SEMI

OPTIONS_BLOCK.set_parse_action(lambda t: {entry.key: entry.value for entry in t.entries})
OPTIONS_BLOCK.ignore(COMMENT)

TRYREDUCE_BLOCK = KW_TRYREDUCE.suppress() - LBRACE - pp.Optional(VARIABLE_LIST("variables") + SEMI) - RBRACE - SEMI

TRYREDUCE_BLOCK.set_parse_action(lambda t: [[variable.name for variable in t.variables]])
TRYREDUCE_BLOCK.ignore(COMMENT)

ASSUMPTION_NAME = pp.one_of(GCN_ASSUMPTIONS, caseless=True)("assumption")


def _unknown_assumption_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    name = toks[0]
    raise GCNParseFailure(
        s,
        loc,
        f"Unknown assumption '{name}'",
        code=ErrorCode.E015,
        found=name,
        suggestions=suggest_assumption(name),
    )


UNKNOWN_ASSUMPTION = (
    (IDENTIFIER("unknown_name") + pp.FollowedBy(LBRACE))
    .add_condition(lambda _s, _loc, toks: toks[0].lower() not in KNOWN_ASSUMPTIONS)
    .set_parse_action(_unknown_assumption_fail)
)

ASSUMPTION_ITEM = VARIABLE_REF | IDENTIFIER.copy().set_parse_action(lambda t: t[0])
ASSUMPTION_LIST = pp.DelimitedList(ASSUMPTION_ITEM)

VALID_ASSUMPTION_SUBBLOCK = pp.Group(ASSUMPTION_NAME - LBRACE - ASSUMPTION_LIST("variables") - SEMI - RBRACE - SEMI)
ASSUMPTION_SUBBLOCK = VALID_ASSUMPTION_SUBBLOCK | UNKNOWN_ASSUMPTION

ASSUMPTIONS_BLOCK = KW_ASSUMPTIONS.suppress() - LBRACE - pp.ZeroOrMore(ASSUMPTION_SUBBLOCK)("subblocks") - RBRACE - SEMI


def _build_assumptions(tokens: pp.ParseResults) -> dict[str, dict[str, bool]]:
    assumption_kwargs = defaultdict(DEFAULT_ASSUMPTIONS.copy)

    for subblock in tokens.subblocks:
        assumption_name = subblock.assumption.lower()
        for item in subblock.variables:
            name = variable_key(item.name) if isinstance(item, Variable) else str(item)
            assumption_kwargs[name][assumption_name] = True
            # ``unit_interval`` is not a sympy predicate. It sits inertly in ``assumptions0`` so the steady-state
            # solver can route the variable to a logit transform. Because it implies positivity, sympy's
            # ``positive`` is set as well, and that predicate does carry algebraic consequences.
            if assumption_name == "unit_interval":
                assumption_kwargs[name]["positive"] = True

    declarations = dict(assumption_kwargs)
    if declarations:
        warnings.warn(DeprecatedAssumptionsBlockWarning(declarations), stacklevel=2)

    return declarations


ASSUMPTIONS_BLOCK.set_parse_action(_build_assumptions)
ASSUMPTIONS_BLOCK.ignore(COMMENT)

SYMBOL_ITEM = VARIABLE_REF | IDENTIFIER.copy().set_parse_action(lambda t: t[0])

BOUND_VALUE = KW_NONE.copy().set_parse_action(lambda _: [None]) | pp.Combine(
    pp.Optional(pp.Literal("-")) + NUMBER
).set_parse_action(lambda t: float(t[0]))
# A bracket says whether the endpoint belongs to the support, as it would in a written interval.
OPEN_LOWER = pp.Literal("(").set_parse_action(lambda: False) | pp.Literal("[").set_parse_action(lambda: True)
OPEN_UPPER = pp.Literal(")").set_parse_action(lambda: False) | pp.Literal("]").set_parse_action(lambda: True)
BOUNDS_VALUE = (OPEN_LOWER - BOUND_VALUE - COMMA - BOUND_VALUE - OPEN_UPPER).set_parse_action(
    lambda t: [((t[1], t[2]), (t[0], t[3]))]
)

BOOL_VALUE = KW_TRUE.copy().set_parse_action(lambda _: True) | KW_FALSE.copy().set_parse_action(lambda _: False)

_COMPLEMENTARY_ASSUMPTION = {
    "positive": "nonpositive",
    "nonpositive": "positive",
    "negative": "nonnegative",
    "nonnegative": "negative",
}


METADATA_FIELD = pp.Group(pp.one_of(SYMBOL_METADATA_FIELDS, caseless=True)("field") - EQUALS - STRING("value") - SEMI)
BOUNDS_FIELD = pp.Group(pp.CaselessKeyword("bounds")("field") - EQUALS - BOUNDS_VALUE("value") - SEMI)


def _reject_false_sign_assumption(s: str, loc: int, tokens: pp.ParseResults) -> None:
    name = tokens[0].field.lower()
    if tokens[0].value is False and name in SIGN_ASSUMPTION_INTERVALS:
        raise GCNParseFailure(
            s,
            loc,
            f"Sign assumption '{name}' set to False",
            code=ErrorCode.E018,
            found=name,
            suggestions=[complement] if (complement := _COMPLEMENTARY_ASSUMPTION.get(name)) else [],
        )


ASSUMPTION_FIELD = pp.Group(
    pp.one_of(GCN_ASSUMPTIONS, caseless=True)("field") - EQUALS - BOOL_VALUE("value") - SEMI
).add_parse_action(_reject_false_sign_assumption)


def _unknown_symbol_field_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    name = toks[0]
    raise GCNParseFailure(
        s,
        loc,
        f"Unknown symbol field '{name}'",
        code=ErrorCode.E020,
        found=name,
        suggestions=suggest_symbol_field(name),
    )


UNKNOWN_SYMBOL_FIELD = (
    (IDENTIFIER("unknown_field") + pp.FollowedBy(EQUALS_LITERAL))
    .add_condition(lambda _s, _loc, toks: toks[0].lower() not in KNOWN_SYMBOL_FIELDS)
    .set_parse_action(_unknown_symbol_field_fail)
)

SYMBOL_FIELD = METADATA_FIELD | BOUNDS_FIELD | ASSUMPTION_FIELD | UNKNOWN_SYMBOL_FIELD


def _intersect(intervals: list[tuple[float | None, float | None]]) -> tuple[float | None, float | None]:
    lowers = [lower for lower, _ in intervals if lower is not None]
    uppers = [upper for _, upper in intervals if upper is not None]
    return max(lowers, default=None), min(uppers, default=None)


def _is_inhabited(bounds: tuple[float | None, float | None]) -> bool:
    lower, upper = bounds
    return (-float("inf") if lower is None else lower) < (float("inf") if upper is None else upper)


def _implied_supports(
    assumptions: dict[str, bool],
) -> list[tuple[tuple[float | None, float | None], tuple[bool, bool]]]:
    return [
        SIGN_ASSUMPTION_INTERVALS[key]
        for key, holds in assumptions.items()
        if holds and key in SIGN_ASSUMPTION_INTERVALS
    ]


def _collect_fields(
    tokens: pp.ParseResults,
) -> tuple[dict[str, str], tuple[float | None, float | None] | None, dict[str, bool]]:
    """
    Sort an entry's fields into metadata, a declared bound, and sympy assumptions.

    Returns
    -------
    metadata : dict mapping str to str
        The ``name``, ``latex`` and ``source`` fields that were given.
    declared_bounds : tuple of (bounds, closed), or None
        The declared bound and which of its ends include their endpoint, or None when the entry has no
        ``bounds`` field.
    assumptions : dict mapping str to bool
        The assumption keywords that were given, before anything is derived from the bound.
    """
    metadata: dict[str, str] = {}
    declared_bounds: tuple[tuple[float | None, float | None], tuple[bool, bool]] | None = None
    assumptions: dict[str, bool] = {}

    for entry in tokens.fields:
        field_name = entry.field.lower()
        value = entry.value

        if field_name in SYMBOL_METADATA_FIELDS:
            metadata[field_name] = value
        elif field_name == "bounds":
            # pyparsing wraps a named multi-token expression, so the pair the parse action built sits one level in.
            declared_bounds = value[0] if isinstance(value, pp.ParseResults) else value
        else:
            assumptions[field_name] = value

    return metadata, declared_bounds, assumptions


def _build_symbol_entry(s: str, loc: int, tokens: pp.ParseResults) -> tuple[int, SymbolDeclaration]:
    item = tokens[0].symbol
    if isinstance(item, pp.ParseResults):
        item = item[0]
    symbol_name = variable_key(item.name) if isinstance(item, Variable) else str(item)

    metadata, declared, assumptions = _collect_fields(tokens[0])
    implied = _implied_supports(assumptions)

    if declared is not None:
        bounds, closed = declared
        if any(end is None and is_closed for end, is_closed in zip(bounds, closed, strict=True)):
            raise GCNParseFailure(
                s,
                loc,
                f"Bound on '{symbol_name}' closes a side that is unbounded",
                code=ErrorCode.E021,
                found=symbol_name,
            )
        for interval, _ in implied:
            if not _is_inhabited(_intersect([bounds, interval])):
                raise GCNParseFailure(
                    s,
                    loc,
                    f"Bound {bounds} on '{symbol_name}' contradicts a sign assumption declared beside it",
                    code=ErrorCode.E017,
                    found=symbol_name,
                )
    elif implied:
        bounds = _intersect([interval for interval, _ in implied])
        closed = tuple(any(ends[side] for _, ends in implied) for side in (0, 1))
    else:
        bounds, closed = (None, None), (False, False)

    if not _is_inhabited(bounds):
        raise GCNParseFailure(
            s,
            loc,
            f"Declaration of '{symbol_name}' has an empty support {bounds}",
            code=ErrorCode.E017,
            found=symbol_name,
        )

    # A unit_interval declared alongside a wider bound keeps the bound, so positivity has to be set here as well.
    if assumptions.get("unit_interval"):
        assumptions.setdefault("positive", True)

    return loc, SymbolDeclaration(
        symbol=symbol_name,
        name=metadata.get("name"),
        latex=metadata.get("latex"),
        source=metadata.get("source"),
        bounds=bounds,
        closed=closed,
        assumptions={**DEFAULT_ASSUMPTIONS, **assumptions_implied_by_bounds(bounds, closed), **assumptions},
    )


SYMBOL_ENTRY = pp.Group(
    SYMBOL_ITEM("symbol") - LBRACE - pp.ZeroOrMore(SYMBOL_FIELD)("fields") - RBRACE - SEMI
).set_parse_action(_build_symbol_entry)

SYMBOLS_BLOCK = KW_SYMBOLS.suppress() - LBRACE - pp.ZeroOrMore(SYMBOL_ENTRY)("entries") - RBRACE - SEMI


def _build_symbols(s: str, _loc: int, tokens: pp.ParseResults) -> dict[str, SymbolDeclaration]:
    declarations: dict[str, SymbolDeclaration] = {}

    for entry_loc, declaration in tokens.entries:
        if declaration.symbol in declarations:
            raise GCNParseFailure(
                s,
                entry_loc,
                f"Symbol '{declaration.symbol}' is declared more than once",
                code=ErrorCode.E019,
                found=declaration.symbol,
            )
        declarations[declaration.symbol] = declaration

    return declarations


SYMBOLS_BLOCK.set_parse_action(_build_symbols)
SYMBOLS_BLOCK.ignore(COMMENT)

SPECIAL_BLOCK = OPTIONS_BLOCK | TRYREDUCE_BLOCK | ASSUMPTIONS_BLOCK | SYMBOLS_BLOCK


__all__ = [
    "ASSUMPTIONS_BLOCK",
    "OPTIONS_BLOCK",
    "SPECIAL_BLOCK",
    "SYMBOLS_BLOCK",
    "TRYREDUCE_BLOCK",
    "extract_special_block_content",
    "parse_assumptions",
    "parse_options",
    "parse_symbols",
    "parse_tryreduce",
    "remove_special_block",
]
