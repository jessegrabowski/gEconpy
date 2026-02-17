import re

from collections import defaultdict

import pyparsing as pp

from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS, GCN_ASSUMPTIONS
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure
from gEconpy.parser.grammar.statements import VARIABLE_LIST, VARIABLE_REF
from gEconpy.parser.grammar.tokens import (
    COMMENT,
    EQUALS,
    IDENTIFIER,
    KW_ASSUMPTIONS,
    KW_FALSE,
    KW_OPTIONS,
    KW_TRUE,
    KW_TRYREDUCE,
    LBRACE,
    RBRACE,
    SEMI,
)
from gEconpy.parser.suggestions import suggest_assumption

OPTION_KEY = pp.Combine(IDENTIFIER + pp.ZeroOrMore(pp.White(" ") + IDENTIFIER))
OPTION_VALUE = (
    KW_TRUE.copy().set_parse_action(lambda _: True) | KW_FALSE.copy().set_parse_action(lambda _: False) | IDENTIFIER
)
OPTION_ENTRY = pp.Group(OPTION_KEY("key") - EQUALS - OPTION_VALUE("value") - SEMI)
OPTIONS_BLOCK = KW_OPTIONS.suppress() - LBRACE - pp.ZeroOrMore(OPTION_ENTRY)("entries") - RBRACE - SEMI


def _build_options(tokens) -> dict[str, bool | str]:
    result = {}
    for entry in tokens.entries:
        result[entry.key] = entry.value
    return result


OPTIONS_BLOCK.set_parse_action(_build_options)
OPTIONS_BLOCK.ignore(COMMENT)

TRYREDUCE_BLOCK = KW_TRYREDUCE.suppress() - LBRACE - pp.Optional(VARIABLE_LIST("variables") + SEMI) - RBRACE - SEMI


def _build_tryreduce(tokens) -> list[str]:
    if not tokens.variables:
        return []
    return [v.name for v in tokens.variables]


TRYREDUCE_BLOCK.set_parse_action(lambda t: [_build_tryreduce(t)])
TRYREDUCE_BLOCK.ignore(COMMENT)

ASSUMPTION_NAME = pp.one_of(GCN_ASSUMPTIONS, caseless=True)("assumption")
_KNOWN_ASSUMPTIONS = frozenset(a.lower() for a in GCN_ASSUMPTIONS)


def _unknown_assumption_fail(s: str, loc: int, toks) -> None:
    name = toks[0]
    suggestions = suggest_assumption(name)
    raise GCNParseFailure(
        s,
        loc,
        f"Unknown assumption '{name}'",
        code=ErrorCode.E015,
        found=name,
        suggestions=suggestions,
    )


UNKNOWN_ASSUMPTION = (
    (IDENTIFIER("unknown_name") + pp.FollowedBy(LBRACE))
    .add_condition(lambda _s, _loc, toks: toks[0].lower() not in _KNOWN_ASSUMPTIONS)
    .set_parse_action(_unknown_assumption_fail)
)

ASSUMPTION_ITEM = VARIABLE_REF | IDENTIFIER.copy().set_parse_action(lambda t: t[0])
ASSUMPTION_LIST = pp.DelimitedList(ASSUMPTION_ITEM)

VALID_ASSUMPTION_SUBBLOCK = pp.Group(ASSUMPTION_NAME - LBRACE - ASSUMPTION_LIST("variables") - SEMI - RBRACE - SEMI)
ASSUMPTION_SUBBLOCK = VALID_ASSUMPTION_SUBBLOCK | UNKNOWN_ASSUMPTION

ASSUMPTIONS_BLOCK = KW_ASSUMPTIONS.suppress() - LBRACE - pp.ZeroOrMore(ASSUMPTION_SUBBLOCK)("subblocks") - RBRACE - SEMI


def _build_assumptions(tokens) -> dict[str, dict[str, bool]]:
    assumption_kwargs = defaultdict(DEFAULT_ASSUMPTIONS.copy)

    for subblock in tokens.subblocks:
        assumption_name = subblock.assumption.lower()
        for item in subblock.variables:
            var_name = item.name if hasattr(item, "name") else str(item)
            assumption_kwargs[var_name][assumption_name] = True

    return dict(assumption_kwargs)


ASSUMPTIONS_BLOCK.set_parse_action(_build_assumptions)
ASSUMPTIONS_BLOCK.ignore(COMMENT)

SPECIAL_BLOCK = OPTIONS_BLOCK | TRYREDUCE_BLOCK | ASSUMPTIONS_BLOCK


def parse_options(text: str) -> dict[str, bool | str]:
    """Parse an options block from GCN text."""
    try:
        for result, _start, _end in OPTIONS_BLOCK.scan_string(text):
            return result[0]
    except pp.ParseException:
        pass
    return {}


def parse_tryreduce(text: str) -> list[str]:
    """Parse a tryreduce block from GCN text."""
    try:
        for result, _start, _end in TRYREDUCE_BLOCK.scan_string(text):
            return result[0]
    except pp.ParseException:
        pass
    return []


def parse_assumptions(text: str) -> dict[str, dict[str, bool]]:
    """Parse an assumptions block from GCN text."""
    try:
        for result, _start, _end in ASSUMPTIONS_BLOCK.scan_string(text):
            return result[0]
    except pp.ParseException:
        pass
    return defaultdict(DEFAULT_ASSUMPTIONS.copy)


def extract_special_block_content(text: str, block_name: str) -> str | None:
    """Extract the raw content of a special block from text."""
    pattern = rf"{block_name}\s*\{{.*?\}};"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def remove_special_block(text: str, block_name: str) -> str:
    """Remove a special block from text."""
    pattern = rf"{block_name}\s*\{{.*?\}};"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)


__all__ = [
    "ASSUMPTIONS_BLOCK",
    "OPTIONS_BLOCK",
    "SPECIAL_BLOCK",
    "TRYREDUCE_BLOCK",
    "extract_special_block_content",
    "parse_assumptions",
    "parse_options",
    "parse_tryreduce",
    "remove_special_block",
]
