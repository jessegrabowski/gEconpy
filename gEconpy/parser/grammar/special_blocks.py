import contextlib
import re

from collections import defaultdict
from typing import Any

import pyparsing as pp

from gEconpy.classes.time_aware_symbol import DEFAULT_ASSUMPTIONS
from gEconpy.parser.ast import Variable
from gEconpy.parser.constants import GCN_ASSUMPTIONS, KNOWN_ASSUMPTIONS
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
            name = item.name if isinstance(item, Variable) else str(item)
            assumption_kwargs[name][assumption_name] = True
            # ``unit_interval`` is not a sympy predicate. It sits inertly in ``assumptions0`` so the steady-state
            # solver can route the variable to a logit transform. Because it implies positivity, sympy's
            # ``positive`` is set as well, and that predicate does carry algebraic consequences.
            if assumption_name == "unit_interval":
                assumption_kwargs[name]["positive"] = True

    return dict(assumption_kwargs)


ASSUMPTIONS_BLOCK.set_parse_action(_build_assumptions)
ASSUMPTIONS_BLOCK.ignore(COMMENT)

SPECIAL_BLOCK = OPTIONS_BLOCK | TRYREDUCE_BLOCK | ASSUMPTIONS_BLOCK


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
