import operator

from collections.abc import Callable

import pyparsing as pp

from gEconpy.parser.ast import (
    GCNDistribution,
    GCNEquation,
    Node,
    Tag,
    Variable,
)
from gEconpy.parser.constants import PRELIZ_DIST_WRAPPERS, PRELIZ_DISTS
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure, ParseLocation
from gEconpy.parser.grammar.expressions import EXPR, _location_at, _parse_time_index
from gEconpy.parser.grammar.tokens import (
    ARROW,
    COLON,
    COMMA,
    EQUALS,
    IDENTIFIER,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    SEMI,
    TILDE,
    TIME_INDEX_CONTENT,
)
from gEconpy.parser.suggestions import suggest_distribution, suggest_wrapper


def parse_equation(text: str) -> GCNEquation:
    """
    Parse a single equation, including any Lagrange multiplier, calibrating parameter, and tags.

    Parameters
    ----------
    text : str
        Text of the equation, with or without a trailing semicolon.

    Returns
    -------
    equation : GCNEquation
        The parsed equation.
    """
    return EQUATION.parse_string(text.strip(), parse_all=True)[0]


def parse_distribution(text: str) -> GCNDistribution:
    """
    Parse a single distribution declaration, of the form ``name ~ Dist(arg=value)``.

    Parameters
    ----------
    text : str
        Text of the declaration, with or without a trailing semicolon.

    Returns
    -------
    distribution : GCNDistribution
        The parsed distribution.
    """
    return DISTRIBUTION.parse_string(text.strip(), parse_all=True)[0]


def _parse_variable_ref(s: str, loc: int, toks: pp.ParseResults) -> Variable:
    variable_text = f"{toks.name}[{toks.time}]"
    return Variable(
        name=toks.name,
        time_index=_parse_time_index(toks.time),
        location=_location_at(s, loc, length=len(variable_text)),
    )


VARIABLE_REF = (
    IDENTIFIER("name") + LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="")("time") + RBRACKET
).set_parse_action(_parse_variable_ref)

VARIABLE_LIST = pp.DelimitedList(VARIABLE_REF)

VALID_TAGS = frozenset(["exclude", "minimize", "maximize"])


def _parse_tag(s: str, loc: int, toks: pp.ParseResults) -> Tag:
    tag_text = toks[0]
    tag_name = tag_text[1:]
    if tag_name.lower() not in VALID_TAGS:
        raise GCNParseFailure(
            s,
            loc,
            f"Unknown tag '{tag_text}'",
            code=ErrorCode.E014,
            found=tag_text,
        )
    return Tag.from_string(tag_name)


TAG = pp.Combine(pp.Literal("@") + IDENTIFIER).set_parse_action(_parse_tag)

LAGRANGE_MULT = COLON + IDENTIFIER("name") + LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="") + RBRACKET

CALIBRATING_PARAM = ARROW + IDENTIFIER("param")


def _missing_lhs_fail(s: str, loc: int, _toks: pp.ParseResults) -> None:
    raise GCNParseFailure(
        s,
        loc,
        "Missing left-hand side of equation",
        code=ErrorCode.E005,
        found="=",
    )


MISSING_LHS = (pp.ZeroOrMore(TAG) + pp.FollowedBy(pp.Literal("="))).set_parse_action(_missing_lhs_fail)


def _missing_rhs_fail(s: str, loc: int, _toks: pp.ParseResults) -> None:
    raise GCNParseFailure(
        s,
        loc,
        "Missing right-hand side of equation",
        code=ErrorCode.E005,
        found=";",
    )


MISSING_RHS = (
    pp.ZeroOrMore(TAG) + EXPR("lhs") + pp.Suppress(pp.Literal("=")) + pp.FollowedBy(pp.Regex(r"\s*[;:]"))
).set_parse_action(_missing_rhs_fail)


def _missing_equals_fail(s: str, loc: int, _toks: pp.ParseResults) -> None:
    raise GCNParseFailure(
        s,
        loc,
        "Missing '=' in equation",
        code=ErrorCode.E012,
        found="",
    )


MISSING_EQUALS = (
    pp.ZeroOrMore(TAG) + EXPR("expr") + ~pp.FollowedBy(pp.Regex(r"\s*=")) + pp.FollowedBy(SEMI)
).set_parse_action(_missing_equals_fail)


def _unmatched_close_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    char = toks[0]
    raise GCNParseFailure(
        s,
        loc,
        f"Unmatched '{char}': extra closing bracket with no matching opener",
        code=ErrorCode.E007,
        found=char,
    )


# Without this check a stray closing bracket after the RHS surfaces as a generic "Expected ';'" error.
_UNMATCHED_CLOSE = pp.Regex(r"[)\]}]").copy().set_parse_action(_unmatched_close_fail)

_VALID_EQUATION = (
    pp.ZeroOrMore(TAG)("tags")
    + EXPR("lhs")
    + pp.Suppress(pp.Literal("="))
    + EXPR("rhs")
    + pp.Optional(_UNMATCHED_CLOSE)
    + pp.Optional(LAGRANGE_MULT)("lagrange")
    + pp.Optional(CALIBRATING_PARAM)("calibrating")
    + SEMI
)

EQUATION = MISSING_LHS | MISSING_RHS | MISSING_EQUALS | _VALID_EQUATION


def _find_location_in_node(node: Node) -> ParseLocation | None:
    if node.location is not None:
        return node.location

    for child_attribute in ("left", "operand", "expr"):
        child = getattr(node, child_attribute, None)
        if child is None:
            continue
        location = _find_location_in_node(child)
        if location is not None:
            return location

    return None


def _equation_end(s: str, loc: int, terminator: str, fallback: tuple[int, int]) -> tuple[int, int]:
    end_loc = s.find(terminator, loc)
    if end_loc == -1:
        return fallback

    end_col = pp.col(end_loc, s)
    if terminator == ";":
        end_col += 1
    return pp.lineno(end_loc, s), end_col


def _build_equation(s: str, loc: int, tokens: pp.ParseResults) -> GCNEquation:
    tags = frozenset(tokens.tags) if tokens.tags else frozenset()
    lagrange_name = tokens.lagrange[0] if tokens.lagrange else None
    calibrating_param = tokens.calibrating[0] if tokens.calibrating else None

    lhs_location = _find_location_in_node(tokens.lhs)
    if lhs_location is not None:
        line, col, source_line = lhs_location.line, lhs_location.column, lhs_location.source_line
    else:
        line = pp.lineno(loc, s)
        col = pp.col(loc, s)
        lines = s.splitlines()
        source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    # A calibrating equation is underlined up to its arrow, a regular one through its semicolon.
    terminator = "->" if calibrating_param else ";"
    fallback = (line, col + len(source_line.strip()))
    end_line, end_col = _equation_end(s, loc, terminator, fallback)

    location = ParseLocation(
        line=line,
        column=col,
        end_line=end_line,
        end_column=end_col,
        source_line=source_line,
    )

    return GCNEquation(
        lhs=tokens.lhs,
        rhs=tokens.rhs,
        lagrange_multiplier=lagrange_name,
        calibrating_parameter=calibrating_param,
        tags=tags,
        location=location,
    )


EQUATION.set_parse_action(_build_equation)

_ARITHMETIC: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def _evaluate_number_expr(value):
    if isinstance(value, pp.ParseResults):
        value = value.as_list()
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, list):
        return value

    while len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if len(value) == 1:
        return _evaluate_number_expr(value[0])

    result = _evaluate_number_expr(value[0])
    for op, operand in zip(value[1::2], value[2::2], strict=True):
        result = _ARITHMETIC[op](result, _evaluate_number_expr(operand))
    return result


def _collect_kwargs(args: pp.ParseResults | str) -> dict[str, float | str | None]:
    if not args:
        return {}
    return {arg.arg_name: _evaluate_number_expr(arg.arg_value) for arg in args}


_DIST_NUMBER = pp.pyparsing_common.number
_DIST_NUMBER_EXPR = pp.infix_notation(
    _DIST_NUMBER,
    [
        (pp.Literal("/"), 2, pp.OpAssoc.LEFT),
        (pp.Literal("*"), 2, pp.OpAssoc.LEFT),
        (pp.Literal("+"), 2, pp.OpAssoc.LEFT),
        (pp.Literal("-"), 2, pp.OpAssoc.LEFT),
    ],
)

DIST_NAME = pp.one_of(PRELIZ_DISTS, caseless=False)("dist_name")
WRAPPER_NAME = pp.one_of(PRELIZ_DIST_WRAPPERS, caseless=False)("wrapper_name")

NONE_KEYWORD = pp.Keyword("None").set_parse_action(lambda _: [None])
DIST_ARG_VALUE = _DIST_NUMBER_EXPR | NONE_KEYWORD | IDENTIFIER
DIST_ARG = pp.Group(IDENTIFIER("arg_name") + EQUALS + DIST_ARG_VALUE("arg_value"))

DIST_CALL = DIST_NAME + LPAREN + pp.Optional(pp.DelimitedList(DIST_ARG))("dist_args") + RPAREN

WRAPPED_DIST = (
    WRAPPER_NAME + LPAREN - DIST_CALL + pp.Optional(COMMA + pp.DelimitedList(DIST_ARG))("wrapper_args") - RPAREN
)


def _unknown_wrapper_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    name = toks[0]
    raise GCNParseFailure(
        s,
        loc,
        f"Unknown distribution wrapper '{name}'",
        code=ErrorCode.E103,
        found=name,
        suggestions=suggest_wrapper(name),
    )


UNKNOWN_WRAPPER = (
    pp.NotAny(DIST_NAME | WRAPPER_NAME) + IDENTIFIER("unknown_name") + pp.FollowedBy(LPAREN + DIST_NAME)
).set_parse_action(_unknown_wrapper_fail)


def _unknown_distribution_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    name = toks[0]
    raise GCNParseFailure(
        s,
        loc,
        f"Unknown distribution '{name}'",
        code=ErrorCode.E102,
        found=name,
        suggestions=suggest_distribution(name),
    )


UNKNOWN_DIST = (
    pp.NotAny(DIST_NAME | WRAPPER_NAME) + IDENTIFIER("unknown_name") + pp.FollowedBy(LPAREN + ~DIST_NAME)
).set_parse_action(_unknown_distribution_fail)

DIST_EXPR = WRAPPED_DIST | DIST_CALL | UNKNOWN_WRAPPER | UNKNOWN_DIST

DISTRIBUTION = IDENTIFIER("param_name") + TILDE - DIST_EXPR + pp.Optional(EQUALS + _DIST_NUMBER_EXPR)("initial") - SEMI


def _missing_tilde_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    param_name = toks.param_name
    dist_or_wrapper = toks.dist_or_wrapper
    raise GCNParseFailure(
        s,
        loc,
        f"Used '=' instead of '~' for distribution prior on '{param_name}'. "
        f"Use '{param_name} ~ {dist_or_wrapper}(...)' instead of '{param_name} = {dist_or_wrapper}(...)'",
        code=ErrorCode.E009,
        found="=",
    )


MISSING_TILDE = (
    IDENTIFIER("param_name")
    + pp.FollowedBy(pp.Literal("=") + (WRAPPER_NAME | DIST_NAME))
    + pp.Literal("=").suppress()
    + (WRAPPER_NAME | DIST_NAME)("dist_or_wrapper")
).set_parse_action(_missing_tilde_fail)


def _build_distribution(tokens: pp.ParseResults) -> GCNDistribution:
    wrapper_name = tokens.wrapper_name or None
    initial_value = _evaluate_number_expr(tokens.initial) if tokens.initial else None

    return GCNDistribution(
        parameter_name=tokens.param_name,
        dist_name=tokens.dist_name,
        dist_kwargs=_collect_kwargs(tokens.dist_args),
        wrapper_name=wrapper_name,
        wrapper_kwargs=_collect_kwargs(tokens.wrapper_args) if wrapper_name else {},
        initial_value=initial_value,
    )


DISTRIBUTION.set_parse_action(_build_distribution)


__all__ = [
    "CALIBRATING_PARAM",
    "DISTRIBUTION",
    "DIST_CALL",
    "DIST_EXPR",
    "EQUATION",
    "LAGRANGE_MULT",
    "MISSING_TILDE",
    "TAG",
    "VARIABLE_LIST",
    "VARIABLE_REF",
    "parse_distribution",
    "parse_equation",
]
