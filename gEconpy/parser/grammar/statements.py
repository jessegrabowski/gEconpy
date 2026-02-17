import pyparsing as pp

from gEconpy.parser.ast import (
    STEADY_STATE,
    GCNDistribution,
    GCNEquation,
    T,
    Tag,
    TimeIndex,
    Variable,
)
from gEconpy.parser.constants import PRELIZ_DIST_WRAPPERS, PRELIZ_DISTS
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure
from gEconpy.parser.grammar.expressions import EXPR
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


def _parse_time_index_content(content: str) -> TimeIndex:
    if content == "":
        return T
    if content == "ss":
        return STEADY_STATE
    return TimeIndex(int(content))


VARIABLE_REF = (
    IDENTIFIER("name") + LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="")("time") + RBRACKET
).set_parse_action(lambda t: Variable(name=t.name, time_index=_parse_time_index_content(t.time)))

VARIABLE_LIST = pp.DelimitedList(VARIABLE_REF)

VALID_TAGS = frozenset(["exclude"])


def _parse_tag(s: str, loc: int, toks):
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


def _missing_lhs_fail(s: str, loc: int, _toks) -> None:
    raise GCNParseFailure(
        s,
        loc,
        "Missing left-hand side of equation",
        code=ErrorCode.E005,
        found="=",
    )


MISSING_LHS = (pp.ZeroOrMore(TAG) + pp.FollowedBy(pp.Literal("="))).set_parse_action(_missing_lhs_fail)


def _missing_rhs_fail(s: str, loc: int, _toks) -> None:
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


def _missing_equals_fail(s: str, loc: int, _toks) -> None:
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


def _check_after_expr(s: str, loc: int, toks):
    while loc < len(s) and s[loc] in " \t\n":
        loc += 1
    if loc < len(s):
        next_char = s[loc]
        if next_char in ")]}":
            raise pp.ParseException(s, loc, f"Unmatched '{next_char}'")
    return toks


_VALID_EQUATION = (
    pp.ZeroOrMore(TAG)("tags")
    + EXPR("lhs")
    + pp.Suppress(pp.Literal("="))
    + EXPR("rhs").add_parse_action(_check_after_expr)
    + pp.Optional(LAGRANGE_MULT)("lagrange")
    + pp.Optional(CALIBRATING_PARAM)("calibrating")
    + SEMI
)

EQUATION = MISSING_LHS | MISSING_RHS | MISSING_EQUALS | _VALID_EQUATION


def _build_equation(tokens) -> GCNEquation:
    tags = frozenset(tokens.tags) if tokens.tags else frozenset()

    lagrange_name = None
    if tokens.lagrange and len(tokens.lagrange) > 0:
        lagrange_name = tokens.lagrange[0]

    calibrating_param = None
    if tokens.calibrating and len(tokens.calibrating) > 0:
        calibrating_param = tokens.calibrating[0]

    return GCNEquation(
        lhs=tokens.lhs,
        rhs=tokens.rhs,
        lagrange_multiplier=lagrange_name,
        calibrating_parameter=calibrating_param,
        tags=tags,
    )


EQUATION.set_parse_action(_build_equation)


def _evaluate_number_expr(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, pp.ParseResults):
        value = value.as_list()
    if isinstance(value, list):
        while len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if len(value) == 1:
            return _evaluate_number_expr(value[0])
        result = float(value[0])
        i = 1
        while i < len(value):
            op = str(value[i])
            operand = float(value[i + 1])
            if op == "+":
                result += operand
            elif op == "-":
                result -= operand
            elif op == "*":
                result *= operand
            elif op == "/":
                result /= operand
            i += 2
        return result
    return value


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


def _unknown_wrapper_fail(s: str, loc: int, toks) -> None:
    name = toks[0]

    suggestions = suggest_wrapper(name)
    raise GCNParseFailure(
        s,
        loc,
        f"Unknown distribution wrapper '{name}'",
        code=ErrorCode.E103,
        found=name,
        suggestions=suggestions,
    )


UNKNOWN_WRAPPER = (
    pp.NotAny(DIST_NAME | WRAPPER_NAME) + IDENTIFIER("unknown_name") + pp.FollowedBy(LPAREN + DIST_NAME)
).set_parse_action(_unknown_wrapper_fail)


def _unknown_distribution_fail(s: str, loc: int, toks) -> None:
    name = toks[0]

    suggestions = suggest_distribution(name)
    raise GCNParseFailure(
        s,
        loc,
        f"Unknown distribution '{name}'",
        code=ErrorCode.E102,
        found=name,
        suggestions=suggestions,
    )


UNKNOWN_DIST = (
    pp.NotAny(DIST_NAME | WRAPPER_NAME) + IDENTIFIER("unknown_name") + pp.FollowedBy(LPAREN + ~DIST_NAME)
).set_parse_action(_unknown_distribution_fail)

DIST_EXPR = WRAPPED_DIST | DIST_CALL | UNKNOWN_WRAPPER | UNKNOWN_DIST

DISTRIBUTION = IDENTIFIER("param_name") + TILDE - DIST_EXPR + pp.Optional(EQUALS + _DIST_NUMBER_EXPR)("initial") - SEMI


def _build_distribution(tokens) -> GCNDistribution:
    param_name = tokens.param_name
    dist_name = tokens.dist_name

    dist_kwargs = {}
    if tokens.dist_args:
        for arg in tokens.dist_args:
            dist_kwargs[arg.arg_name] = _evaluate_number_expr(arg.arg_value)

    wrapper_name = tokens.wrapper_name if hasattr(tokens, "wrapper_name") and tokens.wrapper_name else None
    wrapper_kwargs = {}
    if wrapper_name and hasattr(tokens, "wrapper_args") and tokens.wrapper_args:
        for arg in tokens.wrapper_args:
            wrapper_kwargs[arg.arg_name] = _evaluate_number_expr(arg.arg_value)

    initial_value = None
    if tokens.initial:
        initial_value = _evaluate_number_expr(tokens.initial)

    return GCNDistribution(
        parameter_name=param_name,
        dist_name=dist_name,
        dist_kwargs=dist_kwargs,
        wrapper_name=wrapper_name,
        wrapper_kwargs=wrapper_kwargs,
        initial_value=initial_value,
    )


DISTRIBUTION.set_parse_action(_build_distribution)


def parse_equation(text: str) -> GCNEquation:
    """Parse a single equation from text."""
    return EQUATION.parse_string(text.strip(), parse_all=True)[0]


def parse_distribution(text: str) -> GCNDistribution:
    """Parse a single distribution from text."""
    return DISTRIBUTION.parse_string(text.strip(), parse_all=True)[0]


__all__ = [
    "CALIBRATING_PARAM",
    "DISTRIBUTION",
    "DIST_CALL",
    "DIST_EXPR",
    "EQUATION",
    "LAGRANGE_MULT",
    "TAG",
    "VARIABLE_LIST",
    "VARIABLE_REF",
    "parse_distribution",
    "parse_equation",
]
