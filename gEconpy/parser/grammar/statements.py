import math
import operator

from collections.abc import Callable
from typing import Any

import pyparsing as pp

from gEconpy.parser.ast import (
    STEADY_STATE,
    Expectation,
    GCNDistribution,
    GCNEquation,
    Node,
    Number,
    Operator,
    Parameter,
    SteadyStateSolverCall,
    Tag,
    UnaryOp,
    Variable,
    collect_nodes_of_type,
)
from gEconpy.parser.constants import (
    CONTROL_TAGS,
    EQUATION_TAGS,
    INTERVAL_SOLVER_KEYWORDS,
    PRELIZ_DIST_WRAPPERS,
    PRELIZ_DISTS,
    SOLVER_METHODS,
    VALUE_TAGS,
)
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNParseFailure, ParseLocation
from gEconpy.parser.grammar.expressions import (
    EXPR,
    SOLVER_NAME,
    location_at,
    misplaced_solver_call_fail,
    parse_time_index,
)
from gEconpy.parser.grammar.tokens import (
    ARROW,
    COLON,
    COMMA,
    EQUALS,
    IDENTIFIER,
    LBRACKET,
    LPAREN,
    NUMBER,
    RBRACKET,
    RPAREN,
    SEMI,
    STRING,
    TILDE,
    TIME_INDEX_CONTENT,
)
from gEconpy.parser.suggestions import find_similar_names, suggest_distribution, suggest_wrapper


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
        time_index=parse_time_index(toks.time),
        location=location_at(s, loc, length=len(variable_text)),
    )


VARIABLE_REF = (
    IDENTIFIER("name") + LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="")("time") + RBRACKET
).set_parse_action(_parse_variable_ref)

VARIABLE_LIST = pp.DelimitedList(VARIABLE_REF)

VALID_FLAG_TAGS = frozenset(EQUATION_TAGS)
VALID_EQUATION_VALUE_TAGS = frozenset(VALUE_TAGS)
VALID_CONTROL_TAGS = frozenset(CONTROL_TAGS)

TAG_NAME = pp.Combine(pp.Literal("@") + IDENTIFIER)

# Only a quoted string is a legal value. The unquoted branches and the surrounding Optional exist so that a
# malformed value reaches the parse action and gets a catalogued error rather than a raw pyparsing "Expected".
_TAG_VALUE = STRING("quoted") | NUMBER | IDENTIFIER


def _bad_tag(s: str, loc: int, tag_text: str, message: str, code: ErrorCode) -> GCNParseFailure:
    return GCNParseFailure(s, loc, message, code=code, found=tag_text)


def _parse_tag(s: str, loc: int, toks: pp.ParseResults) -> Tag:
    tag_text = toks[0]
    tag_name = tag_text[1:].lower()

    if tag_name in VALID_EQUATION_VALUE_TAGS:
        message = f"Tag '{tag_text}' takes a value, as in '{tag_text} = \"...\"'"
        raise _bad_tag(s, loc, tag_text, message, ErrorCode.E014)
    if tag_name not in VALID_FLAG_TAGS:
        raise _bad_tag(s, loc, tag_text, f"Unknown tag '{tag_text}'", ErrorCode.E014)

    return Tag.from_string(tag_name)


def _value_tag(
    valid_names: frozenset[str],
    code: ErrorCode,
    flag_names: frozenset[str] = frozenset(),
) -> pp.ParserElement:
    """
    Build a grammar element for ``@tag = "value"``.

    Parameters
    ----------
    valid_names : frozenset of str
        Tag names this element accepts, without the leading ``@``.
    code : ErrorCode
        The catalog code for a malformed tag in this position.
    flag_names : frozenset of str, optional
        Tag names that stand alone in this position, reported as taking no value. Defaults to none.

    Returns
    -------
    element : pyparsing.ParserElement
        An element parsing to a ``(name, value)`` pair.
    """

    def parse(s: str, loc: int, toks: pp.ParseResults) -> tuple[str, str]:
        tag_text = toks[0]
        tag_name = tag_text[1:].lower()

        if tag_name in flag_names:
            raise _bad_tag(s, loc, tag_text, f"Tag '{tag_text}' stands alone and takes no value", code)
        if tag_name not in valid_names:
            raise _bad_tag(s, loc, tag_text, f"Unknown tag '{tag_text}'", code)
        if len(toks) == 1:
            raise _bad_tag(s, loc, tag_text, f"Tag '{tag_text}' is missing its value", code)
        if "quoted" not in toks:
            raise _bad_tag(
                s, loc, tag_text, f"Tag '{tag_text}' takes a quoted string, as in '{tag_text} = \"...\"'", code
            )

        return tag_name, toks[1]

    return (TAG_NAME + pp.Suppress(pp.Literal("=")) + pp.Optional(_TAG_VALUE)).set_parse_action(parse)


def _bare_control_tag_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    tag_text = toks[0]
    message = f"Tag '{tag_text}' takes a value, as in '{tag_text} = \"...\"'"
    raise _bad_tag(s, loc, tag_text, message, ErrorCode.E024)


VALUE_TAG = _value_tag(VALID_EQUATION_VALUE_TAGS, ErrorCode.E014, flag_names=VALID_FLAG_TAGS)

# No tag stands alone in a controls block, so a bare one is caught here rather than falling through to a raw
# "Expected '}'" from the variable list that follows it.
_BARE_CONTROL_TAG = (TAG_NAME + ~pp.FollowedBy(pp.Literal("="))).set_parse_action(_bare_control_tag_fail)

CONTROL_TAG = _value_tag(VALID_CONTROL_TAGS, ErrorCode.E024) | _BARE_CONTROL_TAG

# A tag followed by '=' carries a value and one that is not stands alone. An equation's left-hand side never
# starts with '@', so this lookahead separates the two without backtracking into the equation.
TAG = (TAG_NAME + ~pp.FollowedBy(pp.Literal("="))).set_parse_action(_parse_tag)

ANY_TAG = VALUE_TAG | TAG

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


MISSING_LHS = (pp.ZeroOrMore(ANY_TAG) + pp.FollowedBy(pp.Literal("="))).set_parse_action(_missing_lhs_fail)


def _missing_rhs_fail(s: str, loc: int, _toks: pp.ParseResults) -> None:
    raise GCNParseFailure(
        s,
        loc,
        "Missing right-hand side of equation",
        code=ErrorCode.E005,
        found=";",
    )


MISSING_RHS = (
    pp.ZeroOrMore(ANY_TAG) + EXPR("lhs") + pp.Suppress(pp.Literal("=")) + pp.FollowedBy(pp.Regex(r"\s*[;:]"))
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
    pp.ZeroOrMore(ANY_TAG) + EXPR("expr") + ~pp.FollowedBy(pp.Regex(r"\s*=")) + pp.FollowedBy(SEMI)
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


def _brace_before_semicolon_fail(s: str, loc: int, _toks: pp.ParseResults) -> None:
    raise GCNParseFailure(
        s,
        loc,
        "Missing semicolon",
        code=ErrorCode.E001,
        found="}",
    )


# Without these checks a stray closing bracket after the RHS surfaces as a generic "Expected ';'" error. A '}'
# right after the RHS is the component's own closing brace, so the equation is missing its semicolon.
_UNMATCHED_CLOSE = pp.Regex(r"[)\]]").copy().set_parse_action(_unmatched_close_fail)
_BRACE_BEFORE_SEMICOLON = pp.Literal("}").copy().set_parse_action(_brace_before_semicolon_fail)


def _literal_number(node: Node) -> float | None:
    if isinstance(node, Number):
        return node.value
    if isinstance(node, UnaryOp) and node.op == Operator.NEG and isinstance(node.operand, Number):
        return -node.operand.value
    return None


def _solver_keyword_value(
    s: str, loc: int, call: str, keyword: str, value: Node | pp.ParseResults
) -> float | tuple[float, float]:
    if keyword in INTERVAL_SOLVER_KEYWORDS:
        numbers = [_literal_number(node) for node in value] if isinstance(value, pp.ParseResults) else []
        if len(numbers) != 2 or not all(number is not None and math.isfinite(number) for number in numbers):
            message = f"'{keyword}' in '{call}' takes two finite numbers, as in '{keyword}=[0.1, 5]'"
            raise GCNParseFailure(s, loc, message, code=ErrorCode.E025, found=keyword)
        low, high = numbers
        if low >= high:
            message = f"'{keyword}' in '{call}' takes an interval whose first number is smaller than its second"
            raise GCNParseFailure(s, loc, message, code=ErrorCode.E025, found=keyword)
        return low, high

    number = None if isinstance(value, pp.ParseResults) else _literal_number(value)
    if number is None or not math.isfinite(number):
        message = f"'{keyword}' in '{call}' takes a finite number, as in '{keyword}=1'"
        raise GCNParseFailure(s, loc, message, code=ErrorCode.E025, found=keyword)
    return number


def _build_solver_call(s: str, loc: int, toks: pp.ParseResults) -> SteadyStateSolverCall:
    name = toks.solver_name
    call = f"{name}(...)"
    methods = SOLVER_METHODS[name]

    def fail(at: int, message: str, found: str, suggestions: list[str] | None = None) -> GCNParseFailure:
        return GCNParseFailure(s, at, message, code=ErrorCode.E025, found=found, suggestions=suggestions)

    # Each keyword keeps the offset it starts at, so an error about it points at the keyword rather than the call.
    arguments: dict[str, tuple[int, Node | pp.ParseResults]] = {}
    for argument in toks.solver_args:
        if argument.arg_name in arguments:
            raise fail(argument.arg_loc, f"'{argument.arg_name}' is given twice in '{call}'", argument.arg_name)
        value = argument.arg_list if "arg_list" in argument else argument.arg_value
        arguments[argument.arg_name] = (argument.arg_loc, value)

    method_loc, method_value = arguments.pop("method", (loc, None))
    if not isinstance(method_value, Parameter):
        raise fail(method_loc, f"'{call}' takes a method name, as in 'method={next(iter(methods))}'", name)
    method = method_value.name
    if method not in methods:
        raise fail(method_loc, f"Unknown {name} method '{method}'", method, find_similar_names(method, methods))

    keyword, required = methods[method]
    unexpected = sorted(arguments.keys() - {keyword})
    if unexpected:
        message = f"Method '{method}' of '{call}' takes '{keyword}', not '{unexpected[0]}'"
        raise fail(arguments[unexpected[0]][0], message, unexpected[0])
    if required and keyword not in arguments:
        raise fail(method_loc, f"Method '{method}' of '{call}' needs '{keyword}'", method)

    values = {key: _solver_keyword_value(s, at, call, key, value) for key, (at, value) in arguments.items()}
    return SteadyStateSolverCall(
        func_name=name,
        expr=toks.solver_expr,
        method=method,
        **values,
        location=location_at(s, loc, length=len(name)),
    )


def _positional_solver_argument_fail(s: str, loc: int, _toks: pp.ParseResults) -> None:
    raise GCNParseFailure(
        s,
        loc,
        "A solver call takes its expression first and every other argument by keyword, as in "
        "'root(expr, method=brentq, bounds=[0.1, 5])'",
        code=ErrorCode.E025,
        found=",",
    )


def _trailing_solver_comma_fail(s: str, loc: int, _toks: pp.ParseResults) -> None:
    raise GCNParseFailure(
        s, loc, "A solver call ends without a comma after its last argument", code=ErrorCode.E025, found=")"
    )


_ARGUMENT_START = pp.Empty().set_parse_action(lambda _s, loc, _toks: loc)

# Each alternative carries its own results name. A name on the alternation as a whole would wrap the matched node in a
# ParseResults instead of returning the node.
_SOLVER_LIST = pp.Group(pp.Suppress("[") + pp.Optional(pp.DelimitedList(EXPR)) + pp.Suppress("]"))
_SOLVER_ARG = pp.Group(
    _ARGUMENT_START("arg_loc") + IDENTIFIER("arg_name") + EQUALS + (_SOLVER_LIST("arg_list") | EXPR("arg_value"))
)
_POSITIONAL_SOLVER_ARG = EXPR.copy().set_parse_action(_positional_solver_argument_fail)
_TRAILING_SOLVER_COMMA = pp.FollowedBy(pp.Literal(")")).set_parse_action(_trailing_solver_comma_fail)

SOLVER_CALL = (
    SOLVER_NAME("solver_name")
    + LPAREN
    - EXPR("solver_expr")
    - pp.ZeroOrMore(COMMA + (_SOLVER_ARG | _TRAILING_SOLVER_COMMA | _POSITIONAL_SOLVER_ARG))("solver_args")
    - RPAREN
).set_parse_action(_build_solver_call)

# The lookahead leaves the calibrating arrow '->' to the equation grammar instead of reading its '-' as an operator.
_TRAILING_OPERATOR = pp.Regex(r"(?!->)[-+*/^]").set_parse_action(misplaced_solver_call_fail)
_SOLVER_RHS = SOLVER_CALL("rhs") + pp.Optional(_TRAILING_OPERATOR)

_VALID_EQUATION = (
    pp.ZeroOrMore(ANY_TAG)("tags")
    + EXPR("lhs")
    + pp.Suppress(pp.Literal("="))
    + (_SOLVER_RHS | EXPR("rhs"))
    + pp.Optional(_UNMATCHED_CLOSE | _BRACE_BEFORE_SEMICOLON)
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
    tags = frozenset(tag for tag in tokens.tags if isinstance(tag, Tag))
    annotations = dict(pair for pair in tokens.tags if not isinstance(pair, Tag))
    lagrange_name = tokens.lagrange[0] if tokens.lagrange else None
    calibrating_param = tokens.calibrating[0] if tokens.calibrating else None

    lhs_location = _find_location_in_node(tokens.lhs) or location_at(s, loc, length=0)
    if isinstance(tokens.rhs, SteadyStateSolverCall):
        _check_solver_identity(
            s,
            _offset_of(s, lhs_location),
            lhs=tokens.lhs,
            call=tokens.rhs,
            lagrange_name=lagrange_name,
            calibrating_param=calibrating_param,
        )

    line, col, source_line = lhs_location.line, lhs_location.column, lhs_location.source_line

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
        annotations=annotations,
        location=location,
    )


def _check_solver_identity(
    s: str,
    loc: int,
    lhs: Node,
    call: SteadyStateSolverCall,
    lagrange_name: str | None,
    calibrating_param: str | None,
) -> None:
    def fail(message: str) -> GCNParseFailure:
        return GCNParseFailure(s, loc, message, code=ErrorCode.E025, found=str(lhs))

    if not (isinstance(lhs, Variable) and lhs.time_index == STEADY_STATE):
        raise fail(
            f"The left-hand side of '{call.func_name}(...)' is the steady-state variable it solves for, such as 'w[ss]'"
        )
    if lagrange_name or calibrating_param:
        raise fail(f"A '{call.func_name}(...)' identity takes no Lagrange multiplier and calibrates no parameter")
    if collect_nodes_of_type(call.expr, Expectation):
        raise fail(f"The expression of '{call.func_name}(...)' is a steady-state relation and takes no expectation")


def _offset_of(s: str, location: ParseLocation) -> int:
    """Return the character offset in ``s`` of a 1-based line and column."""
    preceding_lines = s.split("\n")[: location.line - 1]
    return sum(len(line) + 1 for line in preceding_lines) + location.column - 1


EQUATION.set_parse_action(_build_equation)

_ARITHMETIC: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def _evaluate_number_expr(value: pp.ParseResults | list[Any] | float | str | None) -> float | str | None:
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


def collect_kwargs(args: pp.ParseResults | str) -> dict[str, float | str | None]:
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
        f"A prior on '{param_name}' is declared with '='. Declare it with '~': '{param_name} ~ {dist_or_wrapper}(...)'",
        code=ErrorCode.E009,
        found="=",
    )


MISSING_TILDE = (
    IDENTIFIER("param_name")
    + pp.FollowedBy(pp.Literal("=") + (WRAPPER_NAME | DIST_NAME))
    + pp.Literal("=").suppress()
    + (WRAPPER_NAME | DIST_NAME)("dist_or_wrapper")
).set_parse_action(_missing_tilde_fail)


def _build_distribution(s: str, loc: int, tokens: pp.ParseResults) -> GCNDistribution:
    wrapper_name = tokens.wrapper_name or None
    initial_value = _evaluate_number_expr(tokens.initial) if tokens.initial else None

    return GCNDistribution(
        parameter_name=tokens.param_name,
        dist_name=tokens.dist_name,
        dist_kwargs=collect_kwargs(tokens.dist_args),
        wrapper_name=wrapper_name,
        wrapper_kwargs=collect_kwargs(tokens.wrapper_args) if wrapper_name else {},
        initial_value=initial_value,
        location=_statement_location(s, loc),
    )


def _statement_location(s: str, loc: int) -> ParseLocation:
    start = location_at(s, loc, length=0)
    fallback = (start.line, start.column + len(start.source_line.strip()))
    end_line, end_col = _equation_end(s, loc, ";", fallback)

    return ParseLocation(
        line=start.line,
        column=start.column,
        end_line=end_line,
        end_column=end_col,
        source_line=start.source_line,
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
    "SOLVER_CALL",
    "TAG",
    "VARIABLE_LIST",
    "VARIABLE_REF",
    "parse_distribution",
    "parse_equation",
]
