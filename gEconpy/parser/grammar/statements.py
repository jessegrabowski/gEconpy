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
from gEconpy.parser.errors import GCNParseFailure, ParseLocation
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


def _parse_variable_ref(s: str, loc: int, toks) -> Variable:
    """Parse a variable reference with location tracking."""
    line = pp.lineno(loc, s)
    col = pp.col(loc, s)
    lines = s.splitlines()
    source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    # Calculate end column based on the full variable text (name + brackets + time)
    var_text = f"{toks.name}[{toks.time}]"
    end_column = col + len(var_text)

    location = ParseLocation(
        line=line,
        column=col,
        end_line=line,
        end_column=end_column,
        source_line=source_line,
    )

    return Variable(
        name=toks.name,
        time_index=_parse_time_index_content(toks.time),
        location=location,
    )


VARIABLE_REF = (
    IDENTIFIER("name") + LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="")("time") + RBRACKET
).set_parse_action(_parse_variable_ref)

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


def _parse_calibrating_param(s: str, loc: int, toks):
    """Parse calibrating parameter with location tracking."""
    # Skip past the arrow to find the parameter name location
    param_name = toks.param
    # Find actual location of the parameter name (after "->")
    arrow_pos = s.find("->", loc)
    if arrow_pos != -1:
        param_loc = arrow_pos + 2  # Skip past "->"
        # Skip whitespace
        while param_loc < len(s) and s[param_loc] in " \t\n":
            param_loc += 1
    else:
        param_loc = loc

    line = pp.lineno(param_loc, s)
    col = pp.col(param_loc, s)
    lines = s.splitlines()
    source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    location = ParseLocation(
        line=line,
        column=col,
        end_line=line,
        end_column=col + len(param_name),
        source_line=source_line,
    )
    return (param_name, location)


CALIBRATING_PARAM = (ARROW + IDENTIFIER("param")).set_parse_action(_parse_calibrating_param)


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


def _find_location_in_node(node):
    """Recursively find a location from an AST node or its children."""
    # Direct location
    loc = getattr(node, "location", None)
    if loc is not None:
        return loc

    # Try left child (for BinaryOp)
    left = getattr(node, "left", None)
    if left is not None:
        loc = _find_location_in_node(left)
        if loc is not None:
            return loc

    # Try operand (for UnaryOp)
    operand = getattr(node, "operand", None)
    if operand is not None:
        loc = _find_location_in_node(operand)
        if loc is not None:
            return loc

    # Try expr (for Expectation)
    expr = getattr(node, "expr", None)
    if expr is not None:
        loc = _find_location_in_node(expr)
        if loc is not None:
            return loc

    return None


def _build_equation(s: str, loc: int, tokens) -> GCNEquation:
    tags = frozenset(tokens.tags) if tokens.tags else frozenset()

    lagrange_name = None
    if tokens.lagrange and len(tokens.lagrange) > 0:
        lagrange_name = tokens.lagrange[0]

    calibrating_param = None
    if tokens.calibrating and len(tokens.calibrating) > 0:
        # calibrating is now a tuple of (name, location)
        calib_data = tokens.calibrating[0]
        if isinstance(calib_data, tuple):
            calibrating_param, _ = calib_data  # Discard parameter location, use LHS location
        else:
            calibrating_param = calib_data

    # Always use the LHS location for the equation start
    lhs_location = _find_location_in_node(tokens.lhs)
    if lhs_location is not None:
        line = lhs_location.line
        col = lhs_location.column
        source_line = lhs_location.source_line
    else:
        line = pp.lineno(loc, s)
        col = pp.col(loc, s)
        lines = s.splitlines()
        source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    # Find the end of the LHS expression
    # For calibrating equations (with ->), find the arrow and back up
    # For regular equations, find the equals sign
    if calibrating_param:
        # Find "-> param" and determine where LHS ends (at the "= value" part)
        # The LHS is everything before "= value -> param"
        # We want to underline from start of expression to end of "= value"
        arrow_pos = s.find("->", loc)
        if arrow_pos != -1:
            end_line = pp.lineno(arrow_pos, s)
            end_col = pp.col(arrow_pos, s)
        else:
            end_line = line
            end_col = col + 10
    else:
        # For regular equations, end_column covers the whole line for now
        end_loc = s.find(";", loc)
        if end_loc != -1:
            end_line = pp.lineno(end_loc, s)
            end_col = pp.col(end_loc, s) + 1
        else:
            end_line = line
            end_col = col + len(source_line.strip())

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
