import pyparsing as pp

from gEconpy.parser.ast import (
    STEADY_STATE,
    BinaryOp,
    Expectation,
    FunctionCall,
    Number,
    Operator,
    Parameter,
    T,
    TimeIndex,
    UnaryOp,
    Variable,
)
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNGrammarError, GCNParseFailure, ParseLocation
from gEconpy.parser.grammar.tokens import (
    COMMENT,
    IDENTIFIER,
    KW_E,
    LBRACKET,
    LPAREN,
    NUMBER_PATTERN,
    RBRACKET,
    RPAREN,
    TIME_INDEX_CONTENT,
)


def _parse_number(tokens) -> Number:
    return Number(value=float(tokens[0]))


NUMBER = pp.Regex(NUMBER_PATTERN).set_parse_action(_parse_number)


def _parse_time_index(content: str) -> TimeIndex:
    if content == "":
        return T
    if content == "ss":
        return STEADY_STATE
    return TimeIndex(int(content))


TIME_INDEX = (LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="") + RBRACKET).set_parse_action(
    lambda t: _parse_time_index(t[0])
)

_VALID_TIME_INDEX_INNER = pp.Regex(r"-?\d+") | pp.Keyword("ss")
_INVALID_CONTENT = pp.Regex(r"[^\]]+")


def _invalid_time_index_fail(s: str, loc: int, toks) -> None:
    var_name = toks.var_name
    invalid_content = toks.invalid_content
    invalid_index = f"[{invalid_content}]"
    raise GCNParseFailure(
        s,
        loc,
        f"Invalid time index '{invalid_index}' for variable '{var_name}'",
        code=ErrorCode.E010,
        found=invalid_index,
    )


INVALID_TIME_INDEX_VAR = (
    IDENTIFIER("var_name")
    + pp.Literal("[").suppress()
    + ~pp.FollowedBy(_VALID_TIME_INDEX_INNER + pp.Literal("]"))
    + ~pp.FollowedBy(pp.Literal("]"))
    + _INVALID_CONTENT("invalid_content")
    + pp.Literal("]").suppress()
).set_parse_action(_invalid_time_index_fail)

EXPR = pp.Forward()


def _parse_variable(s: str, loc: int, toks) -> Variable:
    line = pp.lineno(loc, s)
    col = pp.col(loc, s)
    lines = s.splitlines()
    source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    name = toks[0]
    time_index = toks[1]
    # Estimate token length (name + [] or [time])
    time_str = str(time_index) if time_index != T else ""
    token_len = len(name) + 2 + len(time_str)

    location = ParseLocation(
        line=line,
        column=col,
        end_line=line,
        end_column=col + token_len,
        source_line=source_line,
    )
    return Variable(name=name, time_index=time_index, location=location)


VARIABLE = (IDENTIFIER + TIME_INDEX).set_parse_action(_parse_variable)


def _parse_parameter(s: str, loc: int, toks) -> Parameter:
    line = pp.lineno(loc, s)
    col = pp.col(loc, s)
    lines = s.splitlines()
    source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    name = toks[0]
    location = ParseLocation(
        line=line,
        column=col,
        end_line=line,
        end_column=col + len(name),
        source_line=source_line,
    )
    return Parameter(name=name, location=location)


PARAMETER = (IDENTIFIER + ~pp.FollowedBy(pp.Literal("[") | pp.Literal("("))).set_parse_action(_parse_parameter)

_EMPTY_BRACKETS = pp.Literal("[]")
_OPEN_BRACKET = pp.Literal("[")

EXPECTATION = (pp.Combine(KW_E + _EMPTY_BRACKETS + _OPEN_BRACKET) - EXPR - RBRACKET).set_parse_action(
    lambda t: Expectation(expr=t[1])
)

FUNC_ARGS = pp.DelimitedList(EXPR, min=1)("args")


def _parse_function_call(tokens) -> FunctionCall:
    func_name = tokens[0]
    args = tuple(tokens.args)
    return FunctionCall(func_name=func_name, args=args)


def _empty_function_fail(s: str, loc: int, toks) -> None:
    func_name = toks[0]
    raise GCNParseFailure(
        s,
        loc,
        f"Empty function call '{func_name}()'",
        code=ErrorCode.E008,
        found=f"{func_name}()",
    )


EMPTY_FUNC_CALL = (
    IDENTIFIER("func_name") + pp.Literal("(").suppress() + pp.FollowedBy(pp.Literal(")"))
).set_parse_action(_empty_function_fail)

FUNC_CALL = EMPTY_FUNC_CALL | (IDENTIFIER("func_name") + LPAREN - FUNC_ARGS - RPAREN).set_parse_action(
    _parse_function_call
)

PAREN_EXPR = LPAREN - EXPR - RPAREN

ATOM = EXPECTATION | FUNC_CALL | VARIABLE | INVALID_TIME_INDEX_VAR | NUMBER | PARAMETER | PAREN_EXPR

_OP_MAP = {
    "+": Operator.ADD,
    "-": Operator.SUB,
    "*": Operator.MUL,
    "/": Operator.DIV,
    "^": Operator.POW,
    "**": Operator.POW,
}


def _make_unary_op(tokens) -> UnaryOp:
    toks = tokens[0]
    return UnaryOp(op=Operator.NEG, operand=toks[1])


def _make_binary_op_left(tokens) -> BinaryOp:
    toks = tokens[0]
    result = toks[0]
    i = 1
    while i < len(toks):
        op_str = toks[i]
        right = toks[i + 1]
        result = BinaryOp(left=result, op=_OP_MAP[op_str], right=right)
        i += 2
    return result


def _make_binary_op_right(tokens) -> BinaryOp:
    toks = tokens[0]
    result = toks[0]
    i = 1
    while i < len(toks):
        op_str = toks[i]
        right = toks[i + 1]
        result = BinaryOp(left=result, op=_OP_MAP[op_str], right=right)
        i += 2
    return result


EXPR <<= pp.infix_notation(
    ATOM,
    [
        (pp.Literal("-"), 1, pp.OpAssoc.RIGHT, _make_unary_op),
        (pp.one_of("^ **"), 2, pp.OpAssoc.RIGHT, _make_binary_op_right),
        (pp.one_of("* /"), 2, pp.OpAssoc.LEFT, _make_binary_op_left),
        (pp.one_of("+ -"), 2, pp.OpAssoc.LEFT, _make_binary_op_left),
    ],
)

EXPR.ignore(COMMENT)


def _convert_parse_exception(exc: pp.ParseBaseException, text: str, context: str = "") -> GCNGrammarError:
    line = exc.lineno
    col = exc.col
    source_line = exc.line if hasattr(exc, "line") and exc.line else ""

    if not source_line and text:
        lines = text.split("\n")
        if 0 < line <= len(lines):
            source_line = lines[line - 1]

    location = ParseLocation(line=line, column=col, source_line=source_line)

    expected = str(exc.expected) if hasattr(exc, "expected") else ""
    found = exc.found if hasattr(exc, "found") and exc.found else ""

    message = _classify_expression_error(exc, expected, found)

    return GCNGrammarError(
        message=message,
        expected=expected if expected else None,
        found=found if found else None,
        location=location,
        context=context,
        code=_get_error_code(exc, expected),
    )


def _classify_expression_error(exc: pp.ParseBaseException, expected: str, found: str) -> str:  # noqa: PLR0911
    msg = str(exc.msg) if hasattr(exc, "msg") else str(exc)

    if "Empty function call" in msg:
        return msg.split(". E008")[0]

    if "Invalid time index" in msg:
        return msg.split(". E010")[0]

    if "end of text" in msg.lower():
        if found in ("^", "**", "*", "/", "+", "-"):
            return f"Unexpected operator '{found}' at end of expression"
        return "Incomplete expression"

    if (expected and "+" in expected) or "-" in expected:
        return "Missing operand after operator"

    if "(" in msg or ")" in msg:
        return "Unbalanced parentheses"

    if "[" in msg or "]" in msg:
        return "Invalid variable syntax"

    return f"Invalid expression syntax: {msg}"


def _get_error_code(exc: pp.ParseBaseException, expected: str) -> ErrorCode:
    msg = str(exc.msg) if hasattr(exc, "msg") else str(exc)

    if "E008" in msg or "Empty function call" in msg:
        return ErrorCode.E008

    if "E010" in msg or "Invalid time index" in msg:
        return ErrorCode.E010

    if "(" in msg or ")" in msg:
        return ErrorCode.E007
    if expected and ("+" in expected or "-" in expected or "*" in expected):
        return ErrorCode.E006
    if "[" in msg or "]" in msg:
        return ErrorCode.E010

    return ErrorCode.E006


def parse_expression(text: str, context: str = ""):
    """
    Parse a mathematical expression string into an AST node.

    Raises GCNGrammarError if the expression cannot be parsed.
    """
    try:
        result = EXPR.parse_string(text, parse_all=True)
        return result[0]
    except pp.ParseBaseException as exc:
        raise _convert_parse_exception(exc, text, context) from None
