import pyparsing as pp

from gEconpy.parser.ast import (
    STEADY_STATE,
    BinaryOp,
    Expectation,
    FunctionCall,
    Node,
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


def parse_expression(text: str, context: str = "") -> Node:
    """
    Parse a mathematical expression string into an abstract syntax tree node.

    Parameters
    ----------
    text : str
        The expression to parse.
    context : str, optional
        Name to report in error messages, identifying where the expression came from. Defaults to an empty string.

    Returns
    -------
    node : Node
        Root of the parsed expression.
    """
    try:
        return EXPR.parse_string(text, parse_all=True)[0]
    except pp.ParseBaseException as exc:
        raise _convert_parse_exception(exc, text, context) from None


def _parse_time_index(content: str) -> TimeIndex:
    if content == "":
        return T
    if content == "ss":
        return STEADY_STATE
    return TimeIndex(int(content))


def _location_at(s: str, loc: int, length: int) -> ParseLocation:
    line = pp.lineno(loc, s)
    col = pp.col(loc, s)
    lines = s.splitlines()
    source_line = lines[line - 1] if 0 < line <= len(lines) else ""

    return ParseLocation(
        line=line,
        column=col,
        end_line=line,
        end_column=col + length,
        source_line=source_line,
    )


NUMBER = pp.Regex(NUMBER_PATTERN).set_parse_action(lambda t: Number(value=float(t[0])))

TIME_INDEX = (LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="") + RBRACKET).set_parse_action(
    lambda t: _parse_time_index(t[0])
)


def _invalid_time_index_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    invalid_index = f"[{toks.invalid_content}]"
    raise GCNParseFailure(
        s,
        loc,
        f"Invalid time index '{invalid_index}' for variable '{toks.var_name}'",
        code=ErrorCode.E010,
        found=invalid_index,
    )


_INVALID_CONTENT = pp.Regex(r"[^\]]+")

INVALID_TIME_INDEX_VAR = (
    IDENTIFIER("var_name")
    + pp.Literal("[").suppress()
    + ~pp.FollowedBy(TIME_INDEX_CONTENT + pp.Literal("]"))
    + ~pp.FollowedBy(pp.Literal("]"))
    + _INVALID_CONTENT("invalid_content")
    + pp.Literal("]").suppress()
).set_parse_action(_invalid_time_index_fail)

EXPR = pp.Forward()


def _parse_variable(s: str, loc: int, toks: pp.ParseResults) -> Variable:
    name, time_index = toks[0], toks[1]
    time_text = "" if time_index == T else str(time_index)
    location = _location_at(s, loc, length=len(name) + 2 + len(time_text))
    return Variable(name=name, time_index=time_index, location=location)


VARIABLE = (IDENTIFIER + TIME_INDEX).set_parse_action(_parse_variable)


def _parse_parameter(s: str, loc: int, toks: pp.ParseResults) -> Parameter:
    name = toks[0]
    return Parameter(name=name, location=_location_at(s, loc, length=len(name)))


PARAMETER = (IDENTIFIER + ~pp.FollowedBy(pp.Literal("[") | pp.Literal("("))).set_parse_action(_parse_parameter)

EXPECTATION = (pp.Combine(KW_E + pp.Literal("[]") + pp.Literal("[")) - EXPR - RBRACKET).set_parse_action(
    lambda t: Expectation(expr=t[1])
)

FUNC_ARGS = pp.DelimitedList(EXPR, min=1)("args")


def _empty_function_fail(s: str, loc: int, toks: pp.ParseResults) -> None:
    call_text = f"{toks[0]}()"
    raise GCNParseFailure(
        s,
        loc,
        f"Empty function call '{call_text}'",
        code=ErrorCode.E008,
        found=call_text,
    )


EMPTY_FUNC_CALL = (
    IDENTIFIER("func_name") + pp.Literal("(").suppress() + pp.FollowedBy(pp.Literal(")"))
).set_parse_action(_empty_function_fail)

FUNC_CALL = EMPTY_FUNC_CALL | (IDENTIFIER("func_name") + LPAREN - FUNC_ARGS - RPAREN).set_parse_action(
    lambda t: FunctionCall(func_name=t[0], args=tuple(t.args))
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


def _make_unary_op(tokens: pp.ParseResults) -> UnaryOp:
    _sign, operand = tokens[0]
    return UnaryOp(op=Operator.NEG, operand=operand)


def _make_binary_op(tokens: pp.ParseResults) -> BinaryOp:
    operands = tokens[0]
    result = operands[0]
    for op_str, right in zip(operands[1::2], operands[2::2], strict=True):
        result = BinaryOp(left=result, op=_OP_MAP[op_str], right=right)
    return result


EXPR <<= pp.infix_notation(
    ATOM,
    [
        (pp.Literal("-"), 1, pp.OpAssoc.RIGHT, _make_unary_op),
        (pp.one_of("^ **"), 2, pp.OpAssoc.RIGHT, _make_binary_op),
        (pp.one_of("* /"), 2, pp.OpAssoc.LEFT, _make_binary_op),
        (pp.one_of("+ -"), 2, pp.OpAssoc.LEFT, _make_binary_op),
    ],
)

EXPR.ignore(COMMENT)


def _convert_parse_exception(exc: pp.ParseBaseException, text: str, context: str = "") -> GCNGrammarError:
    line, col = exc.lineno, exc.col
    source_line = exc.line
    if not source_line and text:
        lines = text.split("\n")
        if 0 < line <= len(lines):
            source_line = lines[line - 1]

    found = exc.found or ""
    message = str(exc.msg)

    return GCNGrammarError(
        message=_classify_expression_error(message, found),
        found=found,
        location=ParseLocation(line=line, column=col, source_line=source_line),
        context=context,
        code=_get_error_code(message),
    )


def _classify_expression_error(message: str, found: str) -> str:
    if "Empty function call" in message:
        return message.split(". E008", maxsplit=1)[0]

    if "Invalid time index" in message:
        return message.split(". E010", maxsplit=1)[0]

    if "end of text" in message.lower():
        if found in _OP_MAP:
            return f"Unexpected operator '{found}' at end of expression"
        return "Incomplete expression"

    for brackets, explanation in (("()", "Unbalanced parentheses"), ("[]", "Invalid variable syntax")):
        if any(bracket in message for bracket in brackets):
            return explanation

    return f"Invalid expression syntax: {message}"


def _get_error_code(message: str) -> ErrorCode:
    if "E008" in message or "Empty function call" in message:
        return ErrorCode.E008

    if "E010" in message or "Invalid time index" in message:
        return ErrorCode.E010

    if "(" in message or ")" in message:
        return ErrorCode.E007

    if "[" in message or "]" in message:
        return ErrorCode.E010

    return ErrorCode.E006


__all__ = [
    "EXPR",
    "parse_expression",
]
