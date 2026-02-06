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

# Enable packrat caching for better performance with recursive grammars
pp.ParserElement.enablePackrat()

# Basic tokens
LPAREN = pp.Suppress("(")
RPAREN = pp.Suppress(")")
LBRACKET = pp.Literal("[")
RBRACKET = pp.Literal("]")
COMMA = pp.Suppress(",")

# Numbers - integers and floats, including scientific notation
# Note: negative numbers are handled via unary minus, not in the number itself
NUMBER = pp.Regex(r"\d+\.?\d*(?:[eE][+-]?\d+)?").setParseAction(lambda t: Number(value=float(t[0])))

# Time index inside brackets: [], [-1], [1], [ss]
TIME_INDEX_CONTENT = pp.Regex(r"-?\d+|ss")
TIME_INDEX = (pp.Suppress("[") + pp.Optional(TIME_INDEX_CONTENT, default="") + pp.Suppress("]")).setParseAction(
    lambda t: _parse_time_index(t[0])
)


def _parse_time_index(content: str) -> TimeIndex:
    if content == "":
        return T
    if content == "ss":
        return STEADY_STATE
    return TimeIndex(int(content))


# Identifier (parameter or function name)
IDENTIFIER = pp.Word(pp.alphas + "_", pp.alphanums + "_")

# Variable: identifier followed by time index
VARIABLE = (IDENTIFIER + TIME_INDEX).setParseAction(lambda t: Variable(name=t[0], time_index=t[1]))

# Expectation operator: E[] followed by bracketed expression
# We need to use Combine to make it a single token that won't match as identifier
EXPECTATION_MARKER = pp.Combine(pp.Literal("E") + pp.Literal("[") + pp.Literal("]"))

# Parameter: identifier not followed by bracket
PARAMETER = IDENTIFIER.copy().setParseAction(lambda t: Parameter(name=t[0]))

# Forward declaration for recursive expression
EXPR = pp.Forward()

# Function call: identifier followed by parenthesized arguments
FUNC_ARGS = pp.Optional(pp.delimitedList(EXPR))
FUNC_CALL = (IDENTIFIER + LPAREN + FUNC_ARGS + RPAREN).setParseAction(
    lambda t: FunctionCall(func_name=t[0], args=tuple(t[1:]))
)

# Expectation: E[][expr]
EXPECTATION = (pp.Suppress(EXPECTATION_MARKER) + pp.Suppress("[") + EXPR + pp.Suppress("]")).setParseAction(
    lambda t: Expectation(expr=t[0])
)

# Atom: the smallest unit of an expression
# Order matters: try more specific patterns first
ATOM = EXPECTATION | FUNC_CALL | VARIABLE | NUMBER | PARAMETER | (LPAREN + EXPR + RPAREN)


def _make_unary_op(tokens):
    """Handle unary + and -."""
    toks = tokens[0]
    if len(toks) == 1:
        return toks[0]
    # Unary operator: [op, operand]
    op_str = toks[0]
    operand = toks[1]
    if op_str == "-":
        return UnaryOp(op=Operator.NEG, operand=operand)
    # Unary + is a no-op
    return operand


def _make_binary_op(tokens):
    """Fold [operand, op, operand, op, ...] into nested BinaryOps."""
    toks = tokens[0]
    result = toks[0]
    i = 1
    while i < len(toks):
        op_str = toks[i]
        right = toks[i + 1]
        op = {
            "+": Operator.ADD,
            "-": Operator.SUB,
            "*": Operator.MUL,
            "/": Operator.DIV,
            "^": Operator.POW,
            "**": Operator.POW,
        }[op_str]
        result = BinaryOp(left=result, op=op, right=right)
        i += 2
    return result


# Expression with operator precedence (lowest to highest):
# 1. Addition/Subtraction (left-associative)
# 2. Multiplication/Division (left-associative)
# 3. Unary minus
# 4. Exponentiation (right-associative)
# 5. Atoms

EXPR <<= pp.infixNotation(
    ATOM,
    [
        (pp.oneOf("^ **"), 2, pp.opAssoc.RIGHT, _make_binary_op),
        (pp.oneOf("+ -"), 1, pp.opAssoc.RIGHT, _make_unary_op),
        (pp.oneOf("* /"), 2, pp.opAssoc.LEFT, _make_binary_op),
        (pp.oneOf("+ -"), 2, pp.opAssoc.LEFT, _make_binary_op),
    ],
)


def parse_expression(text: str):
    """
    Parse a mathematical expression string into an AST node.

    Parameters
    ----------
    text : str
        The expression to parse (e.g., "C[] + I[]", "log(K[-1])", "beta * E[][U[1]]")

    Returns
    -------
    Node
        The AST representation of the expression.
    """
    result = EXPR.parseString(text, parseAll=True)
    return result[0]
