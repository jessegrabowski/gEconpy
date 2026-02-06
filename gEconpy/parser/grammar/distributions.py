import operator

import pyparsing as pp

from gEconpy.parser.ast import GCNDistribution
from gEconpy.parser.dist_syntax import (
    PRELIZ_DIST_WRAPPERS,
    PRELIZ_DISTS,
)

_SAFE_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def _safe_eval_tokens(tokens: list) -> float:
    """Evaluate a flat list of [number, op, number, op, ...] tokens safely."""
    if not tokens:
        raise ValueError("Empty token list")

    # Handle single value
    if len(tokens) == 1:
        return float(tokens[0])

    # Process left-to-right (no precedence - pyparsing handles that via infixNotation)
    result = float(tokens[0])
    i = 1
    while i < len(tokens):
        op_str = str(tokens[i])
        if op_str not in _SAFE_OPERATORS:
            raise ValueError(f"Unknown operator: {op_str}")
        operand = float(tokens[i + 1])
        result = _SAFE_OPERATORS[op_str](result, operand)
        i += 2

    return result


def _evaluate_expression(parsed_expr):
    """Safely evaluate a simple numeric expression without using eval()."""
    if isinstance(parsed_expr, int | float):
        return float(parsed_expr)
    if parsed_expr is None:
        return None
    if isinstance(parsed_expr, pp.ParseResults):
        parsed_expr = parsed_expr.as_list()
    if isinstance(parsed_expr, list):
        # Flatten nested single-element lists
        while len(parsed_expr) == 1 and isinstance(parsed_expr[0], list):
            parsed_expr = parsed_expr[0]
        return _safe_eval_tokens(parsed_expr)
    return float(parsed_expr)


# Basic tokens
EQUALS = pp.Literal("=").suppress()
LPAREN = pp.Literal("(").suppress()
RPAREN = pp.Literal(")").suppress()
COMMA = pp.Literal(",").suppress()
TILDE = pp.Literal("~").suppress()
SEMICOLON = pp.Optional(pp.Literal(";")).suppress()

# Identifiers
PARAM_NAME = pp.Word(pp.alphas, pp.alphanums + "_")

# Numbers and expressions
NUMBER = pp.pyparsing_common.number
NUMBER_EXPR = pp.infixNotation(
    NUMBER,
    [
        (pp.Literal("/"), 2, pp.opAssoc.LEFT),
        (pp.Literal("*"), 2, pp.opAssoc.LEFT),
        (pp.Literal("+"), 2, pp.opAssoc.LEFT),
        (pp.Literal("-"), 2, pp.opAssoc.LEFT),
    ],
)

# None literal for bounds
NONE_LITERAL = pp.Keyword("None").setParseAction(lambda _: [None])

# Value can be number expression, None, or identifier (for hyper-parameters)
VALUE = NUMBER_EXPR | NONE_LITERAL | PARAM_NAME

# Key-value pair: alpha=2, lower=0.5, etc.
KEY_VALUE_PAIR = pp.Group(PARAM_NAME + EQUALS + VALUE)
KWARG_LIST = pp.Optional(pp.delimitedList(KEY_VALUE_PAIR, delim=COMMA), default=None)

# Distribution names
WRAPPER_FUNCS = pp.MatchFirst([pp.Keyword(wrapper) for wrapper in PRELIZ_DIST_WRAPPERS])
DISTRIBUTION_ID = pp.MatchFirst([pp.Keyword(dist) for dist in PRELIZ_DISTS])

# Core distribution: Normal(), Beta(alpha=2, beta=5), etc.
DIST = DISTRIBUTION_ID("dist_name") + LPAREN + KWARG_LIST("dist_kwargs") + RPAREN

# Wrapped distribution: maxent(Normal(), lower=0, upper=1)
WRAPPED_DIST = (
    WRAPPER_FUNCS("wrapper_name") + LPAREN + DIST + pp.Optional(COMMA + KWARG_LIST("wrapper_kwargs")) + RPAREN
)

# Initial value: = 0.35
INITIAL_VALUE = EQUALS + NUMBER_EXPR("initial_value")

# Full distribution declaration (without parameter name): Beta(alpha=2) = 0.35
DIST_DECLARATION = (WRAPPED_DIST | DIST) + pp.Optional(INITIAL_VALUE)

# Full prior: alpha ~ Beta(alpha=2, beta=5) = 0.35
PRIOR_DECLARATION = PARAM_NAME("parameter_name") + TILDE + DIST_DECLARATION + SEMICOLON


def _kwargs_to_dict(tokens, key: str) -> dict:
    """Convert parsed key-value pairs to a dictionary."""
    kwargs = tokens.get(key)
    if not kwargs:
        return {}
    result = {}
    for item in kwargs:
        if item is None:
            continue
        item_list = item.as_list() if isinstance(item, pp.ParseResults) else item
        if isinstance(item_list, list) and len(item_list) >= 2:  # noqa: PLR2004
            name = item_list[0]
            value = item_list[1] if len(item_list) == 2 else item_list[1:]  # noqa: PLR2004
            # Evaluate expressions
            if isinstance(value, list | pp.ParseResults):
                value = _evaluate_expression(value)
            result[name] = value
    return result


def _parse_action(tokens) -> GCNDistribution:
    """Convert parsed tokens to a GCNDistribution AST node."""
    dist_kwargs = _kwargs_to_dict(tokens, "dist_kwargs")
    wrapper_kwargs = _kwargs_to_dict(tokens, "wrapper_kwargs")

    initial_value = tokens.get("initial_value")
    if initial_value is not None:
        initial_value = _evaluate_expression(initial_value)

    return GCNDistribution(
        parameter_name=tokens["parameter_name"],
        dist_name=tokens["dist_name"],
        dist_kwargs=dist_kwargs,
        wrapper_name=tokens.get("wrapper_name"),
        wrapper_kwargs=wrapper_kwargs,
        initial_value=initial_value,
    )


PRIOR_DECLARATION.setParseAction(_parse_action)


def parse_distribution(text: str) -> GCNDistribution:
    """
    Parse a prior distribution declaration into a GCNDistribution AST node.

    Parameters
    ----------
    text : str
        The distribution declaration to parse, e.g.:
        - "alpha ~ Beta(alpha=2, beta=5) = 0.35"
        - "sigma ~ maxent(Normal(), lower=0, upper=1) = 0.5"

    Returns
    -------
    GCNDistribution
        The AST representation of the distribution.
    """
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1]
    result = PRIOR_DECLARATION.parseString(text, parseAll=True)
    return result[0]
