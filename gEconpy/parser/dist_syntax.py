import pyparsing as pp


def evaluate_expression(parsed_expr):
    if isinstance(parsed_expr, int | float):
        return float(parsed_expr)
    elif not parsed_expr:
        return None
    elif isinstance(parsed_expr, pp.ParseResults):
        parsed_expr = parsed_expr.as_list()
        if len(parsed_expr) == 1 and isinstance(parsed_expr[0], list):
            parsed_expr = parsed_expr[0]
        expr_str = "".join(map(str, parsed_expr))
        return eval(expr_str, {"__builtins__": None}, {})
    return parsed_expr


var_name = pp.Word(pp.alphas, pp.alphanums + "_")
dist_name = pp.Word(pp.alphas, pp.alphanums + "_")
equals = pp.Literal("=").suppress()

number = pp.pyparsing_common.number
numeric_expr = pp.infixNotation(
    number,
    [
        (pp.Literal("/"), 2, pp.opAssoc.LEFT),
        (pp.Literal("*"), 2, pp.opAssoc.LEFT),
        (pp.Literal("+"), 2, pp.opAssoc.LEFT),
        (pp.Literal("-"), 2, pp.opAssoc.LEFT),
    ],
)

value = numeric_expr | var_name

key = pp.Word(pp.alphas, pp.alphanums + "_")
key_value_pair = pp.Group(key + equals + value)

args = (
    pp.Suppress("(") + pp.Optional(pp.delimitedList(key_value_pair)) + pp.Suppress(")")
)
initial_value = pp.Optional(equals + numeric_expr, default=None)("initial_value")

dist_syntax = dist_name("dist_name") + args("kwargs") + initial_value + pp.StringEnd()


__all__ = ["dist_syntax", "evaluate_expression"]
