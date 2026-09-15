from collections import defaultdict
from collections.abc import Iterable

from gEconpy.parser.ast import (
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Node,
    Operator,
    UnaryOp,
    Variable,
)

PRECEDENCE = {
    Operator.ADD: 1,
    Operator.SUB: 1,
    Operator.MUL: 2,
    Operator.DIV: 2,
    Operator.NEG: 3,
    Operator.POW: 4,
}


def print_expression(node: Node, parent_precedence: int = 0) -> str:
    """
    Render an expression node as GCN source text.

    Parameters
    ----------
    node : Node
        The expression node to print.
    parent_precedence : int, optional
        Precedence of the enclosing operator. An operation binding more loosely than its parent is wrapped in
        parentheses. Defaults to 0, so a top-level expression is never wrapped.

    Returns
    -------
    text : str
        The expression in GCN syntax.
    """
    match node:
        case UnaryOp(op=op, operand=operand):
            text = f"{op}{print_expression(operand, PRECEDENCE[op])}"
            return f"({text})" if PRECEDENCE[op] < parent_precedence else text

        case BinaryOp(left=left, op=op, right=right):
            return _print_binary_op(left, op, right, parent_precedence)

        case FunctionCall(func_name=func_name, args=args):
            args_str = ", ".join(print_expression(arg) for arg in args)
            return f"{func_name}({args_str})"

        case Expectation(expr=expr):
            return f"E[][{print_expression(expr)}]"

        case _:
            return str(node)


def print_equation(eq: GCNEquation) -> str:
    """
    Render an equation as GCN source text, including its tags, Lagrange multiplier, and calibrating parameter.

    Parameters
    ----------
    eq : GCNEquation
        The equation to print.

    Returns
    -------
    text : str
        The equation in GCN syntax, without the trailing semicolon.
    """
    tags = "".join(f"{tag} " for tag in sorted(eq.tags, key=lambda tag: tag.value))
    text = f"{tags}{print_expression(eq.lhs)} = {print_expression(eq.rhs)}"

    if eq.lagrange_multiplier:
        text += f" : {eq.lagrange_multiplier}[]"

    if eq.calibrating_parameter:
        text += f" -> {eq.calibrating_parameter}"

    return text


def print_distribution(dist: GCNDistribution) -> str:
    """
    Render a prior declaration as GCN source text.

    Parameters
    ----------
    dist : GCNDistribution
        The distribution to print.

    Returns
    -------
    text : str
        The declaration in GCN syntax, without the trailing semicolon.
    """
    return f"{dist.parameter_name} ~ {_distribution_call(dist)}"


def print_block(block: GCNBlock, indent: str = "    ") -> str:
    """
    Render a block as GCN source text. Empty components are omitted.

    Parameters
    ----------
    block : GCNBlock
        The block to print.
    indent : str, optional
        String used for one level of indentation. Defaults to four spaces.

    Returns
    -------
    text : str
        The block in GCN syntax.
    """
    lines = [f"block {block.name}", "{"]

    lines.extend(_component_lines("definitions", (print_equation(eq) for eq in block.definitions), indent))
    lines.extend(_component_lines("controls", _variable_list(block.controls), indent))
    lines.extend(_component_lines("objective", (print_equation(eq) for eq in block.objective), indent))
    lines.extend(_component_lines("constraints", (print_equation(eq) for eq in block.constraints), indent))
    lines.extend(_component_lines("identities", (print_equation(eq) for eq in block.identities), indent))
    lines.extend(_component_lines("shocks", _shock_lines(block), indent))
    lines.extend(_component_lines("calibration", (_print_calibration_item(item) for item in block.calibration), indent))

    lines.append("};")
    return "\n".join(lines)


def print_model(model: GCNModel, indent: str = "    ") -> str:
    """
    Render a model as GCN source text: its options, tryreduce, and assumptions sections, then every block.

    Parameters
    ----------
    model : GCNModel
        The model to print.
    indent : str, optional
        String used for one level of indentation. Defaults to four spaces.

    Returns
    -------
    text : str
        The model in GCN syntax, with sections separated by blank lines.
    """
    sections = []

    if model.options:
        lines = ["options", "{"]
        for key, value in model.options.items():
            lines.append(f"{indent}{key} = {_print_option_value(value)};")
        lines.append("};")
        sections.append("\n".join(lines))

    if model.tryreduce:
        names = ", ".join(f"{name}[]" for name in model.tryreduce)
        sections.append("\n".join(["tryreduce", "{", f"{indent}{names};", "};"]))

    assumption_groups = _group_assumptions(model.assumptions)
    if assumption_groups:
        lines = ["assumptions", "{"]
        for assumption, var_names in assumption_groups.items():
            lines.extend([f"{indent}{assumption}", f"{indent}{{", f"{indent}{indent}{', '.join(sorted(var_names))};"])
            lines.append(f"{indent}}};")
        lines.append("};")
        sections.append("\n".join(lines))

    sections.extend(print_block(block, indent) for block in model.blocks)

    return "\n\n".join(sections)


print_ast = print_expression


def _print_binary_op(left: Node, op: Operator, right: Node, parent_precedence: int) -> str:
    # Equal-precedence operands need parentheses on the side the operator does not associate toward: the left
    # operand of the right-associative power, and the right operand of every other operator. Wrapping the right
    # operand of ADD and MUL as well keeps the printed tree identical to the parsed one.
    precedence = PRECEDENCE[op]
    if op == Operator.POW:
        left_parent, right_parent = precedence + 1, precedence
    else:
        left_parent, right_parent = precedence, precedence + 1

    text = f"{print_expression(left, left_parent)} {op} {print_expression(right, right_parent)}"

    if precedence < parent_precedence:
        return f"({text})"
    return text


def _component_lines(header: str, entries: Iterable[str], indent: str) -> list[str]:
    entries = list(entries)
    if not entries:
        return []

    lines = [f"{indent}{header}", f"{indent}{{"]
    lines.extend(f"{indent}{indent}{entry};" for entry in entries)
    lines.extend([f"{indent}}};", ""])
    return lines


def _variable_list(variables: list[Variable]) -> list[str]:
    if not variables:
        return []
    return [", ".join(print_expression(v) for v in variables)]


def _shock_lines(block: GCNBlock) -> list[str]:
    distributions = {dist.parameter_name: dist for dist in block.shock_distributions}
    plain = [shock for shock in block.shocks if shock.name not in distributions]

    lines = _variable_list(plain)
    lines.extend(f"{name}[] ~ {_distribution_call(dist)}" for name, dist in distributions.items())
    return lines


def _distribution_call(dist: GCNDistribution) -> str:
    kwargs = ", ".join(f"{key}={value}" for key, value in dist.dist_kwargs.items())
    text = f"{dist.dist_name}({kwargs})"

    if dist.wrapper_name:
        wrapper_args = [text, *(f"{key}={value}" for key, value in dist.wrapper_kwargs.items())]
        text = f"{dist.wrapper_name}({', '.join(wrapper_args)})"

    if dist.initial_value is not None:
        text += f" = {dist.initial_value}"
    return text


def _print_calibration_item(item: GCNEquation | GCNDistribution) -> str:
    if isinstance(item, GCNDistribution):
        return print_distribution(item)
    return print_equation(item)


def _print_option_value(value: str | bool) -> str:
    if value is True:
        return "TRUE"
    if value is False:
        return "FALSE"
    return str(value)


def _group_assumptions(assumptions: dict[str, dict[str, bool]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for var_name, var_assumptions in assumptions.items():
        for assumption, holds in var_assumptions.items():
            if holds:
                groups[assumption].append(var_name)
    return groups
