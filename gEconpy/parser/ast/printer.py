from gEconpy.parser.ast import (
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Node,
    Number,
    Operator,
    Parameter,
    UnaryOp,
    Variable,
)

OPERATOR_SYMBOLS = {
    Operator.ADD: "+",
    Operator.SUB: "-",
    Operator.MUL: "*",
    Operator.DIV: "/",
    Operator.POW: "^",
    Operator.NEG: "-",
}

PRECEDENCE = {
    Operator.ADD: 1,
    Operator.SUB: 1,
    Operator.MUL: 2,
    Operator.DIV: 2,
    Operator.POW: 3,
    Operator.NEG: 4,
}


def print_expression(node: Node, parent_precedence: int = 0) -> str:  # noqa: PLR0911
    """
    Convert an AST expression node to a string.

    Parameters
    ----------
    node : Node
        The expression node to print.
    parent_precedence : int
        Precedence of the parent operator (for parenthesization).

    Returns
    -------
    str
        String representation of the expression.
    """
    match node:
        case Number(value=value):
            if value == int(value):
                return str(int(value))
            return str(value)

        case Parameter(name=name):
            return name

        case Variable(name=name, time_index=time_index):
            return f"{name}{time_index}"

        case UnaryOp(op=op, operand=operand):
            operand_str = print_expression(operand, PRECEDENCE[op])
            return f"{OPERATOR_SYMBOLS[op]}{operand_str}"

        case BinaryOp(left=left, op=op, right=right):
            my_precedence = PRECEDENCE[op]
            left_str = print_expression(left, my_precedence)
            right_str = print_expression(right, my_precedence)

            # Right-associativity for power
            if op == Operator.POW:
                right_str = print_expression(right, my_precedence - 1)

            result = f"{left_str} {OPERATOR_SYMBOLS[op]} {right_str}"

            if my_precedence < parent_precedence:
                return f"({result})"
            return result

        case FunctionCall(func_name=func_name, args=args):
            args_str = ", ".join(print_expression(arg) for arg in args)
            return f"{func_name}({args_str})"

        case Expectation(expr=expr):
            return f"E[][{print_expression(expr)}]"

        case _:
            return str(node)


def print_equation(eq: GCNEquation) -> str:
    """
    Convert a GCNEquation to a string.

    Parameters
    ----------
    eq : GCNEquation
        The equation to print.

    Returns
    -------
    str
        String representation of the equation.
    """
    lhs_str = print_expression(eq.lhs)
    rhs_str = print_expression(eq.rhs)

    result = f"{lhs_str} = {rhs_str}"

    if eq.lagrange_multiplier:
        result += f" : {eq.lagrange_multiplier}[]"

    if eq.is_calibrating and eq.calibrating_parameter:
        result = f"{result} -> {eq.calibrating_parameter}"

    return result


def print_distribution(dist: GCNDistribution) -> str:
    """
    Convert a GCNDistribution to a string.

    Parameters
    ----------
    dist : GCNDistribution
        The distribution to print.

    Returns
    -------
    str
        String representation of the distribution.
    """
    # Build distribution kwargs string
    kwargs_str = ", ".join(f"{k}={v}" for k, v in dist.dist_kwargs.items())
    dist_str = f"{dist.dist_name}({kwargs_str})"

    # Wrap if needed
    if dist.wrapper_name:
        wrapper_kwargs_str = ", ".join(f"{k}={v}" for k, v in dist.wrapper_kwargs.items())
        if wrapper_kwargs_str:
            dist_str = f"{dist.wrapper_name}({dist_str}, {wrapper_kwargs_str})"
        else:
            dist_str = f"{dist.wrapper_name}({dist_str})"

    result = f"{dist.parameter_name} ~ {dist_str}"

    if dist.initial_value is not None:
        result += f" = {dist.initial_value}"

    return result


def print_block(block: GCNBlock, indent: str = "    ") -> str:
    """
    Convert a GCNBlock to a string.

    Parameters
    ----------
    block : GCNBlock
        The block to print.
    indent : str
        Indentation string.

    Returns
    -------
    str
        String representation of the block.
    """
    lines = [f"block {block.name}", "{"]

    # Definitions
    if block.definitions:
        lines.append(f"{indent}definitions")
        lines.append(f"{indent}{{")
        lines.extend(f"{indent}{indent}{print_equation(eq)};" for eq in block.definitions)
        lines.append(f"{indent}}};")
        lines.append("")

    # Controls
    if block.controls:
        lines.append(f"{indent}controls")
        lines.append(f"{indent}{{")
        controls_str = ", ".join(print_expression(v) for v in block.controls)
        lines.append(f"{indent}{indent}{controls_str};")
        lines.append(f"{indent}}};")
        lines.append("")

    # Objective
    if block.objective:
        lines.append(f"{indent}objective")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}{indent}{print_equation(block.objective)};")
        lines.append(f"{indent}}};")
        lines.append("")

    # Constraints
    if block.constraints:
        lines.append(f"{indent}constraints")
        lines.append(f"{indent}{{")
        lines.extend(f"{indent}{indent}{print_equation(eq)};" for eq in block.constraints)
        lines.append(f"{indent}}};")
        lines.append("")

    # Identities
    if block.identities:
        lines.append(f"{indent}identities")
        lines.append(f"{indent}{{")
        lines.extend(f"{indent}{indent}{print_equation(eq)};" for eq in block.identities)
        lines.append(f"{indent}}};")
        lines.append("")

    # Shocks
    if block.shocks:
        lines.append(f"{indent}shocks")
        lines.append(f"{indent}{{")
        shocks_str = ", ".join(print_expression(v) for v in block.shocks)
        lines.append(f"{indent}{indent}{shocks_str};")
        lines.append(f"{indent}}};")
        lines.append("")

    # Calibration
    if block.calibration:
        lines.append(f"{indent}calibration")
        lines.append(f"{indent}{{")
        for item in block.calibration:
            if isinstance(item, GCNDistribution):
                lines.append(f"{indent}{indent}{print_distribution(item)};")
            elif isinstance(item, GCNEquation):
                lines.append(f"{indent}{indent}{print_equation(item)};")
        lines.append(f"{indent}}};")
        lines.append("")

    lines.append("};")
    return "\n".join(lines)


def print_model(model: GCNModel, indent: str = "    ") -> str:
    """
    Convert a GCNModel to a string.

    Parameters
    ----------
    model : GCNModel
        The model to print.
    indent : str
        Indentation string.

    Returns
    -------
    str
        String representation of the model.
    """
    sections = []

    # Options
    if model.options:
        lines = ["options", "{"]
        for key, value in model.options.items():
            value_str = "TRUE" if value is True else "FALSE" if value is False else str(value)
            lines.append(f"{indent}{key} = {value_str};")
        lines.append("};")
        sections.append("\n".join(lines))

    # Tryreduce
    if model.tryreduce:
        lines = ["tryreduce", "{"]
        lines.append(f"{indent}{', '.join(model.tryreduce)};")
        lines.append("};")
        sections.append("\n".join(lines))

    # Assumptions
    if model.assumptions:
        # Group assumptions by type
        assumption_groups: dict[str, list[str]] = {}
        for var_name, assumptions in model.assumptions.items():
            for assumption, value in assumptions.items():
                if value:
                    if assumption not in assumption_groups:
                        assumption_groups[assumption] = []
                    assumption_groups[assumption].append(var_name)

        if assumption_groups:
            lines = ["assumptions", "{"]
            for assumption, var_list in assumption_groups.items():
                lines.append(f"{indent}{assumption}")
                lines.append(f"{indent}{{")
                lines.append(f"{indent}{indent}{', '.join(sorted(var_list))};")
                lines.append(f"{indent}}};")
            lines.append("};")
            sections.append("\n".join(lines))

    # Blocks
    sections.extend(print_block(block, indent) for block in model.blocks)

    return "\n\n".join(sections)


# Convenience aliases
print_ast = print_expression
