from collections import defaultdict
from collections.abc import Callable
from typing import cast

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, merge_assumptions
from gEconpy.parser.ast import (
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNEquation,
    Node,
    Number,
    Operator,
    Parameter,
    UnaryOp,
    Variable,
)

SYMPY_FUNCTIONS: dict[str, Callable[..., sp.Expr]] = {
    "log": sp.log,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "sign": sp.sign,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
}

OPERATOR_MAP: dict[Operator, Callable[[sp.Expr, sp.Expr], sp.Expr]] = {
    Operator.ADD: lambda a, b: a + b,
    Operator.SUB: lambda a, b: a - b,
    Operator.MUL: lambda a, b: a * b,
    Operator.DIV: lambda a, b: a / b,
    Operator.POW: lambda a, b: a**b,
}


class ASTToSympyConverter:
    """
    Convert AST nodes to SymPy expressions.

    Parameters
    ----------
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per variable or parameter name, as in ``{"C": {"positive": True, "real": True}}``.
        Defaults to no assumptions.
    """

    def __init__(self, assumptions: dict[str, dict[str, bool]] | None = None):
        self.assumptions = assumptions or defaultdict(dict)

    def convert(self, node: Node) -> sp.Basic:  # noqa: PLR0911
        """
        Convert an AST node to a SymPy expression.

        Parameters
        ----------
        node : Node
            An expression node or a :class:`~gEconpy.parser.ast.nodes.GCNEquation`.

        Returns
        -------
        expr : sp.Basic
            The SymPy expression, or an :class:`~sympy.core.relational.Equality` for an equation node.
        """
        match node:
            case Number():
                return self._convert_number(node)
            case Parameter():
                return self._convert_parameter(node)
            case Variable():
                return self._convert_variable(node)
            case BinaryOp():
                return self._convert_binary_op(node)
            case UnaryOp():
                return self._convert_unary_op(node)
            case FunctionCall():
                return self._convert_function_call(node)
            case Expectation():
                return self._convert_expectation(node)
            case GCNEquation():
                return self._convert_equation(node)
            case _:
                raise TypeError(f"Cannot convert {type(node).__name__} to SymPy. Pass an expression or equation node.")

    def convert_expr(self, node: Node) -> sp.Expr:
        """
        Convert an expression node to a SymPy expression, rejecting equation nodes.

        Parameters
        ----------
        node : Node
            An expression node.

        Returns
        -------
        expr : sp.Expr
            The SymPy expression.
        """
        if isinstance(node, GCNEquation):
            raise TypeError("Expected an expression node, got a GCNEquation. Use convert() for equations.")
        return cast(sp.Expr, self.convert(node))

    def _convert_number(self, node: Number) -> sp.Number:
        if node.value == int(node.value):
            return sp.Integer(int(node.value))
        return sp.Float(node.value)

    def _convert_parameter(self, node: Parameter) -> sp.Symbol:
        return sp.Symbol(node.name, **merge_assumptions(self.assumptions.get(node.name)))

    def _convert_variable(self, node: Variable) -> TimeAwareSymbol:
        return TimeAwareSymbol(node.name, node.time_index.value, **merge_assumptions(self.assumptions.get(node.name)))

    def _convert_binary_op(self, node: BinaryOp) -> sp.Expr:
        return OPERATOR_MAP[node.op](self.convert_expr(node.left), self.convert_expr(node.right))

    def _convert_unary_op(self, node: UnaryOp) -> sp.Expr:
        if node.op != Operator.NEG:
            raise ValueError(f"Unknown unary operator: {node.op}")
        return -self.convert_expr(node.operand)

    def _convert_function_call(self, node: FunctionCall) -> sp.Expr:
        func = SYMPY_FUNCTIONS.get(node.func_name)
        if func is None:
            raise ValueError(
                f"Unknown function '{node.func_name}'. Supported functions are: {', '.join(SYMPY_FUNCTIONS)}"
            )
        return func(*(self.convert_expr(arg) for arg in node.args))

    def _convert_expectation(self, node: Expectation) -> sp.Basic:
        # First-order perturbation is certainty equivalent, so the expectation operator is dropped and its argument
        # converted directly.
        return self.convert(node.expr)

    def _convert_equation(self, node: GCNEquation) -> sp.Eq:
        return _checked_eq(self.convert_expr(node.lhs), self.convert_expr(node.rhs), self.assumptions)


def ast_to_sympy(node: Node, assumptions: dict[str, dict[str, bool]] | None = None) -> sp.Basic:
    """
    Convert an AST node to a SymPy expression.

    Parameters
    ----------
    node : Node
        An expression node or a :class:`~gEconpy.parser.ast.nodes.GCNEquation`.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per variable or parameter name. Defaults to no assumptions.

    Returns
    -------
    expr : sp.Basic
        The SymPy expression, or an :class:`~sympy.core.relational.Equality` for an equation node.
    """
    return ASTToSympyConverter(assumptions).convert(node)


def _checked_eq(lhs: sp.Expr, rhs: sp.Expr, assumptions: dict[str, dict[str, bool]]) -> sp.Eq:
    """
    Build an equation, rejecting one SymPy settles from the declared assumptions alone.

    Parameters
    ----------
    lhs, rhs : sympy expression
        The two sides of the equation.
    assumptions : dict mapping str to dict
        SymPy assumptions per symbol name.

    Returns
    -------
    equation : sp.Eq
        The equation, when SymPy leaves it standing.

    Raises
    ------
    ValueError
        If SymPy settles the equation to a boolean, which drops both sides.
    """
    equation = sp.Eq(lhs, rhs)

    # A settled equation evaluates to a boolean. This is the last point where the symbols are still
    # available to name in the error.
    if equation is sp.true or equation is sp.false:
        raise ValueError(_contradictory_equation_message(lhs, rhs, equation, assumptions))

    return cast(sp.Eq, equation)


def _contradictory_equation_message(
    lhs: sp.Expr,
    rhs: sp.Expr,
    verdict: sp.logic.boolalg.BooleanAtom,
    assumptions: dict[str, dict[str, bool]],
) -> str:
    """
    Explain an equation that SymPy settled from the declared assumptions alone.

    Parameters
    ----------
    lhs, rhs : sympy expression
        The two sides of the equation, before SymPy collapsed it.
    verdict : sympy BooleanAtom
        What the equation evaluated to.
    assumptions : dict mapping str to dict
        SymPy assumptions per symbol name.

    Returns
    -------
    message : str
        Error message naming the equation and the assumptions that decided it.
    """
    outcome = "always true" if verdict else "impossible"

    declared = []
    for name in sorted({str(symbol) for symbol in lhs.free_symbols | rhs.free_symbols}):
        flags = sorted(flag for flag, holds in assumptions.get(name, {}).items() if holds)
        if flags:
            declared.append(f"{name} ({', '.join(flags)})")
    if declared:
        detail = f" Declared assumptions: {'; '.join(declared)}."
        advice = "Either change the value or drop the conflicting assumption."
    else:
        detail = ""
        advice = "Remove it or correct the value."

    return f"The equation '{lhs} = {rhs}' is {outcome}, so it carries no information.{detail} {advice}"
