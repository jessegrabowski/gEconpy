from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, merge_assumptions
from gEconpy.parser.ast import (
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
    GCNEquation,
    GCNModel,
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
        return cast(sp.Eq, sp.Eq(self.convert_expr(node.lhs), self.convert_expr(node.rhs)))


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


def equation_to_sympy(
    eq: GCNEquation, assumptions: dict[str, dict[str, bool]] | None = None
) -> tuple[sp.Eq, dict[str, Any]]:
    """
    Convert an equation to SymPy and extract its calibrating parameter and Lagrange multiplier.

    A calibrating equation ``lhs = rhs -> param`` is returned as ``Eq(param, lhs - rhs)``.

    Parameters
    ----------
    eq : GCNEquation
        The equation to convert.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per variable or parameter name. Defaults to no assumptions.

    Returns
    -------
    equation : sp.Eq
        The SymPy equation.
    metadata : dict
        The keys ``is_calibrating`` (bool), ``calibrating_parameter`` (:class:`~sympy.core.symbol.Symbol` or None), and
        ``lagrange_multiplier`` (:class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol` or None).
    """
    assumptions = assumptions or {}
    converter = ASTToSympyConverter(assumptions)
    lhs = converter.convert_expr(eq.lhs)
    rhs = converter.convert_expr(eq.rhs)
    sympy_eq = cast(sp.Eq, sp.Eq(lhs, rhs))

    metadata: dict[str, Any] = {
        "is_calibrating": eq.is_calibrating,
        "calibrating_parameter": None,
        "lagrange_multiplier": None,
    }

    if eq.calibrating_parameter:
        param_assumptions = merge_assumptions(assumptions.get(eq.calibrating_parameter))
        param = sp.Symbol(eq.calibrating_parameter, **param_assumptions)
        metadata["calibrating_parameter"] = param
        sympy_eq = cast(sp.Eq, sp.Eq(param, lhs - rhs))

    if eq.lagrange_multiplier:
        mult_assumptions = merge_assumptions(assumptions.get(eq.lagrange_multiplier))
        metadata["lagrange_multiplier"] = TimeAwareSymbol(eq.lagrange_multiplier, 0, **mult_assumptions)

    return sympy_eq, metadata


def block_to_sympy(
    block: GCNBlock, assumptions: dict[str, dict[str, bool]] | None = None
) -> dict[str, list[tuple[sp.Eq, dict[str, Any]]]]:
    """
    Convert every equation in a block with :func:`equation_to_sympy`, grouped by component.

    Parameters
    ----------
    block : GCNBlock
        The block to convert.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per variable or parameter name. Defaults to no assumptions.

    Returns
    -------
    equations : dict mapping str to list
        The keys ``definitions``, ``objective``, ``constraints``, ``identities``, and ``calibration``, each holding
        the ``(equation, metadata)`` pairs of that component. Distribution declarations in the calibration
        component are skipped.
    """
    return {
        "definitions": [equation_to_sympy(eq, assumptions) for eq in block.definitions],
        "objective": [equation_to_sympy(eq, assumptions) for eq in block.objective],
        "constraints": [equation_to_sympy(eq, assumptions) for eq in block.constraints],
        "identities": [equation_to_sympy(eq, assumptions) for eq in block.identities],
        "calibration": [
            equation_to_sympy(item, assumptions) for item in block.calibration if isinstance(item, GCNEquation)
        ],
    }


def model_to_sympy(model: GCNModel) -> dict[str, dict[str, list[tuple[sp.Eq, dict[str, Any]]]]]:
    """
    Convert every block in a model with :func:`~gEconpy.parser.transform.to_sympy.block_to_sympy`.

    The model's own assumptions are applied to every block.

    Parameters
    ----------
    model : GCNModel
        The model to convert.

    Returns
    -------
    equations : dict mapping str to dict
        The result of :func:`~gEconpy.parser.transform.to_sympy.block_to_sympy` for each block, keyed by block name.
    """
    return {block.name: block_to_sympy(block, model.assumptions) for block in model.blocks}
