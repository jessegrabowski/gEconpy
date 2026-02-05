from collections import defaultdict
from collections.abc import Callable
from typing import Any

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
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

SYMPY_FUNCTIONS: dict[str, Callable] = {
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

OPERATOR_MAP = {
    Operator.ADD: lambda a, b: a + b,
    Operator.SUB: lambda a, b: a - b,
    Operator.MUL: lambda a, b: a * b,
    Operator.DIV: lambda a, b: a / b,
    Operator.POW: lambda a, b: a**b,
}


class ASTToSympyConverter:
    """
    Convert AST nodes to sympy expressions.

    Parameters
    ----------
    assumptions : dict, optional
        A dictionary mapping variable/parameter names to assumption dictionaries.
        For example: {"C": {"positive": True, "real": True}}
    """

    def __init__(self, assumptions: dict[str, dict[str, bool]] | None = None):
        self.assumptions = assumptions or defaultdict(dict)

    def convert(self, node: Node) -> sp.Basic:  # noqa: PLR0911
        """
        Convert an AST node to a sympy expression.

        Parameters
        ----------
        node : Node
            Any AST node (expression, equation, etc.)

        Returns
        -------
        sp.Basic
            The sympy representation.
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
                raise TypeError(f"Unknown node type: {type(node)}")

    def _convert_number(self, node: Number) -> sp.Number:
        if node.value == int(node.value):
            return sp.Integer(int(node.value))
        return sp.Float(node.value)

    def _convert_parameter(self, node: Parameter) -> sp.Symbol:
        assumptions = self.assumptions.get(node.name, {})
        return sp.Symbol(node.name, **assumptions)

    def _convert_variable(self, node: Variable) -> TimeAwareSymbol:
        assumptions = self.assumptions.get(node.name, {})
        time_index = node.time_index

        if time_index.is_steady_state:
            return TimeAwareSymbol(node.name, "ss", **assumptions)

        return TimeAwareSymbol(node.name, time_index.value, **assumptions)

    def _convert_binary_op(self, node: BinaryOp) -> sp.Basic:
        left = self.convert(node.left)
        right = self.convert(node.right)
        return OPERATOR_MAP[node.op](left, right)

    def _convert_unary_op(self, node: UnaryOp) -> sp.Basic:
        operand = self.convert(node.operand)
        if node.op == Operator.NEG:
            return -operand
        raise ValueError(f"Unknown unary operator: {node.op}")

    def _convert_function_call(self, node: FunctionCall) -> sp.Basic:
        func = SYMPY_FUNCTIONS.get(node.func_name)
        if func is None:
            raise ValueError(f"Unknown function: {node.func_name}")
        args = [self.convert(arg) for arg in node.args]
        return func(*args)

    def _convert_expectation(self, node: Expectation) -> sp.Basic:
        # For now, expectations are transparent - just return the inner expression
        # TODO: Implement proper expectation handling if needed
        return self.convert(node.expr)

    def _convert_equation(self, node: GCNEquation) -> sp.Eq:
        lhs = self.convert(node.lhs)
        rhs = self.convert(node.rhs)
        return sp.Eq(lhs, rhs)


def ast_to_sympy(node: Node, assumptions: dict[str, dict[str, bool]] | None = None) -> sp.Basic:
    """
    Convert an AST node to a sympy expression.

    Parameters
    ----------
    node : Node
        Any AST node.
    assumptions : dict, optional
        Variable/parameter assumptions.

    Returns
    -------
    sp.Basic
        The sympy representation.
    """
    converter = ASTToSympyConverter(assumptions)
    return converter.convert(node)


def equation_to_sympy(
    eq: GCNEquation, assumptions: dict[str, dict[str, bool]] | None = None
) -> tuple[sp.Eq, dict[str, Any]]:
    """
    Convert a GCNEquation to a sympy equation with metadata.

    Parameters
    ----------
    eq : GCNEquation
        The equation to convert.
    assumptions : dict, optional
        Variable/parameter assumptions.

    Returns
    -------
    tuple[sp.Eq, dict]
        The sympy equation and a metadata dictionary containing:
        - is_calibrating: bool
        - calibrating_parameter: sp.Symbol or None
        - lagrange_multiplier: TimeAwareSymbol or None
    """
    converter = ASTToSympyConverter(assumptions)
    sympy_eq = converter._convert_equation(eq)

    metadata = {
        "is_calibrating": eq.is_calibrating,
        "calibrating_parameter": None,
        "lagrange_multiplier": None,
    }

    if eq.is_calibrating and eq.calibrating_parameter:
        param_assumptions = (assumptions or {}).get(eq.calibrating_parameter, {})
        metadata["calibrating_parameter"] = sp.Symbol(eq.calibrating_parameter, **param_assumptions)
        # For calibrating equations, rearrange to: param = lhs - rhs
        sympy_eq = sp.Eq(metadata["calibrating_parameter"], sympy_eq.lhs - sympy_eq.rhs)

    if eq.lagrange_multiplier:
        mult_assumptions = (assumptions or {}).get(eq.lagrange_multiplier, {})
        metadata["lagrange_multiplier"] = TimeAwareSymbol(eq.lagrange_multiplier, 0, **mult_assumptions)

    return sympy_eq, metadata


def block_to_sympy(
    block: GCNBlock, assumptions: dict[str, dict[str, bool]] | None = None
) -> dict[str, list[tuple[sp.Eq, dict[str, Any]]]]:
    """
    Convert all equations in a block to sympy.

    Parameters
    ----------
    block : GCNBlock
        The block to convert.
    assumptions : dict, optional
        Variable/parameter assumptions.

    Returns
    -------
    dict
        A dictionary with keys for each component type containing lists of
        (equation, metadata) tuples.
    """
    result = {
        "definitions": [],
        "objective": None,
        "constraints": [],
        "identities": [],
        "calibration": [],
    }

    for eq in block.definitions:
        result["definitions"].append(equation_to_sympy(eq, assumptions))

    if block.objective:
        result["objective"] = equation_to_sympy(block.objective, assumptions)

    for eq in block.constraints:
        result["constraints"].append(equation_to_sympy(eq, assumptions))

    for eq in block.identities:
        result["identities"].append(equation_to_sympy(eq, assumptions))

    for item in block.calibration:
        if isinstance(item, GCNEquation):
            result["calibration"].append(equation_to_sympy(item, assumptions))
        # GCNDistribution items are kept as-is (handled separately)

    return result


def model_to_sympy(
    model: GCNModel,
) -> dict[str, dict[str, list[tuple[sp.Eq, dict[str, Any]]]]]:
    """
    Convert all equations in a model to sympy.

    Parameters
    ----------
    model : GCNModel
        The model to convert.

    Returns
    -------
    dict
        A dictionary mapping block names to their converted equations.
    """
    assumptions = model.assumptions
    result = {}

    for block in model.blocks:
        result[block.name] = block_to_sympy(block, assumptions)

    return result
