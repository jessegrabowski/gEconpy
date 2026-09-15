import pytest

from gEconpy.parser.grammar.expressions import parse_expression
from gEconpy.parser.transform.to_sympy import ast_to_sympy
from tests.conftest import parsed_var


class TestExpectationIsTransparent:
    """
    The expectation operator ``E[][]`` has no semantic effect during AST-to-sympy conversion.

    First-order perturbation is certainty equivalent, so replacing ``E_t[x_{t+1}]`` by ``x_{t+1}`` and letting the
    policy function supply the conditional expectation is exact at that order.
    """

    @pytest.mark.parametrize(
        ("with_expectation", "without_expectation"),
        [
            ("E[][x[1]]", "x[1]"),
            ("E[][beta * C[1] ^ (-sigma)]", "beta * C[1] ^ (-sigma)"),
            ("beta * E[][C[1] ^ (-sigma) * (1 + r[1])]", "beta * C[1] ^ (-sigma) * (1 + r[1])"),
        ],
    )
    def test_expectation_is_stripped(self, with_expectation, without_expectation):
        assert ast_to_sympy(parse_expression(with_expectation)) == ast_to_sympy(parse_expression(without_expectation))

    @pytest.mark.parametrize(("expression", "offset"), [("E[][x[-1]]", -1), ("E[][x[]]", 0)])
    def test_expectation_of_known_quantity_is_accepted_silently(self, expression, offset):
        assert ast_to_sympy(parse_expression(expression)) == parsed_var("x", offset)
