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


class TestExpectationShouldMatterForModels:
    """
    Cases where an expectations operator with semantics would change the converted expression.

    Each test is a strict xfail documenting behavior the transparent ``E[][]`` cannot provide.
    """

    @pytest.mark.xfail(
        reason=(
            "E[][] is transparent: E[][x[1]^2] gives x[1]^2, but a second-order expansion needs the Jensen "
            "correction E_t[x_{t+1}^2] = (E_t[x_{t+1}])^2 + Var_t(x_{t+1}). This only matters at order >= 2."
        ),
        strict=True,
    )
    def test_jensens_inequality_second_order(self):
        e_of_x_squared = ast_to_sympy(parse_expression("E[][x[1] ^ 2]"))
        x_squared = ast_to_sympy(parse_expression("x[1] ^ 2"))
        assert e_of_x_squared != x_squared

    @pytest.mark.xfail(
        reason=(
            "gEconpy has no syntax for different information sets. E_{t-1}[x_{t+1}] cannot be expressed because "
            "there is no E[-1][x[1]] syntax. This matters for predetermined pricing and sticky information models, "
            "where E_{t-1}[x_t] != x_t because x_t includes shocks unknown at t-1."
        ),
        strict=True,
    )
    def test_predetermined_pricing_different_information_set(self):
        e_x = ast_to_sympy(parse_expression("E[][x[]]"))
        x = ast_to_sympy(parse_expression("x[]"))
        assert e_x != x

    @pytest.mark.xfail(
        reason=(
            "E[][] is transparent, so P[] = E[][beta * D[1] / D[]] and P[] = beta * D[1] / D[] convert to the same "
            "expression. At second order they differ because E_t[D_{t+1} / D_t] != E_t[D_{t+1}] / D_t when D_{t+1} "
            "and D_t are correlated."
        ),
        strict=True,
    )
    def test_identity_with_expectation_differs_from_without(self):
        with_e = ast_to_sympy(parse_expression("E[][beta * D[1] / D[]]"))
        without_e = ast_to_sympy(parse_expression("beta * D[1] / D[]"))
        assert with_e != without_e
