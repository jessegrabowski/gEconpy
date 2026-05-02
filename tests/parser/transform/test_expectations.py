"""
Tests documenting that the expectations operator E[][] is currently a no-op.

The expectation operator is stripped during AST-to-sympy conversion (see
``_convert_expectation`` in ``to_sympy.py``). For first-order perturbation this
is correct: the solution method implicitly replaces forward-looking variables
with their conditional expectations via the policy function x_{t+1} = T x_t.

These tests document cases where a proper expectations implementation would
produce different results, serving as a regression suite for when E[][] is
eventually given semantic meaning.

Cases where E[][] matters:
1. Different information sets: E_{t-1}[x_{t+1}] ≠ E_t[x_{t+1}]
   (predetermined pricing, sticky information models)
2. Second-order perturbation: E_t[f(x)] ≠ f(E_t[x]) due to Jensen's inequality
3. Validation: E_t[x_{t-1}] should equal x_{t-1} (known quantity), but there is
   no check that the argument of E[][] is actually uncertain.
"""

import numpy as np
import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.ast import (
    T_PLUS_1,
    BinaryOp,
    Expectation,
    Number,
    Operator,
    Parameter,
    T,
    TimeIndex,
    Variable,
)
from gEconpy.parser.grammar.expressions import parse_expression
from gEconpy.parser.transform.to_sympy import ast_to_sympy
from tests.conftest import parsed_symbol, parsed_symbols, parsed_var


class TestExpectationIsTransparent:
    """Document that E[][] currently has zero semantic effect."""

    def test_E_of_variable_equals_variable(self):
        """E[][x[1]] produces the same sympy expression as x[1]."""
        with_E = parse_expression("E[][x[1]]")
        without_E = parse_expression("x[1]")

        sym_with = ast_to_sympy(with_E)
        sym_without = ast_to_sympy(without_E)

        # This passes — the expectation is stripped
        assert sym_with == sym_without

    def test_E_of_expression_equals_expression(self):
        """E[][beta * C[1]^(-sigma)] produces the same result as beta * C[1]^(-sigma)."""
        with_E = parse_expression("E[][beta * C[1] ^ (-sigma)]")
        without_E = parse_expression("beta * C[1] ^ (-sigma)")

        sym_with = ast_to_sympy(with_E)
        sym_without = ast_to_sympy(without_E)

        assert sym_with == sym_without

    def test_euler_equation_identical_with_and_without_E(self):
        """
        The Euler equation gives the same sympy expression with and without E[][].

        ``C[]^(-sigma) = beta * E[][C[1]^(-sigma) * (1 + r[1])]``
        """
        with_E = parse_expression("beta * E[][C[1] ^ (-sigma) * (1 + r[1])]")
        without_E = parse_expression("beta * C[1] ^ (-sigma) * (1 + r[1])")

        sym_with = ast_to_sympy(with_E)
        sym_without = ast_to_sympy(without_E)

        assert sym_with == sym_without


class TestExpectationDoesNotValidateContent:
    """
    Document that E[][] does not check whether its argument is uncertain.

    A proper implementation should warn or error when the expectation is applied
    to known quantities (e.g., E_t[x_{t-1}] = x_{t-1}).
    """

    def test_E_of_lagged_variable_is_not_flagged(self):
        """
        E[][x[-1]] should be redundant, but gEconpy accepts it silently.

        x_{t-1} is known at time t, yet the E[][] is just stripped.
        """
        node = parse_expression("E[][x[-1]]")
        result = ast_to_sympy(node)

        # Currently: just strips E, gives x[-1]
        x_lag = parsed_var("x", -1)
        assert result == x_lag

    def test_E_of_current_variable_is_not_flagged(self):
        """
        E[][x[]] should be redundant, but gEconpy accepts it silently.

        x_t is known at time t, yet the E[][] is just stripped.
        """
        node = parse_expression("E[][x[]]")
        result = ast_to_sympy(node)

        x_now = parsed_var("x", 0)
        assert result == x_now


class TestExpectationShouldMatterForModels:
    """
    Demonstrate cases where a real expectations operator would change solutions.

    These tests are marked xfail because they document *desired future behavior*
    that the current transparent E[][] cannot support.
    """

    @pytest.mark.xfail(
        reason=(
            "E[][] is transparent: E[][x[1]^2] gives x[1]^2, but a proper "
            "implementation at second order should add a Jensen's inequality "
            "correction term E_t[x_{t+1}^2] = (E_t[x_{t+1}])^2 + Var_t(x_{t+1}). "
            "This only matters at order >= 2."
        ),
        strict=True,
    )
    def test_jensens_inequality_second_order(self):
        """
        At second order, E[x^2] ≠ (E[x])^2.

        For a model with the identity:
            y[] = E[][x[1]^2]

        A proper second-order expansion should produce:
            y_t = (E_t[x_{t+1}])^2 + Var_t(x_{t+1})

        But with E[][] stripped, we get:
            y_t = x_{t+1}^2

        which, when linearized at any order, is missing the variance correction.
        """
        # E[][x[1]^2]
        e_of_x_squared = ast_to_sympy(parse_expression("E[][x[1] ^ 2]"))
        # x[1]^2
        x_squared = ast_to_sympy(parse_expression("x[1] ^ 2"))

        # Currently these are identical (E is stripped). At second order they
        # should differ by the variance correction term.
        assert e_of_x_squared != x_squared

    @pytest.mark.xfail(
        reason=(
            "gEconpy has no syntax for different information sets. "
            "E_{t-1}[x_{t+1}] cannot be expressed — there is no E[-1][x[1]] syntax. "
            "This matters for predetermined pricing / sticky information models."
        ),
        strict=True,
    )
    def test_predetermined_pricing_different_information_set(self):
        """
        Predetermined-pricing models need E_{t-1}[...], which has no gEconpy syntax.

        In a sticky-information or predetermined-pricing model, firms set
        prices at t-1 based on E_{t-1}[...]. This means:

            pi_t = beta * E_{t-1}[pi_{t+1}] + kappa * E_{t-1}[x_t]

        The crucial difference from the standard NKPC:
            pi_t = beta * E_t[pi_{t+1}] + kappa * x_t

        is that E_{t-1}[x_t] ≠ x_t. The current-period output gap x_t
        includes surprise shocks that weren't known at t-1. Under
        predetermined pricing, pi_t should NOT respond to contemporaneous
        shocks — only to shocks that were anticipated at t-1.

        This test verifies that E_{t-1}[x[]] produces a *different* expression
        than E_t[x[]] = x[]. Currently fails because there is no E[-1][] syntax.
        """
        # The model needs E_{t-1}[x[]] to differ from x[]
        # With proper implementation, E_{t-1}[x_t] would become an auxiliary
        # variable (like Dynare's AUX_EXPECT_LAG_1), giving the equation a
        # different Jacobian structure.

        # Currently: E[][x[]] and x[] are identical
        e_x = ast_to_sympy(parse_expression("E[][x[]]"))
        x = ast_to_sympy(parse_expression("x[]"))

        # In a world with proper E[-1][x[]], these should differ
        # because E_{t-1}[x_t] is a predetermined variable that doesn't
        # respond to current-period shocks.
        # For now, even E_t[x_t] = x_t, so this test documents that
        # the expectation is a no-op.
        assert e_x != x  # Fails: they are identical

    @pytest.mark.xfail(
        reason=(
            "E[][] is transparent, so wrapping vs not wrapping an identity "
            "equation produces identical model solutions. With a proper "
            "implementation, E[][f(x[1])] in an identity should mark the "
            "equation for special handling during linearization."
        ),
        strict=True,
    )
    def test_identity_with_expectation_differs_from_without(self):
        """
        Wrapping an identity in E[][] should change the model, but currently does not.

        Consider two models with identical structure except for E[][]:

        Model A (with E):   P[] = E[][beta * D[1] / D[]]
        Model B (without):  P[] = beta * D[1] / D[]

        At first order these are the same (certainty equivalence). At second
        order they differ because E_t[D_{t+1}/D_t] ≠ E_t[D_{t+1}]/D_t when
        D_{t+1} and D_t are correlated.

        This test checks that the two formulations produce *different* sympy
        expressions, which they currently don't.
        """
        with_E = ast_to_sympy(parse_expression("E[][beta * D[1] / D[]]"))
        without_E = ast_to_sympy(parse_expression("beta * D[1] / D[]"))

        # Should differ: E[D(+1)/D] ≠ D(+1)/D in general
        # Currently identical because E is stripped
        assert with_E != without_E
