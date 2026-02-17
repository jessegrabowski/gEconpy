import pytest

from pyparsing import ParseBaseException, ParseException

from gEconpy.parser.ast import (
    T_MINUS_1,
    BinaryOp,
    Expectation,
    Number,
    Operator,
    Parameter,
    T,
    Tag,
    Variable,
)
from gEconpy.parser.errors import GCNGrammarError
from gEconpy.parser.grammar import parse_equation


class TestSimpleEquations:
    def test_variable_equals_expression(self):
        eq = parse_equation("Y[] = C[] + I[];")
        assert eq.lhs == Variable(name="Y")
        assert isinstance(eq.rhs, BinaryOp)
        assert eq.rhs.op == Operator.ADD
        assert not eq.is_calibrating
        assert not eq.has_lagrange_multiplier

    def test_definition_equation(self):
        eq = parse_equation("u[] = log(C[]) - Theta * L[];")
        assert eq.lhs == Variable(name="u")
        assert isinstance(eq.rhs, BinaryOp)

    def test_with_trailing_semicolon(self):
        eq = parse_equation("Y[] = C[];")
        assert eq.lhs == Variable(name="Y")

    def test_bellman_equation(self):
        eq = parse_equation("U[] = u[] + beta * E[][U[1]];")
        assert eq.lhs == Variable(name="U")
        assert isinstance(eq.rhs, BinaryOp)


class TestLagrangeMultipliers:
    def test_budget_constraint_with_lambda(self):
        eq = parse_equation("C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];")
        assert eq.has_lagrange_multiplier
        assert eq.lagrange_multiplier == "lambda"
        assert isinstance(eq.lhs, BinaryOp)

    def test_production_constraint_with_mc(self):
        eq = parse_equation("Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[];")
        assert eq.lagrange_multiplier == "mc"

    def test_capital_accumulation_with_q(self):
        eq = parse_equation("K[] = (1 - delta) * K[-1] + I[] : q[];")
        assert eq.lagrange_multiplier == "q"


class TestCalibratingEquations:
    def test_simple_calibration(self):
        eq = parse_equation("beta = 0.99 -> beta;")
        assert eq.is_calibrating
        assert eq.calibrating_parameter == "beta"
        assert eq.lhs == Parameter(name="beta")
        assert eq.rhs == Number(value=0.99)

    def test_calibration_from_steady_state(self):
        eq = parse_equation("r[ss] = 0.04 -> delta;")
        assert eq.is_calibrating
        assert eq.calibrating_parameter == "delta"


class TestComplexEquations:
    def test_nk_wage_setting(self):
        # From full_nk.gcn
        eq = parse_equation("LHS_w[] = 1 / (1 + psi_w) * w_star[] * lambda[] * L_d_star[];")
        assert eq.lhs == Variable(name="LHS_w")

    def test_constraint_with_complex_rhs(self):
        eq = parse_equation("C[] + I[] + B[] / r_G[] = r[] * K[-1] + w[] * L[] + B[-1] / pi[] + Div[] : lambda[];")
        assert eq.has_lagrange_multiplier
        assert eq.lagrange_multiplier == "lambda"

    def test_investment_adjustment_cost(self):
        eq = parse_equation("K[] = (1 - delta) * K[-1] + I[] * (1 - gamma_I / 2 * (I[] / I[-1] - 1) ^ 2) : q[];")
        assert eq.lagrange_multiplier == "q"

    def test_ar1_process(self):
        eq = parse_equation("log(A[]) = rho_A * log(A[-1]) + epsilon_A[];")
        assert eq.lhs.func_name == "log"

    def test_steady_state_identity(self):
        eq = parse_equation("K[ss] = alpha * Y[ss] * mc[ss] / r[ss];")
        assert eq.lhs.time_index.is_steady_state


class TestEdgeCases:
    def test_equation_with_spaces_around_colon(self):
        eq = parse_equation("Y[] = C[]   :   lambda[];")
        assert eq.lagrange_multiplier == "lambda"

    def test_equation_with_spaces_around_arrow(self):
        eq = parse_equation("beta = 0.99   ->   beta;")
        assert eq.calibrating_parameter == "beta"

    def test_negative_rhs(self):
        eq = parse_equation("TC[] = -(r[] * K[-1] + w[] * L[]);")
        assert eq.lhs == Variable(name="TC")

    def test_equation_with_expectation_and_lagrange(self):
        # Edge case: expectation in constraint
        eq = parse_equation("V[] = u[] + beta * E[][V[1]] : mu[];")
        assert eq.has_lagrange_multiplier
        assert eq.lagrange_multiplier == "mu"

    def test_brackets_in_expression_dont_confuse_equals(self):
        # The = should be found correctly despite brackets in the expression
        eq = parse_equation("Y[] = A[] * K[-1] ^ alpha;")
        assert eq.lhs == Variable(name="Y")

    def test_multiple_time_indices_in_equation(self):
        eq = parse_equation("K[] = (1 - delta) * K[-1] + I[-1];")
        assert eq.lhs == Variable(name="K", time_index=T)
        assert not eq.has_lagrange_multiplier

    def test_very_long_equation(self):
        # Stress test with a long equation
        eq = parse_equation(
            "Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) + "
            "B[] * M[-1] ^ beta * N[] ^ (1 - beta) + "
            "C[] * P[-1] ^ gamma * Q[] ^ (1 - gamma);"
        )
        assert eq.lhs == Variable(name="Y")

    def test_nested_parentheses_deeply(self):
        eq = parse_equation("Y[] = ((((C[] + I[]))));")
        assert eq.lhs == Variable(name="Y")

    def test_whitespace_variations(self):
        # Tabs, multiple spaces, newlines within equation
        eq = parse_equation("Y[]\t=\t\tC[]  +   I[];")
        assert eq.lhs == Variable(name="Y")

    def test_underscore_in_variable_names(self):
        eq = parse_equation("w_star[] = w[] * (1 + markup);")
        assert eq.lhs == Variable(name="w_star")

    def test_numeric_suffix_in_variable_names(self):
        eq = parse_equation("K1[] = K2[] + K3[];")
        assert eq.lhs == Variable(name="K1")

    def test_lagrange_multiplier_with_underscore(self):
        eq = parse_equation("Y[] = C[] : lambda_1[];")
        assert eq.lagrange_multiplier == "lambda_1"

    def test_calibrating_param_with_underscore(self):
        eq = parse_equation("r[ss] = 0.04 -> r_star;")
        assert eq.calibrating_parameter == "r_star"


class TestErrorCases:
    def test_missing_equals_raises(self):
        with pytest.raises((ParseException, ParseBaseException)):
            parse_equation("Y[] + C[];")

    def test_empty_string_raises(self):
        with pytest.raises(ParseException):
            parse_equation("")

    def test_only_semicolon_raises(self):
        with pytest.raises(ParseException):
            parse_equation(";")

    def test_malformed_variable_raises(self):
        with pytest.raises(ParseBaseException):
            parse_equation("Y[ = C[];")

    def test_unclosed_parenthesis_raises(self):
        with pytest.raises(ParseBaseException, match="Expected '\\)'"):
            parse_equation("Y[] = (C[] + I[];")

    def test_invalid_time_index_raises(self):
        with pytest.raises(ParseBaseException, match="Invalid time index"):
            parse_equation("Y[abc] = C[];")


class TestAmbiguousCases:
    def test_arrow_in_expression_vs_calibrating(self):
        # -> should only be treated as calibrating marker at top level
        # This tests that we find the RIGHT arrow
        eq = parse_equation("x = 0.5 -> x;")
        assert eq.is_calibrating
        assert eq.calibrating_parameter == "x"

    def test_equation_with_only_parameters(self):
        eq = parse_equation("alpha = beta * gamma;")
        assert eq.lhs == Parameter(name="alpha")
        assert isinstance(eq.rhs, BinaryOp)

    def test_equation_with_only_numbers(self):
        eq = parse_equation("x = 1 + 2 * 3;")
        assert eq.lhs == Parameter(name="x")

    def test_function_on_lhs(self):
        # log(X[]) = ... is valid
        eq = parse_equation("log(A[]) = rho * log(A[-1]);")
        assert eq.lhs.func_name == "log"

    def test_complex_lhs(self):
        # LHS can be an expression, not just a variable
        eq = parse_equation("C[] + I[] = Y[];")
        assert isinstance(eq.lhs, BinaryOp)
        assert eq.rhs == Variable(name="Y")

    def test_lagrange_looks_like_variable_in_rhs(self):
        # Make sure lambda[] in RHS isn't confused with Lagrange marker
        eq = parse_equation("C[] = lambda[] * w[];")
        assert not eq.has_lagrange_multiplier
        assert "lambda" not in str(eq.lagrange_multiplier or "")

    def test_steady_state_all_around(self):
        eq = parse_equation("C[ss] + I[ss] = Y[ss] : lambda[ss];")
        # Note: Lagrange multipliers are typically written as var[] not var[ss]
        # This tests that we handle the ss case if someone writes it
        # The current implementation expects lambda[] format
        assert eq.lhs.left.time_index.is_steady_state

    def test_double_equals_raises(self):
        # Two equals signs in one equation is invalid GCN syntax
        with pytest.raises(ParseException):
            parse_equation("Y[] = C[] = I[];")

    def test_arrow_like_minus_greater(self):
        # "- >" with space should NOT be arrow
        eq = parse_equation("Y[] = C[] - 1;")
        assert not eq.is_calibrating

    def test_exponent_with_negative(self):
        eq = parse_equation("Y[] = K[] ^ (-1);")
        assert eq.lhs == Variable(name="Y")

    def test_scientific_notation_in_calibration(self):
        eq = parse_equation("tiny = 1e-10 -> tiny;")
        assert eq.is_calibrating
        assert eq.rhs.value == 1e-10

    def test_division_chain(self):
        eq = parse_equation("Y[] = A[] / B[] / C[];")
        # Should be left-associative: (A / B) / C
        assert eq.rhs.op == Operator.DIV
        assert eq.rhs.left.op == Operator.DIV


class TestEquationTags:
    def test_exclude_tag(self):
        eq = parse_equation("@exclude C[] = Y[];")
        assert eq.has_tag(Tag.EXCLUDE)
        assert eq.is_excluded
        assert eq.lhs == Variable(name="C")

    def test_exclude_with_lagrange(self):
        eq = parse_equation("@exclude C[] = w[] * L[] : lambda[];")
        assert eq.is_excluded
        assert eq.has_lagrange_multiplier
        assert eq.lagrange_multiplier == "lambda"

    def test_exclude_with_whitespace(self):
        eq = parse_equation("  @exclude   C[] = Y[]  ;")
        assert eq.is_excluded

    def test_exclude_on_separate_line(self):
        eq = parse_equation("@exclude\nC[] = Y[];")
        assert eq.is_excluded
        assert eq.lhs == Variable(name="C")

    def test_no_tags_by_default(self):
        eq = parse_equation("Y[] = C[];")
        assert eq.tags == frozenset()
        assert not eq.is_excluded

    def test_unknown_tag_raises(self):
        with pytest.raises(ParseBaseException, match="Unknown tag"):
            parse_equation("@unknown Y[] = C[];")

    def test_tag_case_insensitive(self):
        eq = parse_equation("@EXCLUDE Y[] = C[];")
        assert eq.is_excluded

    def test_with_tags_method(self):
        eq = parse_equation("Y[] = C[];")
        tagged = eq.with_tags(frozenset([Tag.EXCLUDE]))
        assert tagged.is_excluded
        assert not eq.is_excluded  # Original unchanged
