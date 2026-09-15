import pytest

from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    T_PLUS_1,
    BinaryOp,
    Expectation,
    FunctionCall,
    Number,
    Operator,
    Parameter,
    T,
    UnaryOp,
    Variable,
    collect_nodes_of_type,
)
from gEconpy.parser.ast.printer import print_expression
from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNGrammarError, GCNParseFailure
from gEconpy.parser.grammar.expressions import parse_expression


class TestAtoms:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("42", 42.0),
            ("007", 7.0),
            ("3.14", 3.14),
            ("123.", 123.0),
            (".5", 0.5),
            (".123", 0.123),
            ("1e10", 1e10),
            ("1E10", 1e10),
            ("1e+10", 1e10),
            ("1e-10", 1e-10),
            ("1.5e10", 1.5e10),
            (".5e10", 0.5e10),
            ("123.e10", 123e10),
        ],
    )
    def test_numbers(self, text, expected):
        result = parse_expression(text)
        assert isinstance(result, Number)
        assert result.value == expected

    def test_negative_number_is_unary_minus(self):
        result = parse_expression("-5")
        assert result == UnaryOp(op=Operator.NEG, operand=Number(value=5.0))

    def test_number_rejects_partial_match(self):
        with pytest.raises(GCNGrammarError):
            parse_expression("123abc")

    def test_parameter(self):
        result = parse_expression("alpha")
        assert result == Parameter(name="alpha")

    @pytest.mark.parametrize(
        "text,name,time",
        [
            ("C[]", "C", T),
            ("K[-1]", "K", T_MINUS_1),
            ("Y[1]", "Y", T_PLUS_1),
            ("A[ss]", "A", STEADY_STATE),
        ],
    )
    def test_variables(self, text, name, time):
        result = parse_expression(text)
        assert isinstance(result, Variable)
        assert result.name == name
        assert result.time_index == time


class TestBinaryOperators:
    def test_addition(self):
        result = parse_expression("C[] + I[]")
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.ADD
        assert result.left == Variable(name="C")
        assert result.right == Variable(name="I")

    def test_subtraction(self):
        result = parse_expression("Y[] - C[]")
        assert result.op == Operator.SUB

    def test_multiplication(self):
        result = parse_expression("alpha * K[]")
        assert result.op == Operator.MUL
        assert result.left == Parameter(name="alpha")

    def test_division(self):
        result = parse_expression("Y[] / L[]")
        assert result.op == Operator.DIV

    def test_exponentiation(self):
        result = parse_expression("K[] ^ alpha")
        assert result.op == Operator.POW

    def test_exponentiation_double_star(self):
        result = parse_expression("K[] ** alpha")
        assert result.op == Operator.POW
        assert result.left == Variable(name="K")
        assert result.right == Parameter(name="alpha")


class TestOperatorPrecedence:
    def test_multiplication_before_addition(self):
        result = parse_expression("a + b * c")
        assert result.op == Operator.ADD
        assert result.right.op == Operator.MUL

    def test_exponentiation_before_multiplication(self):
        result = parse_expression("a * b ^ c")
        assert result.op == Operator.MUL
        assert result.right.op == Operator.POW

    def test_parentheses_override_precedence(self):
        result = parse_expression("(a + b) * c")
        assert result.op == Operator.MUL
        assert result.left.op == Operator.ADD

    def test_exponentiation_right_associative(self):
        result = parse_expression("a ^ b ^ c")
        assert result.op == Operator.POW
        assert result.right.op == Operator.POW
        assert result.right.left == Parameter(name="b")

    def test_double_star_right_associative(self):
        result = parse_expression("a ** b ** c")
        assert result.op == Operator.POW
        assert result.right.op == Operator.POW

    def test_mixed_exponentiation_operators(self):
        result = parse_expression("a ^ b ** c")
        assert result.op == Operator.POW
        assert result.right.op == Operator.POW

    def test_addition_left_associative(self):
        result = parse_expression("a + b + c")
        assert result.op == Operator.ADD
        assert result.left.op == Operator.ADD
        assert result.left.left == Parameter(name="a")

    def test_multiplication_left_associative(self):
        result = parse_expression("a * b * c")
        assert result.op == Operator.MUL
        assert result.left.op == Operator.MUL
        assert result.left.left == Parameter(name="a")


class TestFunctionCalls:
    def test_single_arg_function(self):
        result = parse_expression("log(C[])")
        assert isinstance(result, FunctionCall)
        assert result.func_name == "log"
        assert len(result.args) == 1
        assert result.args[0] == Variable(name="C")

    def test_nested_expression_in_function(self):
        result = parse_expression("exp(alpha * K[])")
        assert isinstance(result, FunctionCall)
        assert result.func_name == "exp"
        assert isinstance(result.args[0], BinaryOp)

    @pytest.mark.parametrize(
        "text,code,found",
        [("log()", ErrorCode.E008, "log()"), ("Y[abc]", ErrorCode.E010, "[abc]")],
        ids=["empty_call", "bad_time_index"],
    )
    def test_encoded_failure_is_decoded(self, text, code, found):
        with pytest.raises(GCNGrammarError) as exc_info:
            parse_expression(text)

        error = exc_info.value
        assert error.code == code
        assert error.found == found
        assert GCNParseFailure.SEPARATOR not in error.message


class TestGrammarErrors:
    @pytest.mark.parametrize(
        "text, code, message, found, column",
        [
            ("a +", ErrorCode.E006, "Unexpected operator '+' at end of expression. Found '+'", "+", 3),
            ("a b", ErrorCode.E006, "Incomplete expression. Found 'b'", "b", 3),
            ("(a + b", ErrorCode.E007, "Unbalanced parentheses. Found 'end of text'", "end of text", 7),
            ("log(a", ErrorCode.E007, "Unbalanced parentheses. Found 'end of text'", "end of text", 6),
            ("Y[", ErrorCode.E010, "Invalid variable syntax. Found 'end of text'", "end of text", 3),
            ("a + b)", ErrorCode.E007, "Unbalanced parentheses. Found ')'", ")", 6),
        ],
        ids=[
            "trailing_operator",
            "juxtaposed_atoms",
            "unclosed_paren",
            "unclosed_call",
            "unclosed_bracket",
            "stray_paren",
        ],
    )
    def test_structural_failure_maps_to_catalog_code(self, text, code, message, found, column):
        with pytest.raises(GCNGrammarError) as exc_info:
            parse_expression(text, context="objective of HOUSEHOLD")

        error = exc_info.value
        assert error.code == code
        assert error.message == message
        assert error.found == found
        assert error.context == "objective of HOUSEHOLD"
        assert (error.location.line, error.location.column) == (1, column)
        assert error.location.source_line == text


class TestNodeLocations:
    def test_atoms_record_column_span(self):
        result = parse_expression("C[] + alpha")

        assert (result.left.location.column, result.left.location.end_column) == (1, 4)
        assert (result.right.location.column, result.right.location.end_column) == (7, 12)
        assert result.left.location.source_line == "C[] + alpha"

    @pytest.mark.parametrize("text", ["K[-1]", "K[ss]", "K[10]"])
    def test_variable_span_covers_exactly_its_text(self, text):
        location = parse_expression(text).location
        assert (location.column, location.end_column) == (1, 1 + len(text))

    def test_location_tracks_line_of_multiline_source(self):
        result = parse_expression("a +\n  K[]")
        assert (result.right.location.line, result.right.location.column) == (2, 3)


class TestExpectation:
    def test_simple_expectation(self):
        result = parse_expression("E[][U[1]]")
        assert isinstance(result, Expectation)
        assert result.expr == Variable(name="U", time_index=T_PLUS_1)

    def test_expectation_with_expression(self):
        result = parse_expression("E[][beta * U[1]]")
        assert isinstance(result, Expectation)
        assert isinstance(result.expr, BinaryOp)
        assert result.expr.left == Parameter(name="beta")


class TestComplexExpressions:
    def test_bellman_rhs(self):
        result = parse_expression("u[] + beta * E[][U[1]]")
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.ADD
        assert result.left == Variable(name="u")
        assert result.right.op == Operator.MUL
        assert isinstance(result.right.right, Expectation)

    def test_nested_function_calls(self):
        result = parse_expression("log(exp(x))")
        assert isinstance(result, FunctionCall)
        assert result.func_name == "log"
        assert isinstance(result.args[0], FunctionCall)
        assert result.args[0].func_name == "exp"


class TestRealWorldExpressions:
    @pytest.mark.parametrize(
        "expr",
        [
            "A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)",
            "(1 - alpha) * mc[ss] ^ (1 / (1 - alpha)) * (alpha / r[ss]) ^ (alpha / (1 - alpha))",
            "psi / 2 * (K[] - K[-1]) ^ 2",
            "I[] * (1 - gamma_I / 2 * (I[] / I[-1] - 1) ^ 2)",
            "eta_p * pi[] ^ (1 / psi_p) + (1 - eta_p) * pi_star[] ^ (-1 / psi_p)",
            "beta * eta_w * E[][pi[1] * (w_star[1] / w_star[]) ^ (1 / psi_w) * LHS_w[1]]",
        ],
        ids=["production", "nk_wage", "capital_adjustment", "investment_adjustment", "price_evolution", "wage_ratio"],
    )
    def test_canonical_source_prints_back_unchanged(self, expr):
        assert print_expression(parse_expression(expr)) == expr

    def test_utility_with_habit_formation(self):
        expr = "(C[] - phi_H * C[-1]) ^ (1 - sigma_C) / (1 - sigma_C)"
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.DIV

    def test_labor_disutility(self):
        expr = "L[] ^ (1 + sigma_L) / (1 + sigma_L)"
        result = parse_expression(expr)
        assert result.op == Operator.DIV

    def test_steady_state_output(self):
        expr = (
            "w[ss] ^ ((sigma_L + 1) / (sigma_C + sigma_L)) * "
            "(r[ss] / ((1 - phi_H) * (r[ss] - alpha * delta * mc[ss]))) ^ (sigma_C / (sigma_C + sigma_L))"
        )
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.MUL

    def test_nested_expectation_with_ratio(self):
        expr = "beta * eta_w * E[][pi[1] * (w_star[1] / w_star[]) ^ (1 / psi_w) * LHS_w[1]]"
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)
        assert len(collect_nodes_of_type(result, Expectation)) == 1

    def test_deeply_nested_exponents(self):
        expr = "(pi[1] * w_star[1] / w_star[]) ^ ((1 + psi_w) * (1 + sigma_L) / psi_w)"
        result = parse_expression(expr)
        assert result.op == Operator.POW

    def test_log_ar1_process(self):
        expr = "rho_A * log(A[-1]) + epsilon_A[]"
        result = parse_expression(expr)
        assert result.op == Operator.ADD
        assert result.left.op == Operator.MUL
        assert isinstance(result.left.right, FunctionCall)
        assert result.left.right.func_name == "log"

    def test_price_evolution_equation(self):
        expr = "eta_p * pi[] ^ (1 / psi_p) + (1 - eta_p) * pi_star[] ^ (-1 / psi_p)"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_interest_rate_with_exp(self):
        expr = "rstar + psi2 * (exp(IIPbar - IIP[]) - 1)"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_monetary_policy_rule(self):
        expr = "gamma_R * log(r_G[-1] / r_G[ss]) + (1 - gamma_R) * gamma_pi * log(pi[] / pi[ss])"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_capital_labor_ratio_ss(self):
        expr = "(alpha * beta * A[ss] / (1 - beta * (1 - delta))) ^ (1 / (1 - alpha))"
        result = parse_expression(expr)
        assert result.op == Operator.POW

    def test_recursive_firm_value(self):
        expr = "pi[] + beta * E[][lambda[1] / lambda[] * Pi[1]]"
        result = parse_expression(expr)
        assert result.op == Operator.ADD
        assert isinstance(result.right.right, Expectation)

    def test_investment_adjustment_cost(self):
        expr = "I[] * (1 - gamma_I / 2 * (I[] / I[-1] - 1) ^ 2)"
        result = parse_expression(expr)
        assert result.op == Operator.MUL

    def test_deeply_nested_parentheses(self):
        expr = "((((a + b))))"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_chained_divisions(self):
        result = parse_expression("a / b / c")
        assert result.op == Operator.DIV
        assert result.left.op == Operator.DIV
        assert result.left.left == Parameter(name="a")

    def test_mixed_operators_complex(self):
        expr = "a + b * c ^ d / e - f"
        result = parse_expression(expr)
        assert result.op == Operator.SUB
        assert result.left.op == Operator.ADD
        assert result.left.right.op == Operator.DIV
        assert result.left.right.left.op == Operator.MUL
        assert result.left.right.left.right.op == Operator.POW

    def test_unary_minus_in_exponent(self):
        expr = "x ^ (-1)"
        result = parse_expression(expr)
        assert result.op == Operator.POW
        assert isinstance(result.right, UnaryOp)
        assert result.right.op == Operator.NEG

    def test_negative_coefficient_binds_before_multiplication(self):
        expr = "-alpha * K[]"
        result = parse_expression(expr)
        assert result.op == Operator.MUL
        assert result.left == UnaryOp(op=Operator.NEG, operand=Parameter(name="alpha"))

    def test_subtraction_vs_negative(self):
        expr = "a - -b"
        result = parse_expression(expr)
        assert result.op == Operator.SUB
        assert isinstance(result.right, UnaryOp)

    def test_exponent_with_unary_minus_on_rhs(self):
        expr = "x ^ -y"
        result = parse_expression(expr)
        assert result.op == Operator.POW
        assert isinstance(result.right, UnaryOp)
        assert result.right.op == Operator.NEG
        assert result.right.operand == Parameter(name="y")

    def test_variable_to_negative_parameter(self):
        expr = "C_R[ss] ^ -sigma_R"
        result = parse_expression(expr)
        assert result.op == Operator.POW
        assert result.left == Variable(name="C_R", time_index=STEADY_STATE)
        assert isinstance(result.right, UnaryOp)
        assert result.right.operand == Parameter(name="sigma_R")

    def test_power_with_negative_number(self):
        expr = "x ^ -2"
        result = parse_expression(expr)
        assert result.op == Operator.POW
        assert isinstance(result.right, UnaryOp)

    @pytest.mark.parametrize("expr", ["-x ^ 2", "-x ** 2", "--x ^ 2 ^ 3"], ids=["caret", "double_star", "nested"])
    def test_negative_base_to_power_negates_the_power(self, expr):
        result = parse_expression(expr)
        assert isinstance(result, UnaryOp)

        power = result.operand
        while isinstance(power, UnaryOp):
            power = power.operand
        assert isinstance(power, BinaryOp)
        assert power.op == Operator.POW
        assert power.left == Parameter(name="x")

    def test_parenthesized_negative_base_keeps_the_sign_inside(self):
        result = parse_expression("(-x) ^ 2")
        assert result == BinaryOp(
            left=UnaryOp(op=Operator.NEG, operand=Parameter(name="x")), op=Operator.POW, right=Number(value=2.0)
        )
