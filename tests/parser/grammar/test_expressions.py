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
)
from gEconpy.parser.errors import GCNGrammarError
from gEconpy.parser.grammar.expressions import parse_expression


class TestAtoms:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Integers
            ("42", 42.0),
            ("007", 7.0),
            # Floats
            ("3.14", 3.14),
            ("123.", 123.0),
            (".5", 0.5),
            (".123", 0.123),
            # Scientific notation
            ("1e10", 1e10),
            ("1E10", 1e10),
            ("1e+10", 1e10),
            ("1e-10", 1e-10),
            ("1.5e10", 1.5e10),
            (".5e10", 0.5e10),
            ("123.e10", 123e10),
            # Negative (handled by unary minus)
            ("-5", -5.0),
        ],
    )
    def test_numbers(self, text, expected):
        result = parse_expression(text)
        if isinstance(result, UnaryOp):
            assert result.op == Operator.NEG
        else:
            assert isinstance(result, Number)
            assert result.value == expected

    def test_number_rejects_partial_match(self):
        # 123abc should not parse as number 123 followed by garbage
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
        # a + b * c should parse as a + (b * c)
        result = parse_expression("a + b * c")
        assert result.op == Operator.ADD
        assert result.right.op == Operator.MUL

    def test_exponentiation_before_multiplication(self):
        # a * b ^ c should parse as a * (b ^ c)
        result = parse_expression("a * b ^ c")
        assert result.op == Operator.MUL
        assert result.right.op == Operator.POW

    def test_parentheses_override_precedence(self):
        # (a + b) * c should parse as (a + b) * c
        result = parse_expression("(a + b) * c")
        assert result.op == Operator.MUL
        assert result.left.op == Operator.ADD

    def test_exponentiation_right_associative(self):
        # a ^ b ^ c should parse as a ^ (b ^ c)
        result = parse_expression("a ^ b ^ c")
        assert result.op == Operator.POW
        assert result.right.op == Operator.POW
        assert result.right.left == Parameter(name="b")

    def test_double_star_right_associative(self):
        # a ** b ** c should parse as a ** (b ** c)
        result = parse_expression("a ** b ** c")
        assert result.op == Operator.POW
        assert result.right.op == Operator.POW

    def test_mixed_exponentiation_operators(self):
        # Can mix ^ and **
        result = parse_expression("a ^ b ** c")
        assert result.op == Operator.POW
        assert result.right.op == Operator.POW

    def test_addition_left_associative(self):
        # a + b + c should parse as (a + b) + c
        result = parse_expression("a + b + c")
        assert result.op == Operator.ADD
        assert result.left.op == Operator.ADD
        assert result.left.left == Parameter(name="a")

    def test_multiplication_left_associative(self):
        # a * b * c should parse as (a * b) * c
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

    def test_function_with_no_args_raises(self):
        with pytest.raises(GCNGrammarError):
            parse_expression("func()")


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
        # u[] + beta * E[][U[1]]
        result = parse_expression("u[] + beta * E[][U[1]]")
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.ADD
        assert result.left == Variable(name="u")
        assert result.right.op == Operator.MUL
        assert isinstance(result.right.right, Expectation)

    def test_production_function(self):
        # A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)
        result = parse_expression("A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)")
        assert isinstance(result, BinaryOp)

    def test_nested_function_calls(self):
        result = parse_expression("log(exp(x))")
        assert isinstance(result, FunctionCall)
        assert result.func_name == "log"
        assert isinstance(result.args[0], FunctionCall)
        assert result.args[0].func_name == "exp"


class TestRealWorldExpressions:
    """Test expressions extracted from actual GCN model files."""

    def test_utility_with_habit_formation(self):
        # From full_nk.gcn - habit formation in consumption
        expr = "(C[] - phi_H * C[-1]) ^ (1 - sigma_C) / (1 - sigma_C)"
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.DIV

    def test_labor_disutility(self):
        expr = "L[] ^ (1 + sigma_L) / (1 + sigma_L)"
        result = parse_expression(expr)
        assert result.op == Operator.DIV

    def test_wage_from_full_nk(self):
        # Complex wage equation
        expr = "(1 - alpha) * mc[ss] ^ (1 / (1 - alpha)) * (alpha / r[ss]) ^ (alpha / (1 - alpha))"
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)

    def test_steady_state_output(self):
        # Multi-line expression from full_nk.gcn steady state
        expr = (
            "w[ss] ^ ((sigma_L + 1) / (sigma_C + sigma_L)) * "
            "(r[ss] / ((1 - phi_H) * (r[ss] - alpha * delta * mc[ss]))) ^ (sigma_C / (sigma_C + sigma_L))"
        )
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.MUL

    def test_capital_adjustment_cost(self):
        # From open_rbc.gcn
        expr = "psi / 2 * (K[] - K[-1]) ^ 2"
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)

    def test_nested_expectation_with_ratio(self):
        # From full_nk.gcn wage setting
        expr = "beta * eta_w * E[][pi[1] * (w_star[1] / w_star[]) ^ (1 / psi_w) * LHS_w[1]]"
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)

        # Find the expectation deep in the tree
        def find_expectation(node):
            if isinstance(node, Expectation):
                return node
            if isinstance(node, BinaryOp):
                return find_expectation(node.left) or find_expectation(node.right)
            return None

        assert find_expectation(result) is not None

    def test_deeply_nested_exponents(self):
        # From full_nk.gcn RHS_w equation
        expr = "(pi[1] * w_star[1] / w_star[]) ^ ((1 + psi_w) * (1 + sigma_L) / psi_w)"
        result = parse_expression(expr)
        assert result.op == Operator.POW

    def test_log_ar1_process(self):
        # Standard AR(1) in logs
        expr = "rho_A * log(A[-1]) + epsilon_A[]"
        result = parse_expression(expr)
        assert result.op == Operator.ADD
        assert result.left.op == Operator.MUL
        assert isinstance(result.left.right, FunctionCall)
        assert result.left.right.func_name == "log"

    def test_price_evolution_equation(self):
        # From full_nk.gcn
        expr = "eta_p * pi[] ^ (1 / psi_p) + (1 - eta_p) * pi_star[] ^ (-1 / psi_p)"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_interest_rate_with_exp(self):
        # From open_rbc.gcn
        expr = "rstar + psi2 * (exp(IIPbar - IIP[]) - 1)"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_monetary_policy_rule(self):
        # Taylor rule component
        expr = "gamma_R * log(r_G[-1] / r_G[ss]) + (1 - gamma_R) * gamma_pi * log(pi[] / pi[ss])"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_capital_labor_ratio_ss(self):
        # From rbc_firm_capital.gcn
        expr = "(alpha * beta * A[ss] / (1 - beta * (1 - delta))) ^ (1 / (1 - alpha))"
        result = parse_expression(expr)
        assert result.op == Operator.POW

    def test_recursive_firm_value(self):
        # Firm value with stochastic discount factor
        expr = "pi[] + beta * E[][lambda[1] / lambda[] * Pi[1]]"
        result = parse_expression(expr)
        assert result.op == Operator.ADD
        assert isinstance(result.right.right, Expectation)

    def test_investment_adjustment_cost(self):
        # From full_nk.gcn
        expr = "I[] * (1 - gamma_I / 2 * (I[] / I[-1] - 1) ^ 2)"
        result = parse_expression(expr)
        assert result.op == Operator.MUL

    def test_deeply_nested_parentheses(self):
        expr = "((((a + b))))"
        result = parse_expression(expr)
        assert result.op == Operator.ADD

    def test_chained_divisions(self):
        # a / b / c should be (a / b) / c (left associative)
        result = parse_expression("a / b / c")
        assert result.op == Operator.DIV
        assert result.left.op == Operator.DIV
        assert result.left.left == Parameter(name="a")

    def test_mixed_operators_complex(self):
        # Ensure correct parsing of complex mixed operators
        expr = "a + b * c ^ d / e - f"
        result = parse_expression(expr)
        # Should be: (a + ((b * (c ^ d)) / e)) - f
        assert result.op == Operator.SUB

    def test_unary_minus_in_exponent(self):
        expr = "x ^ (-1)"
        result = parse_expression(expr)
        assert result.op == Operator.POW
        assert isinstance(result.right, UnaryOp)
        assert result.right.op == Operator.NEG

    def test_negative_coefficient(self):
        expr = "-alpha * K[]"
        result = parse_expression(expr)
        # Could be UnaryOp(NEG, BinaryOp) or BinaryOp with UnaryOp on left
        assert isinstance(result, UnaryOp | BinaryOp)

    def test_subtraction_vs_negative(self):
        # a - -b should parse correctly
        expr = "a - -b"
        result = parse_expression(expr)
        assert result.op == Operator.SUB
        assert isinstance(result.right, UnaryOp)

    def test_exponent_with_unary_minus_on_rhs(self):
        # x ^ -y should work (common in economics: C^-sigma)
        expr = "x ^ -y"
        result = parse_expression(expr)
        assert result.op == Operator.POW
        assert isinstance(result.right, UnaryOp)
        assert result.right.op == Operator.NEG
        assert result.right.operand == Parameter(name="y")

    def test_variable_to_negative_parameter(self):
        # C_R[ss] ^ -sigma_R from RBC_two_household_additive.gcn
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

    def test_negative_base_to_power(self):
        expr = "-x ^ 2"
        result = parse_expression(expr)
        assert isinstance(result, BinaryOp)
        assert result.op == Operator.POW
        assert isinstance(result.left, UnaryOp)
        assert result.left.op == Operator.NEG
