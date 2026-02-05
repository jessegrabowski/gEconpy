import pytest

from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Number,
    Operator,
    Parameter,
    T,
    TimeIndex,
    UnaryOp,
    Variable,
)
from gEconpy.parser.ast.printer import (
    print_block,
    print_distribution,
    print_equation,
    print_expression,
    print_model,
)
from gEconpy.parser.grammar.expressions import parse_expression


class TestPrintExpressionAtoms:
    def test_print_integer(self):
        node = Number(value=42.0)
        assert print_expression(node) == "42"

    def test_print_float(self):
        node = Number(value=3.14)
        assert print_expression(node) == "3.14"

    def test_print_parameter(self):
        node = Parameter(name="alpha")
        assert print_expression(node) == "alpha"

    def test_print_variable_at_t(self):
        node = Variable(name="C", time_index=T)
        assert print_expression(node) == "C[]"

    def test_print_variable_lagged(self):
        node = Variable(name="K", time_index=T_MINUS_1)
        assert print_expression(node) == "K[-1]"

    def test_print_variable_lead(self):
        node = Variable(name="U", time_index=TimeIndex(1))
        assert print_expression(node) == "U[1]"

    def test_print_variable_steady_state(self):
        node = Variable(name="Y", time_index=STEADY_STATE)
        assert print_expression(node) == "Y[ss]"


class TestPrintExpressionOperations:
    def test_print_addition(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.ADD,
            right=Parameter(name="b"),
        )
        assert print_expression(node) == "a + b"

    def test_print_subtraction(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.SUB,
            right=Parameter(name="b"),
        )
        assert print_expression(node) == "a - b"

    def test_print_multiplication(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.MUL,
            right=Parameter(name="b"),
        )
        assert print_expression(node) == "a * b"

    def test_print_division(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.DIV,
            right=Parameter(name="b"),
        )
        assert print_expression(node) == "a / b"

    def test_print_power(self):
        node = BinaryOp(
            left=Parameter(name="K"),
            op=Operator.POW,
            right=Parameter(name="alpha"),
        )
        assert print_expression(node) == "K ^ alpha"

    def test_print_negation(self):
        node = UnaryOp(op=Operator.NEG, operand=Parameter(name="x"))
        assert print_expression(node) == "-x"

    def test_print_nested_needs_parens(self):
        # (a + b) * c needs parens around a + b
        node = BinaryOp(
            left=BinaryOp(
                left=Parameter(name="a"),
                op=Operator.ADD,
                right=Parameter(name="b"),
            ),
            op=Operator.MUL,
            right=Parameter(name="c"),
        )
        assert print_expression(node) == "(a + b) * c"

    def test_print_nested_no_parens_needed(self):
        # a * b + c doesn't need parens
        node = BinaryOp(
            left=BinaryOp(
                left=Parameter(name="a"),
                op=Operator.MUL,
                right=Parameter(name="b"),
            ),
            op=Operator.ADD,
            right=Parameter(name="c"),
        )
        assert print_expression(node) == "a * b + c"


class TestPrintExpressionFunctions:
    def test_print_log(self):
        node = FunctionCall(func_name="log", args=(Parameter(name="x"),))
        assert print_expression(node) == "log(x)"

    def test_print_exp(self):
        node = FunctionCall(func_name="exp", args=(Parameter(name="x"),))
        assert print_expression(node) == "exp(x)"

    def test_print_function_with_expression_arg(self):
        node = FunctionCall(
            func_name="log",
            args=(Variable(name="C"),),
        )
        assert print_expression(node) == "log(C[])"


class TestPrintExpectation:
    def test_print_expectation(self):
        node = Expectation(expr=Variable(name="U", time_index=TimeIndex(1)))
        assert print_expression(node) == "E[][U[1]]"


class TestPrintEquation:
    def test_simple_equation(self):
        eq = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=Variable(name="C"),
        )
        assert print_equation(eq) == "Y[] = C[]"

    def test_equation_with_lagrange(self):
        eq = GCNEquation(
            lhs=Variable(name="C"),
            rhs=Variable(name="Y"),
            lagrange_multiplier="lambda",
        )
        assert print_equation(eq) == "C[] = Y[] : lambda[]"

    def test_calibrating_equation(self):
        eq = GCNEquation(
            lhs=Parameter(name="beta"),
            rhs=Number(value=0.99),
            calibrating_parameter="beta",
        )
        result = print_equation(eq)
        assert "beta = 0.99" in result
        assert "-> beta" in result


class TestPrintDistribution:
    def test_simple_distribution(self):
        dist = GCNDistribution(
            parameter_name="alpha",
            dist_name="Beta",
            dist_kwargs={"alpha": 2, "beta": 5},
        )
        result = print_distribution(dist)
        assert "alpha ~ Beta(" in result
        assert "alpha=2" in result
        assert "beta=5" in result

    def test_distribution_with_initial_value(self):
        dist = GCNDistribution(
            parameter_name="alpha",
            dist_name="Beta",
            dist_kwargs={"alpha": 2, "beta": 5},
            initial_value=0.35,
        )
        result = print_distribution(dist)
        assert "= 0.35" in result

    def test_wrapped_distribution(self):
        dist = GCNDistribution(
            parameter_name="beta",
            dist_name="Beta",
            dist_kwargs={},
            wrapper_name="maxent",
            wrapper_kwargs={"lower": 0.95, "upper": 0.999},
            initial_value=0.99,
        )
        result = print_distribution(dist)
        assert "maxent(Beta()" in result
        assert "lower=0.95" in result
        assert "upper=0.999" in result


class TestPrintBlock:
    def test_simple_block(self):
        block = GCNBlock(name="TEST")
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=Variable(name="C"),
            )
        ]
        result = print_block(block)
        assert "block TEST" in result
        assert "identities" in result
        assert "Y[] = C[]" in result

    def test_block_with_controls(self):
        block = GCNBlock(name="HOUSEHOLD")
        block.controls = [Variable(name="C"), Variable(name="L")]
        result = print_block(block)
        assert "controls" in result
        assert "C[], L[]" in result

    def test_block_with_calibration(self):
        block = GCNBlock(name="TEST")
        block.calibration = [
            GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35)),
            GCNDistribution(
                parameter_name="beta",
                dist_name="Beta",
                dist_kwargs={"alpha": 2, "beta": 5},
            ),
        ]
        result = print_block(block)
        assert "calibration" in result
        assert "alpha = 0.35" in result
        assert "beta ~ Beta" in result


class TestPrintModel:
    def test_model_with_options(self):
        model = GCNModel(
            blocks=[],
            options={"output logfile": True, "output LaTeX": False},
            tryreduce=[],
            assumptions={},
        )
        result = print_model(model)
        assert "options" in result
        assert "output logfile = TRUE" in result
        assert "output LaTeX = FALSE" in result

    def test_model_with_tryreduce(self):
        model = GCNModel(
            blocks=[],
            options={},
            tryreduce=["U[]", "TC[]"],
            assumptions={},
        )
        result = print_model(model)
        assert "tryreduce" in result
        assert "U[], TC[]" in result

    def test_full_model(self):
        block = GCNBlock(name="HOUSEHOLD")
        block.controls = [Variable(name="C")]
        block.objective = GCNEquation(
            lhs=Variable(name="U"),
            rhs=Variable(name="u"),
        )
        block.constraints = [
            GCNEquation(
                lhs=Variable(name="C"),
                rhs=Variable(name="Y"),
                lagrange_multiplier="lambda",
            )
        ]

        model = GCNModel(
            blocks=[block],
            options={"output logfile": True},
            tryreduce=["U[]"],
            assumptions={},
        )
        result = print_model(model)

        assert "options" in result
        assert "tryreduce" in result
        assert "block HOUSEHOLD" in result
        assert "controls" in result
        assert "objective" in result
        assert "constraints" in result


class TestRoundTrip:
    """Test that parsing then printing produces equivalent output."""

    @pytest.mark.parametrize(
        "expr_str",
        [
            "alpha",
            "C[]",
            "K[-1]",
            "Y[1]",
            "a + b",
            "a - b",
            "a * b",
            "a / b",
            "K[] ^ alpha",
            "log(C[])",
            "exp(x)",
        ],
    )
    def test_simple_expressions_roundtrip(self, expr_str):
        node = parse_expression(expr_str)
        printed = print_expression(node)
        reparsed = parse_expression(printed)
        reprinted = print_expression(reparsed)
        assert printed == reprinted

    def test_complex_expression_roundtrip(self):
        expr_str = "A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)"
        node = parse_expression(expr_str)
        printed = print_expression(node)
        reparsed = parse_expression(printed)
        reprinted = print_expression(reparsed)
        assert printed == reprinted

    def test_bellman_roundtrip(self):
        expr_str = "u[] + beta * E[][U[1]]"
        node = parse_expression(expr_str)
        printed = print_expression(node)
        reparsed = parse_expression(printed)
        reprinted = print_expression(reparsed)
        assert printed == reprinted
