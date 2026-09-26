import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
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
from gEconpy.parser.grammar.expressions import parse_expression
from gEconpy.parser.transform.to_sympy import (
    ASTToSympyConverter,
    ast_to_sympy,
)
from tests.conftest import parsed_symbol, parsed_symbols, parsed_var


class TestConvertAtoms:
    def test_convert_integer(self):
        result = ast_to_sympy(Number(value=42.0))
        assert result == sp.Integer(42)

    def test_convert_float(self):
        result = ast_to_sympy(Number(value=3.14))
        assert isinstance(result, sp.Float)
        assert float(result) == 3.14

    def test_convert_parameter(self):
        result = ast_to_sympy(Parameter(name="alpha"))
        assert isinstance(result, sp.Symbol)
        assert result.name == "alpha"

    @pytest.mark.parametrize(
        ("time_index", "expected_offset"),
        [(T, 0), (T_MINUS_1, -1), (TimeIndex(1), 1), (TimeIndex(10), 10), (TimeIndex(-10), -10), (STEADY_STATE, "ss")],
    )
    def test_convert_variable(self, time_index, expected_offset):
        result = ast_to_sympy(Variable(name="C", time_index=time_index))
        assert isinstance(result, TimeAwareSymbol)
        assert result.base_name == "C"
        assert result.time_index == expected_offset

    def test_variable_with_numeric_suffix(self):
        result = ast_to_sympy(Variable(name="alpha_1", time_index=T_MINUS_1))
        assert result.base_name == "alpha_1"
        assert result.time_index == -1


class TestConvertOperations:
    @pytest.mark.parametrize(
        ("op", "combine"),
        [
            (Operator.ADD, lambda a, b: a + b),
            (Operator.SUB, lambda a, b: a - b),
            (Operator.MUL, lambda a, b: a * b),
            (Operator.DIV, lambda a, b: a / b),
            (Operator.POW, lambda a, b: a**b),
        ],
    )
    def test_binary_operators(self, op, combine):
        node = BinaryOp(left=Parameter(name="a"), op=op, right=Parameter(name="b"))
        a, b = parsed_symbols("a b")
        assert ast_to_sympy(node) == combine(a, b)

    def test_negation(self):
        node = UnaryOp(op=Operator.NEG, operand=Parameter(name="x"))
        assert ast_to_sympy(node) == -parsed_symbol("x")

    def test_nested_operations(self):
        node = BinaryOp(
            left=BinaryOp(left=Parameter(name="a"), op=Operator.ADD, right=Parameter(name="b")),
            op=Operator.MUL,
            right=Parameter(name="c"),
        )
        a, b, c = parsed_symbols("a b c")
        assert ast_to_sympy(node) == (a + b) * c


class TestConvertFunctions:
    @pytest.mark.parametrize(("func_name", "func"), [("log", sp.log), ("exp", sp.exp), ("sqrt", sp.sqrt)])
    def test_unary_functions(self, func_name, func):
        node = FunctionCall(func_name=func_name, args=(Parameter(name="x"),))
        assert ast_to_sympy(node) == func(parsed_symbol("x"))

    def test_nested_function(self):
        node = FunctionCall(func_name="log", args=(FunctionCall(func_name="exp", args=(Parameter(name="x"),)),))
        assert ast_to_sympy(node) == sp.log(sp.exp(parsed_symbol("x")))

    def test_unknown_function_raises(self):
        node = FunctionCall(func_name="gamma", args=(Parameter(name="x"),))
        with pytest.raises(ValueError, match="Unknown function 'gamma'"):
            ast_to_sympy(node)


class TestConvertRejectsWrongNodeKinds:
    def test_convert_expr_rejects_equation(self):
        eq = GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))
        with pytest.raises(TypeError, match="Expected an expression node, got a GCNEquation"):
            ASTToSympyConverter().convert_expr(eq)

    def test_convert_rejects_non_expression_node(self):
        with pytest.raises(TypeError, match="Cannot convert GCNBlock to SymPy"):
            ast_to_sympy(GCNBlock(name="TEST"))


class TestConvertExpectation:
    def test_expectation_is_transparent(self):
        node = Expectation(expr=Variable(name="U", time_index=TimeIndex(1)))
        assert ast_to_sympy(node) == parsed_var("U", 1)


class TestConvertEquation:
    def test_simple_equation(self):
        eq = GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))
        assert ast_to_sympy(eq) == sp.Eq(parsed_var("Y", 0), parsed_var("C", 0))


class TestConvertEquationRejectsSettledEquations:
    def test_contradictory_equation_raises_instead_of_returning_a_boolean(self):
        eq = GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0))
        with pytest.raises(ValueError, match=r"'alpha = 0' is impossible"):
            ast_to_sympy(eq, {"alpha": {"positive": True}})


class TestRealEquations:
    def test_definition_equation(self):
        result = ast_to_sympy(parse_expression("log(C[]) + log(L[])"))
        assert result == sp.log(parsed_var("C", 0)) + sp.log(parsed_var("L", 0))

    def test_bellman_with_expectation(self):
        result = ast_to_sympy(parse_expression("u[] + beta * E[][U[1]]"))
        assert result == parsed_var("u", 0) + parsed_symbol("beta") * parsed_var("U", 1)

    def test_cobb_douglas_production(self):
        result = ast_to_sympy(parse_expression("A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)"))

        A, K, L = parsed_var("A", 0), parsed_var("K", -1), parsed_var("L", 0)
        alpha = parsed_symbol("alpha")
        assert result == A * K**alpha * L ** (1 - alpha)

    def test_euler_equation(self):
        result = ast_to_sympy(parse_expression("sigma / beta * (E[][C[1]] - C[])"))

        sigma, beta = parsed_symbols("sigma beta")
        assert result == sigma / beta * (parsed_var("C", 1) - parsed_var("C", 0))

    def test_capital_accumulation(self):
        result = ast_to_sympy(parse_expression("(1 - delta) * K[-1] + delta * I[]"))

        delta = parsed_symbol("delta")
        assert result == (1 - delta) * parsed_var("K", -1) + delta * parsed_var("I", 0)

    def test_steady_state_ratio(self):
        result = ast_to_sympy(parse_expression("L[ss] / K[ss]"))
        assert result == parsed_var("L", "ss") / parsed_var("K", "ss")

    def test_complex_steady_state_expression(self):
        result = ast_to_sympy(
            parse_expression("(1 - alpha) ^ (1 / (1 - alpha)) * (alpha / R_ss) ^ (alpha / (1 - alpha))")
        )

        alpha, R_ss = parsed_symbols("alpha R_ss")
        assert result == (1 - alpha) ** (1 / (1 - alpha)) * (alpha / R_ss) ** (alpha / (1 - alpha))

    def test_ar1_shock_process(self):
        result = ast_to_sympy(parse_expression("rho_A * A[-1] + epsilon_A[]"))
        assert result == parsed_symbol("rho_A") * parsed_var("A", -1) + parsed_var("epsilon_A", 0)


class TestAssumptionsPropagation:
    def test_assumptions_on_parameters(self):
        result = ast_to_sympy(Parameter(name="beta"), {"beta": {"positive": True, "real": True}})
        assert result.is_positive is True
        assert result.is_real is True

    def test_assumptions_on_variables(self):
        result = ast_to_sympy(Variable(name="C", time_index=T), {"C": {"positive": True}})
        assert result.is_positive is True

    def test_no_assumptions_gives_none(self):
        result = ast_to_sympy(Parameter(name="x"))
        assert result.is_positive is None
