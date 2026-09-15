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
    ast_to_sympy,
    block_to_sympy,
    equation_to_sympy,
    model_to_sympy,
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


class TestConvertExpectation:
    def test_expectation_is_transparent(self):
        node = Expectation(expr=Variable(name="U", time_index=TimeIndex(1)))
        assert ast_to_sympy(node) == parsed_var("U", 1)


class TestConvertEquation:
    def test_simple_equation(self):
        eq = GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))
        assert ast_to_sympy(eq) == sp.Eq(parsed_var("Y", 0), parsed_var("C", 0))

    def test_equation_to_sympy_with_metadata(self):
        eq = GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))
        result, metadata = equation_to_sympy(eq)
        assert result == sp.Eq(parsed_var("Y", 0), parsed_var("C", 0))
        assert metadata == {"is_calibrating": False, "calibrating_parameter": None, "lagrange_multiplier": None}

    def test_equation_with_lagrange(self):
        eq = GCNEquation(lhs=Variable(name="C"), rhs=Variable(name="Y"), lagrange_multiplier="lambda")
        _result, metadata = equation_to_sympy(eq)
        assert metadata["lagrange_multiplier"] == parsed_var("lambda", 0)

    def test_calibrating_equation(self):
        eq = GCNEquation(lhs=Parameter(name="beta"), rhs=Number(value=0.99), calibrating_parameter="beta")
        _result, metadata = equation_to_sympy(eq)
        assert metadata["is_calibrating"] is True
        assert metadata["calibrating_parameter"] == parsed_symbol("beta")

    def test_calibrating_equation_rearrangement(self):
        # L[ss] / K[ss] = 0.36 -> alpha  becomes  alpha = L[ss] / K[ss] - 0.36
        eq = GCNEquation(
            lhs=BinaryOp(
                left=Variable(name="L", time_index=STEADY_STATE),
                op=Operator.DIV,
                right=Variable(name="K", time_index=STEADY_STATE),
            ),
            rhs=Number(value=0.36),
            calibrating_parameter="alpha",
        )
        result, _metadata = equation_to_sympy(eq)

        assert result.lhs == parsed_symbol("alpha")
        assert result.rhs == parsed_var("L", "ss") / parsed_var("K", "ss") - sp.Float(0.36)


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


class TestBlockToSympy:
    def test_groups_equations_by_component_and_skips_distributions(self):
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="Y"),
                    rhs=BinaryOp(left=Variable(name="C"), op=Operator.ADD, right=Variable(name="I")),
                )
            ],
            calibration=[
                GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35)),
                GCNDistribution(parameter_name="beta", dist_name="Beta", dist_kwargs={"alpha": 2, "beta": 5}),
            ],
        )

        result = block_to_sympy(block)

        assert set(result) == {"definitions", "objective", "constraints", "identities", "calibration"}
        assert result["definitions"] == []
        assert len(result["identities"]) == 1
        assert len(result["calibration"]) == 1
        assert result["identities"][0][0] == sp.Eq(parsed_var("Y", 0), parsed_var("C", 0) + parsed_var("I", 0))


class TestModelToSympy:
    def test_uses_model_assumptions(self):
        block = GCNBlock(name="EQUILIBRIUM", identities=[GCNEquation(lhs=Variable(name="Y"), rhs=Variable(name="C"))])
        model = GCNModel(blocks=[block], assumptions={"Y": {"positive": True}, "C": {"positive": True}})

        result = model_to_sympy(model)

        assert set(result) == {"EQUILIBRIUM"}
        equation, _metadata = result["EQUILIBRIUM"]["identities"][0]
        assert equation.lhs.is_positive is True
        assert equation.rhs.is_positive is True


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
