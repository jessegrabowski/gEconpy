import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
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


class TestConvertAtoms:
    def test_convert_integer(self):
        node = Number(value=42.0)
        result = ast_to_sympy(node)
        assert result == sp.Integer(42)

    def test_convert_float(self):
        node = Number(value=3.14)
        result = ast_to_sympy(node)
        assert isinstance(result, sp.Float)
        assert float(result) == 3.14

    def test_convert_parameter(self):
        node = Parameter(name="alpha")
        result = ast_to_sympy(node)
        assert isinstance(result, sp.Symbol)
        assert result.name == "alpha"

    def test_convert_parameter_with_assumptions(self):
        node = Parameter(name="beta")
        result = ast_to_sympy(node, {"beta": {"positive": True}})
        assert result.is_positive is True

    def test_convert_variable_at_t(self):
        node = Variable(name="C", time_index=T)
        result = ast_to_sympy(node)
        assert isinstance(result, TimeAwareSymbol)
        assert result.base_name == "C"
        assert result.time_index == 0

    def test_convert_variable_lagged(self):
        node = Variable(name="K", time_index=T_MINUS_1)
        result = ast_to_sympy(node)
        assert isinstance(result, TimeAwareSymbol)
        assert result.base_name == "K"
        assert result.time_index == -1

    def test_convert_variable_lead(self):
        node = Variable(name="U", time_index=TimeIndex(1))
        result = ast_to_sympy(node)
        assert isinstance(result, TimeAwareSymbol)
        assert result.time_index == 1

    def test_convert_variable_steady_state(self):
        node = Variable(name="Y", time_index=STEADY_STATE)
        result = ast_to_sympy(node)
        assert isinstance(result, TimeAwareSymbol)
        assert result.time_index == "ss"


class TestConvertOperations:
    def test_addition(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.ADD,
            right=Parameter(name="b"),
        )
        result = ast_to_sympy(node)
        a, b = sp.symbols("a b")
        assert result == a + b

    def test_subtraction(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.SUB,
            right=Parameter(name="b"),
        )
        result = ast_to_sympy(node)
        a, b = sp.symbols("a b")
        assert result == a - b

    def test_multiplication(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.MUL,
            right=Parameter(name="b"),
        )
        result = ast_to_sympy(node)
        a, b = sp.symbols("a b")
        assert result == a * b

    def test_division(self):
        node = BinaryOp(
            left=Parameter(name="a"),
            op=Operator.DIV,
            right=Parameter(name="b"),
        )
        result = ast_to_sympy(node)
        a, b = sp.symbols("a b")
        assert result == a / b

    def test_power(self):
        node = BinaryOp(
            left=Parameter(name="K"),
            op=Operator.POW,
            right=Parameter(name="alpha"),
        )
        result = ast_to_sympy(node)
        K, alpha = sp.symbols("K alpha")
        assert result == K**alpha

    def test_negation(self):
        node = UnaryOp(op=Operator.NEG, operand=Parameter(name="x"))
        result = ast_to_sympy(node)
        x = sp.Symbol("x")
        assert result == -x

    def test_nested_operations(self):
        # (a + b) * c
        node = BinaryOp(
            left=BinaryOp(
                left=Parameter(name="a"),
                op=Operator.ADD,
                right=Parameter(name="b"),
            ),
            op=Operator.MUL,
            right=Parameter(name="c"),
        )
        result = ast_to_sympy(node)
        a, b, c = sp.symbols("a b c")
        assert result == (a + b) * c


class TestConvertFunctions:
    def test_log(self):
        node = FunctionCall(func_name="log", args=(Parameter(name="x"),))
        result = ast_to_sympy(node)
        x = sp.Symbol("x")
        assert result == sp.log(x)

    def test_exp(self):
        node = FunctionCall(func_name="exp", args=(Parameter(name="x"),))
        result = ast_to_sympy(node)
        x = sp.Symbol("x")
        assert result == sp.exp(x)

    def test_sqrt(self):
        node = FunctionCall(func_name="sqrt", args=(Parameter(name="x"),))
        result = ast_to_sympy(node)
        x = sp.Symbol("x")
        assert result == sp.sqrt(x)

    def test_nested_function(self):
        # log(exp(x))
        node = FunctionCall(
            func_name="log",
            args=(FunctionCall(func_name="exp", args=(Parameter(name="x"),)),),
        )
        result = ast_to_sympy(node)
        x = sp.Symbol("x")
        assert result == sp.log(sp.exp(x))


class TestConvertExpectation:
    def test_expectation_is_transparent(self):
        # E[][U[1]] -> U[1]
        node = Expectation(expr=Variable(name="U", time_index=TimeIndex(1)))
        result = ast_to_sympy(node)
        assert isinstance(result, TimeAwareSymbol)
        assert result.base_name == "U"
        assert result.time_index == 1


class TestConvertEquation:
    def test_simple_equation(self):
        # Y[] = C[]
        eq = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=Variable(name="C"),
        )
        result = ast_to_sympy(eq)
        assert isinstance(result, sp.Eq)

    def test_equation_to_sympy_with_metadata(self):
        eq = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=Variable(name="C"),
        )
        result, metadata = equation_to_sympy(eq)
        assert isinstance(result, sp.Eq)
        assert metadata["is_calibrating"] is False
        assert metadata["lagrange_multiplier"] is None

    def test_equation_with_lagrange(self):
        eq = GCNEquation(
            lhs=Variable(name="C"),
            rhs=Variable(name="Y"),
            lagrange_multiplier="lambda",
        )
        _result, metadata = equation_to_sympy(eq)
        assert metadata["lagrange_multiplier"] is not None
        assert metadata["lagrange_multiplier"].base_name == "lambda"

    def test_calibrating_equation(self):
        eq = GCNEquation(
            lhs=Parameter(name="beta"),
            rhs=Number(value=0.99),
            calibrating_parameter="beta",
        )
        _result, metadata = equation_to_sympy(eq)
        assert metadata["is_calibrating"] is True
        assert metadata["calibrating_parameter"].name == "beta"


class TestConvertComplexExpressions:
    def test_production_function(self):
        # A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)
        node = parse_expression("A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)")
        result = ast_to_sympy(node)

        # Verify it's a valid sympy expression with the right atoms
        atoms = result.free_symbols
        names = {a.name if hasattr(a, "name") else str(a) for a in atoms}
        assert "alpha" in names or any("alpha" in str(a) for a in atoms)

    def test_bellman_equation(self):
        # u[] + beta * E[][U[1]]
        node = parse_expression("u[] + beta * E[][U[1]]")
        result = ast_to_sympy(node)
        assert isinstance(result, sp.Basic)

    def test_ar1_process(self):
        # rho * log(A[-1]) + epsilon[]

        node = parse_expression("rho * log(A[-1]) + epsilon[]")
        result = ast_to_sympy(node)
        assert isinstance(result, sp.Basic)


class TestBlockToSympy:
    def test_simple_block(self):
        block = GCNBlock(name="TEST")
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=BinaryOp(
                    left=Variable(name="C"),
                    op=Operator.ADD,
                    right=Variable(name="I"),
                ),
            )
        ]

        result = block_to_sympy(block)
        assert len(result["identities"]) == 1
        eq, _metadata = result["identities"][0]
        assert isinstance(eq, sp.Eq)


class TestModelToSympy:
    def test_simple_model(self):
        block = GCNBlock(name="EQUILIBRIUM")
        block.identities = [
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=Variable(name="C"),
            )
        ]

        model = GCNModel(
            blocks=[block],
            options={},
            tryreduce=[],
            assumptions={"Y": {"positive": True}, "C": {"positive": True}},
        )

        result = model_to_sympy(model)
        assert "EQUILIBRIUM" in result
        assert len(result["EQUILIBRIUM"]["identities"]) == 1


class TestVariableTimeIndices:
    """Test various time index patterns from real GCN files."""

    def test_large_lead(self):
        node = Variable(name="Happy", time_index=TimeIndex(10))
        result = ast_to_sympy(node)
        assert result.time_index == 10
        assert result.base_name == "Happy"

    def test_large_lag(self):
        node = Variable(name="HAPPY", time_index=TimeIndex(-10))
        result = ast_to_sympy(node)
        assert result.time_index == -10

    def test_variable_with_numeric_suffix(self):
        node = Variable(name="alpha_1", time_index=T_MINUS_1)
        result = ast_to_sympy(node)
        assert result.base_name == "alpha_1"
        assert result.time_index == -1


class TestRealEquations:
    """Test conversion of real equations from GCN files."""

    def test_definition_equation(self):
        # u[] = log(C[]) + log(L[])
        node = parse_expression("log(C[]) + log(L[])")
        result = ast_to_sympy(node)

        C = TimeAwareSymbol("C", 0)
        L = TimeAwareSymbol("L", 0)
        expected = sp.log(C) + sp.log(L)
        assert result == expected

    def test_bellman_with_expectation(self):
        # U[] = u[] + beta * E[][U[1]]
        node = parse_expression("u[] + beta * E[][U[1]]")
        result = ast_to_sympy(node)

        u = TimeAwareSymbol("u", 0)
        beta = sp.Symbol("beta")
        U1 = TimeAwareSymbol("U", 1)
        expected = u + beta * U1
        assert result == expected

    def test_cobb_douglas_production(self):
        # Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)
        node = parse_expression("A[] * K[-1] ^ alpha * L[] ^ (1 - alpha)")
        result = ast_to_sympy(node)

        A = TimeAwareSymbol("A", 0)
        K = TimeAwareSymbol("K", -1)
        L = TimeAwareSymbol("L", 0)
        alpha = sp.Symbol("alpha")
        expected = A * K**alpha * L ** (1 - alpha)
        assert result == expected

    def test_euler_equation(self):
        # sigma / beta * (E[][C[1]] - C[]) = R_ss * E[][R[1]]
        lhs = parse_expression("sigma / beta * (E[][C[1]] - C[])")
        result = ast_to_sympy(lhs)

        sigma = sp.Symbol("sigma")
        beta = sp.Symbol("beta")
        C = TimeAwareSymbol("C", 0)
        C1 = TimeAwareSymbol("C", 1)
        expected = sigma / beta * (C1 - C)
        assert result == expected

    def test_capital_accumulation(self):
        # K[] = (1 - delta) * K[-1] + delta * I[]
        node = parse_expression("(1 - delta) * K[-1] + delta * I[]")
        result = ast_to_sympy(node)

        delta = sp.Symbol("delta")
        K = TimeAwareSymbol("K", -1)
        I = TimeAwareSymbol("I", 0)
        expected = (1 - delta) * K + delta * I
        assert result == expected

    def test_steady_state_ratio(self):
        # L[ss] / K[ss]
        node = parse_expression("L[ss] / K[ss]")
        result = ast_to_sympy(node)

        L_ss = TimeAwareSymbol("L", "ss")
        K_ss = TimeAwareSymbol("K", "ss")
        expected = L_ss / K_ss
        assert result == expected

    def test_complex_steady_state_expression(self):
        # W_ss = (1 - alpha) ^ (1 / (1 - alpha)) * (alpha / R_ss) ^ (alpha / (1 - alpha))
        node = parse_expression("(1 - alpha) ^ (1 / (1 - alpha)) * (alpha / R_ss) ^ (alpha / (1 - alpha))")
        result = ast_to_sympy(node)
        assert isinstance(result, sp.Basic)
        # Just verify it parses and converts without error

    def test_ar1_shock_process(self):
        # A[] = rho_A * A[-1] + epsilon_A[]
        node = parse_expression("rho_A * A[-1] + epsilon_A[]")
        result = ast_to_sympy(node)

        rho_A = sp.Symbol("rho_A")
        A = TimeAwareSymbol("A", -1)
        epsilon_A = TimeAwareSymbol("epsilon_A", 0)
        expected = rho_A * A + epsilon_A
        assert result == expected


class TestCalibratingEquationConversion:
    def test_calibrating_equation_rearrangement(self):
        # L[ss] / K[ss] = 0.36 -> alpha
        # Should become: alpha = L[ss]/K[ss] - 0.36
        eq = GCNEquation(
            lhs=BinaryOp(
                left=Variable(name="L", time_index=STEADY_STATE),
                op=Operator.DIV,
                right=Variable(name="K", time_index=STEADY_STATE),
            ),
            rhs=Number(value=0.36),
            calibrating_parameter="alpha",
        )
        result, metadata = equation_to_sympy(eq)

        assert metadata["is_calibrating"] is True
        assert metadata["calibrating_parameter"].name == "alpha"
        # The equation should be: alpha = lhs - rhs
        assert result.lhs == sp.Symbol("alpha")


class TestAssumptionsPropagation:
    def test_assumptions_on_parameters(self):
        node = Parameter(name="beta")
        assumptions = {"beta": {"positive": True, "real": True}}
        result = ast_to_sympy(node, assumptions)
        assert result.is_positive is True
        assert result.is_real is True

    def test_assumptions_on_variables(self):
        node = Variable(name="C", time_index=T)
        assumptions = {"C": {"positive": True}}
        result = ast_to_sympy(node, assumptions)
        assert result.is_positive is True

    def test_no_assumptions_gives_none(self):
        node = Parameter(name="x")
        result = ast_to_sympy(node)
        assert result.is_positive is None
