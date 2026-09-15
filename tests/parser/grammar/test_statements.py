import pyparsing as pp
import pytest

from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    T_PLUS_1,
    BinaryOp,
    GCNDistribution,
    GCNEquation,
    Number,
    Operator,
    Parameter,
    T,
    Tag,
    Variable,
)
from gEconpy.parser.grammar.statements import (
    DISTRIBUTION,
    EQUATION,
    VARIABLE_LIST,
    VARIABLE_REF,
)


class TestVariableRef:
    @pytest.mark.parametrize(
        "text,name,time_index",
        [
            ("C[]", "C", T),
            ("K[-1]", "K", T_MINUS_1),
            ("Y[1]", "Y", T_PLUS_1),
            ("A[ss]", "A", STEADY_STATE),
            ("epsilon_A[]", "epsilon_A", T),
        ],
    )
    def test_variable_ref(self, text, name, time_index):
        result = VARIABLE_REF.parse_string(text)[0]
        assert isinstance(result, Variable)
        assert result.name == name
        assert result.time_index == time_index

    def test_location_spans_name_and_index(self):
        result = VARIABLE_REF.parse_string("  K[-1]")[0]
        assert result.location.column == 3
        assert result.location.end_column == 3 + len("K[-1]")


class TestVariableList:
    def test_single_variable(self):
        result = VARIABLE_LIST.parse_string("C[]")
        assert len(result) == 1
        assert result[0].name == "C"

    def test_multiple_variables(self):
        result = VARIABLE_LIST.parse_string("C[], K[-1], Y[]")
        assert len(result) == 3
        assert result[0].name == "C"
        assert result[1].name == "K"
        assert result[1].time_index == T_MINUS_1
        assert result[2].name == "Y"

    def test_with_whitespace(self):
        result = VARIABLE_LIST.parse_string("C[] ,  K[] ,Y[]")
        assert len(result) == 3


class TestEquation:
    def test_simple_equation(self):
        result = EQUATION.parse_string("Y[] = C[];")[0]
        assert isinstance(result, GCNEquation)
        assert isinstance(result.lhs, Variable)
        assert result.lhs.name == "Y"
        assert isinstance(result.rhs, Variable)
        assert result.rhs.name == "C"

    def test_binary_expression(self):
        result = EQUATION.parse_string("Y[] = C[] + I[];")[0]
        assert isinstance(result.rhs, BinaryOp)
        assert result.rhs.op == Operator.ADD

    def test_parameter_equation(self):
        result = EQUATION.parse_string("beta = 0.99;")[0]
        assert isinstance(result.lhs, Parameter)
        assert result.lhs.name == "beta"
        assert isinstance(result.rhs, Number)
        assert result.rhs.value == 0.99

    def test_lagrange_multiplier(self):
        result = EQUATION.parse_string("C[] + I[] = Y[] : lambda[];")[0]
        assert result.lagrange_multiplier == "lambda"

    def test_calibrating_parameter(self):
        result = EQUATION.parse_string("beta = 0.99 -> beta;")[0]
        assert result.calibrating_parameter == "beta"

    def test_lagrange_and_calibrating(self):
        result = EQUATION.parse_string("C[] = Y[] : lambda[] -> alpha;")[0]
        assert result.lagrange_multiplier == "lambda"
        assert result.calibrating_parameter == "alpha"

    def test_tag_exclude(self):
        result = EQUATION.parse_string("@exclude Y[] = C[];")[0]
        assert Tag.EXCLUDE in result.tags
        assert result.is_excluded

    def test_complex_expression(self):
        result = EQUATION.parse_string("U[] = u[] + beta * E[][U[1]];")[0]
        assert isinstance(result.lhs, Variable)
        assert result.lhs.name == "U"
        assert isinstance(result.rhs, BinaryOp)
        assert result.rhs.op == Operator.ADD

    def test_steady_state_equation(self):
        result = EQUATION.parse_string("r[ss] = 1 / beta - (1 - delta);")[0]
        assert result.lhs.time_index == STEADY_STATE

    def test_multiline_equation(self):
        eq_text = """Y[ss] = (r[ss] / (r[ss] - delta * alpha)) ^ (sigma_C / (sigma_C + sigma_L)) *
            (w[ss] * (w[ss] / (1 - alpha)) ^ sigma_L) ^ (1 / (sigma_C + sigma_L));"""
        result = EQUATION.parse_string(eq_text)[0]
        assert isinstance(result.lhs, Variable)
        assert result.lhs.name == "Y"


class TestDistribution:
    def test_simple_distribution(self):
        result = DISTRIBUTION.parse_string("alpha ~ Beta(a=1, b=1);")[0]
        assert isinstance(result, GCNDistribution)
        assert result.parameter_name == "alpha"
        assert result.dist_name == "Beta"
        assert result.dist_kwargs == {"a": 1.0, "b": 1.0}

    def test_distribution_with_initial(self):
        result = DISTRIBUTION.parse_string("alpha ~ Beta(a=1, b=1) = 0.35;")[0]
        assert result.initial_value == 0.35

    def test_gamma_distribution(self):
        result = DISTRIBUTION.parse_string("sigma ~ Gamma(alpha=2, beta=1) = 1.5;")[0]
        assert result.dist_name == "Gamma"
        assert result.initial_value == 1.5

    def test_normal_distribution(self):
        result = DISTRIBUTION.parse_string("mu ~ Normal(mu=0, sigma=1);")[0]
        assert result.dist_name == "Normal"
        assert "mu" in result.dist_kwargs
        assert "sigma" in result.dist_kwargs

    def test_wrapped_distribution_maxent(self):
        result = DISTRIBUTION.parse_string("beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99;")[0]
        assert result.dist_name == "Beta"
        assert result.wrapper_name == "maxent"
        assert "lower" in result.wrapper_kwargs
        assert "upper" in result.wrapper_kwargs
        assert "mass" in result.wrapper_kwargs
        assert result.initial_value == 0.99

    def test_truncated_distribution(self):
        result = DISTRIBUTION.parse_string("sigma ~ Truncated(Normal(mu=0, sigma=1), lower=0) = 0.5;")[0]
        assert result.dist_name == "Normal"
        assert result.wrapper_name == "Truncated"
        assert "lower" in result.wrapper_kwargs

    def test_distribution_with_none(self):
        result = DISTRIBUTION.parse_string("sigma ~ Truncated(Normal(mu=0, sigma=1), lower=0, upper=None) = 0.5;")[0]
        assert result.wrapper_kwargs.get("upper") is None

    def test_distribution_parameter_reference(self):
        result = DISTRIBUTION.parse_string("eps ~ Normal(mu=0, sigma=sigma_eps);")[0]
        assert result.dist_kwargs["sigma"] == "sigma_eps"

    def test_location_spans_declaration_through_semicolon(self):
        text = "  alpha ~ Beta(a=1, b=1) = 0.35;"
        location = DISTRIBUTION.parse_string(text)[0].location
        assert (location.line, location.column, location.end_column) == (1, 3, len(text) + 1)
        assert location.source_line == text


class TestEquationErrors:
    @pytest.mark.parametrize(
        "text,match",
        [
            ("Y[] = C[]", "Expected ';'"),
            ("Y[] = ;", "Missing right-hand side"),
            ("= C[];", "Missing left-hand side"),
            ("Y[];", "Missing '='"),
            ("Y[] C[];", "Expected '='"),
            ("Y[] = C[]);", "Unmatched '\\)'"),
            ("Y[] = C[]];", "Unmatched '\\]'"),
            ("Y[] = C[] }", "Missing semicolon"),
        ],
        ids=[
            "missing_semicolon",
            "missing_rhs",
            "missing_lhs",
            "missing_equals",
            "juxtaposed",
            "unmatched_paren",
            "unmatched_bracket",
            "brace_before_semicolon",
        ],
    )
    def test_invalid_equation_raises(self, text, match):
        with pytest.raises(pp.ParseBaseException, match=match):
            EQUATION.parse_string(text)


class TestDistributionErrors:
    @pytest.mark.parametrize(
        "text,match",
        [
            ("alpha Beta(a=1, b=1);", "Expected '~'"),
            ("alpha ~ UnknownDist(a=1);", "Unknown distribution 'UnknownDist'"),
            ("alpha ~ unknownwrap(Beta());", "Unknown distribution wrapper 'unknownwrap'"),
            ("alpha ~ Beta(a=1, b=1)", "Expected ';'"),
        ],
        ids=["missing_tilde", "unknown_distribution", "unknown_wrapper", "missing_semicolon"],
    )
    def test_invalid_distribution_raises(self, text, match):
        with pytest.raises(pp.ParseBaseException, match=match):
            DISTRIBUTION.parse_string(text)
