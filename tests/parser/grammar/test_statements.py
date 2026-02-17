"""Tests for statement grammar (equations, distributions, variable references)."""

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
    def test_current_period(self):
        result = VARIABLE_REF.parse_string("C[]")[0]
        assert isinstance(result, Variable)
        assert result.name == "C"
        assert result.time_index == T

    def test_lagged(self):
        result = VARIABLE_REF.parse_string("K[-1]")[0]
        assert result.name == "K"
        assert result.time_index == T_MINUS_1

    def test_lead(self):
        result = VARIABLE_REF.parse_string("Y[1]")[0]
        assert result.name == "Y"
        assert result.time_index == T_PLUS_1

    def test_steady_state(self):
        result = VARIABLE_REF.parse_string("A[ss]")[0]
        assert result.name == "A"
        assert result.time_index == STEADY_STATE

    def test_underscore_name(self):
        result = VARIABLE_REF.parse_string("epsilon_A[]")[0]
        assert result.name == "epsilon_A"


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
        # Both can exist (though rare)
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
        # RHS should be a binary op tree

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
        # Distribution arg can reference another parameter
        result = DISTRIBUTION.parse_string("eps ~ Normal(mu=0, sigma=sigma_eps);")[0]
        assert result.dist_kwargs["sigma"] == "sigma_eps"


class TestEquationErrors:
    def test_missing_semicolon(self):
        with pytest.raises(pp.ParseBaseException):
            EQUATION.parse_string("Y[] = C[]")

    def test_missing_rhs(self):
        with pytest.raises(pp.ParseBaseException):
            EQUATION.parse_string("Y[] = ;")

    def test_missing_equals(self):
        with pytest.raises(pp.ParseBaseException):
            EQUATION.parse_string("Y[] C[];")


class TestDistributionErrors:
    def test_missing_tilde(self):
        with pytest.raises(pp.ParseBaseException):
            DISTRIBUTION.parse_string("alpha Beta(a=1, b=1);")

    def test_unknown_distribution(self):
        with pytest.raises(pp.ParseBaseException):
            DISTRIBUTION.parse_string("alpha ~ UnknownDist(a=1);")

    def test_missing_semicolon(self):
        with pytest.raises(pp.ParseBaseException):
            DISTRIBUTION.parse_string("alpha ~ Beta(a=1, b=1)")
