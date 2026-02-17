import pyparsing as pp
import pytest

from gEconpy.parser.ast import (
    T_MINUS_1,
    BinaryOp,
    GCNDistribution,
    GCNEquation,
    Number,
    Parameter,
    T,
    Variable,
)
from gEconpy.parser.grammar.components import (
    CALIBRATION,
    COMPONENT,
    CONSTRAINTS,
    CONTROLS,
    DEFINITIONS,
    IDENTITIES,
    OBJECTIVE,
    SHOCKS,
)


class TestDefinitions:
    def test_single_definition(self):
        text = """definitions
        {
            u[] = C[] ^ (1 - sigma_C) / (1 - sigma_C);
        };"""
        name, equations = DEFINITIONS.parse_string(text)[0]
        assert name == "definitions"
        assert len(equations) == 1
        assert isinstance(equations[0], GCNEquation)
        assert equations[0].lhs.name == "u"

    def test_multiple_definitions(self):
        text = """definitions
        {
            u[] = log(C[]);
            profit[] = Y[] - w[] * L[];
        };"""
        name, equations = DEFINITIONS.parse_string(text)[0]
        assert name == "definitions"
        assert len(equations) == 2
        assert equations[0].lhs.name == "u"
        assert equations[1].lhs.name == "profit"

    def test_empty_content_allowed(self):
        result = DEFINITIONS.parse_string("definitions { };")
        name, equations = result[0]
        assert name == "definitions"
        assert equations == []


class TestControls:
    def test_single_control(self):
        text = "controls { C[]; };"
        name, variables = CONTROLS.parse_string(text)[0]
        assert name == "controls"
        assert len(variables) == 1
        assert variables[0].name == "C"

    def test_multiple_controls(self):
        text = "controls { C[], L[], I[], K[]; };"
        name, variables = CONTROLS.parse_string(text)[0]
        assert name == "controls"
        assert len(variables) == 4
        assert [v.name for v in variables] == ["C", "L", "I", "K"]

    def test_controls_with_time_index(self):
        text = "controls { K[-1], L[]; };"
        _name, variables = CONTROLS.parse_string(text)[0]
        assert variables[0].time_index == T_MINUS_1
        assert variables[1].time_index == T


class TestObjective:
    def test_bellman_equation(self):
        text = """objective
        {
            U[] = u[] + beta * E[][U[1]];
        };"""
        name, equations = OBJECTIVE.parse_string(text)[0]
        assert name == "objective"
        assert len(equations) == 1
        assert equations[0].lhs.name == "U"

    def test_profit_maximization(self):
        text = """objective
        {
            TC[] = -(r[] * K[-1] + w[] * L[]);
        };"""
        _name, equations = OBJECTIVE.parse_string(text)[0]
        assert equations[0].lhs.name == "TC"


class TestConstraints:
    def test_budget_constraint(self):
        text = """constraints
        {
            C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
        };"""
        name, equations = CONSTRAINTS.parse_string(text)[0]
        assert name == "constraints"
        assert len(equations) == 1
        assert equations[0].lagrange_multiplier == "lambda"

    def test_multiple_constraints(self):
        text = """constraints
        {
            C[] + I[] = Y[] : lambda[];
            K[] = (1 - delta) * K[-1] + I[];
        };"""
        _name, equations = CONSTRAINTS.parse_string(text)[0]
        assert len(equations) == 2
        assert equations[0].lagrange_multiplier == "lambda"
        assert equations[1].lagrange_multiplier is None

    def test_production_constraint(self):
        text = """constraints
        {
            Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[];
        };"""
        _name, equations = CONSTRAINTS.parse_string(text)[0]
        assert equations[0].lagrange_multiplier == "mc"


class TestIdentities:
    def test_market_clearing(self):
        text = """identities
        {
            Y[] = C[] + I[];
        };"""
        name, equations = IDENTITIES.parse_string(text)[0]
        assert name == "identities"
        assert len(equations) == 1

    def test_ar1_process(self):
        text = """identities
        {
            log(A[]) = rho_A * log(A[-1]) + epsilon_A[];
        };"""
        _name, equations = IDENTITIES.parse_string(text)[0]
        assert len(equations) == 1

    def test_steady_state_identities(self):
        text = """identities
        {
            A[ss] = 1;
            r[ss] = 1 / beta - (1 - delta);
        };"""
        _name, equations = IDENTITIES.parse_string(text)[0]
        assert len(equations) == 2

    def test_with_comment(self):
        text = """identities
        {
            # Perfect competition
            mc[] = 1;
        };"""
        _name, equations = IDENTITIES.parse_string(text)[0]
        assert len(equations) == 1
        assert equations[0].lhs.name == "mc"


class TestShocks:
    def test_single_shock(self):
        text = """shocks
        {
            epsilon_A[];
        };"""
        name, (variables, distributions) = SHOCKS.parse_string(text)[0]
        assert name == "shocks"
        assert len(variables) == 1
        assert variables[0].name == "epsilon_A"
        assert len(distributions) == 0

    def test_multiple_shocks(self):
        text = """shocks
        {
            epsilon_A[];
            epsilon_B[];
        };"""
        _name, (variables, _distributions) = SHOCKS.parse_string(text)[0]
        assert len(variables) == 2

    def test_shock_with_distribution(self):
        text = """shocks
        {
            epsilon_A[] ~ Normal(mu=0, sigma=sigma_eps);
        };"""
        _name, (variables, distributions) = SHOCKS.parse_string(text)[0]
        assert len(variables) == 1
        assert variables[0].name == "epsilon_A"
        assert len(distributions) == 1
        assert distributions[0].dist_name == "Normal"

    def test_mixed_shocks(self):
        text = """shocks
        {
            epsilon_A[];
            epsilon_B[] ~ Normal(mu=0, sigma=0.01);
        };"""
        _name, (variables, distributions) = SHOCKS.parse_string(text)[0]
        assert len(variables) == 2
        assert len(distributions) == 1

    def test_empty_shocks(self):
        text = "shocks { };"
        _name, (variables, distributions) = SHOCKS.parse_string(text)[0]
        assert len(variables) == 0
        assert len(distributions) == 0


class TestCalibration:
    def test_simple_parameter(self):
        text = """calibration
        {
            beta = 0.99;
        };"""
        name, items = CALIBRATION.parse_string(text)[0]
        assert name == "calibration"
        assert len(items) == 1
        assert isinstance(items[0], GCNEquation)
        assert isinstance(items[0].lhs, Parameter)
        assert items[0].lhs.name == "beta"

    def test_multiple_parameters(self):
        text = """calibration
        {
            beta = 0.99;
            delta = 0.025;
            alpha = 0.35;
        };"""
        _name, items = CALIBRATION.parse_string(text)[0]
        assert len(items) == 3

    def test_distribution_prior(self):
        text = """calibration
        {
            alpha ~ Beta(alpha=2, beta=5) = 0.35;
        };"""
        _name, items = CALIBRATION.parse_string(text)[0]
        assert len(items) == 1
        assert isinstance(items[0], GCNDistribution)
        assert items[0].parameter_name == "alpha"
        assert items[0].initial_value == 0.35

    def test_wrapped_distribution(self):
        text = """calibration
        {
            beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99;
        };"""
        _name, items = CALIBRATION.parse_string(text)[0]
        assert isinstance(items[0], GCNDistribution)
        assert items[0].wrapper_name == "maxent"

    def test_calibrating_equation(self):
        text = """calibration
        {
            r[ss] = 1 / beta - (1 - delta) -> delta;
        };"""
        _name, items = CALIBRATION.parse_string(text)[0]
        assert isinstance(items[0], GCNEquation)
        assert items[0].calibrating_parameter == "delta"

    def test_mixed_calibration(self):
        text = """calibration
        {
            beta = 0.99;
            alpha ~ Beta(alpha=2, beta=5) = 0.35;
            sigma_C ~ Gamma(alpha=2, beta=1) = 1.5;
        };"""
        _name, items = CALIBRATION.parse_string(text)[0]
        assert len(items) == 3
        assert isinstance(items[0], GCNEquation)
        assert isinstance(items[1], GCNDistribution)
        assert isinstance(items[2], GCNDistribution)

    def test_empty_calibration(self):
        text = "calibration { };"
        _name, items = CALIBRATION.parse_string(text)[0]
        assert len(items) == 0


class TestComponent:
    """Test the combined COMPONENT parser."""

    def test_parses_definitions(self):
        text = "definitions { u[] = log(C[]); };"
        name, _content = COMPONENT.parse_string(text)[0]
        assert name == "definitions"

    def test_parses_controls(self):
        text = "controls { C[], L[]; };"
        name, _content = COMPONENT.parse_string(text)[0]
        assert name == "controls"

    def test_parses_identities(self):
        text = "identities { Y[] = C[]; };"
        name, _content = COMPONENT.parse_string(text)[0]
        assert name == "identities"

    def test_parses_shocks(self):
        text = "shocks { epsilon[]; };"
        name, _content = COMPONENT.parse_string(text)[0]
        assert name == "shocks"

    def test_parses_calibration(self):
        text = "calibration { beta = 0.99; };"
        name, _content = COMPONENT.parse_string(text)[0]
        assert name == "calibration"

    def test_case_insensitive(self):
        text = "DEFINITIONS { u[] = log(C[]); };"
        name, _content = COMPONENT.parse_string(text)[0]
        assert name == "definitions"

        text = "Calibration { beta = 0.99; };"
        name, _content = COMPONENT.parse_string(text)[0]
        assert name == "calibration"


class TestComponentErrors:
    def test_missing_semicolon_after_brace(self):
        with pytest.raises(pp.ParseBaseException):
            DEFINITIONS.parse_string("definitions { u[] = log(C[]); }")

    def test_wrong_content_type(self):
        # controls expects variable list, not equation
        with pytest.raises(pp.ParseBaseException):
            CONTROLS.parse_string("controls { Y[] = C[]; };")
