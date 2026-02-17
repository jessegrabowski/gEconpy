"""Tests for model block grammar."""

import pyparsing as pp
import pytest

from gEconpy.parser.ast import (
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    Variable,
)
from gEconpy.parser.grammar.blocks import MODEL_BLOCK


class TestModelBlockBasic:
    def test_empty_block(self):
        text = "block TEST { };"
        result = MODEL_BLOCK.parse_string(text)[0]
        assert isinstance(result, GCNBlock)
        assert result.name == "TEST"

    def test_block_name_preserved(self):
        text = "block HOUSEHOLD { };"
        result = MODEL_BLOCK.parse_string(text)[0]
        assert result.name == "HOUSEHOLD"

    def test_block_without_trailing_semicolon_raises(self):
        text = "block TEST { }"
        with pytest.raises(pp.ParseBaseException, match="Expected ';'"):
            MODEL_BLOCK.parse_string(text)

    def test_block_case_insensitive_keyword(self):
        text = "BLOCK TEST { };"
        result = MODEL_BLOCK.parse_string(text)[0]
        assert result.name == "TEST"

        text = "Block TEST { };"
        result = MODEL_BLOCK.parse_string(text)[0]
        assert result.name == "TEST"


class TestModelBlockWithComponents:
    def test_block_with_identities(self):
        text = """block TEST {
            identities {
                Y[] = C[] + I[];
            };
        };"""
        result = MODEL_BLOCK.parse_string(text)[0]
        assert len(result.identities) == 1
        assert result.identities[0].lhs.name == "Y"

    def test_block_with_controls(self):
        text = """block TEST {
            controls {
                C[], L[], K[];
            };
        };"""
        result = MODEL_BLOCK.parse_string(text)[0]
        assert len(result.controls) == 3
        assert [v.name for v in result.controls] == ["C", "L", "K"]

    def test_block_with_calibration(self):
        text = """block TEST {
            calibration {
                beta = 0.99;
                alpha ~ Beta(a=2, b=5) = 0.35;
            };
        };"""
        result = MODEL_BLOCK.parse_string(text)[0]
        assert len(result.calibration) == 2
        assert isinstance(result.calibration[0], GCNEquation)
        assert isinstance(result.calibration[1], GCNDistribution)

    def test_block_with_shocks(self):
        text = """block TEST {
            shocks {
                epsilon_A[];
                epsilon_B[] ~ Normal(mu=0, sigma=0.01);
            };
        };"""
        result = MODEL_BLOCK.parse_string(text)[0]
        assert len(result.shocks) == 2
        assert len(result.shock_distributions) == 1

    def test_block_with_constraints_and_lagrange(self):
        text = """block TEST {
            constraints {
                C[] + I[] = Y[] : lambda[];
            };
        };"""
        result = MODEL_BLOCK.parse_string(text)[0]
        assert len(result.constraints) == 1
        assert result.constraints[0].lagrange_multiplier == "lambda"


class TestHouseholdBlock:
    def test_full_household_block(self):
        text = """block HOUSEHOLD
        {
            definitions
            {
                u[] = C[] ^ (1 - sigma_C) / (1 - sigma_C) - L[] ^ (1 + sigma_L) / (1 + sigma_L);
            };

            controls
            {
                C[], L[], I[], K[];
            };

            objective
            {
                U[] = u[] + beta * E[][U[1]];
            };

            constraints
            {
                C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
                K[] = (1 - delta) * K[-1] + I[];
            };

            calibration
            {
                beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99;
                delta ~ maxent(Beta(), lower=0.01, upper=0.05, mass=0.99) = 0.02;
                sigma_C ~ maxent(Gamma(), lower=1.01, upper=10.0, mass=0.99) = 1.5;
                sigma_L ~ maxent(Gamma(), lower=1.0, upper=10.0, mass=0.99) = 2.0;
            };
        };"""

        result = MODEL_BLOCK.parse_string(text)[0]

        assert result.name == "HOUSEHOLD"
        assert len(result.definitions) == 1
        assert len(result.controls) == 4
        assert len(result.objective) == 1
        assert len(result.constraints) == 2
        assert len(result.calibration) == 4

        assert result.constraints[0].lagrange_multiplier == "lambda"
        assert all(isinstance(item, GCNDistribution) for item in result.calibration)


class TestFirmBlock:
    def test_firm_block(self):
        text = """block FIRM
        {
            controls
            {
                K[-1], L[];
            };

            objective
            {
                TC[] = -(r[] * K[-1] + w[] * L[]);
            };

            constraints
            {
                Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[];
            };

            identities
            {
                # Perfect competition
                mc[] = 1;
            };

            calibration
            {
                alpha ~ maxent(Beta(), lower=0.2, upper=0.5, mass=0.99) = 0.35;
            };
        };"""

        result = MODEL_BLOCK.parse_string(text)[0]

        assert result.name == "FIRM"
        assert len(result.controls) == 2
        assert len(result.objective) == 1
        assert len(result.constraints) == 1
        assert len(result.identities) == 1
        assert len(result.calibration) == 1
        assert result.constraints[0].lagrange_multiplier == "mc"


class TestSteadyStateBlock:
    def test_steady_state_block(self):
        text = """block STEADY_STATE
        {
            identities
            {
                A[ss] = 1;
                r[ss] = 1 / beta - (1 - delta);
                K[ss] = alpha * Y[ss] / r[ss];
            };
        };"""

        result = MODEL_BLOCK.parse_string(text)[0]

        assert result.name == "STEADY_STATE"
        assert len(result.identities) == 3


class TestShocksBlock:
    def test_technology_shocks_block(self):
        text = """block TECHNOLOGY_SHOCKS
        {
            identities
            {
                log(A[]) = rho_A * log(A[-1]) + epsilon_A[];
            };

            shocks
            {
                epsilon_A[];
            };

            calibration
            {
                rho_A ~ maxent(Beta(), lower=0.8, upper=0.99) = 0.95;
            };
        };"""

        result = MODEL_BLOCK.parse_string(text)[0]

        assert result.name == "TECHNOLOGY_SHOCKS"
        assert len(result.identities) == 1
        assert len(result.shocks) == 1
        assert result.shocks[0].name == "epsilon_A"
        assert len(result.calibration) == 1


class TestBlockWithComments:
    def test_comments_are_ignored(self):
        text = """block TEST
        {
            # This is a comment
            identities
            {
                # Another comment
                Y[] = C[];  # Inline comment
            };
        };"""

        result = MODEL_BLOCK.parse_string(text)[0]
        assert len(result.identities) == 1


class TestBlockErrors:
    def test_missing_name(self):
        with pytest.raises(pp.ParseBaseException):
            MODEL_BLOCK.parse_string("block { };")

    def test_missing_braces(self):
        with pytest.raises(pp.ParseBaseException):
            MODEL_BLOCK.parse_string("block TEST;")

    def test_unclosed_brace(self):
        with pytest.raises(pp.ParseBaseException):
            MODEL_BLOCK.parse_string("block TEST { identities { Y[] = C[]; };", parse_all=True)
