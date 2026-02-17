"""Tests for special block grammar (options, tryreduce, assumptions)."""

import pytest

from pyparsing import ParseBaseException

from gEconpy.parser.grammar.special_blocks import (
    ASSUMPTIONS_BLOCK,
    OPTIONS_BLOCK,
    TRYREDUCE_BLOCK,
    parse_assumptions,
    parse_options,
    parse_tryreduce,
)


class TestOptionsBlock:
    def test_empty_options(self):
        text = "options { };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result == {}

    def test_single_boolean_true(self):
        text = "options { verbose = TRUE; };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["verbose"] is True

    def test_single_boolean_false(self):
        text = "options { verbose = FALSE; };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["verbose"] is False

    def test_string_value(self):
        text = "options { solver = gensys; };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["solver"] == "gensys"

    def test_multiple_options(self):
        text = """options {
            verbose = TRUE;
            output = latex;
            debug = FALSE;
        };"""
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["verbose"] is True
        assert result["output"] == "latex"
        assert result["debug"] is False

    def test_multi_word_key(self):
        text = "options { output logfile = TRUE; };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["output logfile"] is True

    def test_case_insensitive_keyword(self):
        text = "OPTIONS { verbose = TRUE; };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["verbose"] is True

    def test_case_insensitive_boolean(self):
        text = "options { verbose = true; };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result["verbose"] is True


class TestTryreduceBlock:
    def test_single_variable(self):
        text = "tryreduce { U[]; };"
        result = TRYREDUCE_BLOCK.parse_string(text)[0]
        assert result == ["U"]

    def test_multiple_variables(self):
        text = "tryreduce { U[], TC[], Div[]; };"
        result = TRYREDUCE_BLOCK.parse_string(text)[0]
        assert result == ["U", "TC", "Div"]

    def test_case_insensitive_keyword(self):
        text = "TRYREDUCE { U[]; };"
        result = TRYREDUCE_BLOCK.parse_string(text)[0]
        assert result == ["U"]

    def test_with_whitespace(self):
        text = """tryreduce
        {
            U[], TC[];
        };"""
        result = TRYREDUCE_BLOCK.parse_string(text)[0]
        assert result == ["U", "TC"]


class TestAssumptionsBlock:
    def test_single_assumption_single_variable(self):
        text = "assumptions { positive { C[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert "C" in result
        assert result["C"]["positive"] is True

    def test_single_assumption_multiple_variables(self):
        text = "assumptions { positive { C[], K[], L[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert "C" in result
        assert "K" in result
        assert "L" in result

    def test_multiple_assumptions(self):
        text = """assumptions {
            positive { C[], K[]; };
            real { shock[]; };
        };"""
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result["C"]["positive"] is True
        assert result["K"]["positive"] is True
        assert result["shock"]["real"] is True

    def test_parameter_in_assumptions(self):
        text = "assumptions { positive { alpha, beta; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert "alpha" in result
        assert "beta" in result

    def test_mixed_variables_and_parameters(self):
        text = "assumptions { positive { C[], alpha, K[], beta; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert "C" in result
        assert "alpha" in result
        assert "K" in result
        assert "beta" in result

    def test_case_insensitive_keyword(self):
        text = "ASSUMPTIONS { positive { C[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert "C" in result

    def test_case_insensitive_assumption(self):
        text = "assumptions { POSITIVE { C[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result["C"]["positive"] is True

    def test_empty_assumptions(self):
        text = "assumptions { };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result == {}


class TestParseOptionsFn:
    def test_finds_options_in_larger_text(self):
        text = """
        options { verbose = TRUE; };

        block HOUSEHOLD { };
        """
        result = parse_options(text)
        assert result["verbose"] is True

    def test_returns_empty_when_no_options(self):
        text = "block HOUSEHOLD { };"
        result = parse_options(text)
        assert result == {}


class TestParseTryreduceFn:
    def test_finds_tryreduce_in_larger_text(self):
        text = """
        tryreduce { U[], TC[]; };

        block HOUSEHOLD { };
        """
        result = parse_tryreduce(text)
        assert result == ["U", "TC"]

    def test_returns_empty_when_no_tryreduce(self):
        text = "block HOUSEHOLD { };"
        result = parse_tryreduce(text)
        assert result == []


class TestParseAssumptionsFn:
    def test_finds_assumptions_in_larger_text(self):
        text = """
        assumptions { positive { C[]; }; };

        block HOUSEHOLD { };
        """
        result = parse_assumptions(text)
        assert "C" in result

    def test_returns_default_when_no_assumptions(self):
        text = "block HOUSEHOLD { };"
        result = parse_assumptions(text)
        # Should return empty dict or default
        assert isinstance(result, dict)


class TestSpecialBlockErrors:
    def test_options_missing_semicolon(self):
        with pytest.raises(ParseBaseException):
            OPTIONS_BLOCK.parse_string("options { verbose = TRUE }")

    def test_tryreduce_missing_semicolon(self):
        with pytest.raises(ParseBaseException):
            TRYREDUCE_BLOCK.parse_string("tryreduce { U[] }")

    def test_assumptions_invalid_assumption(self):
        # "invalid" is not a valid assumption name
        with pytest.raises(ParseBaseException):
            ASSUMPTIONS_BLOCK.parse_string("assumptions { invalid { C[]; }; };")
