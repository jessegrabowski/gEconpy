import pytest

from pyparsing import ParseBaseException

from gEconpy.classes.time_aware_symbol import DEFAULT_ASSUMPTIONS
from gEconpy.parser.grammar.special_blocks import (
    ASSUMPTIONS_BLOCK,
    OPTIONS_BLOCK,
    TRYREDUCE_BLOCK,
    extract_special_block_content,
    parse_assumptions,
    parse_options,
    parse_tryreduce,
    remove_special_block,
)


class TestOptionsBlock:
    def test_empty_options(self):
        text = "options { };"
        result = OPTIONS_BLOCK.parse_string(text)[0]
        assert result == {}

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("options { verbose = TRUE; };", {"verbose": True}),
            ("options { verbose = FALSE; };", {"verbose": False}),
            ("options { verbose = true; };", {"verbose": True}),
            ("options { solver = gensys; };", {"solver": "gensys"}),
            ("options { output logfile = TRUE; };", {"output logfile": True}),
            ("OPTIONS { verbose = TRUE; };", {"verbose": True}),
        ],
        ids=["true", "false", "lowercase_true", "identifier", "multi_word_key", "uppercase_keyword"],
    )
    def test_single_option(self, text, expected):
        assert OPTIONS_BLOCK.parse_string(text)[0] == expected

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


class TestTryreduceBlock:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("tryreduce { };", []),
            ("tryreduce { U[]; };", ["U"]),
            ("tryreduce { U[], TC[], Div[]; };", ["U", "TC", "Div"]),
            ("TRYREDUCE { U[]; };", ["U"]),
            ("tryreduce\n{\n    U[], TC[];\n};", ["U", "TC"]),
        ],
        ids=["empty", "single", "multiple", "uppercase_keyword", "multiline"],
    )
    def test_tryreduce(self, text, expected):
        assert TRYREDUCE_BLOCK.parse_string(text)[0] == expected


class TestAssumptionsBlock:
    def test_single_assumption_single_variable(self):
        text = "assumptions { positive { C[]; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert "C" in result
        assert result["C"]["positive"] is True

    @pytest.mark.parametrize(
        "text,expected_names",
        [
            ("assumptions { positive { C[], K[], L[]; }; };", ["C", "K", "L"]),
            ("assumptions { positive { alpha, beta; }; };", ["alpha", "beta"]),
            ("assumptions { positive { C[], alpha, K[], beta; }; };", ["C", "alpha", "K", "beta"]),
            ("ASSUMPTIONS { positive { C[]; }; };", ["C"]),
        ],
        ids=["variables", "parameters", "mixed", "uppercase_keyword"],
    )
    def test_assumption_applies_to_every_listed_name(self, text, expected_names):
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert list(result) == expected_names
        assert all(result[name]["positive"] is True for name in expected_names)

    def test_multiple_assumptions(self):
        text = """assumptions {
            positive { C[], K[]; };
            real { shock[]; };
        };"""
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result["C"]["positive"] is True
        assert result["K"]["positive"] is True
        assert result["shock"]["real"] is True

    def test_unit_interval_implies_positive(self):
        text = "assumptions { unit_interval { alpha; }; };"
        result = ASSUMPTIONS_BLOCK.parse_string(text)[0]
        assert result["alpha"]["unit_interval"] is True
        assert result["alpha"]["positive"] is True

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
        assert result["anything"] == DEFAULT_ASSUMPTIONS


class TestSpecialBlockText:
    def test_extract_block_text(self):
        text = "options { verbose = TRUE; };\nblock HOUSEHOLD { };"
        assert extract_special_block_content(text, "options") == "options { verbose = TRUE; };"
        assert extract_special_block_content(text, "OPTIONS") == "options { verbose = TRUE; };"
        assert extract_special_block_content(text, "tryreduce") is None

    def test_remove_block_text(self):
        text = "options { verbose = TRUE; };\nblock HOUSEHOLD { };"
        assert remove_special_block(text, "options") == "\nblock HOUSEHOLD { };"
        assert remove_special_block(text, "tryreduce") == text


class TestSpecialBlockErrors:
    def test_options_missing_semicolon(self):
        with pytest.raises(ParseBaseException):
            OPTIONS_BLOCK.parse_string("options { verbose = TRUE }")

    def test_tryreduce_missing_semicolon(self):
        with pytest.raises(ParseBaseException):
            TRYREDUCE_BLOCK.parse_string("tryreduce { U[] }")

    def test_assumptions_invalid_assumption(self):
        with pytest.raises(ParseBaseException, match="Unknown assumption 'invalid'"):
            ASSUMPTIONS_BLOCK.parse_string("assumptions { invalid { C[]; }; };")
