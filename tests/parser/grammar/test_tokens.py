"""Tests for grammar tokens."""

import pytest

from pyparsing import ParseException

from gEconpy.parser.grammar.tokens import (
    ARROW,
    COLON,
    COMMA,
    COMMENT,
    EQUALS,
    IDENTIFIER,
    KW_BLOCK,
    KW_CALIBRATION,
    KW_CONSTRAINTS,
    KW_CONTROLS,
    KW_DEFINITIONS,
    KW_E,
    KW_FALSE,
    KW_IDENTITIES,
    KW_OBJECTIVE,
    KW_SHOCKS,
    KW_TRUE,
    LBRACE,
    LBRACKET,
    LPAREN,
    NUMBER,
    RBRACE,
    RBRACKET,
    RPAREN,
    SEMI,
    TILDE,
    TIME_INDEX,
    TIME_INDEX_CONTENT,
)


class TestStructuralTokens:
    def test_braces(self):
        result = (LBRACE + RBRACE).parse_string("{}")
        assert list(result) == []  # Suppressed

    def test_parentheses(self):
        result = (LPAREN + RPAREN).parse_string("()")
        assert list(result) == []

    def test_brackets(self):
        result = (LBRACKET + RBRACKET).parse_string("[]")
        assert list(result) == []

    def test_semicolon(self):
        result = SEMI.parse_string(";")
        assert list(result) == []

    def test_comma(self):
        result = COMMA.parse_string(",")
        assert list(result) == []

    def test_equals(self):
        result = EQUALS.parse_string("=")
        assert list(result) == []

    def test_colon(self):
        result = COLON.parse_string(":")
        assert list(result) == []

    def test_tilde(self):
        result = TILDE.parse_string("~")
        assert list(result) == []

    def test_arrow(self):
        result = ARROW.parse_string("->")
        assert list(result) == []


class TestIdentifier:
    @pytest.mark.parametrize(
        "text",
        [
            "C",
            "alpha",
            "sigma_C",
            "_private",
            "K1",
            "var_123_test",
            "A",
            "LongVariableName",
        ],
    )
    def test_valid_identifiers(self, text):
        result = IDENTIFIER.parse_string(text)
        assert result[0] == text

    @pytest.mark.parametrize(
        "text",
        [
            "123",  # Starts with digit
            "1abc",  # Starts with digit
        ],
    )
    def test_invalid_identifiers(self, text):
        with pytest.raises(ParseException):
            IDENTIFIER.parse_string(text)


class TestNumber:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Integers
            ("42", "42"),
            ("0", "0"),
            ("007", "007"),
            # Floats
            ("3.14", "3.14"),
            ("123.", "123."),
            (".5", ".5"),
            (".123", ".123"),
            ("0.0", "0.0"),
            # Scientific notation
            ("1e10", "1e10"),
            ("1E10", "1E10"),
            ("1e+10", "1e+10"),
            ("1e-10", "1e-10"),
            ("1.5e10", "1.5e10"),
            (".5e10", ".5e10"),
            ("123.e10", "123.e10"),
            ("1.5E-3", "1.5E-3"),
        ],
    )
    def test_valid_numbers(self, text, expected):
        result = NUMBER.parse_string(text)
        assert result[0] == expected

    def test_number_does_not_match_identifier(self):
        # "123abc" should not match as number "123"

        with pytest.raises(ParseException):
            NUMBER.parse_string("123abc", parse_all=True)


class TestTimeIndex:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("[]", ""),  # Current period
            ("[-1]", "-1"),  # Lag
            ("[1]", "1"),  # Lead
            ("[ss]", "ss"),  # Steady state
            ("[-10]", "-10"),  # Multiple digit lag
            ("[100]", "100"),  # Multiple digit lead
        ],
    )
    def test_time_index(self, text, expected):
        result = TIME_INDEX.parse_string(text)
        assert result[0] == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("-1", "-1"),
            ("1", "1"),
            ("0", "0"),
            ("ss", "ss"),
            ("-100", "-100"),
        ],
    )
    def test_time_index_content(self, text, expected):
        result = TIME_INDEX_CONTENT.parse_string(text)
        assert result[0] == expected


class TestKeywords:
    def test_block_keyword(self):
        # Case insensitive
        for variant in ["block", "BLOCK", "Block", "BLoCK"]:
            result = KW_BLOCK.parse_string(variant)
            assert result[0].lower() == "block"

    def test_component_keywords(self):

        keywords = [
            (KW_DEFINITIONS, "definitions"),
            (KW_CONTROLS, "controls"),
            (KW_OBJECTIVE, "objective"),
            (KW_CONSTRAINTS, "constraints"),
            (KW_IDENTITIES, "identities"),
            (KW_SHOCKS, "shocks"),
            (KW_CALIBRATION, "calibration"),
        ]
        for kw, expected in keywords:
            result = kw.parse_string(expected.upper())
            assert result[0].lower() == expected

    def test_boolean_keywords(self):
        result_true = KW_TRUE.parse_string("TRUE")
        assert result_true[0].upper() == "TRUE"

        result_false = KW_FALSE.parse_string("FALSE")
        assert result_false[0].upper() == "FALSE"

        # Case insensitive
        result_true_lower = KW_TRUE.parse_string("true")
        assert result_true_lower[0].lower() == "true"

    def test_expectation_keyword_case_sensitive(self):
        # E must be uppercase
        result = KW_E.parse_string("E")
        assert result[0] == "E"

        # Lowercase should fail
        with pytest.raises(ParseException):
            KW_E.parse_string("e")


class TestComments:
    def test_comment_to_end_of_line(self):
        result = COMMENT.parse_string("# this is a comment")
        assert len(result) == 1
        assert "this is a comment" in result[0]

    def test_comment_with_content_before(self):
        # Comment should only match from # onwards
        grammar = IDENTIFIER + COMMENT
        result = grammar.parse_string("alpha # comment")
        assert result[0] == "alpha"


class TestTokenCombinations:
    def test_variable_pattern(self):
        # IDENTIFIER + TIME_INDEX should work together
        grammar = IDENTIFIER + TIME_INDEX
        result = grammar.parse_string("C[]")
        assert result[0] == "C"
        assert result[1] == ""

        result = grammar.parse_string("K[-1]")
        assert result[0] == "K"
        assert result[1] == "-1"

    def test_assignment_pattern(self):
        # IDENTIFIER + EQUALS + NUMBER
        grammar = IDENTIFIER + EQUALS + NUMBER
        result = grammar.parse_string("beta = 0.99")
        assert result[0] == "beta"
        assert result[1] == "0.99"

    def test_distribution_pattern_structure(self):
        # param ~ Dist(...)
        grammar = IDENTIFIER + TILDE + IDENTIFIER + LPAREN + RPAREN
        result = grammar.parse_string("alpha ~ Beta()")
        assert result[0] == "alpha"
        assert result[1] == "Beta"

    def test_lagrange_pattern(self):
        # : lambda[]
        grammar = COLON + IDENTIFIER + TIME_INDEX
        result = grammar.parse_string(": lambda[]")
        assert result[0] == "lambda"
        assert result[1] == ""

    def test_calibrating_pattern(self):
        # -> param
        grammar = ARROW + IDENTIFIER
        result = grammar.parse_string("-> beta")
        assert result[0] == "beta"
