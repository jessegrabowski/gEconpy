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
    @pytest.mark.parametrize(
        "grammar,text",
        [
            (LBRACE + RBRACE, "{}"),
            (LPAREN + RPAREN, "()"),
            (LBRACKET + RBRACKET, "[]"),
            (SEMI, ";"),
            (COMMA, ","),
            (EQUALS, "="),
            (COLON, ":"),
            (TILDE, "~"),
            (ARROW, "->"),
        ],
        ids=["braces", "parentheses", "brackets", "semicolon", "comma", "equals", "colon", "tilde", "arrow"],
    )
    def test_structural_tokens_are_suppressed(self, grammar, text):
        assert list(grammar.parse_string(text)) == []


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

    @pytest.mark.parametrize("text", ["123", "1abc"])
    def test_identifier_cannot_start_with_digit(self, text):
        with pytest.raises(ParseException):
            IDENTIFIER.parse_string(text)


class TestNumber:
    @pytest.mark.parametrize(
        "text",
        [
            "42",
            "0",
            "007",
            "3.14",
            "123.",
            ".5",
            ".123",
            "0.0",
            "1e10",
            "1E10",
            "1e+10",
            "1e-10",
            "1.5e10",
            ".5e10",
            "123.e10",
            "1.5E-3",
        ],
    )
    def test_valid_numbers(self, text):
        result = NUMBER.parse_string(text)
        assert result[0] == text

    def test_number_does_not_match_identifier(self):
        with pytest.raises(ParseException):
            NUMBER.parse_string("123abc", parse_all=True)


class TestTimeIndex:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("[]", ""),
            ("[-1]", "-1"),
            ("[1]", "1"),
            ("[ss]", "ss"),
            ("[-10]", "-10"),
            ("[100]", "100"),
        ],
    )
    def test_time_index(self, text, expected):
        result = TIME_INDEX.parse_string(text)
        assert result[0] == expected

    @pytest.mark.parametrize("text", ["-1", "1", "0", "ss", "-100"])
    def test_time_index_content(self, text):
        result = TIME_INDEX_CONTENT.parse_string(text)
        assert result[0] == text


class TestKeywords:
    @pytest.mark.parametrize("variant", ["block", "BLOCK", "Block", "BLoCK"])
    def test_block_keyword_is_case_insensitive(self, variant):
        result = KW_BLOCK.parse_string(variant)
        assert result[0].lower() == "block"

    @pytest.mark.parametrize(
        "keyword,expected",
        [
            (KW_DEFINITIONS, "definitions"),
            (KW_CONTROLS, "controls"),
            (KW_OBJECTIVE, "objective"),
            (KW_CONSTRAINTS, "constraints"),
            (KW_IDENTITIES, "identities"),
            (KW_SHOCKS, "shocks"),
            (KW_CALIBRATION, "calibration"),
            (KW_TRUE, "true"),
            (KW_FALSE, "false"),
        ],
    )
    def test_keywords_are_case_insensitive(self, keyword, expected):
        assert keyword.parse_string(expected.upper())[0].lower() == expected
        assert keyword.parse_string(expected.lower())[0].lower() == expected

    def test_expectation_keyword_is_case_sensitive(self):
        assert KW_E.parse_string("E")[0] == "E"

        with pytest.raises(ParseException):
            KW_E.parse_string("e")


class TestComments:
    def test_comment_to_end_of_line(self):
        result = COMMENT.parse_string("# this is a comment")
        assert len(result) == 1
        assert "this is a comment" in result[0]

    def test_comment_with_content_before(self):
        grammar = IDENTIFIER + COMMENT
        result = grammar.parse_string("alpha # comment")
        assert result[0] == "alpha"


class TestTokenCombinations:
    def test_variable_pattern(self):
        grammar = IDENTIFIER + TIME_INDEX
        result = grammar.parse_string("C[]")
        assert result[0] == "C"
        assert result[1] == ""

        result = grammar.parse_string("K[-1]")
        assert result[0] == "K"
        assert result[1] == "-1"

    def test_assignment_pattern(self):
        grammar = IDENTIFIER + EQUALS + NUMBER
        result = grammar.parse_string("beta = 0.99")
        assert result[0] == "beta"
        assert result[1] == "0.99"

    def test_distribution_pattern_structure(self):
        grammar = IDENTIFIER + TILDE + IDENTIFIER + LPAREN + RPAREN
        result = grammar.parse_string("alpha ~ Beta()")
        assert result[0] == "alpha"
        assert result[1] == "Beta"

    def test_lagrange_pattern(self):
        grammar = COLON + IDENTIFIER + TIME_INDEX
        result = grammar.parse_string(": lambda[]")
        assert result[0] == "lambda"
        assert result[1] == ""

    def test_calibrating_pattern(self):
        grammar = ARROW + IDENTIFIER
        result = grammar.parse_string("-> beta")
        assert result[0] == "beta"
