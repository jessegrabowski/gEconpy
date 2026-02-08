import pytest

from gEconpy.parser.errors import GCNLexerError
from gEconpy.parser.lexer import TokenType, tokenize


class TestLexerBasics:
    def test_empty_input_produces_only_eof(self):
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only_produces_eof(self):
        tokens = tokenize("   \n\t\n   ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_comments_are_skipped(self):
        tokens = tokenize("alpha # this is a comment\nbeta")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER]

    def test_line_numbers_tracked_correctly(self):
        tokens = tokenize("a\nb\nc")
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3


class TestVariablesAndParameters:
    @pytest.mark.parametrize(
        "text,expected_value",
        [
            ("C[]", "C[]"),
            ("K[-1]", "K[-1]"),
            ("Y[1]", "Y[1]"),
            ("A[ss]", "A[ss]"),
            ("X[-10]", "X[-10]"),
            ("consumption[]", "consumption[]"),
        ],
    )
    def test_variables_with_time_index(self, text, expected_value):
        tokens = tokenize(text)
        assert tokens[0].type == TokenType.VARIABLE
        assert tokens[0].value == expected_value

    def test_parameter_without_brackets(self):
        tokens = tokenize("alpha beta_1 delta")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert all(t == TokenType.IDENTIFIER for t in types)

    def test_expectation_operator(self):
        tokens = tokenize("E[]")
        assert tokens[0].type == TokenType.EXPECTATION
        assert tokens[0].value == "E[]"

    def test_expectation_in_expression(self):
        # E[][U[1]] should tokenize as E[], [, U[1], ]
        tokens = tokenize("E[][U[1]]")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types[0] == TokenType.EXPECTATION
        assert types[1] == TokenType.LBRACKET
        assert types[2] == TokenType.VARIABLE
        assert types[3] == TokenType.RBRACKET


class TestOperators:
    def test_single_char_operators(self):
        tokens = tokenize("+ - * / ^ = : ~")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.STAR,
            TokenType.SLASH,
            TokenType.CARET,
            TokenType.EQUALS,
            TokenType.COLON,
            TokenType.TILDE,
        ]

    def test_arrow_operator(self):
        tokens = tokenize("->")
        assert tokens[0].type == TokenType.ARROW
        assert tokens[0].value == "->"

    def test_arrow_vs_minus(self):
        # - followed by -> should be MINUS, ARROW
        tokens = tokenize("- ->")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.MINUS, TokenType.ARROW]


class TestNumbers:
    @pytest.mark.parametrize("text", ["42", "3.14", "0.5", "1e10", "2.5e-3"])
    def test_number_formats(self, text):
        tokens = tokenize(text)
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == text


class TestKeywords:
    @pytest.mark.parametrize(
        "keyword",
        ["block", "definitions", "controls", "objective", "constraints", "identities", "shocks", "calibration"],
    )
    def test_keywords_recognized(self, keyword):
        tokens = tokenize(keyword)
        assert tokens[0].type != TokenType.IDENTIFIER

    def test_keywords_case_insensitive(self):
        tokens = tokenize("BLOCK Block block")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert all(t == TokenType.BLOCK for t in types)


class TestBlockEnd:
    def test_block_end_token(self):
        tokens = tokenize("};")
        assert tokens[0].type == TokenType.RBRACE
        assert tokens[0].value == "};"


class TestErrors:
    def test_unexpected_character_raises(self):
        with pytest.raises(GCNLexerError) as exc_info:
            tokenize("alpha @ beta")
        assert "@" in str(exc_info.value)

    def test_error_includes_location(self):
        with pytest.raises(GCNLexerError) as exc_info:
            tokenize("line1\n@")
        assert exc_info.value.location.line == 2


class TestRealGCNSnippets:
    def test_equation(self):
        tokens = tokenize("Y[] = C[] + I[];")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.VARIABLE,
            TokenType.EQUALS,
            TokenType.VARIABLE,
            TokenType.PLUS,
            TokenType.VARIABLE,
            TokenType.SEMICOLON,
        ]

    def test_constraint_with_lagrange(self):
        tokens = tokenize("C[] + I[] = r[] * K[-1] : lambda[];")
        values = [t.value for t in tokens if t.type != TokenType.EOF]
        assert "lambda[]" in values
        assert "K[-1]" in values

    def test_bellman_equation(self):
        tokens = tokenize("U[] = u[] + beta * E[][U[1]];")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.EXPECTATION in types
        assert TokenType.VARIABLE in types

    def test_calibration_with_distribution(self):
        tokens = tokenize("beta ~ Beta(alpha=2, beta=5) = 0.99;")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.TILDE in types
        assert TokenType.IDENTIFIER in types  # Beta (the distribution)
