import pytest

from gEconpy.parser.lexer.tokens import KEYWORDS, SINGLE_CHAR_TOKENS, Token, TokenType


class TestToken:
    def test_to_location_creates_parse_location(self):
        tok = Token(type=TokenType.IDENTIFIER, value="alpha", line=10, column=5)
        loc = tok.to_location()
        assert loc.line == 10
        assert loc.column == 5

    def test_tokens_are_frozen(self):
        tok = Token(type=TokenType.NUMBER, value="42", line=1, column=1)
        with pytest.raises(AttributeError):
            tok.value = "99"

    def test_tokens_can_be_used_in_sets(self):
        t1 = Token(type=TokenType.PLUS, value="+", line=1, column=1)
        t2 = Token(type=TokenType.PLUS, value="+", line=1, column=1)
        t3 = Token(type=TokenType.PLUS, value="+", line=2, column=1)
        assert t1 == t2
        assert t1 != t3
        assert len({t1, t2, t3}) == 2


class TestTokenMappings:
    @pytest.mark.parametrize("keyword", ["block", "definitions", "controls", "objective", "constraints"])
    def test_keywords_map_to_distinct_types(self, keyword):
        assert keyword in KEYWORDS
        assert isinstance(KEYWORDS[keyword], TokenType)

    def test_all_single_char_operators_covered(self):
        expected = set("+-*/^=:~(){}[];,")
        assert set(SINGLE_CHAR_TOKENS.keys()) == expected

    def test_arrow_not_in_single_char(self):
        # -> is two chars, must be handled specially by lexer
        assert "->" not in SINGLE_CHAR_TOKENS
