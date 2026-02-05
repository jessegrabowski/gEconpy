from gEconpy.parser.errors import GCNLexerError, ParseLocation
from gEconpy.parser.tokens import KEYWORDS, SINGLE_CHAR_TOKENS, Token, TokenType


class Lexer:
    """
    Tokenizes GCN source text into a stream of tokens.

    The lexer handles GCN-specific syntax including:
    - Variables with time indices: X[], X[-1], X[1], X[ss]
    - The expectation operator: E[]
    - Multi-character operators: ->
    - Comments starting with #
    """

    def __init__(self, text: str, filename: str = ""):
        self.text = text
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        while not self._at_end():
            self._skip_whitespace_and_comments()
            if self._at_end():
                break
            self._scan_token()

        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens

    def _at_end(self) -> bool:
        return self.pos >= len(self.text)

    def _peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos >= len(self.text):
            return "\0"
        return self.text[pos]

    def _advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _add_token(self, token_type: TokenType, value: str, line: int, column: int):
        self.tokens.append(Token(token_type, value, line, column))

    def _skip_whitespace_and_comments(self):
        while not self._at_end():
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "#":
                self._skip_line_comment()
            else:
                break

    def _skip_line_comment(self):
        while not self._at_end() and self._peek() != "\n":
            self._advance()

    def _scan_token(self):
        start_line = self.line
        start_column = self.column
        ch = self._peek()

        # Two-character tokens
        if ch == "-" and self._peek(1) == ">":
            self._advance()
            self._advance()
            self._add_token(TokenType.ARROW, "->", start_line, start_column)
            return

        if ch == "}" and self._peek(1) == ";":
            self._advance()
            self._advance()
            self._add_token(TokenType.RBRACE, "};", start_line, start_column)
            return

        # Single-character tokens
        if ch in SINGLE_CHAR_TOKENS:
            self._advance()
            self._add_token(SINGLE_CHAR_TOKENS[ch], ch, start_line, start_column)
            return

        # Numbers
        if ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
            self._scan_number(start_line, start_column)
            return

        # Identifiers, keywords, variables
        if ch.isalpha() or ch == "_":
            self._scan_identifier(start_line, start_column)
            return

        # Unknown character
        self._advance()
        raise GCNLexerError(
            "Unexpected character",
            invalid_text=ch,
            location=ParseLocation(start_line, start_column, filename=self.filename),
        )

    def _scan_number(self, start_line: int, start_column: int):
        start_pos = self.pos

        while self._peek().isdigit():
            self._advance()

        if self._peek() == "." and self._peek(1).isdigit():
            self._advance()  # consume '.'
            while self._peek().isdigit():
                self._advance()

        # Scientific notation
        if self._peek() in "eE":
            self._advance()
            if self._peek() in "+-":
                self._advance()
            while self._peek().isdigit():
                self._advance()

        value = self.text[start_pos : self.pos]
        self._add_token(TokenType.NUMBER, value, start_line, start_column)

    def _scan_identifier(self, start_line: int, start_column: int):
        start_pos = self.pos

        while self._peek().isalnum() or self._peek() == "_":
            self._advance()

        name = self.text[start_pos : self.pos]

        # Check for E[] expectation operator
        if name == "E" and self._peek() == "[" and self._peek(1) == "]":
            self._advance()  # [
            self._advance()  # ]
            self._add_token(TokenType.EXPECTATION, "E[]", start_line, start_column)
            return

        # Check for variable with time index: X[], X[-1], X[1], X[ss]
        if self._peek() == "[":
            time_index = self._scan_time_index()
            if time_index is not None:
                full_value = name + time_index
                self._add_token(TokenType.VARIABLE, full_value, start_line, start_column)
                return

        # Check for keyword
        lower_name = name.lower()
        if lower_name in KEYWORDS:
            self._add_token(KEYWORDS[lower_name], name, start_line, start_column)
            return

        # Regular identifier (parameter)
        self._add_token(TokenType.IDENTIFIER, name, start_line, start_column)

    def _scan_time_index(self) -> str | None:
        """Try to scan a time index like [], [-1], [1], [ss]. Returns None if not a valid time index."""
        if self._peek() != "[":
            return None

        # Look ahead to see what's inside the brackets
        lookahead_pos = self.pos + 1
        content = ""

        while lookahead_pos < len(self.text):
            ch = self.text[lookahead_pos]
            if ch == "]":
                break
            if ch in " \t":
                lookahead_pos += 1
                continue
            content += ch
            lookahead_pos += 1

        if lookahead_pos >= len(self.text):
            return None

        # Valid time indices: empty, -1, 1, -2, 2, ss, etc.
        content = content.strip()
        valid = bool(content in {"", "ss"} or content.lstrip("-").isdigit())

        if not valid:
            return None

        # Actually consume the bracket and content
        result = ""
        result += self._advance()  # [

        while self._peek() != "]":
            result += self._advance()

        result += self._advance()  # ]

        return result


def tokenize(text: str, filename: str = "") -> list[Token]:
    return Lexer(text, filename).tokenize()
