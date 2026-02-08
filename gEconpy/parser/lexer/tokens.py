from dataclasses import dataclass
from enum import Enum, auto

from gEconpy.parser.errors import ParseLocation


class TokenType(Enum):
    """Token types for the GCN lexer."""

    # Literals
    NUMBER = auto()
    IDENTIFIER = auto()

    # Variables and parameters
    VARIABLE = auto()  # X[], X[-1], X[1], X[ss]
    PARAMETER = auto()  # alpha, beta (no brackets)

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    CARET = auto()
    EQUALS = auto()
    ARROW = auto()  # -> (calibration assignment)
    COLON = auto()  # : (Lagrange multiplier marker)
    TILDE = auto()  # ~ (distribution assignment)

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()

    # Keywords
    BLOCK = auto()
    DEFINITIONS = auto()
    CONTROLS = auto()
    OBJECTIVE = auto()
    CONSTRAINTS = auto()
    IDENTITIES = auto()
    SHOCKS = auto()
    CALIBRATION = auto()
    TRYREDUCE = auto()
    OPTIONS = auto()
    ASSUMPTIONS = auto()

    # Special
    EXPECTATION = auto()  # E[]
    EOF = auto()


KEYWORDS = {
    "block": TokenType.BLOCK,
    "definitions": TokenType.DEFINITIONS,
    "controls": TokenType.CONTROLS,
    "objective": TokenType.OBJECTIVE,
    "constraints": TokenType.CONSTRAINTS,
    "identities": TokenType.IDENTITIES,
    "shocks": TokenType.SHOCKS,
    "calibration": TokenType.CALIBRATION,
    "tryreduce": TokenType.TRYREDUCE,
    "options": TokenType.OPTIONS,
    "assumptions": TokenType.ASSUMPTIONS,
}

SINGLE_CHAR_TOKENS = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "^": TokenType.CARET,
    "=": TokenType.EQUALS,
    ":": TokenType.COLON,
    "~": TokenType.TILDE,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ";": TokenType.SEMICOLON,
    ",": TokenType.COMMA,
}


@dataclass(frozen=True)
class Token:
    """
    A token produced by the lexer.

    Attributes
    ----------
    type : TokenType
        The type of the token.
    value : str
        The raw string value from the source.
    line : int
        1-based line number.
    column : int
        1-based column number.
    """

    type: TokenType
    value: str
    line: int
    column: int

    def to_location(self) -> ParseLocation:
        return ParseLocation(line=self.line, column=self.column)

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"
