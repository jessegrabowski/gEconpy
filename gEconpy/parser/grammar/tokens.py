import pyparsing as pp

pp.ParserElement.enable_packrat()

LBRACE = pp.Suppress("{")
RBRACE = pp.Suppress("}")
LPAREN = pp.Suppress("(")
RPAREN = pp.Suppress(")")
LBRACKET = pp.Suppress("[")
RBRACKET = pp.Suppress("]")
SEMI = pp.Suppress(";")
COMMA = pp.Suppress(",")
EQUALS = pp.Suppress("=")
COLON = pp.Suppress(":")
TILDE = pp.Suppress("~")
ARROW = pp.Suppress("->")

EQUALS_LITERAL = pp.Literal("=")
COLON_LITERAL = pp.Literal(":")
ARROW_LITERAL = pp.Literal("->")

COMMENT = pp.python_style_comment

IDENTIFIER = pp.Word(pp.alphas + "_", pp.alphanums + "_")

NUMBER_PATTERN = r"(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?(?![.\w])"
NUMBER = pp.Regex(NUMBER_PATTERN)

TIME_INDEX_CONTENT = pp.Regex(r"-?\d+") | pp.Keyword("ss")
TIME_INDEX = LBRACKET + pp.Optional(TIME_INDEX_CONTENT, default="") + RBRACKET

KW_BLOCK = pp.CaselessKeyword("block")
KW_DEFINITIONS = pp.CaselessKeyword("definitions")
KW_CONTROLS = pp.CaselessKeyword("controls")
KW_OBJECTIVE = pp.CaselessKeyword("objective")
KW_CONSTRAINTS = pp.CaselessKeyword("constraints")
KW_IDENTITIES = pp.CaselessKeyword("identities")
KW_SHOCKS = pp.CaselessKeyword("shocks")
KW_CALIBRATION = pp.CaselessKeyword("calibration")
KW_OPTIONS = pp.CaselessKeyword("options")
KW_TRYREDUCE = pp.CaselessKeyword("tryreduce")
KW_ASSUMPTIONS = pp.CaselessKeyword("assumptions")
KW_TRUE = pp.CaselessKeyword("TRUE")
KW_FALSE = pp.CaselessKeyword("FALSE")
KW_E = pp.Keyword("E")

__all__ = [
    "ARROW",
    "ARROW_LITERAL",
    "COLON",
    "COLON_LITERAL",
    "COMMA",
    "COMMENT",
    "EQUALS",
    "EQUALS_LITERAL",
    "IDENTIFIER",
    "KW_ASSUMPTIONS",
    "KW_BLOCK",
    "KW_CALIBRATION",
    "KW_CONSTRAINTS",
    "KW_CONTROLS",
    "KW_DEFINITIONS",
    "KW_E",
    "KW_FALSE",
    "KW_IDENTITIES",
    "KW_OBJECTIVE",
    "KW_OPTIONS",
    "KW_SHOCKS",
    "KW_TRUE",
    "KW_TRYREDUCE",
    "LBRACE",
    "LBRACKET",
    "LPAREN",
    "NUMBER",
    "NUMBER_PATTERN",
    "RBRACE",
    "RBRACKET",
    "RPAREN",
    "SEMI",
    "TILDE",
    "TIME_INDEX",
    "TIME_INDEX_CONTENT",
]
