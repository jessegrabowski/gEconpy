import pyparsing as pp

from gEconpy.parser.ast import GCNBlock
from gEconpy.parser.grammar.components import COMPONENT
from gEconpy.parser.grammar.tokens import (
    COMMENT,
    IDENTIFIER,
    KW_BLOCK,
    LBRACE,
    RBRACE,
    SEMI,
)

MODEL_BLOCK = KW_BLOCK.suppress() - IDENTIFIER("name") - LBRACE - pp.ZeroOrMore(COMPONENT)("components") - RBRACE - SEMI


def _build_block(tokens) -> GCNBlock:
    block = GCNBlock(name=tokens.name)

    for comp_name, content in tokens.components:
        if comp_name == "definitions":
            block.definitions = content
        elif comp_name == "controls":
            block.controls = content
        elif comp_name == "objective":
            block.objective = content
        elif comp_name == "constraints":
            block.constraints = content
        elif comp_name == "identities":
            block.identities = content
        elif comp_name == "shocks":
            variables, distributions = content
            block.shocks = variables
            block.shock_distributions = distributions
        elif comp_name == "calibration":
            block.calibration = content

    return block


MODEL_BLOCK.set_parse_action(_build_block)
MODEL_BLOCK.ignore(COMMENT)


def parse_block(name: str, content: str) -> GCNBlock:
    """Parse block content into a GCNBlock (compatibility function)."""
    text = f"block {name} {{ {content} }}"
    result = MODEL_BLOCK.parse_string(text, parse_all=True)
    return result[0]


def parse_block_from_text(text: str) -> GCNBlock:
    """Parse a complete block definition from text."""
    result = MODEL_BLOCK.parse_string(text, parse_all=True)
    return result[0]


__all__ = [
    "MODEL_BLOCK",
    "parse_block",
    "parse_block_from_text",
]
