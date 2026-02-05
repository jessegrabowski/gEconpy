import re

from gEconpy.parser.ast import GCNModel
from gEconpy.parser.grammar.blocks import parse_block
from gEconpy.parser.grammar.special_blocks import (
    parse_assumptions,
    parse_options,
    parse_tryreduce,
    remove_special_block,
)


def _extract_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract all block declarations from GCN text.

    Returns list of (block_name, block_content) tuples.
    """
    blocks = []
    pattern = r"\bblock\s+(\w+)\s*\{"

    pos = 0
    while pos < len(text):
        match = re.search(pattern, text[pos:], re.IGNORECASE)
        if not match:
            break

        block_name = match.group(1)
        brace_start = pos + match.end() - 1

        # Find matching closing brace
        depth = 0
        end_pos = brace_start
        while end_pos < len(text):
            if text[end_pos] == "{":
                depth += 1
            elif text[end_pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            end_pos += 1

        # Extract content between braces
        content = text[brace_start + 1 : end_pos]
        blocks.append((block_name, content))

        # Move past this block (including optional trailing semicolon)
        pos = end_pos + 1
        if pos < len(text) and text[pos] == ";":
            pos += 1

    return blocks


def _remove_comments(text: str) -> str:
    """Remove # comments from text."""
    lines = text.split("\n")
    result = []
    for line in lines:
        # Find # that's not inside quotes
        comment_pos = -1
        in_quotes = False
        for i, ch in enumerate(line):
            if ch in {'"', "'"}:
                in_quotes = not in_quotes
            elif ch == "#" and not in_quotes:
                comment_pos = i
                break
        if comment_pos >= 0:
            result.append(line[:comment_pos])
        else:
            result.append(line)
    return "\n".join(result)


def parse_gcn(text: str) -> GCNModel:
    """
    Parse a complete GCN file into a GCNModel AST node.

    Parameters
    ----------
    text : str
        The full GCN file content.

    Returns
    -------
    GCNModel
        The AST representation of the complete model.
    """
    # Remove comments first
    text = _remove_comments(text)

    # Parse special blocks
    options = parse_options(text)
    tryreduce = parse_tryreduce(text)
    assumptions = parse_assumptions(text)

    # Remove special blocks from text before parsing model blocks
    remaining = text
    for block_name in ["options", "tryreduce", "assumptions"]:
        remaining = remove_special_block(remaining, block_name)

    # Extract and parse model blocks
    block_tuples = _extract_blocks(remaining)
    blocks = []
    for name, content in block_tuples:
        block = parse_block(name, content)
        blocks.append(block)

    return GCNModel(
        blocks=blocks,
        options=options,
        tryreduce=tryreduce,
        assumptions=dict(assumptions) if assumptions else {},
    )
