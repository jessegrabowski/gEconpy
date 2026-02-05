import re

from gEconpy.parser.ast import (
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    Variable,
)
from gEconpy.parser.grammar.distributions import parse_distribution
from gEconpy.parser.grammar.equations import parse_equation
from gEconpy.parser.grammar.expressions import parse_expression

COMPONENT_NAMES = {
    "definitions",
    "controls",
    "objective",
    "constraints",
    "identities",
    "shocks",
    "calibration",
}


def _split_statements(text: str) -> list[str]:
    """Split text into individual statements by semicolons, respecting brackets."""
    statements = []
    current = []
    depth = 0

    for char in text:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == ";" and depth == 0:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)

    # Handle any remaining content
    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)

    return statements


def _parse_variable_list(text: str) -> list[Variable]:
    """Parse a comma-separated list of variables like 'C[], L[], I[], K[];'."""
    # Remove trailing semicolon and whitespace
    text = text.strip().rstrip(";").strip()
    if not text:
        return []

    variables = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        # Parse as expression and extract variable
        expr = parse_expression(item)
        if isinstance(expr, Variable):
            variables.append(expr)
    return variables


def _extract_component_content(block_text: str, component_name: str) -> str | None:
    """Extract the content between braces for a component."""
    pattern = rf"\b{component_name}\s*\{{"
    match = re.search(pattern, block_text, re.IGNORECASE)
    if not match:
        return None

    start = match.end()
    depth = 1
    pos = start

    while pos < len(block_text) and depth > 0:
        if block_text[pos] == "{":
            depth += 1
        elif block_text[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        return None

    return block_text[start : pos - 1]


def _is_distribution_line(line: str) -> bool:
    """Check if a line is a distribution declaration (contains ~)."""
    # Must have ~ outside of any brackets
    depth = 0
    for char in line:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "~" and depth == 0:
            return True
    return False


def _parse_calibration_statements(
    statements: list[str],
) -> list[GCNEquation | GCNDistribution]:
    """Parse calibration statements which can be equations or distributions."""
    results = []
    for stmt in statements:
        if _is_distribution_line(stmt):
            results.append(parse_distribution(stmt))
        else:
            results.append(parse_equation(stmt))
    return results


def parse_block(name: str, content: str) -> GCNBlock:
    """
    Parse a block's content into a GCNBlock AST node.

    Parameters
    ----------
    name : str
        The block name (e.g., "HOUSEHOLD", "FIRM")
    content : str
        The content between the block's outer braces

    Returns
    -------
    GCNBlock
        The AST representation of the block.
    """
    block = GCNBlock(name=name)

    # Parse definitions
    definitions_content = _extract_component_content(content, "definitions")
    if definitions_content:
        statements = _split_statements(definitions_content)
        block.definitions = [parse_equation(s) for s in statements]

    # Parse controls
    controls_content = _extract_component_content(content, "controls")
    if controls_content:
        block.controls = _parse_variable_list(controls_content)

    # Parse objective
    objective_content = _extract_component_content(content, "objective")
    if objective_content:
        statements = _split_statements(objective_content)
        if statements:
            block.objective = parse_equation(statements[0])

    # Parse constraints
    constraints_content = _extract_component_content(content, "constraints")
    if constraints_content:
        statements = _split_statements(constraints_content)
        block.constraints = [parse_equation(s) for s in statements]

    # Parse identities
    identities_content = _extract_component_content(content, "identities")
    if identities_content:
        statements = _split_statements(identities_content)
        block.identities = [parse_equation(s) for s in statements]

    # Parse shocks
    shocks_content = _extract_component_content(content, "shocks")
    if shocks_content:
        block.shocks = _parse_variable_list(shocks_content)

    # Parse calibration
    calibration_content = _extract_component_content(content, "calibration")
    if calibration_content:
        statements = _split_statements(calibration_content)
        block.calibration = _parse_calibration_statements(statements)

    return block


def parse_block_from_text(text: str) -> GCNBlock:
    """
    Parse a complete block declaration from text.

    Parameters
    ----------
    text : str
        The full block text starting with 'block NAME { ... }'

    Returns
    -------
    GCNBlock
        The AST representation of the block.
    """
    # Extract block name
    match = re.match(r"\s*block\s+(\w+)\s*\{", text, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid block format: must start with 'block NAME {'")

    name = match.group(1)

    # Find the content between outer braces
    start = match.end() - 1  # Position of opening brace
    depth = 0
    pos = start

    while pos < len(text):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                break
        pos += 1

    content = text[start + 1 : pos]
    return parse_block(name, content)
