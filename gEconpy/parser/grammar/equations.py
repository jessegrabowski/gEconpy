import pyparsing as pp

from gEconpy.parser.ast import GCNEquation
from gEconpy.parser.grammar.expressions import parse_expression


def _parse_equation_string(text: str) -> GCNEquation:
    """
    Parse an equation string into a GCNEquation AST node.

    Handles:
    - Simple equations: Y[] = C[] + I[]
    - Equations with Lagrange multipliers: C[] + I[] = Y[] : lambda[]
    - Calibrating equations: beta = 0.99 -> beta
    """
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1].strip()

    lagrange_mult = None
    calibrating_param = None

    # Check for calibrating parameter (->)
    if " -> " in text or "\t->" in text or "-> " in text:
        # Find the -> that's not inside brackets
        arrow_pos = _find_outside_brackets(text, "->")
        if arrow_pos != -1:
            calibrating_param = text[arrow_pos + 2 :].strip()
            text = text[:arrow_pos].strip()

    # Check for Lagrange multiplier (: identifier[...])
    colon_pos = _find_lagrange_colon(text)
    if colon_pos != -1:
        lagrange_part = text[colon_pos + 1 :].strip()
        # Extract identifier from "lambda[]", "lambda[ss]", "mc[-1]", etc.
        # Find the identifier before the [
        bracket_pos = lagrange_part.find("[")
        if bracket_pos != -1:
            lagrange_mult = lagrange_part[:bracket_pos].strip()
        text = text[:colon_pos].strip()

    # Now split on = to get LHS and RHS
    equals_pos = _find_outside_brackets(text, "=")
    if equals_pos == -1:
        raise ValueError(f"No '=' found in equation: {text}")

    lhs_str = text[:equals_pos].strip()
    rhs_str = text[equals_pos + 1 :].strip()

    lhs = parse_expression(lhs_str)
    rhs = parse_expression(rhs_str)

    return GCNEquation(
        lhs=lhs,
        rhs=rhs,
        lagrange_multiplier=lagrange_mult,
        calibrating_parameter=calibrating_param,
    )


def _find_outside_brackets(text: str, target: str) -> int:
    """Find target string outside of any brackets or parentheses."""
    depth_paren = 0
    depth_bracket = 0
    i = 0
    while i < len(text) - len(target) + 1:
        ch = text[i]
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif depth_paren == 0 and depth_bracket == 0 and text[i : i + len(target)] == target:
            return i
        i += 1
    return -1


def _find_lagrange_colon(text: str) -> int:
    """
    Find the colon that marks a Lagrange multiplier.

    The colon must be followed by an identifier with brackets (possibly with time index),
    not just any colon. This distinguishes from colons in other contexts.
    """
    depth_paren = 0
    depth_bracket = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == ":" and depth_paren == 0 and depth_bracket == 0:
            # Check if followed by identifier with brackets (with optional content)
            rest = text[i + 1 :].strip()
            # Match: identifier[], identifier[ss], identifier[-1], identifier[1], etc.
            if pp.Regex(r"\w+\s*\[\s*(-?\d+|ss)?\s*\]").matches(rest):
                return i
        i += 1
    return -1


def parse_equation(text: str) -> GCNEquation:
    """
    Parse an equation string into a GCNEquation AST node.

    Parameters
    ----------
    text : str
        The equation to parse. Can include:
        - Simple equations: "Y[] = C[] + I[]"
        - Lagrange multipliers: "C[] + I[] = Y[] : lambda[]"
        - Calibrating equations: "beta = 0.99 -> beta"

    Returns
    -------
    GCNEquation
        The AST representation of the equation.
    """
    return _parse_equation_string(text)
