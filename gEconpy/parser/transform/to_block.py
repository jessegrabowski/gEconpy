"""
Bridge to convert new parser AST to Block objects.

This module converts the AST directly to sympy equations and creates Block
objects using Block.from_sympy(), bypassing the old token-based parsing.
"""

from collections import defaultdict

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.block import Block
from gEconpy.parser.ast import (
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Tag,
)
from gEconpy.parser.transform.to_sympy import ast_to_sympy


def _equation_to_sympy(
    eq: GCNEquation,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> sp.Eq:
    """Convert a GCNEquation AST to a sympy equation."""
    assumptions = assumptions or {}
    lhs = ast_to_sympy(eq.lhs, assumptions)
    rhs = ast_to_sympy(eq.rhs, assumptions)

    # For calibrating equations with `-> param`, rewrite as `Eq(param, rhs - lhs)`
    # This matches what Block._get_param_dict_and_calibrating_equations expects
    if eq.calibrating_parameter:
        param_assumptions = assumptions.get(eq.calibrating_parameter, {})
        param = sp.Symbol(eq.calibrating_parameter, **param_assumptions)
        # The calibrating equation form is: original_lhs = original_rhs -> param
        # Block expects: param = original_rhs - original_lhs (as equation to solve)
        # Actually, Block stores param -> original_rhs in calib_dict
        # where original_rhs is derived from solving original_lhs = original_rhs for param
        # For now, just put param on LHS and the RHS expression
        return sp.Eq(param, rhs - lhs)

    return sp.Eq(lhs, rhs)


def _variable_to_sympy(
    var,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> TimeAwareSymbol:
    """Convert a Variable AST to TimeAwareSymbol."""
    assumptions = assumptions or {}
    return ast_to_sympy(var, assumptions)


def _extract_flags(eq: GCNEquation) -> dict[str, bool]:
    """Extract equation flags from a GCNEquation."""
    flags = {}
    if eq.has_tag(Tag.EXCLUDE):
        flags["exclude"] = True
    if eq.calibrating_parameter:
        flags["is_calibrating"] = True
    else:
        flags["is_calibrating"] = False
    return flags


def _extract_multiplier(
    eq: GCNEquation,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> TimeAwareSymbol | None:
    """Extract Lagrange multiplier from a GCNEquation."""
    if not eq.lagrange_multiplier:
        return None
    assumptions = assumptions or {}
    name = eq.lagrange_multiplier
    var_assumptions = assumptions.get(name, {})
    return TimeAwareSymbol(name, 0, **var_assumptions)


def _convert_equation_list(
    equations: list[GCNEquation],
    assumptions: dict,
    eq_num: int,
    equation_flags: dict,
    multipliers: dict,
) -> tuple[dict[int, sp.Eq], int]:
    """Convert a list of GCNEquations to a dict of sympy equations."""
    result = {}
    for eq in equations:
        result[eq_num] = _equation_to_sympy(eq, assumptions)
        equation_flags[eq_num] = _extract_flags(eq)
        multipliers[eq_num] = _extract_multiplier(eq, assumptions)
        eq_num += 1
    return result, eq_num


def ast_block_to_block(
    ast_block: GCNBlock,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> Block:
    """
    Convert a GCNBlock AST directly to a Block object using Block.from_sympy().

    Parameters
    ----------
    ast_block : GCNBlock
        The AST block to convert.
    assumptions : dict, optional
        Variable assumptions for sympy symbols.

    Returns
    -------
    Block
        A Block object ready for solve_optimization().
    """
    assumptions = assumptions or defaultdict(dict)

    eq_num = 0
    equation_flags: dict[int, dict[str, bool]] = {}
    multipliers: dict[int, TimeAwareSymbol | None] = {}

    controls = [_variable_to_sympy(v, assumptions) for v in ast_block.controls] if ast_block.controls else None
    shocks = [_variable_to_sympy(v, assumptions) for v in ast_block.shocks] if ast_block.shocks else None

    definitions = None
    if ast_block.definitions:
        definitions, eq_num = _convert_equation_list(
            ast_block.definitions, assumptions, eq_num, equation_flags, multipliers
        )

    objective = None
    if ast_block.objective:
        objective, eq_num = _convert_equation_list(
            ast_block.objective, assumptions, eq_num, equation_flags, multipliers
        )

    constraints = None
    if ast_block.constraints:
        constraints, eq_num = _convert_equation_list(
            ast_block.constraints, assumptions, eq_num, equation_flags, multipliers
        )

    identities = None
    if ast_block.identities:
        identities, eq_num = _convert_equation_list(
            ast_block.identities, assumptions, eq_num, equation_flags, multipliers
        )

    calibration, eq_num = _convert_calibration(ast_block.calibration, assumptions, eq_num, equation_flags, multipliers)

    return Block.from_sympy(
        name=ast_block.name,
        definitions=definitions,
        controls=controls,
        objective=objective,
        constraints=constraints,
        identities=identities,
        calibration=calibration,
        shocks=shocks,
        multipliers=multipliers,
        equation_flags=equation_flags,
    )


def _convert_calibration(
    calibration_items: list,
    assumptions: dict,
    eq_num: int,
    equation_flags: dict,
    multipliers: dict,
) -> tuple[dict[int, sp.Eq] | None, int]:
    """Convert calibration items (equations and distributions) to sympy equations."""
    calib_items = []
    for item in calibration_items:
        if isinstance(item, GCNEquation):
            calib_items.append((item, _equation_to_sympy(item, assumptions)))
        elif isinstance(item, GCNDistribution) and item.initial_value is not None:
            param_assumptions = assumptions.get(item.parameter_name, {})
            param = sp.Symbol(item.parameter_name, **param_assumptions)
            value = sp.Float(item.initial_value)
            calib_items.append((None, sp.Eq(param, value)))

    if not calib_items:
        return None, eq_num

    calibration = {}
    for ast_eq, sympy_eq in calib_items:
        calibration[eq_num] = sympy_eq
        if ast_eq is not None:
            equation_flags[eq_num] = _extract_flags(ast_eq)
            multipliers[eq_num] = _extract_multiplier(ast_eq, assumptions)
        else:
            equation_flags[eq_num] = {"is_calibrating": False}
            multipliers[eq_num] = None
        eq_num += 1

    return calibration, eq_num


def ast_model_to_block_dict(
    model: GCNModel,
    assumptions: dict[str, dict[str, bool]] | None = None,
    simplify_blocks: bool = False,
) -> dict[str, Block]:
    """
    Convert a GCNModel AST to a dictionary of Block objects.

    This creates Block objects directly from the AST using Block.from_sympy(),
    then calls solve_optimization() on each block.

    Parameters
    ----------
    model : GCNModel
        The parsed model AST.
    assumptions : dict, optional
        Variable/parameter assumptions. If None, uses model.assumptions.
    simplify_blocks : bool, optional
        Whether to simplify block equations during optimization.

    Returns
    -------
    dict[str, Block]
        Dictionary mapping block names to Block objects.
    """
    if assumptions is None:
        assumptions = defaultdict(dict, model.assumptions or {})

    block_dict = {}

    for ast_block in model.blocks:
        # Skip STEADY_STATE blocks (handled separately)
        block_name_upper = ast_block.name.upper().replace("_", "")
        if block_name_upper in ("STEADYSTATE", "SS"):
            continue

        # Convert AST directly to Block using from_sympy
        block = ast_block_to_block(ast_block, assumptions)

        # Solve optimization (derives FOCs, creates Lagrange multipliers)
        block.solve_optimization(try_simplify=simplify_blocks)
        block_dict[block.name] = block

    return block_dict
