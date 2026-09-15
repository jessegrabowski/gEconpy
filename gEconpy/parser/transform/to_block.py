from collections import defaultdict
from dataclasses import dataclass
from typing import cast

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, merge_assumptions
from gEconpy.model.block import Block, dispatch_block
from gEconpy.parser.ast import (
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Parameter,
    Tag,
    Variable,
)
from gEconpy.parser.errors import ParseLocation
from gEconpy.parser.transform.expand_time_indices import expand_block_time_indices
from gEconpy.parser.transform.to_sympy import ASTToSympyConverter, ast_to_sympy

EQUATION_COMPONENTS = ("definitions", "objective", "constraints", "identities", "calibration")


def ast_model_to_block_dict(
    model: GCNModel,
    assumptions: dict[str, dict[str, bool]] | None = None,
    simplify_blocks: bool = False,
    source: str | None = None,
    ss_solution_dict: SymbolDictionary | None = None,
) -> dict[str, Block]:
    """
    Convert every block of a model AST to a solved :class:`~gEconpy.model.block.basic.Block`.

    Each block has its deep time indices expanded with :func:`expand_block_time_indices`, is converted with
    :func:`ast_block_to_block`, and then has its optimization problem solved. The steady-state block is skipped.

    Parameters
    ----------
    model : GCNModel
        The parsed model AST.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per variable or parameter name. Defaults to the assumptions declared in the model.
    simplify_blocks : bool, optional
        Whether to simplify the equations of each block while solving its optimization problem. Defaults to False.
    source : str, optional
        The GCN source text, used to point error messages at the offending line. Defaults to None.
    ss_solution_dict : SymbolDictionary, optional
        Analytically known steady-state values, used to resolve calibration expressions that reference
        steady-state variables. Defaults to None.

    Returns
    -------
    block_dict : dict mapping str to Block
        The solved blocks, keyed by block name.
    """
    if assumptions is None:
        assumptions = defaultdict(dict, model.assumptions or {})

    block_dict = {}
    for ast_block in model.blocks:
        if _is_steady_state_block(ast_block):
            continue

        expanded_block = expand_block_time_indices(ast_block)
        block = ast_block_to_block(expanded_block, assumptions, source=source, ss_solution_dict=ss_solution_dict)
        block.solve_optimization(try_simplify=simplify_blocks)
        block_dict[block.name] = block

    return block_dict


def ast_block_to_block(
    ast_block: GCNBlock,
    assumptions: dict[str, dict[str, bool]] | None = None,
    source: str | None = None,
    ss_solution_dict: SymbolDictionary | None = None,
) -> Block:
    """
    Convert a block AST to a :class:`~gEconpy.model.block.basic.Block`.

    Equations are numbered consecutively across the definitions, objective, constraints, identities, and
    calibration components, in that order.

    Parameters
    ----------
    ast_block : GCNBlock
        The block AST to convert.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per variable or parameter name. Defaults to no assumptions.
    source : str, optional
        The GCN source text, used to point error messages at the offending line. Defaults to None.
    ss_solution_dict : SymbolDictionary, optional
        Analytically known steady-state values, forwarded to the block for resolving calibration expressions that
        reference steady-state variables. Defaults to None.

    Returns
    -------
    block : Block
        A block ready for ``solve_optimization()``.
    """
    assumptions = assumptions or defaultdict(dict)

    converted = {
        "definitions": [_convert_equation(eq, assumptions) for eq in ast_block.definitions],
        "objective": [_convert_equation(eq, assumptions) for eq in ast_block.objective],
        "constraints": [_convert_equation(eq, assumptions) for eq in ast_block.constraints],
        "identities": [_convert_equation(eq, assumptions) for eq in ast_block.identities],
        "calibration": _convert_calibration(ast_block.calibration, assumptions),
    }

    components: dict[str, dict[int, sp.Eq] | None] = {}
    equation_flags: dict[int, dict[str, bool]] = {}
    multipliers: dict[int, TimeAwareSymbol | None] = {}
    eq_num = 0
    for name in EQUATION_COMPONENTS:
        equations = converted[name]
        if not equations:
            components[name] = None
            continue

        numbered: dict[int, sp.Eq] = {}
        for item in equations:
            numbered[eq_num] = item.equation
            equation_flags[eq_num] = item.flags
            multipliers[eq_num] = item.multiplier
            eq_num += 1
        components[name] = numbered

    symbol_locations = {
        var.name: var.location for var in [*ast_block.controls, *ast_block.shocks] if var.location is not None
    }
    symbol_locations.update(_calibration_locations(ast_block.calibration))

    return dispatch_block(
        name=ast_block.name,
        controls=[_variable_to_symbol(v, assumptions) for v in ast_block.controls] or None,
        shocks=[_variable_to_symbol(v, assumptions) for v in ast_block.shocks] or None,
        multipliers=multipliers,
        equation_flags=equation_flags,
        source=source,
        symbol_locations=symbol_locations,
        ss_solution_dict=ss_solution_dict,
        **components,
    )


@dataclass(frozen=True)
class _ConvertedEquation:
    equation: sp.Eq
    flags: dict[str, bool]
    multiplier: TimeAwareSymbol | None


def _variable_to_symbol(var: Variable, assumptions: dict[str, dict[str, bool]]) -> TimeAwareSymbol:
    return cast(TimeAwareSymbol, ast_to_sympy(var, assumptions))


def _is_steady_state_block(block: GCNBlock) -> bool:
    return block.name.upper().replace("_", "") in ("STEADYSTATE", "SS")


def _convert_equation(eq: GCNEquation, assumptions: dict[str, dict[str, bool]]) -> _ConvertedEquation:
    return _ConvertedEquation(
        equation=_equation_to_sympy(eq, assumptions),
        flags=_extract_flags(eq),
        multiplier=_extract_multiplier(eq, assumptions),
    )


def _equation_to_sympy(eq: GCNEquation, assumptions: dict[str, dict[str, bool]]) -> sp.Eq:
    converter = ASTToSympyConverter(assumptions)
    lhs = converter.convert_expr(eq.lhs)
    rhs = converter.convert_expr(eq.rhs)

    # The block reads a calibrating equation as ``param = residual``, so ``lhs = rhs -> param`` is stored with the
    # parameter on the left and the residual ``rhs - lhs`` on the right.
    if eq.calibrating_parameter:
        param_assumptions = merge_assumptions(assumptions.get(eq.calibrating_parameter))
        param = sp.Symbol(eq.calibrating_parameter, **param_assumptions)
        return cast(sp.Eq, sp.Eq(param, rhs - lhs))

    return cast(sp.Eq, sp.Eq(lhs, rhs))


def _extract_flags(eq: GCNEquation) -> dict[str, bool]:
    flags = {}
    if eq.has_tag(Tag.EXCLUDE):
        flags["exclude"] = True
    if eq.has_tag(Tag.MINIMIZE):
        flags["minimize"] = True
    if eq.has_tag(Tag.MAXIMIZE):
        flags["maximize"] = True
    flags["is_calibrating"] = bool(eq.calibrating_parameter)
    return flags


def _extract_multiplier(eq: GCNEquation, assumptions: dict[str, dict[str, bool]]) -> TimeAwareSymbol | None:
    if not eq.lagrange_multiplier:
        return None
    var_assumptions = merge_assumptions(assumptions.get(eq.lagrange_multiplier))
    return TimeAwareSymbol(eq.lagrange_multiplier, 0, **var_assumptions)


def _convert_calibration(
    calibration_items: list[GCNEquation | GCNDistribution],
    assumptions: dict[str, dict[str, bool]],
) -> list[_ConvertedEquation]:
    # A distribution declaration contributes an equation only when it carries an initial value.
    converted = []
    for item in calibration_items:
        if isinstance(item, GCNEquation):
            converted.append(_convert_equation(item, assumptions))
        elif item.initial_value is not None:
            param_assumptions = merge_assumptions(assumptions.get(item.parameter_name))
            param = sp.Symbol(item.parameter_name, **param_assumptions)
            converted.append(
                _ConvertedEquation(
                    equation=cast(sp.Eq, sp.Eq(param, sp.Float(item.initial_value))),
                    flags={"is_calibrating": False},
                    multiplier=None,
                )
            )
    return converted


def _calibration_locations(
    calibration_items: list[GCNEquation | GCNDistribution],
) -> dict[str, ParseLocation]:
    locations = {}
    for item in calibration_items:
        if item.location is None:
            continue
        if isinstance(item, GCNDistribution):
            locations[item.parameter_name] = item.location
        elif item.calibrating_parameter:
            locations[item.calibrating_parameter] = item.location
        elif isinstance(item.lhs, Parameter | Variable):
            locations[item.lhs.name] = item.location
    return locations
