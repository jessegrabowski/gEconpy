import os

from pathlib import Path

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.ast import GCNBlock, GCNDistribution, GCNEquation, GCNModel
from gEconpy.parser.ast_to_block import ast_model_to_block_dict
from gEconpy.parser.ast_to_distribution import ast_to_distribution_with_metadata
from gEconpy.parser.ast_to_sympy import ast_to_sympy, equation_to_sympy
from gEconpy.parser.constants import STEADY_STATE_NAMES
from gEconpy.parser.file_loaders import (
    block_dict_to_equation_list,
    block_dict_to_model_primitives,
    block_dict_to_param_dict,
    block_dict_to_variables_and_shocks,
    gcn_to_block_dict,
    parsed_model_to_data,
)
from gEconpy.parser.gEcon_parser import preprocess_gcn
from gEconpy.parser.preprocessor import preprocess, preprocess_file
from gEconpy.utilities import substitute_repeatedly

# Feature flag for parser switching
# Set to True to use the new AST-based parser, False for the old parser
# Can be controlled via environment variable GECONPY_USE_NEW_PARSER
_USE_NEW_PARSER = os.environ.get("GECONPY_USE_NEW_PARSER", "1").lower() in ("1", "true", "yes")


def use_new_parser() -> bool:
    """Return whether the new parser is enabled."""
    return _USE_NEW_PARSER


def set_parser(use_new: bool) -> None:
    """
    Set which parser to use.

    Parameters
    ----------
    use_new : bool
        If True, use the new AST-based parser. If False, use the old parser.
    """
    global _USE_NEW_PARSER  # noqa: PLW0603
    _USE_NEW_PARSER = use_new


def ast_block_to_equations(
    block: GCNBlock,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> dict[str, list[tuple[sp.Eq, dict]]]:
    """
    Convert a GCNBlock's equations to sympy.

    Parameters
    ----------
    block : GCNBlock
        The block to convert.
    assumptions : dict, optional
        Variable/parameter assumptions.

    Returns
    -------
    dict
        Dictionary with keys for each component type containing
        lists of (equation, metadata) tuples.
    """
    assumptions = assumptions or {}
    result = {
        "definitions": [],
        "objective": None,
        "constraints": [],
        "identities": [],
    }

    for eq in block.definitions:
        result["definitions"].append(equation_to_sympy(eq, assumptions))

    if block.objective:
        result["objective"] = equation_to_sympy(block.objective, assumptions)

    for eq in block.constraints:
        result["constraints"].append(equation_to_sympy(eq, assumptions))

    for eq in block.identities:
        result["identities"].append(equation_to_sympy(eq, assumptions))

    return result


def ast_block_to_calibration(
    block: GCNBlock,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> tuple[SymbolDictionary, SymbolDictionary, SymbolDictionary]:
    """
    Extract calibration from a GCNBlock.

    Parameters
    ----------
    block : GCNBlock
        The block to extract calibration from.
    assumptions : dict, optional
        Variable/parameter assumptions.

    Returns
    -------
    tuple[SymbolDictionary, SymbolDictionary, SymbolDictionary]
        param_dict, calib_dict, distributions
    """
    assumptions = assumptions or {}
    param_dict = SymbolDictionary()
    calib_dict = SymbolDictionary()
    distributions = SymbolDictionary()

    for item in block.calibration:
        if isinstance(item, GCNDistribution):
            dist, metadata = ast_to_distribution_with_metadata(item)
            distributions[item.parameter_name] = dist
            # Also add initial value to param_dict
            if metadata["initial_value"] is not None:
                param_dict[item.parameter_name] = metadata["initial_value"]

        elif isinstance(item, GCNEquation):
            lhs_sympy = ast_to_sympy(item.lhs, assumptions)
            rhs_sympy = ast_to_sympy(item.rhs, assumptions)

            if item.is_calibrating:
                # Calibrating equation: param = expr -> 0
                calib_dict[item.calibrating_parameter] = sp.Eq(lhs_sympy, rhs_sympy)
            # Simple assignment: param = value
            elif hasattr(lhs_sympy, "name"):
                param_dict[lhs_sympy.name] = rhs_sympy
            else:
                param_dict[str(lhs_sympy)] = rhs_sympy

    return param_dict, calib_dict, distributions


def ast_block_to_variables_and_shocks(
    block: GCNBlock,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Extract variables and shocks from a GCNBlock.

    Parameters
    ----------
    block : GCNBlock
        The block to extract from.
    assumptions : dict, optional
        Variable/parameter assumptions.

    Returns
    -------
    tuple[list, list]
        List of variables, list of shocks.
    """
    assumptions = assumptions or {}
    variables = []
    shocks = []

    for var in block.controls:
        sympy_var = ast_to_sympy(var, assumptions)
        if isinstance(sympy_var, TimeAwareSymbol):
            variables.append(sympy_var.set_t(0))

    for shock in block.shocks:
        sympy_shock = ast_to_sympy(shock, assumptions)
        if isinstance(sympy_shock, TimeAwareSymbol):
            shocks.append(sympy_shock.set_t(0))

    # Also extract from equations (LHS of definitions, identities, objective)
    for eq in block.definitions + block.identities:
        lhs = ast_to_sympy(eq.lhs, assumptions)
        if isinstance(lhs, TimeAwareSymbol):
            variables.append(lhs.set_t(0))

    if block.objective:
        lhs = ast_to_sympy(block.objective.lhs, assumptions)
        if isinstance(lhs, TimeAwareSymbol):
            variables.append(lhs.set_t(0))

    # Also add constraint LHS
    for eq in block.constraints:
        lhs = ast_to_sympy(eq.lhs, assumptions)
        if isinstance(lhs, TimeAwareSymbol):
            variables.append(lhs.set_t(0))

    return list(set(variables)), list(set(shocks))


def _extract_ss_solution_dict(model: GCNModel, assumptions: dict) -> SymbolDictionary:
    """Extract user-provided steady state equations from STEADY_STATE block."""
    # Find STEADY_STATE block
    ss_blocks = [
        b for b in model.blocks if b.name.upper().replace("_", "") in [n.replace("_", "") for n in STEADY_STATE_NAMES]
    ]

    if not ss_blocks:
        return SymbolDictionary()

    ss_block = ss_blocks[0]
    sub_dict = SymbolDictionary()
    steady_state_dict = SymbolDictionary()

    # Process definitions first (they define substitution rules)
    for eq in ss_block.definitions:
        sympy_eq = _equation_to_sympy_simple(eq, assumptions)
        sub_dict[sympy_eq.lhs] = sympy_eq.rhs

    # Process identities (these define the steady state values)
    for eq in ss_block.identities:
        sympy_eq = _equation_to_sympy_simple(eq, assumptions)
        subbed_rhs = substitute_repeatedly(sympy_eq.rhs, sub_dict)
        steady_state_dict[sympy_eq.lhs] = subbed_rhs
        sub_dict[sympy_eq.lhs] = subbed_rhs

    # Substitute within steady state dict
    for k, eq in steady_state_dict.items():
        steady_state_dict[k] = substitute_repeatedly(eq, steady_state_dict)

    return steady_state_dict.sort_keys().to_string().values_to_float()


def _equation_to_sympy_simple(eq, assumptions):
    """Convert a GCNEquation to sympy Eq (helper for SS block)."""
    lhs = ast_to_sympy(eq.lhs, assumptions)
    rhs = ast_to_sympy(eq.rhs, assumptions)
    return sp.Eq(lhs, rhs)


def ast_model_to_primitives(
    model: GCNModel,
    simplify_blocks: bool = False,
) -> dict:
    """
    Convert a GCNModel to model primitives.

    This function builds Block objects from the AST, calls solve_optimization()
    on each block to derive FOCs, then extracts model primitives.

    Parameters
    ----------
    model : GCNModel
        The model to convert.
    simplify_blocks : bool, optional
        Whether to simplify block equations during optimization.

    Returns
    -------
    dict
        Dictionary containing:
        - equations: list of sympy equations
        - variables: list of TimeAwareSymbol
        - shocks: list of TimeAwareSymbol
        - param_dict: SymbolDictionary of parameters
        - calib_dict: SymbolDictionary of calibrating equations
        - deterministic_dict: SymbolDictionary of deterministic relationships
        - distributions: SymbolDictionary of prior distributions
        - ss_solution_dict: SymbolDictionary of user-provided steady state solutions
        - options: dict of model options
        - tryreduce: list of variables to try to reduce
        - assumptions: dict of variable assumptions
        - block_dict: dict of Block objects (for advanced use)
    """
    # Build Block objects and solve optimization
    block_dict, assumptions, options, tryreduce = ast_model_to_block_dict(model, simplify_blocks=simplify_blocks)

    # Extract primitives using the same functions as the old parser
    equations = block_dict_to_equation_list(block_dict)
    param_dict = block_dict_to_param_dict(block_dict, "param_dict")
    calib_dict = block_dict_to_param_dict(block_dict, "calib_dict")
    deterministic_dict = block_dict_to_param_dict(block_dict, "deterministic_dict")
    variables, shocks = block_dict_to_variables_and_shocks(block_dict)

    # Extract steady state solutions from STEADY_STATE block
    ss_solution_dict = _extract_ss_solution_dict(model, assumptions)

    # Handle distributions from calibration blocks
    distributions = SymbolDictionary()
    for ast_block in model.blocks:
        for item in ast_block.calibration:
            if isinstance(item, GCNDistribution):
                try:
                    dist, metadata = ast_to_distribution_with_metadata(item)
                    distributions[item.parameter_name] = dist
                except (ValueError, TypeError):
                    # Distribution has parameter references that can't be resolved yet
                    pass
        # Also include shock distributions
        for item in ast_block.shock_distributions:
            try:
                dist, metadata = ast_to_distribution_with_metadata(item)
                distributions[item.parameter_name] = dist
            except (ValueError, TypeError):
                pass

    return {
        "equations": equations,
        "variables": variables,
        "shocks": shocks,
        "param_dict": param_dict,
        "calib_dict": calib_dict,
        "deterministic_dict": deterministic_dict,
        "distributions": distributions,
        "ss_solution_dict": ss_solution_dict,
        "options": options,
        "tryreduce": tryreduce,
        "assumptions": assumptions,
        "block_dict": block_dict,
    }


def load_gcn_file(filepath: str | Path, simplify_blocks: bool = True) -> dict:
    """
    Load a GCN file and return model primitives.

    This is the main entry point for loading GCN files. Uses the new AST-based
    parser by default, but can fall back to the old parser via feature flag.

    Parameters
    ----------
    filepath : str | Path
        Path to the GCN file.
    simplify_blocks : bool, optional
        Whether to simplify block equations during optimization. Default True.

    Returns
    -------
    dict
        Dictionary of model primitives ready for Model construction.

    Notes
    -----
    The parser can be switched using:
    - Environment variable: GECONPY_USE_NEW_PARSER=0
    - Programmatically: gEconpy.parser.loader.set_parser(False)
    """
    if not _USE_NEW_PARSER:
        # Use old parser via adapter
        filepath = Path(filepath)
        block_dict, assumptions, options, tryreduce, ss_solution_dict, prior_dict = gcn_to_block_dict(
            filepath, simplify_blocks=simplify_blocks
        )
        (
            equations,
            param_dict,
            calib_dict,
            deterministic_dict,
            variables,
            shocks,
            param_priors,
            _shock_priors,
            _eliminated_variables,
            _singletons,
        ) = block_dict_to_model_primitives(
            block_dict,
            assumptions,
            tryreduce,
            prior_dict,
            simplify_tryreduce=False,
            simplify_constants=False,
        )
        return {
            "equations": equations,
            "variables": variables,
            "shocks": shocks,
            "param_dict": param_dict,
            "calib_dict": calib_dict,
            "deterministic_dict": deterministic_dict,
            "distributions": param_priors,
            "ss_solution_dict": ss_solution_dict,
            "options": options,
            "tryreduce": tryreduce,
            "assumptions": assumptions,
            "block_dict": block_dict,
        }

    result = preprocess_file(filepath, validate=True)
    return ast_model_to_primitives(result.ast, simplify_blocks=simplify_blocks)


def load_gcn_string(source: str) -> dict:
    """
    Load a GCN model from a string and return model primitives.

    Parameters
    ----------
    source : str
        GCN model source text.

    Returns
    -------
    dict
        Dictionary of model primitives ready for Model construction.

    Notes
    -----
    The parser can be switched using:
    - Environment variable: GECONPY_USE_NEW_PARSER=0
    - Programmatically: gEconpy.parser.loader.set_parser(False)
    """
    if not _USE_NEW_PARSER:
        # Use old parser via adapter

        parsed_model, prior_dict = preprocess_gcn(source)
        block_dict, assumptions, options, tryreduce, _ss_solution_dict = parsed_model_to_data(
            parsed_model, simplify_blocks=False
        )
        (
            equations,
            param_dict,
            calib_dict,
            _deterministic_dict,
            variables,
            shocks,
            param_priors,
            _shock_priors,
            _eliminated_variables,
            _singletons,
        ) = block_dict_to_model_primitives(
            block_dict,
            assumptions,
            tryreduce,
            prior_dict,
            simplify_tryreduce=False,
            simplify_constants=False,
        )
        return {
            "equations": equations,
            "variables": variables,
            "shocks": shocks,
            "param_dict": param_dict,
            "calib_dict": calib_dict,
            "distributions": param_priors,
            "options": options,
            "tryreduce": tryreduce,
            "assumptions": assumptions,
        }

    result = preprocess(source, validate=True)
    return ast_model_to_primitives(result.ast)
