import os

from pathlib import Path

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.ast import GCNBlock, GCNDistribution, GCNEquation, GCNModel
from gEconpy.parser.ast_to_distribution import ast_to_distribution_with_metadata
from gEconpy.parser.ast_to_sympy import ast_to_sympy, equation_to_sympy
from gEconpy.parser.preprocessor import preprocess, preprocess_file

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


def ast_model_to_primitives(
    model: GCNModel,
) -> dict:
    """
    Convert a GCNModel to model primitives.

    Parameters
    ----------
    model : GCNModel
        The model to convert.

    Returns
    -------
    dict
        Dictionary containing:
        - equations: list of sympy equations
        - variables: list of TimeAwareSymbol
        - shocks: list of TimeAwareSymbol
        - param_dict: SymbolDictionary of parameters
        - calib_dict: SymbolDictionary of calibrating equations
        - distributions: SymbolDictionary of prior distributions
        - options: dict of model options
        - tryreduce: list of variables to try to reduce
        - assumptions: dict of variable assumptions
    """
    assumptions = model.assumptions
    all_variables = []
    all_shocks = []
    all_equations = []
    param_dict = SymbolDictionary()
    calib_dict = SymbolDictionary()
    distributions = SymbolDictionary()

    for block in model.blocks:
        # Get equations
        block_eqs = ast_block_to_equations(block, assumptions)
        for eq, _metadata in block_eqs["definitions"]:
            all_equations.append(eq)
        if block_eqs["objective"]:
            eq, _metadata = block_eqs["objective"]
            all_equations.append(eq)
        for eq, _metadata in block_eqs["constraints"]:
            all_equations.append(eq)
        for eq, _metadata in block_eqs["identities"]:
            all_equations.append(eq)

        # Get variables and shocks
        block_vars, block_shocks = ast_block_to_variables_and_shocks(block, assumptions)
        all_variables.extend(block_vars)
        all_shocks.extend(block_shocks)

        # Get calibration
        block_params, block_calib, block_dists = ast_block_to_calibration(block, assumptions)
        param_dict = param_dict | block_params
        calib_dict = calib_dict | block_calib
        distributions = distributions | block_dists

    # Deduplicate
    all_variables = sorted(set(all_variables), key=lambda x: x.name)
    all_shocks = sorted(set(all_shocks), key=lambda x: x.name)

    # Remove shocks from variables
    shock_names = {s.name for s in all_shocks}
    all_variables = [v for v in all_variables if v.name not in shock_names]

    return {
        "equations": all_equations,
        "variables": all_variables,
        "shocks": all_shocks,
        "param_dict": param_dict,
        "calib_dict": calib_dict,
        "distributions": distributions,
        "options": model.options,
        "tryreduce": model.tryreduce,
        "assumptions": model.assumptions,
    }


def load_gcn_file(filepath: str | Path) -> dict:
    """
    Load a GCN file and return model primitives.

    This is the main entry point for loading GCN files. Uses the new AST-based
    parser by default, but can fall back to the old parser via feature flag.

    Parameters
    ----------
    filepath : str | Path
        Path to the GCN file.

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
        from gEconpy.parser.file_loaders import block_dict_to_model_primitives, gcn_to_block_dict  # noqa: PLC0415

        filepath = Path(filepath)
        block_dict, assumptions, options, tryreduce, _ss_solution_dict, prior_dict = gcn_to_block_dict(
            filepath, simplify_blocks=False
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

    result = preprocess_file(filepath, validate=True)
    return ast_model_to_primitives(result.ast)


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
        from gEconpy.parser.file_loaders import block_dict_to_model_primitives, parsed_model_to_data  # noqa: PLC0415
        from gEconpy.parser.gEcon_parser import preprocess_gcn  # noqa: PLC0415

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
