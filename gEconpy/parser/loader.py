from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import DuplicateParameterError
from gEconpy.parser.ast import GCNBlock, GCNDistribution, GCNEquation, GCNModel
from gEconpy.parser.constants import STEADY_STATE_NAMES
from gEconpy.parser.preprocessor import preprocess, preprocess_file
from gEconpy.parser.transform.to_block import ast_model_to_block_dict
from gEconpy.parser.transform.to_distribution import ast_to_distribution_with_metadata
from gEconpy.parser.transform.to_sympy import ast_to_sympy, equation_to_sympy
from gEconpy.utilities import substitute_repeatedly

if TYPE_CHECKING:
    from gEconpy.model.block import Block

PARAM_DICTS = Literal["param_dict", "deterministic_dict", "calib_dict"]


@dataclass
class ModelPrimitives:
    """Container for all primitives extracted from a parsed GCN model."""

    equations: list[sp.Expr]
    variables: list[TimeAwareSymbol]
    shocks: list[TimeAwareSymbol]
    param_dict: SymbolDictionary
    calib_dict: SymbolDictionary
    deterministic_dict: SymbolDictionary
    distributions: SymbolDictionary
    shock_distributions: SymbolDictionary
    distribution_param_names: set[str]
    ss_solution_dict: SymbolDictionary
    options: dict[str, str]
    tryreduce: list[TimeAwareSymbol]
    assumptions: dict[str, dict[str, bool]]
    block_dict: dict[str, "Block"] = field(repr=False)


def _create_shock_distribution(
    item: GCNDistribution,
    param_distributions: SymbolDictionary,
) -> CompositeDistribution | None:
    """
    Create a CompositeDistribution for shock distributions with parameter references.

    Shock distributions like `epsilon[] ~ Normal(mu=0, sigma=sigma_A)` where `sigma_A`
    is a parameter need to be converted to CompositeDistribution objects that link
    the shock's distribution parameter to the hyper-parameter and its prior.

    Parameters
    ----------
    item : GCNDistribution
        The shock distribution AST node.
    param_distributions : SymbolDictionary
        Dictionary of parameter priors (for looking up hyper-parameter distributions).

    Returns
    -------
    CompositeDistribution or None
        A CompositeDistribution if the shock has parameter references, None otherwise.
    """
    if item.dist_name != "Normal":
        # Only Normal distributions are currently supported for shocks
        return None

    fixed_params = {}
    hyper_param_dict = {}
    param_name_to_hyper_name = {}

    for param_name, value in item.dist_kwargs.items():
        if isinstance(value, str):
            # Check if it's a number
            try:
                fixed_params[param_name] = float(value)
            except ValueError:
                # It's a parameter reference
                hyper_name = value
                # Look up the parameter's distribution
                if hyper_name in param_distributions:
                    hyper_param_dict[param_name] = param_distributions[hyper_name]
                    param_name_to_hyper_name[param_name] = hyper_name
                else:
                    # Parameter has no distribution - just use a placeholder
                    param_name_to_hyper_name[param_name] = hyper_name
        elif isinstance(value, (int, float)):
            fixed_params[param_name] = float(value)

    # If there are no hyper-parameters, this is a simple distribution
    if not param_name_to_hyper_name:
        return None

    return CompositeDistribution(
        name=item.parameter_name,
        dist_name=item.dist_name,
        fixed_params=fixed_params,
        hyper_param_dict=hyper_param_dict,
        param_name_to_hyper_name=param_name_to_hyper_name,
    )


def _block_dict_to_equation_list(block_dict: dict[str, "Block"]) -> list[sp.Expr]:
    """Extract all system equations from a dictionary of Block objects."""
    equations = []
    for block in block_dict.values():
        equations.extend(block.system_equations)
    return equations


def _block_dict_to_param_dict(
    block_dict: dict[str, "Block"], dict_name: PARAM_DICTS = "param_dict"
) -> SymbolDictionary:
    """Extract and merge parameter dictionaries from all blocks."""
    param_dict = SymbolDictionary()
    duplicates = set()

    for block in block_dict.values():
        current_keys = set(param_dict.keys())
        new_keys = set(getattr(block, dict_name).keys())
        new_duplicates = current_keys.intersection(new_keys)
        duplicates = duplicates.union(new_duplicates)
        param_dict = param_dict | getattr(block, dict_name)

    if len(duplicates) > 0:
        raise DuplicateParameterError(duplicates)

    return param_dict.sort_keys().to_string().values_to_float()


def _block_dict_to_variables_and_shocks(
    block_dict: dict[str, "Block"],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """Extract variables and shocks from all blocks."""
    variables = []
    shocks = []

    for block in block_dict.values():
        if block.variables is not None:
            variables.extend(block.variables)
        if block.shocks is not None:
            shocks.extend(block.shocks)

    shocks = sorted({x.set_t(0) for x in shocks}, key=lambda x: x.name)
    variables = sorted(
        {x.set_t(0) for x in variables if x.set_t(0) not in shocks},
        key=lambda x: x.name,
    )
    return variables, shocks


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
        "objective": [],
        "constraints": [],
        "identities": [],
    }

    for eq in block.definitions:
        result["definitions"].append(equation_to_sympy(eq, assumptions))

    for eq in block.objective:
        result["objective"].append(equation_to_sympy(eq, assumptions))

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
                # Create parameter symbol with appropriate assumptions
                param_assumptions = assumptions.get(item.calibrating_parameter, {})
                calib_param = sp.Symbol(item.calibrating_parameter, **param_assumptions)
                calib_dict[calib_param] = sp.Eq(lhs_sympy, rhs_sympy)
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

    for eq in block.objective:
        lhs = ast_to_sympy(eq.lhs, assumptions)
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


def _extract_tryreduce(
    model: GCNModel,
    variables: list[TimeAwareSymbol],
) -> list[TimeAwareSymbol]:
    """
    Convert tryreduce block entries to TimeAwareSymbol objects.

    Parameters
    ----------
    model : GCNModel
        The model AST containing tryreduce strings like "Pi[]", "U[]".
    variables : list[TimeAwareSymbol]
        List of model variables to match against.

    Returns
    -------
    list[TimeAwareSymbol]
        List of variables marked for reduction.
    """
    tryreduce = []
    for tr_str in model.tryreduce:
        base_name = tr_str.replace("[]", "").strip()
        for var in variables:
            if var.base_name == base_name:
                tryreduce.append(var)
                break
    return tryreduce


def ast_model_to_primitives(
    model: GCNModel,
    simplify_blocks: bool = False,
) -> ModelPrimitives:
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
    ModelPrimitives
        Dataclass containing all extracted model primitives.
    """
    # Extract model-level properties
    assumptions = dict(model.assumptions) if model.assumptions else {}
    options = model.options

    # Build Block objects and solve optimization
    block_dict = ast_model_to_block_dict(model, assumptions=assumptions, simplify_blocks=simplify_blocks)

    # Extract primitives from blocks
    equations = _block_dict_to_equation_list(block_dict)
    param_dict = _block_dict_to_param_dict(block_dict, "param_dict")
    calib_dict = _block_dict_to_param_dict(block_dict, "calib_dict")
    deterministic_dict = _block_dict_to_param_dict(block_dict, "deterministic_dict")
    variables, shocks = _block_dict_to_variables_and_shocks(block_dict)

    # Convert tryreduce strings to symbols
    tryreduce = _extract_tryreduce(model, variables)

    # Extract steady state solutions from STEADY_STATE block
    ss_solution_dict = _extract_ss_solution_dict(model, assumptions)

    # Handle distributions from calibration and shock blocks
    distributions = SymbolDictionary()
    shock_distributions = SymbolDictionary()
    distribution_param_names: set[str] = set()

    for ast_block in model.blocks:
        for item in ast_block.calibration:
            if isinstance(item, GCNDistribution):
                try:
                    dist, _metadata = ast_to_distribution_with_metadata(item)
                    distributions[item.parameter_name] = dist
                except (ValueError, TypeError):
                    # Distribution has parameter references that can't be resolved yet
                    pass

        # Handle shock distributions - may need CompositeDistribution for parameter references
        for item in ast_block.shock_distributions:
            shock_dist = _create_shock_distribution(item, distributions)
            if shock_dist is not None:
                shock_distributions[item.parameter_name] = shock_dist

            # Track parameter names used in distribution kwargs
            for kwarg_value in item.dist_kwargs.values():
                if isinstance(kwarg_value, str) and not kwarg_value.replace(".", "").replace("-", "").isdigit():
                    distribution_param_names.add(kwarg_value)

    return ModelPrimitives(
        equations=equations,
        variables=variables,
        shocks=shocks,
        param_dict=param_dict,
        calib_dict=calib_dict,
        deterministic_dict=deterministic_dict,
        distributions=distributions,
        shock_distributions=shock_distributions,
        distribution_param_names=distribution_param_names,
        ss_solution_dict=ss_solution_dict,
        options=options,
        tryreduce=tryreduce,
        assumptions=assumptions,
        block_dict=block_dict,
    )


def load_gcn_file(filepath: str | Path, simplify_blocks: bool = True) -> ModelPrimitives:
    """
    Load a GCN file and return model primitives.

    This is the main entry point for loading GCN files.

    Parameters
    ----------
    filepath : str | Path
        Path to the GCN file.
    simplify_blocks : bool, optional
        Whether to simplify block equations during optimization. Default True.

    Returns
    -------
    ModelPrimitives
        Dataclass containing model primitives ready for Model construction.
    """
    result = preprocess_file(filepath, validate=True)
    return ast_model_to_primitives(result.ast, simplify_blocks=simplify_blocks)


def load_gcn_string(source: str) -> ModelPrimitives:
    """
    Load a GCN model from a string and return model primitives.

    Parameters
    ----------
    source : str
        GCN model source text.

    Returns
    -------
    ModelPrimitives
        Dataclass containing model primitives ready for Model construction.
    """
    result = preprocess(source, validate=True)
    return ast_model_to_primitives(result.ast)
