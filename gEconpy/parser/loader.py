from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, merge_assumptions
from gEconpy.exceptions import DuplicateParameterError
from gEconpy.model.block import Block
from gEconpy.parser.ast import GCNBlock, GCNDistribution, GCNEquation, GCNModel, Node
from gEconpy.parser.constants import STEADY_STATE_BLOCK_KEYS
from gEconpy.parser.preprocessor import preprocess, preprocess_file
from gEconpy.parser.transform.to_block import ast_model_to_block_dict
from gEconpy.parser.transform.to_distribution import ast_to_distribution_with_metadata
from gEconpy.parser.transform.to_sympy import (
    _checked_eq,
    ast_to_sympy,
    equation_to_sympy,
)
from gEconpy.utilities import flatten_substitution_dict

ParamDictName = Literal["param_dict", "deterministic_dict", "calib_dict"]

_EQUATION_COMPONENTS = ("definitions", "objective", "constraints", "identities")


@dataclass
class ModelPrimitives:
    """
    Everything extracted from a parsed GCN file that :class:`~gEconpy.model.model.Model` construction needs.

    Parameters
    ----------
    equations : list of sympy expressions
        The model equations after each block's optimization problem is solved.
    variables : list of TimeAwareSymbol
        Model variables at time zero, sorted by name.
    shocks : list of TimeAwareSymbol
        Exogenous shocks at time zero, sorted by name.
    param_dict : SymbolDictionary
        Parameters with fixed numeric values.
    calib_dict : SymbolDictionary
        Parameters pinned down by a calibrating equation, mapped to that equation.
    deterministic_dict : SymbolDictionary
        Parameters defined as expressions of other parameters.
    distributions : SymbolDictionary
        Prior distributions declared in calibration blocks, keyed by parameter name.
    shock_distributions : SymbolDictionary
        Shock distributions whose parameters reference model parameters, keyed by shock name.
    distribution_param_names : set of str
        Parameter names referenced inside shock distribution arguments.
    ss_solution_dict : SymbolDictionary
        Steady-state values the user supplied in the steady state block.
    options : dict mapping str to str or bool
        Entries of the ``options`` block.
    tryreduce : list of TimeAwareSymbol
        Variables the ``tryreduce`` block marks for elimination.
    assumptions : dict mapping str to dict
        SymPy assumptions per symbol name.
    block_dict : dict mapping str to Block
        The solved blocks, keyed by block name.
    """

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
    options: dict[str, str | bool]
    tryreduce: list[TimeAwareSymbol]
    assumptions: dict[str, dict[str, bool]]
    block_dict: dict[str, Block] = field(repr=False)


def load_gcn_file(filepath: str | Path, simplify_blocks: bool = True) -> ModelPrimitives:
    """
    Parse a GCN file, solve each block's optimization problem, and collect the model primitives.

    Parameters
    ----------
    filepath : str or Path
        Path to the GCN file.
    simplify_blocks : bool, optional
        Simplify block equations while deriving first-order conditions. Defaults to True.

    Returns
    -------
    primitives : ModelPrimitives
        The extracted primitives, ready for :class:`~gEconpy.model.model.Model` construction.
    """
    parsed = preprocess_file(filepath, validate=True)
    return ast_model_to_primitives(parsed.ast, simplify_blocks=simplify_blocks, source=parsed.source)


def load_gcn_string(source: str) -> ModelPrimitives:
    """
    Parse GCN source text, solve each block's optimization problem, and collect the model primitives.

    Parameters
    ----------
    source : str
        GCN model source text.

    Returns
    -------
    primitives : ModelPrimitives
        The extracted primitives, ready for :class:`~gEconpy.model.model.Model` construction.
    """
    parsed = preprocess(source, validate=True)
    return ast_model_to_primitives(parsed.ast, source=parsed.source)


def ast_model_to_primitives(
    model: GCNModel,
    simplify_blocks: bool = False,
    source: str | None = None,
) -> ModelPrimitives:
    """
    Build :class:`~gEconpy.model.block.basic.Block` objects from a parsed model, solve them, and collect the primitives.

    Parameters
    ----------
    model : GCNModel
        The parsed model.
    simplify_blocks : bool, optional
        Simplify block equations while deriving first-order conditions. Defaults to False.
    source : str, optional
        The GCN source text, used for error reporting. Defaults to None.

    Returns
    -------
    primitives : ModelPrimitives
        The extracted primitives.
    """
    assumptions = dict(model.assumptions) if model.assumptions else {}

    ss_solution_dict = _extract_ss_solution_dict(model, assumptions)
    block_dict = ast_model_to_block_dict(
        model,
        assumptions=assumptions,
        simplify_blocks=simplify_blocks,
        source=source,
        ss_solution_dict=ss_solution_dict,
    )

    variables, shocks = _block_dict_to_variables_and_shocks(block_dict)
    distributions, shock_distributions, distribution_param_names = _extract_distributions(model)

    return ModelPrimitives(
        equations=_block_dict_to_equation_list(block_dict),
        variables=variables,
        shocks=shocks,
        param_dict=_block_dict_to_param_dict(block_dict, "param_dict"),
        calib_dict=_block_dict_to_param_dict(block_dict, "calib_dict"),
        deterministic_dict=_block_dict_to_param_dict(block_dict, "deterministic_dict"),
        distributions=distributions,
        shock_distributions=shock_distributions,
        distribution_param_names=distribution_param_names,
        ss_solution_dict=ss_solution_dict,
        options=model.options,
        tryreduce=_extract_tryreduce(model, variables),
        assumptions=assumptions,
        block_dict=block_dict,
    )


def ast_block_to_equations(
    block: GCNBlock,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> dict[str, list[tuple[sp.Eq, dict[str, Any]]]]:
    """
    Convert the equations of a parsed block to SymPy, grouped by component.

    Parameters
    ----------
    block : GCNBlock
        The block to convert.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per symbol name. Defaults to no assumptions.

    Returns
    -------
    equations : dict mapping str to list
        For each of ``definitions``, ``objective``, ``constraints``, and ``identities``, a list of
        ``(equation, metadata)`` pairs.
    """
    assumptions = assumptions or {}
    return {
        component: [equation_to_sympy(eq, assumptions) for eq in getattr(block, component)]
        for component in _EQUATION_COMPONENTS
    }


def ast_block_to_calibration(
    block: GCNBlock,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> tuple[SymbolDictionary, SymbolDictionary, SymbolDictionary]:
    """
    Split the calibration block of a parsed block into fixed values, calibrating equations, and priors.

    Parameters
    ----------
    block : GCNBlock
        The block whose calibration section is read.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per symbol name. Defaults to no assumptions.

    Returns
    -------
    param_dict : SymbolDictionary
        Parameters with fixed values, including the initial values of parameters that carry a prior.
    calib_dict : SymbolDictionary
        Parameters pinned down by a calibrating equation, mapped to that equation.
    distributions : SymbolDictionary
        Prior distributions, keyed by parameter name.
    """
    assumptions = assumptions or {}
    param_dict = SymbolDictionary()
    calib_dict = SymbolDictionary()
    distributions = SymbolDictionary()

    for item in block.calibration:
        if isinstance(item, GCNDistribution):
            dist, metadata = ast_to_distribution_with_metadata(item)
            distributions[item.parameter_name] = dist
            if metadata["initial_value"] is not None:
                param_dict[item.parameter_name] = metadata["initial_value"]

        elif isinstance(item, GCNEquation):
            lhs_sympy = ast_to_sympy(item.lhs, assumptions)
            rhs_sympy = ast_to_sympy(item.rhs, assumptions)

            if item.is_calibrating:
                param_assumptions = merge_assumptions(assumptions.get(item.calibrating_parameter))
                calib_param = sp.Symbol(item.calibrating_parameter, **param_assumptions)
                calib_dict[calib_param] = sp.Eq(lhs_sympy, rhs_sympy)
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
    Collect the variables and shocks a parsed block declares.

    Variables are the block's controls plus the left-hand sides of its definitions, identities, objective, and
    constraints.

    Parameters
    ----------
    block : GCNBlock
        The block to read.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per symbol name. Defaults to no assumptions.

    Returns
    -------
    variables : list of TimeAwareSymbol
        The distinct variables, at time zero.
    shocks : list of TimeAwareSymbol
        The distinct shocks, at time zero.
    """
    assumptions = assumptions or {}

    equation_lhs = [eq.lhs for eq in block.all_equations()]
    variables = _time_aware_symbols([*block.controls, *equation_lhs], assumptions)
    shocks = _time_aware_symbols(block.shocks, assumptions)

    return list(set(variables)), list(set(shocks))


def _time_aware_symbols(nodes: Sequence[Node], assumptions: dict[str, dict[str, bool]]) -> list[TimeAwareSymbol]:
    symbols = [ast_to_sympy(node, assumptions) for node in nodes]
    return [symbol.set_t(0) for symbol in symbols if isinstance(symbol, TimeAwareSymbol)]


def _block_dict_to_equation_list(block_dict: dict[str, Block]) -> list[sp.Expr]:
    return [eq for block in block_dict.values() for eq in block.system_equations]


def _block_dict_to_param_dict(block_dict: dict[str, Block], dict_name: ParamDictName) -> SymbolDictionary:
    merged = SymbolDictionary()
    duplicates = set()

    for block in block_dict.values():
        block_params = getattr(block, dict_name)
        duplicates |= set(merged.keys()) & set(block_params.keys())
        merged = merged | block_params

    if duplicates:
        raise DuplicateParameterError(duplicates)

    return merged.sort_keys().to_string().values_to_float()


def _block_dict_to_variables_and_shocks(
    block_dict: dict[str, Block],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol]]:
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


def _extract_ss_solution_dict(model: GCNModel, assumptions: dict[str, dict[str, bool]]) -> SymbolDictionary:
    ss_blocks = [b for b in model.blocks if b.name.upper().replace("_", "") in STEADY_STATE_BLOCK_KEYS]
    if not ss_blocks:
        return SymbolDictionary()

    ss_block = ss_blocks[0]

    # Definitions and identities both contribute substitution rules, but only identities are surfaced as steady-state
    # values. One combined dict in source order lets dependencies resolve forward. Flattening it once by topological
    # sort leaves every value free of other keys, so downstream code needs no fixed-point loop.
    sub_dict: dict[sp.Expr, sp.Expr] = {}
    identity_keys: list[sp.Expr] = []

    for eq in ss_block.definitions:
        sympy_eq = _equation_to_sympy_eq(eq, assumptions)
        sub_dict[sympy_eq.lhs] = sympy_eq.rhs

    for eq in ss_block.identities:
        sympy_eq = _equation_to_sympy_eq(eq, assumptions)
        sub_dict[sympy_eq.lhs] = sympy_eq.rhs
        identity_keys.append(sympy_eq.lhs)

    flat = flatten_substitution_dict(sub_dict)

    steady_state_dict = SymbolDictionary({k: flat[k] for k in identity_keys})
    return steady_state_dict.sort_keys().to_string().values_to_float()


def _equation_to_sympy_eq(eq: GCNEquation, assumptions: dict[str, dict[str, bool]]) -> sp.Eq:
    lhs = ast_to_sympy(eq.lhs, assumptions)
    rhs = ast_to_sympy(eq.rhs, assumptions)
    return _checked_eq(lhs, rhs, assumptions)


def _extract_tryreduce(model: GCNModel, variables: list[TimeAwareSymbol]) -> list[TimeAwareSymbol]:
    variables_by_base_name = {var.base_name: var for var in variables}
    base_names = (entry.replace("[]", "").strip() for entry in model.tryreduce)
    return [variables_by_base_name[name] for name in base_names if name in variables_by_base_name]


def _extract_distributions(model: GCNModel) -> tuple[SymbolDictionary, SymbolDictionary, set[str]]:
    distributions = SymbolDictionary()
    shock_distributions = SymbolDictionary()
    distribution_param_names: set[str] = set()

    for ast_block in model.blocks:
        for item in ast_block.calibration:
            if not isinstance(item, GCNDistribution):
                continue
            # PreliZ rejects a distribution whose arguments name other parameters. Those priors are resolved later,
            # once the hyper-parameters they reference have distributions of their own.
            try:
                dist, _metadata = ast_to_distribution_with_metadata(item)
            except (ValueError, TypeError):
                continue
            distributions[item.parameter_name] = dist

        for item in ast_block.shock_distributions:
            shock_dist = _create_shock_distribution(item, distributions)
            if shock_dist is not None:
                shock_distributions[item.parameter_name] = shock_dist

            for kwarg_value in item.dist_kwargs.values():
                if isinstance(kwarg_value, str) and not kwarg_value.replace(".", "").replace("-", "").isdigit():
                    distribution_param_names.add(kwarg_value)

    return distributions, shock_distributions, distribution_param_names


def _create_shock_distribution(
    item: GCNDistribution,
    param_distributions: SymbolDictionary,
) -> CompositeDistribution | None:
    """
    Link a shock distribution to the priors of the parameters its arguments name.

    A declaration such as ``epsilon[] ~ Normal(mu=0, sigma=sigma_A)`` ties the shock's ``sigma`` to the parameter
    ``sigma_A`` and, when ``sigma_A`` has a prior, to that prior.

    Parameters
    ----------
    item : GCNDistribution
        The shock distribution AST node.
    param_distributions : SymbolDictionary
        Parameter priors, keyed by parameter name.

    Returns
    -------
    distribution : CompositeDistribution or None
        The linked distribution, or None when the shock is not Normal or none of its arguments name a parameter.
    """
    if item.dist_name != "Normal":
        return None

    fixed_params = {}
    hyper_param_dict = {}
    param_name_to_hyper_name = {}

    for param_name, value in item.dist_kwargs.items():
        if isinstance(value, int | float):
            fixed_params[param_name] = float(value)
            continue
        if not isinstance(value, str):
            continue

        numeric_value = _parse_float(value)
        if numeric_value is not None:
            fixed_params[param_name] = numeric_value
            continue

        param_name_to_hyper_name[param_name] = value
        if value in param_distributions:
            hyper_param_dict[param_name] = param_distributions[value]

    if not param_name_to_hyper_name:
        return None

    return CompositeDistribution(
        name=item.parameter_name,
        dist_name=item.dist_name,
        fixed_params=fixed_params,
        hyper_param_dict=hyper_param_dict,
        param_name_to_hyper_name=param_name_to_hyper_name,
    )


def _parse_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None
