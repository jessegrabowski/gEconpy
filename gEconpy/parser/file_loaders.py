import logging

from typing import Literal
from warnings import warn

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import (
    DuplicateParameterError,
    ExtraParameterError,
    ExtraParameterWarning,
    MultipleSteadyStateBlocksException,
    OrphanParameterError,
)
from gEconpy.model.block import Block
from gEconpy.model.simplification import simplify_constants, simplify_tryreduce
from gEconpy.parser.constants import STEADY_STATE_NAMES
from gEconpy.parser.gEcon_parser import (
    ASSUMPTION_DICT,
    parsed_block_to_dict,
    preprocess_gcn,
    split_gcn_into_dictionaries,
)
from gEconpy.parser.parse_distributions import (
    create_prior_distribution_dictionary,
)
from gEconpy.parser.parse_equations import single_symbol_to_sympy
from gEconpy.utilities import substitute_repeatedly, unpack_keys_and_values

PARAM_DICTS = Literal["param_dict", "deterministic_dict", "calib_dict"]
_log = logging.getLogger(__name__)


def load_gcn(gcn_path: str) -> str:
    """
    Loads a model file as raw text.

    Parameters
    ----------
    gcn_path : str
        File path to model file (GCN file).

    Returns
    -------
    str
        Raw-text of the model file.
    """

    with open(gcn_path, encoding="utf-8") as file:
        gcn_raw = file.read()
    return gcn_raw


def get_provided_ss_equations(
    raw_blocks: dict[str, str],
    assumptions: ASSUMPTION_DICT = None,
) -> tuple[Block | None, dict[str, sp.Expr]]:
    """
    Extract user-provided steady state equations from the `raw_blocks` dictionary and store the resulting
    relationships in self.steady_state_relationships.

    Parameters
    ----------
    raw_blocks: dict[str, str]
        Dictionary of block names and block contents extracted from a gEcon model.

    assumptions: dict[str, dict[str, bool]]
        Dictionary of assumptions about the model, with keys corresponding to variable names and values
        corresponding to dictionaries of assumptions about the variable. See sympy documentation for more details.

    Raises
    ------
    MultipleSteadyStateBlocksException
        If there is more than one block in `raw_blocks` with a name from `STEADY_STATE_NAMES`.
    """
    block_names = raw_blocks.keys()
    ss_block_names = [name for name in block_names if name in STEADY_STATE_NAMES]
    n_ss_blocks = len(ss_block_names)

    if n_ss_blocks == 0:
        return None, {}
    if n_ss_blocks > 1:
        raise MultipleSteadyStateBlocksException(ss_block_names)

    ss_key = next(iter(ss_block_names))
    block_content = raw_blocks[ss_key]

    block_dict = parsed_block_to_dict(block_content)
    block = Block(name="steady_state", block_dict=block_dict, assumptions=assumptions)

    sub_dict = SymbolDictionary()
    steady_state_dict = SymbolDictionary()

    if block.definitions is not None:
        _, definitions = unpack_keys_and_values(block.definitions)
        sub_dict = SymbolDictionary({eq.lhs: eq.rhs for eq in definitions})

    if block.identities is not None:
        _, identities = unpack_keys_and_values(block.identities)
        for eq in identities:
            subbed_rhs = substitute_repeatedly(eq.rhs, sub_dict)
            steady_state_dict[eq.lhs] = subbed_rhs
            sub_dict[eq.lhs] = subbed_rhs

    for k, eq in steady_state_dict.items():
        steady_state_dict[k] = substitute_repeatedly(eq, steady_state_dict)

    provided_ss_equations = steady_state_dict.sort_keys().to_string().values_to_float()

    del raw_blocks[ss_key]

    return block, provided_ss_equations


def simplify_provided_ss_equations(
    ss_solution_dict: SymbolDictionary, variables: list[TimeAwareSymbol]
) -> SymbolDictionary:
    if not ss_solution_dict:
        return SymbolDictionary()

    ss_variables = [x.to_ss() for x in variables]
    extra_equations = SymbolDictionary(
        {k: v for k, v in ss_solution_dict.to_sympy().items() if k not in ss_variables}
    )
    if not extra_equations:
        return ss_solution_dict

    simplified_ss_dict = SymbolDictionary(
        {k: v for k, v in ss_solution_dict.to_sympy().items() if k in ss_variables}
    )
    for var, eq in simplified_ss_dict.items():
        if not hasattr(eq, "subs"):
            continue
        simplified_ss_dict[var] = substitute_repeatedly(eq, extra_equations)

    return simplified_ss_dict


def block_dict_to_equation_list(block_dict: dict[str, Block]) -> list[sp.Expr]:
    equations = []
    block_names, blocks = unpack_keys_and_values(block_dict)
    for block in blocks:
        equations.extend(block.system_equations)

    return equations


def block_dict_to_sub_dict(
    block_dict: dict[str, Block],
) -> dict[TimeAwareSymbol, sp.Expr]:
    sub_dict = {}
    block_names, blocks = unpack_keys_and_values(block_dict)
    for block in blocks:
        for group in ["identities", "objective", "constraints"]:
            if getattr(block, group) is not None:
                _, equations = unpack_keys_and_values(getattr(block, group))
                for eq in equations:
                    sub_dict[eq.lhs] = eq.rhs

    return sub_dict


def block_dict_to_param_dict(
    block_dict: dict[str, Block], dict_name: PARAM_DICTS = "param_dict"
) -> SymbolDictionary:
    param_dict = SymbolDictionary()
    block_names, blocks = unpack_keys_and_values(block_dict)
    duplicates = set()

    for block in blocks:
        current_keys = set(param_dict.keys())
        new_keys = set(getattr(block, dict_name).keys())

        new_duplicates = current_keys.intersection(new_keys)
        duplicates = duplicates.union(new_duplicates)
        param_dict = param_dict | getattr(block, dict_name)

    if len(duplicates) > 0:
        raise DuplicateParameterError(duplicates)

    return param_dict.sort_keys().to_string().values_to_float()


def block_dict_to_variables_and_shocks(
    block_dict: dict[str, Block],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    variables = []
    shocks = []
    block_names, blocks = unpack_keys_and_values(block_dict)
    for block in blocks:
        if block.variables is not None:
            variables.extend(block.variables)
        if block.shocks is not None:
            shocks.extend(block.shocks)

    # Sort variables and shocks alphabetically by name, and set all time indices to 0
    shocks = sorted(list({x.set_t(0) for x in shocks}), key=lambda x: x.name)
    variables = sorted(
        list({x.set_t(0) for x in variables if x.set_t(0) not in shocks}),
        key=lambda x: x.name,
    )
    return variables, shocks


def prior_info_to_prior_dict(
    prior_info: dict[str, str],
    assumptions: dict[str, dict[str, bool]],
) -> tuple[SymbolDictionary, SymbolDictionary]:
    """
    Parse prior information extracted from GCN file and return dictionaries of parameter and shock priors.

    Parameters
    ----------
    prior_info: dict[str, str]
        Dictionary mapping shock and parameter names to priors. The priors are strings that can be parsed by the
        `parse_distributions` module.
    assumptions: dict[str, dict[str, bool]]
        Dictionary of assumptions about model parameters, with keys corresponding to variable names and values
        corresponding to dictionaries of assumptions about the variable. See sympy documentation for more details.

    Returns
    -------
    param_priors: SymbolDictionary
        Dictionary of parameter priors
    """
    params, shocks = create_prior_distribution_dictionary(prior_info)
    param_priors = SymbolDictionary()
    shock_priors = SymbolDictionary()

    for param_name, dist in params.items():
        sympy_key = single_symbol_to_sympy(param_name, assumptions=assumptions)
        param_priors[sympy_key.name] = dist

    for shock_name, dist in shocks.items():
        if not shock_name.endswith("[]"):
            # Add the [] back to the end of the shock name, because single_symbol_to_sympy uses it to decide
            # whether to create a time-aware symbol or a regular symbol
            shock_name = shock_name + "[]"
        sympy_key = single_symbol_to_sympy(shock_name, assumptions=assumptions)
        shock_priors[sympy_key.name] = dist

    return param_priors, shock_priors


def parsed_model_to_data(
    parsed_model: str, simplify_blocks: bool, include_ss_block: bool = False
) -> tuple[
    dict[str, Block], ASSUMPTION_DICT, dict[str, str], list[str], dict[str, sp.Expr]
]:
    """
    Builds blocks of the gEconpy model using strings parsed from the GCN file.

    Parameters
    ----------
    parsed_model: str
        The GCN model as a string.
    simplify_blocks : bool
        Whether to try to simplify equations or not.
    include_ss_block: bool
        If True, the user-provided steady-state are included among the model blocks. Otherwise, the raw block
        is discarded. Default is False.

    Returns
    -------
    blocks: dict[str, Block]
        Dictionary of block names and block objects. If `include_ss_block` is True, the steady-state block is
        included in the dictionary.
    assumptions: dict[str, dict[str, bool]]
        Dictionary of Sympy assumptions about model variables and parameters. Default is that variables are real, with
        unknown sign. See Sympy documentation for more details.
    options: dict[str, str]
        Dictionary of model options.
    tryreduce: list[str]
        List of variables to try to eliminate from model equations via substitution.
    provided_ss_equations: dict[str, sp.Expr]
        Dictionary of user-provided steady-state equations. Keys are variable names, and values should be expressions
        giving the steady-state value of the variable in terms of parameters only.
    """

    block_dict: dict[str, Block] = {}
    raw_blocks, options, tryreduce, assumptions = split_gcn_into_dictionaries(
        parsed_model
    )
    ss_block, provided_ss_equations = get_provided_ss_equations(raw_blocks, assumptions)

    if include_ss_block and ss_block:
        block_dict[ss_block.name] = ss_block

    for block_name, block_content in raw_blocks.items():
        parsed_block_dict = parsed_block_to_dict(block_content)
        block = Block(
            name=block_name, block_dict=parsed_block_dict, assumptions=assumptions
        )
        block.solve_optimization(try_simplify=simplify_blocks)
        block_dict[block.name] = block

    return block_dict, assumptions, options, tryreduce, provided_ss_equations


def gcn_to_block_dict(
    gcn_path: str, simplify_blocks: bool, include_ss_block=False
) -> tuple[
    dict[str, Block],
    ASSUMPTION_DICT,
    dict[str, str],
    list[TimeAwareSymbol],
    dict[str, sp.Expr],
    dict[str, str],
]:
    raw_model = load_gcn(gcn_path)
    parsed_model, prior_dict = preprocess_gcn(raw_model)
    block_dict, assumptions, options, tryreduce, ss_solution_dict = (
        parsed_model_to_data(
            parsed_model,
            simplify_blocks=simplify_blocks,
            include_ss_block=include_ss_block,
        )
    )

    tryreduce = [single_symbol_to_sympy(x, assumptions) for x in tryreduce]

    return block_dict, assumptions, options, tryreduce, ss_solution_dict, prior_dict


def check_for_orphan_params(
    equations: list[sp.Expr], param_dict: SymbolDictionary
) -> None:
    parameters = list(param_dict.to_sympy().keys())
    param_equations = [x for x in param_dict.values() if isinstance(x, sp.Expr)]

    orphans = [
        atom
        for eq in equations
        for atom in eq.atoms()
        if (
            isinstance(atom, sp.Symbol)
            and not isinstance(atom, TimeAwareSymbol)
            and atom not in parameters
            and not any(eq.has(atom) for eq in param_equations)
        )
    ]

    if len(orphans) > 0:
        raise OrphanParameterError(orphans)


def check_for_extra_params(
    equations: list[sp.Expr], param_dict: SymbolDictionary, on_unused_parameters="raise"
):
    parameters = list(param_dict.to_sympy().keys())
    param_equations = [x for x in param_dict.values() if isinstance(x, sp.Expr)]

    all_atoms = {atom for eq in equations + param_equations for atom in eq.atoms()}
    extras = [parameter for parameter in parameters if parameter not in all_atoms]

    if len(extras) > 0:
        if on_unused_parameters == "raise":
            raise ExtraParameterError(extras)
        elif on_unused_parameters == "warn":
            warn(ExtraParameterWarning(extras))
        else:
            return


def apply_simplifications(
    try_reduce_vars: list[TimeAwareSymbol],
    equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    tryreduce_sub_dict: dict[TimeAwareSymbol, sp.Expr] | None = None,
    do_simplify_tryreduce: bool = True,
    do_simplify_constants: bool = True,
) -> tuple[
    list[sp.Expr],
    list[TimeAwareSymbol],
    list[TimeAwareSymbol] | None,
    list[TimeAwareSymbol] | None,
]:
    eliminated_variables = None
    singletons = None

    if do_simplify_tryreduce:
        equations, variables, eliminated_variables = simplify_tryreduce(
            try_reduce_vars, equations, variables, tryreduce_sub_dict
        )

    if do_simplify_constants:
        equations, variables, singletons = simplify_constants(equations, variables)

    return equations, variables, eliminated_variables, singletons


def validate_results(
    equations,
    steady_state_relationships,
    param_dict,
    calib_dict,
    deterministic_dict,
    on_unused_parameters="raise",
):
    joint_dict = param_dict | calib_dict | deterministic_dict
    check_for_orphan_params(equations + steady_state_relationships, joint_dict)
    check_for_extra_params(
        equations + steady_state_relationships, joint_dict, on_unused_parameters
    )


def block_dict_to_model_primitives(
    block_dict: dict[str, Block],
    assumptions: ASSUMPTION_DICT,
    try_reduce_vars: list[TimeAwareSymbol],
    prior_info: dict[str, str],
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
):
    equations = block_dict_to_equation_list(block_dict)
    param_dict = block_dict_to_param_dict(block_dict, "param_dict")
    calib_dict = block_dict_to_param_dict(block_dict, "calib_dict")
    deterministic_dict = block_dict_to_param_dict(block_dict, "deterministic_dict")
    variables, shocks = block_dict_to_variables_and_shocks(block_dict)
    param_priors, shock_priors = prior_info_to_prior_dict(prior_info, assumptions)

    tryreduce_sub_dict = block_dict_to_sub_dict(block_dict)

    equations, variables, eliminated_variables, singletons = apply_simplifications(
        try_reduce_vars,
        equations,
        variables,
        tryreduce_sub_dict,
        do_simplify_tryreduce=simplify_tryreduce,
        do_simplify_constants=simplify_constants,
    )

    return (
        equations,
        param_dict,
        calib_dict,
        deterministic_dict,
        variables,
        shocks,
        param_priors,
        shock_priors,
        eliminated_variables,
        singletons,
    )


def build_report(
    equations: list[sp.Expr],
    param_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol],
    shocks: list[TimeAwareSymbol],
    param_priors: SymbolDictionary,
    shock_priors: SymbolDictionary,
    reduced_vars: list[TimeAwareSymbol],
    reduced_params: list[sp.Symbol],
    singletons: list[TimeAwareSymbol],
) -> None:
    """
    Write a diagnostic message after building the model. Note that successfully building the model does not
    guarantee that the model is correctly specified. For example, it is possible to build a model with more
    equations than parameters. This message will warn the user in this case.

    Parameters
    ----------
    equations: list[sp.Expr]

    param_dict: SymbolDictionary

    calib_dict: SymbolDictionary

    variables: list[TimeAwareSymbol]

    shocks: list[TimeAwareSymbol]

    param_priors: SymbolDictionary

    shock_priors: SymbolDictionary

    reduced_vars: list[TimeAwareSymbol]
        A list of variables reduced by the `try_reduce` method. Used to print the names of eliminated variables.

    reduced_params: list of Symbol
        A list of "deterministic" parameters eliminated via substitution. These are parameters that are only used
        in the definiton of other parameters.

    singletons: list[TimeAwareSymbol]
        A list of "singleton" variables -- those defined as time-invariant constants. Used ot print the sames of
        eliminated variables.

    verbose: bool
        Flag to print the build report to the terminal. Default is True. Regardless of the flag, the function will
        always issue a warning to the user if the system is not fully defined.

    Returns
    -------
    None
    """

    n_equations = len(equations)
    n_variables = len(variables)
    n_shocks = len(shocks)
    n_params_to_calibrate = len(calib_dict)
    n_free_params = len(param_dict)

    if singletons and len(singletons) == 0:
        singletons = None

    eq_str = "equation" if n_equations == 1 else "equations"
    var_str = "variable" if n_variables == 1 else "variables"
    shock_str = "shock" if n_shocks == 1 else "shocks"
    free_par_str = "parameter" if len(param_dict) == 1 else "parameters"
    calib_par_str = "parameter" if n_params_to_calibrate == 1 else "parameters"

    n_params = n_free_params + n_params_to_calibrate

    param_priors = param_priors.keys()
    shock_priors = shock_priors.keys()

    report = "Model Building Complete.\nFound:\n"
    report += f"\t{n_equations} {eq_str}\n"
    report += f"\t{n_variables} {var_str}\n"

    if reduced_vars:
        report += "\t\tThe following variables were eliminated at user request:\n"
        report += "\t\t\t" + ", ".join([x.name for x in reduced_vars]) + "\n"

    if singletons:
        report += '\t\tThe following "variables" were defined as constants and have been substituted away:\n'
        report += "\t\t\t" + ", ".join([x.name for x in singletons]) + "\n"

    report += f"\t{n_shocks} stochastic {shock_str}\n"
    report += (
        f"\t\t {len(shock_priors)} / {n_shocks} {'have' if len(shock_priors) == 1 else 'has'}"
        f" a defined prior. \n"
    )

    report += f"\t{n_params} {free_par_str}\n"
    if reduced_params:
        report += "\t\tThe following parameters were eliminated via substitution into other parameters:\n"
        report += "\t\t\t" + ", ".join([x.name for x in reduced_params]) + "\n"

    report += (
        f"\t\t {len(param_priors)} / {n_params} parameters {'have' if len(param_priors) == 1 else 'has'} "
        f"a defined prior. \n"
    )

    report += f"\t{n_params_to_calibrate} {calib_par_str} to calibrate.\n"

    if n_equations == n_variables:
        report += "Model appears well defined and ready to proceed to solving.\n"
    else:
        message = (
            f"The model does not appear correctly specified, there are {n_equations} {eq_str} but "
            f"{n_variables} {var_str}. It will not be possible to solve this model. Please check the "
            f"specification using available diagnostic tools, and check the GCN file for typos."
        )
        warn(message)

    _log.info(report)
