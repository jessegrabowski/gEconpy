from typing import Literal, Optional
from warnings import warn

import sympy as sp

from gEconpy.classes.block import Block
from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.simplification import simplify_singletons, try_reduce
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions.exceptions import (
    DuplicateParameterError,
    ExtraParameterError,
    MultipleSteadyStateBlocksException,
    OrphanParameterError,
)
from gEconpy.parser.constants import STEADY_STATE_NAMES
from gEconpy.parser.gEcon_parser import (
    ASSUMPTION_DICT,
    parsed_block_to_dict,
    preprocess_gcn,
    split_gcn_into_dictionaries,
)
from gEconpy.parser.parse_distributions import create_prior_distribution_dictionary
from gEconpy.parser.parse_equations import single_symbol_to_sympy
from gEconpy.shared.utilities import unpack_keys_and_values

PARAM_DICTS = Literal["param_dict", "deterministic_dict", "calib_dict"]


def read_gcn(
    gcn_path: str,
):
    outputs = gcn_to_block_dict(gcn_path, simplify_blocks=True)
    block_dict, assumptions, options, try_reduce, steady_state_equations, prior_info = outputs


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


def get_steady_state_equations(
    raw_blocks: dict[str, str],
    assumptions: ASSUMPTION_DICT = None,
) -> dict[str, sp.Expr]:
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
        return {}
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
            subbed_rhs = eq.rhs.subs(sub_dict)
            steady_state_dict[eq.lhs] = subbed_rhs
            sub_dict[eq.lhs] = subbed_rhs

    for k, eq in steady_state_dict.items():
        steady_state_dict[k] = eq.subs(steady_state_dict)

    steady_state_relationships = steady_state_dict.sort_keys().to_string().values_to_float()

    del raw_blocks[ss_key]

    return steady_state_relationships


def block_dict_to_equation_list(block_dict: dict[str, Block]) -> list[sp.Expr]:
    equations = []
    block_names, blocks = unpack_keys_and_values(block_dict)
    for block in blocks:
        equations.extend(block.system_equations)

    return equations


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
    shocks = sorted(list(set([x.set_t(0) for x in shocks])), key=lambda x: x.name)
    variables = sorted(
        list(set([x.set_t(0) for x in variables if x.set_t(0) not in shocks])), key=lambda x: x.name
    )
    return variables, shocks


def prior_info_to_prior_dict(
    prior_info: dict[str, str],
    assumptions: dict[str, dict[str, bool]],
    param_dict: SymbolDictionary,
    backend: Literal["scipy", "pymc"] = "scipy",
) -> tuple[SymbolDictionary, SymbolDictionary, SymbolDictionary]:
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
    param_dict: SymbolDictionary
        Dictionary of model parameters.
    backend: Literal["scipy", "pymc"]
        The backend into which the priors should be parsed.

    Returns
    -------
    param_priors: SymbolDictionary
        Dictionary of parameter priors
    shock_priors: SymbolDictionary
        Dictionary of shock priors
    hyper_priors_final: SymbolDictionary
        Dictionary of hyperparameter priors
    """
    priors, hyper_priors = create_prior_distribution_dictionary(prior_info)
    hyper_parameters = set(prior_info.keys()) - set(priors.keys())

    # Remove hyperparameters from the free parameters
    for parameter in hyper_parameters:
        del param_dict[parameter]

    param_priors = SymbolDictionary()
    shock_priors = SymbolDictionary()
    hyper_priors_final = SymbolDictionary()

    for key, value in priors.items():
        sympy_key = single_symbol_to_sympy(key, assumptions=assumptions)
        if isinstance(sympy_key, TimeAwareSymbol):
            shock_priors[sympy_key.base_name] = value
        else:
            param_priors[sympy_key.name] = value

    for key, value in hyper_priors.items():
        parent_rv, param_type, dist = value
        parent_key = single_symbol_to_sympy(parent_rv, assumptions=assumptions)
        param_key = single_symbol_to_sympy(key, assumptions=assumptions)

        hyper_priors_final[param_key] = (parent_key, param_type, dist)

    return param_priors, shock_priors, hyper_priors_final


def parsed_model_to_data(
    parsed_model: str, simplify_blocks: bool
) -> tuple[dict[str, Block], ASSUMPTION_DICT, dict[str, str], list[str], dict[str, sp.Expr]]:
    """
    Builds blocks of the gEconpy model using strings parsed from the GCN file.

    Parameters
    ----------
    parsed_model: str
        The GCN model as a string.
    simplify_blocks : bool
        Whether to try to simplify equations or not.

    Returns
    -------
    blocks: dict[str, Block]
        Dictionary of block names and block objects.
    assumptions: dict[str, dict[str, bool]]
        Dictionary of Sympy assumptions about model variables and parameters. Default is that variables are real, with
        unknown sign. See Sympy documentation for more details.
    options: dict[str, str]
        Dictionary of model options.
    tryreduce: list[str]
        List of variables to try to eliminate from model equations via substitution.
    steady_state_equations: dict[str, sp.Expr]
        Dictionary of user-provided steady-state equations. Keys are variable names, and values should be expressions
        giving the steady-state value of the variable in terms of parameters only.
    """

    block_dict: dict[str, Block] = {}
    raw_blocks, options, tryreduce, assumptions = split_gcn_into_dictionaries(parsed_model)
    steady_state_equations = get_steady_state_equations(raw_blocks, assumptions)

    for block_name, block_content in raw_blocks.items():
        parsed_block_dict = parsed_block_to_dict(block_content)
        block = Block(name=block_name, block_dict=parsed_block_dict, assumptions=assumptions)
        block.solve_optimization(try_simplify=simplify_blocks)

        block_dict[block.name] = block

    return block_dict, assumptions, options, tryreduce, steady_state_equations


def gcn_to_block_dict(
    gcn_path: str, simplify_blocks: bool
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
    block_dict, assumptions, options, tryreduce, steady_state_equations = parsed_model_to_data(
        parsed_model, simplify_blocks
    )

    tryreduce = [single_symbol_to_sympy(x, assumptions) for x in tryreduce]

    return block_dict, assumptions, options, tryreduce, steady_state_equations, prior_dict


def check_for_orphan_params(equations: list[sp.Expr], param_dict: SymbolDictionary) -> None:
    parameters = list(param_dict.to_sympy().keys())
    orphans = [
        atom
        for eq in equations
        for atom in eq.atoms()
        if (
            isinstance(atom, sp.Symbol)
            and not isinstance(atom, TimeAwareSymbol)
            and atom not in parameters
        )
    ]

    if len(orphans) > 0:
        raise OrphanParameterError(orphans)


def check_for_extra_params(equations: list[sp.Expr], param_dict: SymbolDictionary):
    parameters = list(param_dict.to_sympy().keys())
    all_atoms = set([atom for eq in equations for atom in eq.atoms()])
    extras = [parameter for parameter in parameters if parameter not in all_atoms]

    if len(extras) > 0:
        raise ExtraParameterError(extras)


def apply_simplifications(
    try_reduce_vars: list[TimeAwareSymbol],
    equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    do_try_reduce: bool = True,
    do_simplify_singletons: bool = True,
) -> tuple[
    list[sp.Expr],
    list[TimeAwareSymbol],
    Optional[list[TimeAwareSymbol]],
    Optional[list[TimeAwareSymbol]],
]:
    eliminated_variables = None
    singletons = None

    if do_try_reduce:
        equations, variables, eliminated_variables = try_reduce(
            try_reduce_vars, equations, variables
        )
    if do_simplify_singletons:
        equations, variables, singletons = simplify_singletons(equations, variables)

    return equations, variables, eliminated_variables, singletons


def validate_results(equations, param_dict):
    check_for_orphan_params(equations, param_dict)
    check_for_extra_params(equations, param_dict)


def block_dict_to_model(
    block_dict: dict[str, Block],
    assumptions: ASSUMPTION_DICT,
    options: dict[str, str],
    try_reduce_vars: list[TimeAwareSymbol],
    steady_state_equations: dict[str, sp.Expr],
    prior_info: dict[str, str],
    verbose=True,
):
    equations = block_dict_to_equation_list(block_dict)
    param_dict = block_dict_to_param_dict(block_dict, "param_dict")
    calib_dict = block_dict_to_param_dict(block_dict, "calib_dict")
    deterministic_dict = block_dict_to_param_dict(block_dict, "deterministic_dict")
    variables, shocks = block_dict_to_variables_and_shocks(block_dict)
    param_priors, shock_priors, hyper_priors_final = prior_info_to_prior_dict(
        prior_info, assumptions, param_dict
    )
    equations, variables, eliminated_variables, singletons = apply_simplifications(
        try_reduce_vars, equations, variables, do_try_reduce=True, do_simplify_singletons=True
    )
    validate_results(equations, param_dict)

    if verbose:
        build_report(
            equations,
            param_dict,
            calib_dict,
            variables,
            shocks,
            param_priors,
            shock_priors,
            eliminated_variables,
            singletons,
        )

    return


def build_report(
    equations: list[sp.Expr],
    param_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol],
    shocks: list[TimeAwareSymbol],
    param_priors: SymbolDictionary,
    shock_priors: SymbolDictionary,
    reduced_vars: list[TimeAwareSymbol],
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
        report += "\tThe following variables were eliminated at user request:\n"
        report += "\t\t" + ",".join([x.name for x in reduced_vars]) + "\n"

    if singletons:
        report += '\tThe following "variables" were defined as constants and have been substituted away:\n'
        report += "\t\t" + ",".join([x.name for x in singletons]) + "\n"

    report += f"\t{n_shocks} stochastic {shock_str}\n"
    report += (
        f'\t\t {len(shock_priors)} / {n_shocks} {"have" if len(shock_priors) == 1 else "has"}'
        f" a defined prior. \n"
    )

    report += f"\t{n_params} {free_par_str}\n"
    report += (
        f'\t\t {len(param_priors)} / {n_params} {"have" if len(param_priors) == 1 else "has"} '
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

    print(report)
