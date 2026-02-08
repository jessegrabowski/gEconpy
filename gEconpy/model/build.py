import logging

from pathlib import Path
from warnings import warn

import pytensor.tensor as pt
import sympy as sp

from pymc.pytensorf import rewrite_pregrad
from pytensor import graph_replace

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.model.compile import BACKENDS
from gEconpy.model.model import Model
from gEconpy.model.perturbation import compile_linearized_system
from gEconpy.model.simplification import simplify_constants, simplify_tryreduce
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.steady_state import (
    ERROR_FUNCTIONS,
    compile_model_ss_functions,
    simplify_provided_ss_equations,
    system_to_steady_state,
)
from gEconpy.parser.loader import load_gcn_file
from gEconpy.parser.validation import validate_results
from gEconpy.utilities import get_name, substitute_repeatedly

_log = logging.getLogger(__name__)


def split_out_hyper_params(
    param_dict: SymbolDictionary[str, float],
    shock_prior: SymbolDictionary[str, CompositeDistribution],
) -> SymbolDictionary[str, float]:
    """
    Remove shock hyper parameters from the parameter dictionary.

    Parameters
    ----------
    param_dict: SymbolDictionary
        Dictionary of initial parameter values

    shock_prior: SymbolDictionary
        Dictionary of shock priors

    Returns
    -------
    param_dict: SymbolDictionary
        Dictionary of initial parameter values with shock hyper parameters removed.
    """
    new_param_dict = param_dict.copy()
    hyper_param_dict = SymbolDictionary()

    all_hyper_params = [param for dist in shock_prior.values() for param in dist.param_name_to_hyper_name.values()]

    for param in all_hyper_params:
        if param in new_param_dict:
            del new_param_dict[param]
            hyper_param_dict[param] = param_dict[param]

    return new_param_dict, hyper_param_dict


def _block_dict_to_sub_dict(block_dict):
    """Extract substitution dictionary from block equations for tryreduce."""
    sub_dict = {}
    for block in block_dict.values():
        for group in ["identities", "objective", "constraints"]:
            group_dict = getattr(block, group, None)
            if group_dict is not None:
                for eq in group_dict.values():
                    sub_dict[eq.lhs] = eq.rhs
    return sub_dict


def _apply_simplifications(
    try_reduce_vars,
    equations,
    variables,
    tryreduce_sub_dict=None,
    do_simplify_tryreduce_flag=True,
    do_simplify_constants_flag=True,
):
    """Apply tryreduce and constant simplifications to equations."""
    eliminated_variables = None
    singletons = None

    if do_simplify_tryreduce_flag:
        equations, variables, eliminated_variables = simplify_tryreduce(
            try_reduce_vars, equations, variables, tryreduce_sub_dict
        )

    if do_simplify_constants_flag:
        equations, variables, singletons = simplify_constants(equations, variables)

    return equations, variables, eliminated_variables, singletons


def _compile_gcn(
    gcn_path: Path,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    backend: BACKENDS = "numpy",
    return_symbolic: bool = False,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    **kwargs,
) -> tuple[tuple, tuple, tuple, dict, tuple, dict]:
    # Load using new parser
    primitives = load_gcn_file(gcn_path, simplify_blocks=simplify_blocks)

    equations = primitives.equations
    variables = primitives.variables
    shocks = primitives.shocks
    param_dict = primitives.param_dict
    calib_dict = primitives.calib_dict
    deterministic_dict = primitives.deterministic_dict
    ss_solution_dict = primitives.ss_solution_dict
    options = primitives.options
    try_reduce = primitives.tryreduce
    block_dict = primitives.block_dict
    distributions = primitives.distributions
    shock_distributions = primitives.shock_distributions
    distribution_param_names = primitives.distribution_param_names

    # Build sub_dict for tryreduce from block equations
    tryreduce_sub_dict = _block_dict_to_sub_dict(block_dict)

    # Apply simplifications
    equations, variables, reduced_vars, singletons = _apply_simplifications(
        try_reduce,
        equations,
        variables,
        tryreduce_sub_dict,
        do_simplify_tryreduce_flag=simplify_tryreduce,
        do_simplify_constants_flag=simplify_constants,
    )

    # Split distributions into param_priors and shock_priors
    param_priors = SymbolDictionary()
    shock_priors = SymbolDictionary()
    shock_names = {s.base_name for s in shocks}

    # Shock priors come from shock_distributions (CompositeDistribution objects)
    # Also collect hyper-param names to exclude from param_priors
    shock_hyper_param_names = set()
    for name, dist in shock_distributions.items():
        shock_priors[name] = dist
        if hasattr(dist, "param_name_to_hyper_name"):
            shock_hyper_param_names.update(dist.param_name_to_hyper_name.values())

    # Parameter priors are distributions not associated with shocks or shock hyper-params
    for name, dist in distributions.items():
        if name not in shock_names and name not in shock_hyper_param_names:
            param_priors[name] = dist

    param_dict, hyper_param_dict = split_out_hyper_params(param_dict, shock_priors)

    ss_solution_dict = simplify_provided_ss_equations(ss_solution_dict, variables)
    steady_state_relationships = [sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()]

    # TODO: Move this to a separate function
    # TODO: Add option to not eliminate deterministic parameters (the user might be interested in them)

    deterministic_dict.to_sympy(inplace=True)
    for param, expr in deterministic_dict.items():
        deterministic_dict[param] = substitute_repeatedly(expr, deterministic_dict)

    # If a deterministic parameter is only used in other parameters, it will now have been completely substituted away
    # and can be removed
    reduced_params = []
    final_deterministics = deterministic_dict.copy()

    for param in deterministic_dict:
        if not any(eq.has(param) for eq in equations + steady_state_relationships):
            reduced_params.append(param)
            del final_deterministics[param]

    deterministic_dict = final_deterministics.to_string()

    validate_results(
        equations,
        steady_state_relationships,
        param_dict,
        calib_dict,
        deterministic_dict,
        on_unused_parameters=on_unused_parameters,
        distributions=distributions,
        distribution_param_names=distribution_param_names,
    )
    steady_state_equations = system_to_steady_state(equations, shocks)

    variables = sorted(variables, key=lambda x: x.base_name)
    shocks = sorted(shocks, key=lambda x: x.base_name)

    functions, cache = compile_model_ss_functions(
        steady_state_equations,
        ss_solution_dict,
        variables,
        param_dict,
        deterministic_dict,
        calib_dict,
        error_func=error_function,
        backend=backend,
        return_symbolic=return_symbolic,
        **kwargs,
    )

    f_params, f_ss, resid_funcs, error_funcs = functions
    f_ss_resid, f_ss_jac = resid_funcs
    f_ss_error, f_ss_grad, f_ss_hess, f_ss_hessp = error_funcs

    f_linearize, cache = compile_linearized_system(
        equations,
        variables,
        param_dict,
        deterministic_dict,
        calib_dict,
        shocks,
        backend=backend,
        return_symbolic=return_symbolic,
        cache=cache,
    )

    if verbose:
        build_report(
            equations,
            param_dict,
            calib_dict,
            variables,
            shocks,
            param_priors,
            shock_priors,
            reduced_vars,
            reduced_params,
            singletons,
        )

    objects = (variables, shocks, equations, steady_state_relationships)
    dictionaries = (param_dict, hyper_param_dict, deterministic_dict, calib_dict)
    functions = (
        f_ss,
        f_ss_jac,
        f_params,
        f_ss_resid,
        f_ss_error,
        f_ss_grad,
        f_ss_hess,
        f_ss_hessp,
        f_linearize,
    )
    priors = (param_priors, shock_priors)

    return objects, dictionaries, functions, cache, priors, options


def model_from_gcn(
    gcn_path: str | Path,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    backend: BACKENDS = "numpy",
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    **kwargs,
) -> Model:
    gcn_path = Path(gcn_path)
    objects, dictionaries, functions, _cache, priors, options = _compile_gcn(
        gcn_path,
        simplify_blocks=simplify_blocks,
        simplify_tryreduce=simplify_tryreduce,
        simplify_constants=simplify_constants,
        verbose=verbose,
        backend=backend,
        error_function=error_function,
        on_unused_parameters=on_unused_parameters,
        **kwargs,
    )

    variables, shocks, equations, ss_relationships = objects
    param_dict, hyper_param_dict, deterministic_dict, calib_dict = dictionaries

    (
        f_ss,
        f_ss_jac,
        f_params,
        f_ss_resid,
        f_ss_error,
        f_ss_grad,
        f_ss_hess,
        f_ss_hessp,
        f_linearize,
    ) = functions

    return Model(
        variables=variables,
        shocks=shocks,
        equations=equations,
        steady_state_relationships=ss_relationships,
        param_dict=param_dict,
        hyper_param_dict=hyper_param_dict,
        deterministic_dict=deterministic_dict,
        calib_dict=calib_dict,
        f_ss=f_ss,
        f_ss_jac=f_ss_jac,
        f_params=f_params,
        f_ss_resid=f_ss_resid,
        f_ss_error=f_ss_error,
        f_ss_error_grad=f_ss_grad,
        f_ss_error_hess=f_ss_hess,
        f_ss_error_hessp=f_ss_hessp,
        f_linearize=f_linearize,
        backend=backend,
        priors=priors,
        is_linear=options.get("linear", False),
    )


def statespace_from_gcn(
    gcn_path: str,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    log_linearize: bool = True,
    not_loglin_variables: list[str] | None = None,
    **kwargs,
):
    objects, dictionaries, functions, cache, priors, options = _compile_gcn(
        gcn_path,
        simplify_blocks=simplify_blocks,
        simplify_tryreduce=simplify_tryreduce,
        simplify_constants=simplify_constants,
        verbose=verbose,
        backend="pytensor",
        error_function=error_function,
        on_unused_parameters=on_unused_parameters,
        return_symbolic=True,
        **kwargs,
    )

    variables, shocks, equations, _ss_relationships = objects
    param_dict, hyper_param_dict, _deterministic_dict, calib_dict = dictionaries
    param_priors, shock_priors = priors

    if len(calib_dict) > 0:
        raise NotImplementedError("Calibration not yet implemented in StateSpace model")

    (
        steady_state_mapping,
        ss_jac,
        parameter_mapping,
        ss_resid,
        ss_error,
        ss_grad,
        ss_hess,
        _ss_hessp,
        linearized_matrices,
    ) = functions

    # Check that the entire steady state has been provided
    if steady_state_mapping is None or len(steady_state_mapping) != len(variables):
        raise NotImplementedError("Numeric steady state not yet implemented in StateSpace model")

    A, B, C, D = linearized_matrices

    not_loglin_flags = next(x for x in cache.values() if x.name == "not_loglin_variable")

    # First replace deterministic variables with functions of input variables in the user-provided steady state
    # expressiong
    steady_state_mapping = {
        k: graph_replace(v, parameter_mapping, strict=False) for k, v in steady_state_mapping.items()
    }

    ss_vec = pt.stack(list(steady_state_mapping.values()))
    if not_loglin_variables is None:
        not_loglin_variables = []

    var_names = [get_name(x, base_name=True) for x in variables]
    unknown_not_login = set(not_loglin_variables) - set(var_names)

    if len(unknown_not_login) > 0:
        raise ValueError(
            f"The following variables were requested not to be log-linearized, but are unknown to the model: "
            f"{', '.join(unknown_not_login)}"
        )

    if options.get("linear", False):
        log_linearize = False

    if log_linearize:
        not_loglin_mask = pt.as_tensor([x in not_loglin_variables for x in var_names])
        not_loglin_values = pt.le(ss_vec, 0.0).astype(float)
        not_loglin_values = not_loglin_values[not_loglin_mask].set(1.0)
    else:
        not_loglin_values = pt.ones(ss_vec.shape[0])

    not_loglin_replacement = {not_loglin_flags: not_loglin_values}

    replacements = parameter_mapping | steady_state_mapping | not_loglin_replacement

    # Replace all placeholders with functions of the input parameters
    ss_resid, ss_jac, ss_error, ss_grad, ss_hess = graph_replace(
        [ss_resid, ss_jac, ss_error, ss_grad, ss_hess], replacements, strict=False
    )
    A, B, C, D = rewrite_pregrad(graph_replace([A, B, C, D], replacements, strict=False))

    return DSGEStateSpace(
        variables=variables,
        shocks=shocks,
        equations=equations,
        param_dict=param_dict,
        param_priors=param_priors,
        hyper_param_dict=hyper_param_dict,
        shock_priors=shock_priors,
        parameter_mapping=parameter_mapping,
        steady_state_mapping=steady_state_mapping,
        ss_jac=ss_jac,
        ss_resid=ss_resid,
        ss_error=ss_error,
        ss_error_grad=ss_grad,
        ss_error_hess=ss_hess,
        linearized_system=[A, B, C, D],
        verbose=verbose,
    )


def build_report(
    equations: list,
    param_dict: SymbolDictionary,
    calib_dict: SymbolDictionary,
    variables: list,
    shocks: list,
    param_priors: SymbolDictionary,
    shock_priors: SymbolDictionary,
    reduced_vars: list | None,
    reduced_params: list | None,
    singletons: list | None,
) -> None:
    """
    Write a diagnostic message after building the model.

    Note that successfully building the model does not guarantee that the model
    is correctly specified. For example, it is possible to build a model with
    more equations than parameters. This message will warn the user in this case.
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

    param_prior_keys = param_priors.keys()
    shock_prior_keys = shock_priors.keys()

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
    have_has = "have" if len(shock_prior_keys) == 1 else "has"
    report += f"\t\t {len(shock_prior_keys)} / {n_shocks} {have_has} a defined prior.\n"

    report += f"\t{n_params} {free_par_str}\n"
    if reduced_params:
        report += "\t\tThe following parameters were eliminated via substitution into other parameters:\n"
        report += "\t\t\t" + ", ".join([x.name for x in reduced_params]) + "\n"

    report += (
        f"\t\t {len(param_prior_keys)} / {n_params} parameters {'have' if len(param_prior_keys) == 1 else 'has'} "
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
        warn(message, stacklevel=2)

    _log.info(report)
