import logging

import pytensor.tensor as pt
import sympy as sp
from pathlib import Path

from pymc.pytensorf import rewrite_pregrad
from pytensor import graph_replace

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.model.model import Model
from gEconpy.model.perturbation import compile_linearized_system
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.steady_state import (
    compile_model_ss_functions,
    system_to_steady_state,
)
from gEconpy.parser.file_loaders import (
    block_dict_to_model_primitives,
    build_report,
    gcn_to_block_dict,
    simplify_provided_ss_equations,
    validate_results,
)
from gEconpy.parser.parse_distributions import CompositeDistribution
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

    all_hyper_params = [
        param
        for dist in shock_prior.values()
        for param in dist.param_name_to_hyper_name.values()
    ]

    for param in all_hyper_params:
        if param in new_param_dict:
            del new_param_dict[param]
            hyper_param_dict[param] = param_dict[param]

    return new_param_dict, hyper_param_dict


def _compile_gcn(
    gcn_path: str,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    backend: str = "numpy",
    return_symbolic: bool = False,
    error_function: str = "squared",
    on_unused_parameters="raise",
    remove_hyperpriors_from_param_dict=True,
    **kwargs,
) -> tuple[tuple, tuple, tuple, dict, tuple, dict]:
    outputs = gcn_to_block_dict(gcn_path, simplify_blocks=simplify_blocks)
    block_dict, assumptions, options, try_reduce, ss_solution_dict, prior_info = outputs

    (
        equations,
        param_dict,
        calib_dict,
        deterministic_dict,
        variables,
        shocks,
        param_priors,
        shock_priors,
        reduced_vars,
        singletons,
    ) = block_dict_to_model_primitives(
        block_dict,
        assumptions,
        try_reduce,
        prior_info,
        simplify_tryreduce=simplify_tryreduce,
        simplify_constants=simplify_constants,
    )

    param_dict, hyper_param_dict = split_out_hyper_params(param_dict, shock_priors)

    ss_solution_dict = simplify_provided_ss_equations(ss_solution_dict, variables)
    steady_state_relationships = [
        sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()
    ]

    # TODO: Move this to a separate function
    # TODO: Add option to not eliminate deterministic parameters (the user might be interested in them)

    deterministic_dict.to_sympy(inplace=True)
    for param, expr in deterministic_dict.items():
        deterministic_dict[param] = substitute_repeatedly(expr, deterministic_dict)

    # If a deterministic parameter is only used in other parameters, it will now have been completely substituted away
    # and can be removed
    reduced_params = []
    final_deterministics = deterministic_dict.copy()

    for param in deterministic_dict.keys():
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
    backend: str = "numpy",
    error_function: str = "squared",
    on_unused_parameters="raise",
    **kwargs,
) -> Model:
    """
    Create a Model from a GCN file.

    This function parses a GCN specification and constructs a Model object. The build process performs
    optional symbolic simplifications, compiles steady-state residuals, objective and derivatives, and
    generates a callable that returns the first-order linearization of the system. The resulting Model
    stores all parsed primitives (variables, shocks, equations, steady-state relationships) and exposes
    convenience methods to compute or validate the steady state, linearize the model, and obtain a
    perturbation solution.

    Parameters
    ----------
    gcn_path: str or Path
        Path to the model GCN file.
    simplify_blocks: bool, default True
        Simplify equations within each parsed block.
    simplify_tryreduce: bool, default True
        Attempt symbolic reduction by eliminating singletons and redundant relations.
    simplify_constants: bool, default True
        Fold constant expressions during compilation.
    verbose: bool, default True
        If True, print a short build report showing the model composition and reductions performed (if any).
    backend: str, default 'numpy'
        Computational backend used for the compiled functions attached to the Model. One of 'numpy' or 'pytensor'.
    error_function: One of 'squared', 'abs', 'huber', ...}, default 'squared'
        Objective used when forming steady-state error terms and derivatives. Only used when the steady state is
        numerically computed, and only when the solution method is "minimize".
    on_unused_parameters: str, default 'raise'
        Behavior when parameters are defined but unused by equations or steady-state relationships. One of 'raise',
        or 'ignore'.
    **kwargs
        Forwarded to internal compilation routines (e.g., steady-state compilation and linearization).

    Returns
    -------
    model: Model
        A compiled DSGE model instance.
    """

    objects, dictionaries, functions, cache, priors, options = _compile_gcn(
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
    error_function: str = "squared",
    on_unused_parameters: str = "raise",
    log_linearize: bool = True,
    not_loglin_variables: list[str] | None = None,
    exog_state_names: list[str] | dict[str, list[str]] | None = None,
    k_exog: int | dict[str, int] | None = None,
    **kwargs,
):
    """
    Create a statespace model from a gcn file.

    A statespace model is used for solving and estimating linearized DSGE models using PyMC.

    Parameters
    ----------
    gcn_path: str
        Path to the model GCN file.

    simplify_blocks: bool, default True
        Simplify equations within each parsed block.

    simplify_tryreduce: bool, default True
        Attempt symbolic reduction by eliminating singletons/redundant relations.

    simplify_constants: bool, default True
        Fold constant expressions during compilation.

    verbose: bool, default True
        Print a short build report (variables, shocks, reductions, priors).

    error_function: {'squared', 'abs', 'huber', ...}, default 'squared'
        Objective used when forming steady-state error terms and derivatives.

    on_unused_parameters: {'raise', 'ignore'}, default 'raise'
        Behavior when parameters are defined but unused by equations or steady-state.

    log_linearize: bool, default True
        If True, variables are log-linearized around the steady state (unless the model
        is declared linear in options, in which case this is forced to False).

    not_loglin_variables: list[str], optional
        Variable names to keep in levels even when log_linearize=True. Names must match
        model variable base names (no time suffix). Unknown names raise ValueError.

    exog_state_names : list[str] or dict[str, list[str]], optional
        Names of the exogenous state variables. If a list, all endogenous variables will share the same exogenous
        variables. If a dict, keys should be the names of the endogenous variables, and values should be lists of the
        exogenous variable names for that endogenous variable. Endogenous variables not included in the dict will
        be assumed to have no exogenous variables. If None, no exogenous variables will be included.

    k_exog : int or dict[str, int], optional
        Number of exogenous variables. If an int, all endogenous variables will share the same number of exogenous
        variables. If a dict, keys should be the names of the endogenous variables, and values should be the number of
        exogenous variables for that endogenous variable. Endogenous variables not included in the dict will be
        assumed to have no exogenous variables. If None, no exogenous variables will be included.

    **kwargs
        Forwarded to internal compilation routines (e.g., steady-state compilation).

    Returns
    -------
    statespace_model: DSGEStateSpace
        An instance of a PyMCStateSpace model representing the linearized DSGE model.
    """
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

    variables, shocks, equations, ss_relationships = objects
    param_dict, hyper_param_dict, deterministic_dict, calib_dict = dictionaries
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
        ss_hessp,
        linearized_matrices,
    ) = functions

    # Check that the entire steady state has been provided
    if steady_state_mapping is None or len(steady_state_mapping) != len(variables):
        raise NotImplementedError(
            "Numeric steady state not yet implemented in StateSpace model"
        )

    A, B, C, D = linearized_matrices

    not_loglin_flags = next(
        x for x in cache.values() if x.name == "not_loglin_variable"
    )

    # First replace deterministic variables with functions of input variables in the user-provided steady state
    # expressiong
    steady_state_mapping = {
        k: graph_replace(v, parameter_mapping, strict=False)
        for k, v in steady_state_mapping.items()
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
    A, B, C, D = rewrite_pregrad(
        graph_replace([A, B, C, D], replacements, strict=False)
    )

    return DSGEStateSpace(
        variables=variables,
        shocks=shocks,
        equations=equations,
        k_exog=k_exog,
        exog_state_names=exog_state_names,
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
