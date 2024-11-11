import logging

import pytensor.tensor as pt
import sympy as sp

from pymc.pytensorf import rewrite_pregrad
from pytensor import graph_replace

from gEconpy.model.compile import BACKENDS
from gEconpy.model.model import Model
from gEconpy.model.perturbation import compile_linearized_system
from gEconpy.model.statespace import DSGEStateSpace
from gEconpy.model.steady_state import (
    ERROR_FUNCTIONS,
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

_log = logging.getLogger(__name__)


def _compile_gcn(
    gcn_path: str,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    backend: BACKENDS = "numpy",
    return_symbolic: bool = False,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    **kwargs,
) -> tuple[tuple, tuple, tuple, dict, tuple]:
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
        hyper_priors_final,
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

    ss_solution_dict = simplify_provided_ss_equations(ss_solution_dict, variables)
    steady_state_relationships = [
        sp.Eq(var, eq) for var, eq in ss_solution_dict.to_sympy().items()
    ]

    validate_results(
        equations,
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
            singletons,
        )

    objects = (variables, shocks, equations, steady_state_relationships)
    dictionaries = (param_dict, deterministic_dict, calib_dict)
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
    priors = (param_priors, shock_priors, hyper_priors_final)

    return objects, dictionaries, functions, cache, priors


def model_from_gcn(
    gcn_path: str,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    backend: BACKENDS = "numpy",
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    **kwargs,
) -> Model:
    objects, dictionaries, functions, cache, priors = _compile_gcn(
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
    param_dict, deterministic_dict, calib_dict = dictionaries

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
    )


def statespace_from_gcn(
    gcn_path: str,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    error_function: ERROR_FUNCTIONS = "squared",
    on_unused_parameters="raise",
    **kwargs,
):
    objects, dictionaries, functions, cache, priors = _compile_gcn(
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
    param_dict, deterministic_dict, calib_dict = dictionaries
    param_priors, shock_priors, hyper_priors = priors

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
    if len(steady_state_mapping) != len(variables):
        raise NotImplementedError(
            "Numeric steady state not yet implemented in StateSpace model"
        )

    A, B, C, D = linearized_matrices

    not_loglin_variables = next(
        x for x in cache.values() if x.name == "not_loglin_variable"
    )

    # First replace deterministic variables with functions of input variables in the user-provided steady state
    # expressiong
    steady_state_mapping = {
        k: graph_replace(v, parameter_mapping, strict=False)
        for k, v in steady_state_mapping.items()
    }

    # TODO: The user might want to choose this. For now its hardcoded.
    ss_vec = pt.stack(list(steady_state_mapping.values()))
    not_loglin_replacement = {not_loglin_variables: pt.le(ss_vec, 0.0).astype(float)}

    replacements = steady_state_mapping | not_loglin_replacement

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
        param_dict=param_dict,
        priors=priors,
        parameter_mapping=parameter_mapping,
        steady_state_mapping=steady_state_mapping,
        ss_jac=ss_jac,
        ss_resid=ss_resid,
        ss_error=ss_error,
        ss_error_grad=ss_grad,
        ss_error_hess=ss_hess,
        linearized_system=[A, B, C, D],
    )
