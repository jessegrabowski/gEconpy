import logging

from typing import cast

import pytensor.tensor as pt

from pytensor import clone_replace

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
) -> tuple[tuple, tuple, tuple, dict]:
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

    ss_solution_dict = simplify_provided_ss_equations(ss_solution_dict, variables)

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
    f_ss_error, f_ss_grad, f_ss_hess = error_funcs

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

    objects = (variables, shocks, equations)
    dictionaries = (param_dict, deterministic_dict, calib_dict)
    functions = (
        f_ss,
        f_ss_jac,
        f_params,
        f_ss_resid,
        f_ss_error,
        f_ss_grad,
        f_ss_hess,
        f_linearize,
    )

    return objects, dictionaries, functions, cache


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
    objects, dictionaries, functions, _ = _compile_gcn(
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

    variables, shocks, equations = objects
    param_dict, deterministic_dict, calib_dict = dictionaries
    (
        f_ss,
        f_ss_jac,
        f_params,
        f_ss_resid,
        f_ss_error,
        f_ss_grad,
        f_ss_hess,
        f_linearize,
    ) = functions

    return Model(
        variables=variables,
        shocks=shocks,
        equations=equations,
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
        f_linearize=f_linearize,
        backend=backend,
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
    objects, dictionaries, functions, cache = _compile_gcn(
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

    variables, shocks, equations = objects
    param_dict, deterministic_dict, calib_dict = dictionaries
    (
        steady_state_solutions,
        ss_jac,
        param_vec,
        ss_resid,
        ss_error,
        ss_grad,
        ss_hess,
        linearized_matrices,
    ) = functions
    A, B, C, D = linearized_matrices

    # The graphs created by _compile_gcn will have placeholders for intermediate computations like steady-state
    # values. We want to replace these with user-provided values.
    input_params = [v for k, v in cache.items() if k[0] in param_dict.keys()]
    deterministic_params = [
        v for k, v in cache.items() if k[0] in deterministic_dict.keys()
    ]
    calibrated_params = [v for k, v in cache.items() if k[0] in calib_dict.keys()]

    all_inputs_pt = list(cache.values())

    # The last value in the cache is always not_loglin_variables (because linearization is the last step)
    not_loglin_variables = all_inputs_pt.pop(-1)

    ss_vars_pt = [
        x
        for x in all_inputs_pt
        if x
        not in input_params
        + deterministic_params
        + calibrated_params
        + [not_loglin_variables]
    ]

    # First replace deterministic variables with functions of input variables in the user-provided steady state
    # expressiong
    param_replacement_dict = {
        param: param_vec[i]
        for i, param in enumerate(input_params + deterministic_params)
    }
    steady_state_solutions = cast(
        pt.TensorVariable, clone_replace(steady_state_solutions, param_replacement_dict)
    )

    # Then create a mapping from the placeholder steady-state variables to the provided steady-state solutions
    ss_replacement_dict = {
        var: steady_state_solutions[i] for i, var in enumerate(ss_vars_pt)
    }

    # TODO: The user might want to choose this. For now its hardcoded.
    ss_replacement_dict[not_loglin_variables] = pt.gt(
        steady_state_solutions, 0.0
    ).astype(float)

    replacements = param_replacement_dict | ss_replacement_dict

    # Replace all placeholders with functions of the input parameters
    ss_resid = cast(pt.TensorVariable, clone_replace(ss_resid, replacements))
    ss_jac = cast(pt.TensorVariable, clone_replace(ss_jac, replacements))
    ss_error = cast(pt.TensorVariable, clone_replace(ss_error, replacements))
    ss_grad = cast(pt.TensorVariable, clone_replace(ss_grad, replacements))
    ss_hess = cast(pt.TensorVariable, clone_replace(ss_hess, replacements))
    A = cast(pt.TensorVariable, clone_replace(A, replacements))
    B = cast(pt.TensorVariable, clone_replace(B, replacements))
    C = cast(pt.TensorVariable, clone_replace(C, replacements))
    D = cast(pt.TensorVariable, clone_replace(D, replacements)).astype(float)

    return DSGEStateSpace(
        variables=variables,
        shocks=shocks,
        equations=equations,
        param_dict=param_dict,
        input_parameters=input_params,
        deterministic_params=deterministic_params,
        calibrated_params=calibrated_params,
        steady_state_solutions=steady_state_solutions,
        ss_jac=ss_jac,
        ss_resid=ss_resid,
        ss_error=ss_error,
        ss_error_grad=ss_grad,
        ss_error_hess=ss_hess,
        linearized_system=[A, B, C, D],
    )
