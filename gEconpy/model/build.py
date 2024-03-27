from gEconpy.model.compile import BACKENDS
from gEconpy.model.model import Model, compile_model_ss_functions
from gEconpy.model.perturbation.perturbation import compile_linearized_system
from gEconpy.model.steady_state.steady_state import (
    ERROR_FUNCTIONS,
    make_steady_state_shock_dict,
    system_to_steady_state,
)
from gEconpy.parser.file_loaders import (
    block_dict_to_model_primitives,
    build_report,
    gcn_to_block_dict,
    simplify_provided_ss_equations,
    validate_results,
)


def model_from_gcn(
    gcn_path: str,
    simplify_blocks: bool = True,
    simplify_tryreduce: bool = True,
    simplify_constants: bool = True,
    verbose: bool = True,
    backend: BACKENDS = "numpy",
    symbolic_model: bool = False,
    error_function: ERROR_FUNCTIONS = "squared",
    **kwargs,
) -> Model:
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

    validate_results(equations, param_dict, calib_dict, deterministic_dict)
    ss_shock_dict = make_steady_state_shock_dict(shocks)
    steady_state_equations = system_to_steady_state(equations, ss_shock_dict)

    functions, cache = compile_model_ss_functions(
        steady_state_equations,
        ss_solution_dict,
        variables,
        param_dict,
        deterministic_dict,
        calib_dict,
        error_func=error_function,
        backend=backend,
        return_symbolic=symbolic_model,
        **kwargs,
    )
    f_params, f_ss, resid_funcs, error_funcs = functions
    f_ss_resid, f_ss_jac = resid_funcs
    f_ss_error, f_ss_grad, f_ss_hess = error_funcs

    f_linearize = compile_linearized_system(
        variables, equations, shocks, backend=backend, return_symbolic=symbolic_model, cache=cache
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

    return Model(
        variables=variables,
        shocks=shocks,
        equations=equations,
        param_dict=param_dict,
        f_ss=f_ss,
        f_ss_jac=f_ss_jac,
        f_params=f_params,
        f_ss_resid=f_ss_resid,
        f_ss_error=f_ss_error,
        f_ss_error_grad=f_ss_grad,
        f_ss_error_hess=f_ss_hess,
    )
