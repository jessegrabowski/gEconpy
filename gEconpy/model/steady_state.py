import logging

from typing import Literal

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import (
    BACKENDS,
    compile_function,
    dictionary_return_wrapper,
    make_return_dict_and_update_cache,
)
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.shared.utilities import eq_to_ss

_log = logging.getLogger(__name__)

ERROR_FUNCTIONS = Literal["squared", "mean_squared", "abs", "l2-norm"]


def _validate_optimizer_kwargs(
    optimizer_kwargs: dict,
    n_eq: int,
    method: str,
    use_jac: bool,
    use_hess: bool,
) -> dict:
    """
    Validate user-provided keyword arguments to either scipy.optimize.root or scipy.optimize.minimize, and insert
    defaults where not provided.

    Note: This function never overwrites user arguments.

    Parameters
    ----------
    optimizer_kwargs: dict
        User-provided arguments for the optimizer
    n_eq: int
        Number of remaining steady-state equations after reduction
    method: str
        Which family of solution algorithms, minimization or root-finding, to be used.
    use_jac: bool
        Whether computation of the jacobian has been requested
    use_hess: bool
        Whether computation of the hessian has been requested

    Returns
    -------
    optimizer_kwargs: dict
        Keyword arguments for the scipy function, with "reasonable" defaults inserted where not provided
    """

    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    method_given = "method" in optimizer_kwargs.keys()

    if method == "root" and not method_given:
        if use_jac:
            optimizer_kwargs["method"] = "hybr"
        else:
            optimizer_kwargs["method"] = "broyden1"

        if n_eq == 1:
            optimizer_kwargs["method"] = "lm"

    elif method == "minimize" and not method_given:
        # Set optimizer_kwargs for minimization
        if use_hess and use_jac:
            optimizer_kwargs["method"] = "trust-exact"
        elif use_jac:
            optimizer_kwargs["method"] = "BFGS"
        else:
            optimizer_kwargs["method"] = "Nelder-Mead"

    if "tol" not in optimizer_kwargs.keys():
        optimizer_kwargs["tol"] = 1e-9

    return optimizer_kwargs


def make_steady_state_shock_dict(shocks):
    return SymbolDictionary.fromkeys(shocks, 0.0).to_ss()


def make_steady_state_variables(variables):
    return list(map(lambda x: x.to_ss(), variables))


def system_to_steady_state(system, shocks):
    shock_dict = make_steady_state_shock_dict(shocks)
    steady_state_system = [eq_to_ss(eq).subs(shock_dict).simplify() for eq in system]

    return steady_state_system


def faster_simplify(x: sp.Expr, var_list: list[TimeAwareSymbol]):
    # return sp.powsimp(sp.powdenest(x, force=True), force=True)
    return x


def steady_state_error_function(
    steady_state, variables: list[sp.Symbol], func: ERROR_FUNCTIONS = "squared"
) -> sp.Expr:
    ss_vars = [x.to_ss() if isinstance(x, TimeAwareSymbol) else x for x in variables]

    if func == "squared":
        error = sum([faster_simplify(eq**2, ss_vars) for eq in steady_state])
    elif func == "mean_squared":
        error = sum([faster_simplify(eq**2, ss_vars) for eq in steady_state]) / len(
            steady_state
        )
    elif func == "abs":
        error = sum([faster_simplify(sp.Abs(eq), ss_vars) for eq in steady_state])
    elif func == "l2-norm":
        error = sp.sqrt(sum([faster_simplify(eq**2, ss_vars) for eq in steady_state]))
    else:
        raise NotImplementedError(
            f"Error function {func} not implemented, must be one of {ERROR_FUNCTIONS}"
        )

    return error


def compile_ss_resid_and_sq_err(
    steady_state: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    parameters: list[sp.Symbol],
    ss_error: sp.Expr,
    backend: BACKENDS,
    cache: dict,
    return_symbolic: bool,
    **kwargs,
):
    cache = {} if cache is None else cache
    ss_variables = [x.to_ss() if hasattr(x, "to_ss") else x for x in variables]
    resid_jac = sp.Matrix(
        [
            [faster_simplify(eq.diff(x), ss_variables) for x in ss_variables]
            for eq in steady_state
        ]
    )

    f_ss_resid, cache = compile_function(
        ss_variables + parameters,
        steady_state,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        stack_return=True,
        pop_return=False,
        **kwargs,
    )

    f_ss_jac, cache = compile_function(
        ss_variables + parameters,
        resid_jac,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        # for pytensor/numba, the return is a single object; don't stack into a (1,n,n) array
        stack_return=backend == "numpy",
        # Numba directly returns the jacobian as an array, don't pop
        # pytensor and lambdify return a list of one item, so we have to extract it.
        pop_return=backend != "numba",
        **kwargs,
    )

    error_grad = [faster_simplify(ss_error.diff(x), ss_variables) for x in ss_variables]
    error_hess = sp.Matrix(
        [
            [faster_simplify(eq.diff(x), ss_variables) for eq in error_grad]
            for x in ss_variables
        ]
    )

    f_ss_error, cache = compile_function(
        ss_variables + parameters,
        [ss_error],
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        pop_return=True,
        stack_return=False,
        **kwargs,
    )

    f_ss_grad, cache = compile_function(
        ss_variables + parameters,
        error_grad,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        stack_return=True,
        pop_return=False,
        **kwargs,
    )

    f_ss_hess, cache = compile_function(
        ss_variables + parameters,
        error_hess,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        # error_hess is a list of one element; don't stack into a (1,n,n) array
        stack_return=backend != "pytensor",
        # Numba directly returns the hessian as an array, don't pop
        pop_return=backend != "numba",
        **kwargs,
    )

    return (f_ss_resid, f_ss_jac), (f_ss_error, f_ss_grad, f_ss_hess), cache


def compile_known_ss(
    ss_solution_dict: SymbolDictionary,
    variables: list[TimeAwareSymbol | sp.Symbol],
    parameters: list[sp.Symbol],
    backend: BACKENDS,
    cache: dict,
    return_symbolic: bool = False,
    stack_return: bool | None = None,
    **kwargs,
):
    def to_ss(x):
        if isinstance(x, TimeAwareSymbol):
            return x.to_ss()
        return x

    cache = {} if cache is None else cache
    if not ss_solution_dict:
        return None, cache

    ss_solution_dict = ss_solution_dict.to_sympy()
    ss_variables = [to_ss(x) for x in variables]

    sorted_solution_dict = {
        to_ss(k): ss_solution_dict[to_ss(k)]
        for k in ss_variables
        if k in ss_solution_dict.keys()
    }

    output_vars, output_exprs = (
        list(sorted_solution_dict.keys()),
        list(sorted_solution_dict.values()),
    )
    if stack_return is None:
        stack_return = True if not return_symbolic else False

    f_ss, cache = compile_function(
        parameters,
        output_exprs,
        backend=backend,
        cache=cache,
        stack_return=stack_return,
        return_symbolic=return_symbolic,
        **kwargs,
    )
    if return_symbolic and backend == "pytensor":
        return make_return_dict_and_update_cache(
            ss_variables, f_ss, cache, TimeAwareSymbol
        )

    return dictionary_return_wrapper(f_ss, output_vars), cache


def compile_model_ss_functions(
    steady_state_equations,
    ss_solution_dict,
    variables,
    param_dict,
    deterministic_dict,
    calib_dict,
    error_func: ERROR_FUNCTIONS = "squared",
    backend: BACKENDS = "numpy",
    return_symbolic: bool = False,
    **kwargs,
):
    cache = {}
    f_params, cache = compile_param_dict_func(
        param_dict,
        deterministic_dict,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
    )

    calib_eqs = list(calib_dict.to_sympy().values())
    steady_state_equations = steady_state_equations + calib_eqs

    parameters = list((param_dict | deterministic_dict).to_sympy().keys())
    parameters = [x for x in parameters if x not in calib_dict.to_sympy()]

    variables = variables + list(calib_dict.to_sympy().keys())
    ss_error = steady_state_error_function(
        steady_state_equations, variables, error_func
    )

    f_ss, cache = compile_known_ss(
        ss_solution_dict,
        variables,
        parameters,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        **kwargs,
    )

    (f_ss_resid, f_ss_jac), (f_ss_error, f_ss_grad, f_ss_hess), cache = (
        compile_ss_resid_and_sq_err(
            steady_state_equations,
            variables,
            parameters,
            ss_error,
            backend=backend,
            cache=cache,
            return_symbolic=return_symbolic,
            **kwargs,
        )
    )

    return (
        f_params,
        f_ss,
        (f_ss_resid, f_ss_jac),
        (f_ss_error, f_ss_grad, f_ss_hess),
    ), cache


def print_steady_state(ss_dict, success):
    output = []
    if not success:
        output.append(
            "Values come from the latest solver iteration but are NOT a valid steady state."
        )

    max_var_name = max(len(x) for x in list(ss_dict.keys())) + 5

    calibrated_outputs = []
    for key, value in ss_dict.to_sympy().items():
        if isinstance(key, TimeAwareSymbol):
            output.append(f"{key.name:{max_var_name}}{value:>10.3f}")
        else:
            calibrated_outputs.append(f"{key.name:{max_var_name}}{value:>10.3f}")

    if len(calibrated_outputs) > 0:
        output.append("\n")
        output.extend(calibrated_outputs)

    _log.info("\n".join(output))
