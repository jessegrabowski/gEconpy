import functools as ft
import logging

from typing import Union
from collections.abc import Callable

import numpy as np
import sympy as sp

from scipy import optimize

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.optimize_wrapper import (
    CostFuncWrapper,
    optimzer_early_stopping_wrapper,
    postprocess_optimizer_res,
)
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions.exceptions import (
    ModelUnknownParameterError,
)
from gEconpy.model.compile import BACKENDS
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.perturbation.perturbation import override_dummy_wrapper
from gEconpy.model.steady_state.steady_state import (
    ERROR_FUNCTIONS,
    compile_known_ss,
    compile_ss_resid_and_sq_err,
    steady_state_error_function,
)

VariableType = Union[sp.Symbol, TimeAwareSymbol]
_log = logging.getLogger(__name__)


def scipy_wrapper(f, var_names, unknown_var_idxs, f_ss=None):
    if f_ss is not None:

        @ft.wraps(f)
        def inner(ss_values, param_dict):
            given_ss = f_ss(**param_dict)
            ss_dict = dict(zip(var_names, ss_values))
            ss_dict.update(given_ss)
            res = f(**ss_dict, **param_dict)
            if res.ndim == 1:
                res = res[unknown_var_idxs]
            elif res.ndim == 2:
                res = res[unknown_var_idxs, :][:, unknown_var_idxs]
            return res

    else:

        @ft.wraps(f)
        def inner(ss_values, param_dict):
            ss_dict = dict(zip(var_names, ss_values))
            return f(**ss_dict, **param_dict)

    return inner


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
    stack_return: bool | None = None,
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

    calib_eqs = [lhs - rhs for lhs, rhs in calib_dict.to_sympy().items()]
    steady_state_equations = steady_state_equations + calib_eqs

    parameters = list((param_dict | deterministic_dict).to_sympy().keys())
    parameters = [x for x in parameters if x not in calib_dict.to_sympy()]

    variables += list(calib_dict.to_sympy().keys())
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
        stack_return=stack_return,
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
            stack_return=stack_return,
            **kwargs,
        )
    )

    return (
        f_params,
        f_ss,
        (f_ss_resid, f_ss_jac),
        (f_ss_error, f_ss_grad, f_ss_hess),
    ), cache


def infer_variable_bounds(variables):
    bounds = []
    for var in variables:
        assumptions = var.assumptions0
        is_positive = assumptions.get("positive", False)
        is_negative = assumptions.get("negative", False)
        lhs = 0 if is_positive else None
        rhs = 0 if is_negative else None
        bounds.append((lhs, rhs))

    return bounds


class Model:
    def __init__(
        self,
        variables: list[TimeAwareSymbol],
        shocks: list[TimeAwareSymbol],
        equations: list[sp.Expr],
        param_dict: SymbolDictionary,
        f_params: Callable,
        f_ss_resid: Callable,
        f_ss: Callable | None = None,
        f_ss_error: Callable | None = None,
        f_ss_jac: Callable | None = None,
        f_ss_error_grad: Callable | None = None,
        f_ss_error_hess: Callable | None = None,
        f_linearize: Callable | None = None,
        f_perturbation: Callable | None = None,
        backend: BACKENDS = "numpy",
    ) -> None:
        """
        Initialize a DSGE model object from a GCN file.

        Parameters
        ----------
        variables: list[TimeAwareSymbol]
            List of variables in the model
        shocks: list[TimeAwareSymbol]
            List of shocks in the model
        equations: list[sp.Expr]
            List of equations in the model
        param_dict: SymbolDictionary
            Dictionary of parameters in the model
        f_params: Callable
            Function that returns a dictionary of parameter values given a dictionary of parameter values
        f_ss_resid: Callable
            Function that takes a dictionary of parameter values theta and steady-state variable values x_ss and evaluates
            the system of model equations f(x_ss, theta) = 0.
        f_ss: Callable
            Function that takes current parameter values and returns a dictionary of steady-state values.
        f_ss_error: Callable, optional
            Function that takes a dictionary of parameter values theta and steady-state variable values x_ss and returns
            a scalar error measure of x_ss given theta.
            If None, the sum of squared residuals returned by f_ss_resid is used.
        f_ss_error_grad: Callable, optional
            Function that takes a dictionary of parameter values theta and steady-state variable values x_ss and returns
            the gradients of the error function f_ss_error with respect to the steady-state variable values x_ss

            If f_ss_error is not provided, an error will be raised if a gradient function is passed.
        f_ss_error_hess: Callable, optional
            Function that takes a dictionary of parameter values theta and steady-state variable values x_ss and returns
            the Hessian of the error function f_ss_error with respect to the steady-state variable values x_ss

            If f_ss_error is not provided, an error will be raised if a gradient function is passed.

        f_ss_jac: Callable, optional

        f_linearize: Callable, optional

        """

        self.variables = variables
        self.shocks = shocks
        self.equations = equations
        self.params = list(param_dict.to_sympy().keys())

        self._default_params = param_dict
        self.f_params: Callable = f_params
        self.f_ss_resid: Callable = f_ss_resid

        self.f_ss_error: Callable = f_ss_error
        self.f_ss_error_grad: Callable = f_ss_error_grad
        self.f_ss_error_hess: Callable = f_ss_error_hess

        self.f_ss: Callable = f_ss
        self.f_ss_jac: Callable = f_ss_jac

        if backend == "numpy":
            f_linearize = override_dummy_wrapper(f_linearize, "not_loglin_variable")
        self.f_linearize: Callable = f_linearize

        # self.steady_state_relationships: SymbolDictionary[VariableType, sp.Add] = SymbolDictionary()
        #
        # self.param_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        # self.shock_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        # self.hyper_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        # self.observation_noise_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        #
        # self.n_variables: int = 0
        # self.n_shocks: int = 0
        # self.n_equations: int = 0
        # self.n_calibrating_equations: int = 0
        #
        # # Functional representations of the model
        # self.f_ss: Union[Callable, None] = None
        # self.f_ss_resid: Union[Callable, None] = None
        #
        # # Steady state information
        # self.steady_state_solved: bool = False
        # self.steady_state_system: list[sp.Add] = []
        # self.steady_state_dict: SymbolDictionary[sp.Symbol, float] = SymbolDictionary()
        # self.residuals: list[float] = []
        #
        # # Functional representation of the perturbation system
        # self.build_perturbation_matrices: Union[Callable, None] = None
        #
        # # Perturbation solution information
        # self.perturbation_solved: bool = False
        # self.T: pd.DataFrame = None
        # self.R: pd.DataFrame = None
        # self.P: pd.DataFrame = None
        # self.Q: pd.DataFrame = None
        # self.R: pd.DataFrame = None
        # self.S: pd.DataFrame = None
        #
        # # Assign Solvers
        # self.steady_state_solver = SteadyStateSolver(self)
        # self.perturbation_solver = PerturbationSolver(self)

    def parameters(self, **updates: float):
        param_dict = self._default_params.copy()
        unknown_updates = set(updates.keys()) - set(param_dict.keys())
        if unknown_updates:
            raise ModelUnknownParameterError(list(unknown_updates))
        param_dict.update(updates)

        return self.f_params(**param_dict)

    def steady_state(
        self,
        how="analytic",
        use_jac=True,
        use_hess=True,
        progressbar=True,
        optimizer_kwargs: dict | None = None,
        verbose=True,
        **updates: float,
    ):
        param_dict = self.parameters(**updates)

        if how == "analytic" and self.f_ss is None:
            how = "minimize"
        else:
            ss_dict = self.f_ss(**param_dict)
            if len(ss_dict) != len(self.variables):
                how = "minimize"
            else:
                return ss_dict

        ss_variables = [x.to_ss() for x in self.variables]
        known_variables = (
            set()
            if self.f_ss is None
            else set(self.f_ss(**self.parameters()).to_sympy().keys())
        )
        vars_to_solve = sorted(
            list(set(ss_variables) - known_variables), key=lambda x: x.name
        )
        var_names_to_solve = [x.name for x in vars_to_solve]
        unknown_var_idx = np.array(
            [x in vars_to_solve for x in ss_variables], dtype="bool"
        )

        if how == "root":
            if np.any(~unknown_var_idx):
                raise ValueError(
                    "Method root not currently supported for partially defined steady states. Use "
                    "method = 'minimize' instead."
                )
            res = self._solve_steady_state_with_root(
                use_jac,
                var_names_to_solve,
                unknown_var_idx,
                progressbar,
                optimizer_kwargs,
                **updates,
            )
        elif how == "minimize":
            res = self._solve_steady_state_with_minimize(
                use_jac,
                use_hess,
                var_names_to_solve,
                unknown_var_idx,
                progressbar,
                optimizer_kwargs,
                **updates,
            )
        else:
            raise NotImplementedError()

        provided_ss_values = (
            self.f_ss(**param_dict).to_sympy() if self.f_ss is not None else {}
        )
        optimizer_results = SymbolDictionary(
            {var: res.x[i] for i, var in enumerate(vars_to_solve)}
        )
        res_dict = optimizer_results | provided_ss_values
        res_dict = SymbolDictionary({x: res_dict[x] for x in ss_variables})

        return postprocess_optimizer_res(res, res_dict, verbose=verbose)

    def _evaluate_steady_state(self, **updates: float):
        param_dict = self.parameters(**updates)
        ss_dict = self.f_ss(**param_dict)

        return self.f_ss_resid(**param_dict, **ss_dict)

    def _solve_steady_state_with_root(
        self,
        use_jac: bool = True,
        vars_to_solve: list[str] | None = None,
        unknown_var_idx: np.ndarray | None = None,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        n_variables = len(vars_to_solve)
        maxeval = optimizer_kwargs.pop("niter", 5000)
        x0 = optimizer_kwargs.pop("x0", np.full(n_variables, 0.8))
        if jitter_x0:
            x0 += np.random.normal(scale=0.01, size=n_variables)
        method = optimizer_kwargs.pop("method", "hybr")
        param_dict = self.parameters(**param_updates)

        objective = CostFuncWrapper(
            maxeval=maxeval,
            f=scipy_wrapper(self.f_ss_resid, vars_to_solve, unknown_var_idx, self.f_ss),
            f_jac=scipy_wrapper(
                self.f_ss_jac, vars_to_solve, unknown_var_idx, self.f_ss
            )
            if use_jac
            else None,
            progressbar=progressbar,
        )

        f_optim = ft.partial(
            optimize.root,
            objective,
            x0,
            jac=use_jac,
            args=param_dict,
            method=method,
            **optimizer_kwargs,
        )
        res = optimzer_early_stopping_wrapper(f_optim)

        return res

    def _solve_steady_state_with_minimize(
        self,
        use_jac: bool = True,
        use_hess: bool = True,
        vars_to_solve: list[str] | None = None,
        unknown_var_idx: np.ndarray | None = None,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        n_variables = len(vars_to_solve)
        maxeval = optimizer_kwargs.pop("niter", 5000)
        x0 = optimizer_kwargs.pop("x0", np.full(n_variables, 0.8))
        if jitter_x0:
            x0 += np.random.normal(scale=0.01, size=n_variables)
        tol = optimizer_kwargs.pop("tol", 1e-30)
        bounds = optimizer_kwargs.pop("bounds", None)

        if bounds is None:
            bounds = infer_variable_bounds(self.variables)

        has_bounds = any([x != (None, None) for x in bounds])
        method = optimizer_kwargs.pop(
            "method", "trust-krylov" if not has_bounds else "trust-constr"
        )
        if method not in ["trust-constr", "L-BFGS-B"]:
            has_bounds = False

        param_dict = self.parameters(**param_updates)

        objective = CostFuncWrapper(
            maxeval=maxeval,
            f=scipy_wrapper(self.f_ss_error, vars_to_solve, unknown_var_idx, self.f_ss),
            f_jac=scipy_wrapper(
                self.f_ss_error_grad, vars_to_solve, unknown_var_idx, self.f_ss
            )
            if use_jac
            else None,
            f_hess=scipy_wrapper(
                self.f_ss_error_hess, vars_to_solve, unknown_var_idx, self.f_ss
            )
            if use_hess
            else None,
            progressbar=progressbar,
        )

        f_optim = ft.partial(
            optimize.minimize,
            objective,
            x0,
            jac=use_jac,
            hess=scipy_wrapper(
                self.f_ss_error_hess, vars_to_solve, unknown_var_idx, self.f_ss
            )
            if use_hess
            else None,
            args=param_dict,
            callback=objective.callback,
            method=method,
            tol=tol,
            bounds=bounds if has_bounds else None,
            **optimizer_kwargs,
        )
        return optimzer_early_stopping_wrapper(f_optim)

    def linearize_model(
        self,
        order=1,
        not_loglin_variables=None,
        steady_state_dict=None,
        loglin_negative_ss=False,
        **param_updates,
    ):
        if order != 1:
            raise NotImplementedError(
                "Only first order linearization is currently supported."
            )
        if not_loglin_variables is None:
            not_loglin_variables = []

        n_variables = len(self.variables)
        not_loglin_flags = np.zeros(n_variables)
        for i, var in enumerate(self.variables):
            not_loglin_flags[i] = var.base_name in not_loglin_variables

        param_dict = self.parameters(**param_updates)

        if steady_state_dict is None:
            steady_state_dict = self.steady_state()

        ss_values = np.array(list(steady_state_dict.values()))
        ss_zeros = np.abs(ss_values) < 1e-8
        ss_negative = ss_values < 0.0

        if np.any(ss_zeros):
            zero_idxs = np.flatnonzero(ss_zeros)
            zero_vars = [self.variables[i] for i in zero_idxs]
            _log.warning(
                f"The following variables had steady-state values close to zero and will not be log-linearized:"
                f"{[x.base_name for x in zero_vars]}"
            )

            not_loglin_flags[ss_zeros] = 1

        if np.any(ss_negative) and not loglin_negative_ss:
            neg_idxs = np.flatnonzero(ss_negative)
            neg_vars = [self.variables[i] for i in neg_idxs]
            _log.warning(
                f"The following variables had negative steady-state values and will not be log-linearized:"
                f"{[x.base_name for x in neg_vars]}"
            )

            not_loglin_flags[neg_idxs] = 1

        A, B, C, D = self.f_linearize(
            **param_dict, **steady_state_dict, not_loglin_variable=not_loglin_flags
        )

        return A, B, C, D


#
#     def steady_state(
#         self,
#         verbose: Optional[bool] = True,
#         model_is_linear: Optional[bool] = False,
#         apply_user_simplifications=True,
#         method: Optional[str] = "root",
#         optimizer_kwargs: Optional[dict[str, Any]] = None,
#         use_jac: Optional[bool] = True,
#         use_hess: Optional[bool] = True,
#         tol: Optional[float] = 1e-6,
#     ) -> None:
#         """
#         Solves for a function f(params) that computes steady state values and calibrated parameter values given
#         parameter values, stores results, and verifies that the residuals of the solution are zero.
#
#         Parameters
#         ----------
#         verbose: bool
#             Flag controlling whether to print results of the steady state solver. Default is True.
#         model_is_linear: bool, optional
#             If True, the model is assumed to have been linearized by the user. A specialized solving routine is used
#             to find the steady state, which is likely all zeros. If True, all other arguments to this function
#             have no effect (except verbose). Default is False.
#         apply_user_simplifications: bool
#             Whether to simplify system equations using the user-defined steady state relationships defined in the GCN
#             before passing the system to the numerical solver. Default is True.
#         method: str
#             One of "root" or "minimize". Indicates which family of solution algorithms should be used to find a
#             numerical steady state: direct root finding or minimization of squared error. Not that "root" is not
#             suitable if the number of inputs is not equal to the number of outputs, for example if user-provided
#             steady state relationships do not result in elimination of model equations. Default is "root".
#         optimizer_kwargs: dict
#             Dictionary of arguments to be passed to scipy.optimize.root or scipy.optimize.minimize, see those
#             functions for more details.
#         use_jac: bool
#             Whether to symbolically compute the Jacobian matrix of the steady state system (when method is "root") or
#             the Jacobian vector of the loss function (when method is "minimize"). Strongly recommended. Default is True
#         use_hess: bool
#             Whether to symbolically compute the Hessian matrix of the loss function. Ignored if method is "root".
#             If "False", the default BFGS solver will compute a numerical approximation, so not necessarily required.
#             Still recommended. Default is True.
#         tol: float
#             Numerical tolerance for declaring a steady-state solution valid. Default is 1e-6. Note that this only used
#             by the gEconpy model to decide if a steady state has been found, and is **NOT** passed to the scipy
#             solution algorithms. To adjust solution tolerance for these algorithms, use optimizer_kwargs.
#
#         Returns
#         -------
#         None
#         """
##
#         if not self.steady_state_solved:
#             self.f_ss = self.steady_state_solver.solve_steady_state(
#                 apply_user_simplifications=apply_user_simplifications,
#                 model_is_linear=model_is_linear,
#                 method=method,
#                 optimizer_kwargs=optimizer_kwargs,
#                 use_jac=use_jac,
#                 use_hess=use_hess,
#             )
#
#         self._process_steady_state_results(verbose, tol=tol)
#
#     def _process_steady_state_results(self, verbose=True, tol=1e-6) -> None:
#         """Process results from steady state solver.
#
#         This function sets the steady state dictionary, calibrated parameter dictionary, and residuals attribute
#         based on the results of the steady state solver. It also sets the `steady_state_solved` attribute to
#         indicate whether the steady state was successfully found. If `verbose` is True, it prints a message
#         indicating whether the steady state was found and the sum of squared residuals.
#
#         Parameters
#         ----------
#         verbose : bool, optional
#             If True, print a message indicating whether the steady state was found and the sum of squared residuals.
#             Default is True.
#         tol: float, optional
#             Numerical tolerance for declaring a steady-state solution has been found. Default is 1e-6.
#
#         Returns
#         -------
#         None
#         """
#         results = self.f_ss(self.free_param_dict)
#         self.steady_state_dict = results["ss_dict"]
#         self.calib_param_dict = results["calib_dict"]
#         self.residuals = results["resids"]
#
#         self.steady_state_system = self.steady_state_solver.steady_state_system
#         self.steady_state_solved = np.allclose(self.residuals, 0, atol=tol) & results["success"]
#
#         if verbose:
#             if self.steady_state_solved:
#                 print(
#                     f"Steady state found! Sum of squared residuals is {(self.residuals ** 2).sum()}"
#                 )
#             else:
#                 print(
#                     f"Steady state NOT found. Sum of squared residuals is {(self.residuals ** 2).sum()}"
#                 )
#
#     def print_steady_state(self):
#         """
#         Prints the steady state values for the model's variables and calibrated parameters.
#
#         Prints an error message if a valid steady state has not yet been found.
#         """
#         if len(self.steady_state_dict) == 0:
#             print("Run the steady_state method to find a steady state before calling this method.")
#             return
#
#         output = []
#         if not self.steady_state_solved:
#             output.append(
#                 "Values come from the latest solver iteration but are NOT a valid steady state."
#             )
#
#         max_var_name = (
#             max(
#                 len(x)
#                 for x in list(self.steady_state_dict.keys()) + list(self.calib_param_dict.keys())
#             )
#             + 5
#         )
#
#         for key, value in self.steady_state_dict.items():
#             output.append(f"{key:{max_var_name}}{value:>10.3f}")
#
#         if len(self.params_to_calibrate) > 0:
#             output.append("\n")
#             output.append("In addition, the following parameter values were calibrated:")
#             for key, value in self.calib_param_dict.items():
#                 output.append(f"{key:{max_var_name}}{value:>10.3f}")
#
#         print("\n".join(output))
#
#     def solve_model(
#         self,
#         solver="cycle_reduction",
#         not_loglin_variable: Optional[list[str]] = None,
#         order: int = 1,
#         model_is_linear: bool = False,
#         tol: float = 1e-8,
#         max_iter: int = 1000,
#         verbose: bool = True,
#         on_failure="error",
#     ) -> None:
#         """
#         Solve for the linear approximation to the policy function via perturbation. Adapted from R code in the gEcon
#         package by Grzegorz Klima, Karol Podemski, and Kaja Retkiewicz-Wijtiwiak., http://gecon.r-forge.r-project.org/.
#
#         Parameters
#         ----------
#         solver: str, default: 'cycle_reduction'
#             Name of the algorithm to solve the linear solution. Currently "cycle_reduction" and "gensys" are supported.
#             Following Dynare, cycle_reduction is the default, but note that gEcon uses gensys.
#         not_loglin_variable: List, default: None
#             Variables to not log linearize when solving the model. Variables with steady state values close to zero
#             will be automatically selected to not log linearize.
#         order: int, default: 1
#             Order of taylor expansion to use to solve the model. Currently only 1st order approximation is supported.
#         model_is_linear: bool, default: False
#             Flag indicating whether a model has already been linearized by the user.
#         tol: float, default 1e-8
#             Desired level of floating point accuracy in the solution
#         max_iter: int, default: 1000
#             Maximum number of cycle_reduction iterations. Not used if solver is 'gensys'.
#         verbose: bool, default: True
#             Flag indicating whether to print solver results to the terminal
#         on_failure: str, one of ['error', 'ignore'], default: 'error'
#             Instructions on what to do if the algorithm to find a linearized policy matrix. "Error" will raise an error,
#             while "ignore" will return None. "ignore" is useful when repeatedly solving the model, e.g. when sampling.
#
#         Returns
#         -------
#         None
#         """
#
#         if on_failure not in ["error", "ignore"]:
#             raise ValueError(
#                 f'Parameter on_failure must be one of "error" or "ignore", found {on_failure}'
#             )
#
#         if self.options.get("linear", False):
#             model_is_linear = True
#
#         param_dict = self.free_param_dict | self.calib_param_dict
#         steady_state_dict = self.steady_state_dict
#
#         if self.build_perturbation_matrices is None:
#             self._perturbation_setup(not_loglin_variable, order, model_is_linear, verbose, bool)
#
#         A, B, C, D = self.build_perturbation_matrices(
#             np.array(list(param_dict.values())),
#             np.array(list(steady_state_dict.values())),
#         )
#         _, variables, _ = self.perturbation_solver.make_all_variable_time_combinations()
#
#         if solver == "gensys":
#             gensys_results = self.perturbation_solver.solve_policy_function_with_gensys(
#                 A, B, C, D, tol, verbose
#             )
#             G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = gensys_results
#
#             success = all([x == 1 for x in eu[:2]])
#
#             if not success:
#                 if on_failure == "error":
#                     raise GensysFailedException(eu)
#                 elif on_failure == "ignore":
#                     if verbose:
#                         message = interpret_gensys_output(eu)
#                         print(message)
#                     self.P = None
#                     self.Q = None
#                     self.R = None
#                     self.S = None
#
#                     self.perturbation_solved = False
#
#                     return
#
#             if verbose:
#                 message = interpret_gensys_output(eu)
#                 print(message)
#                 print(
#                     "Policy matrices have been stored in attributes model.P, model.Q, model.R, and model.S"
#                 )
#
#             T = G_1[: self.n_variables, :][:, : self.n_variables]
#             R = impact[: self.n_variables, :]
#
#         elif solver == "cycle_reduction":
#             (
#                 T,
#                 R,
#                 result,
#                 log_norm,
#             ) = self.perturbation_solver.solve_policy_function_with_cycle_reduction(
#                 A, B, C, D, max_iter, tol, verbose
#             )
#             if T is None:
#                 if on_failure == "error":
#                     raise GensysFailedException(result)
#         else:
#             raise NotImplementedError(
#                 'Only "cycle_reduction" and "gensys" are valid values for solver'
#             )
#
#         gEcon_matrices = self.perturbation_solver.statespace_to_gEcon_representation(
#             A, T, R, variables, tol
#         )
#         P, Q, _, _, A_prime, R_prime, S_prime = gEcon_matrices
#
#         resid_norms = self.perturbation_solver.residual_norms(
#             B, C, D, Q, P, A_prime, R_prime, S_prime
#         )
#         norm_deterministic, norm_stochastic = resid_norms
#
#         if verbose:
#             print(f"Norm of deterministic part: {norm_deterministic:0.9f}")
#             print(f"Norm of stochastic part:    {norm_deterministic:0.9f}")
#
#         self.T = pd.DataFrame(
#             T,
#             index=[x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)],
#             columns=[x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)],
#         )
#         self.R = pd.DataFrame(
#             R,
#             index=[x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)],
#             columns=[x.base_name for x in sorted(self.shocks, key=lambda x: x.base_name)],
#         )
#
#         self.perturbation_solved = True
#
#     def _perturbation_setup(
#         self,
#         not_loglin_variables=None,
#         order=1,
#         model_is_linear=False,
#         verbose=True,
#         return_F_matrices=False,
#         tol=1e-8,
#     ):
#         """
#         Set up the perturbation matrices needed to simulate the model. Linearizes the model around the steady state and
#         constructs matrices A, B, C, and D needed to solve the system.
#
#         Parameters
#         ----------
#         not_loglin_variables: list of str
#             List of variables that should not be log-linearized. This is useful when a variable has a zero or negative
#              steady state value and cannot be log-linearized.
#         order: int
#             The order of the approximation. Currently only order 1 is implemented.
#         model_is_linear: bool
#             If True, assumes that the model is already linearized in the GCN file and directly
#              returns the matrices A, B, C, D.
#         verbose: bool
#             If True, prints warning messages.
#         return_F_matrices: bool
#             If True, returns the matrices A, B, C, D.
#         tol: float
#             The tolerance used to determine if a steady state value is close to zero.
#
#         Returns
#         -------
#         None or list of sympy matrices
#             If return_F_matrices is True, returns the F matrices. Otherwise, does not return anything.
#
#         """
#
#         if self.options.get("linear", False):
#             model_is_linear = True
#
#         free_param_dict = self.free_param_dict.copy()
#
#         parameters = list(free_param_dict.to_sympy().keys())
#         variables = list(self.steady_state_dict.to_sympy().keys())
#         params_to_calibrate = list(self.calib_param_dict.to_sympy().keys())
#
#         all_params = parameters + params_to_calibrate
#
#         shocks = self.shocks
#         shock_ss_dict = dict(zip([x.to_ss() for x in shocks], np.zeros(self.n_shocks)))
#         variables_and_shocks = self.variables + shocks
#         valid_names = [x.base_name for x in variables_and_shocks]
#
#         steady_state_dict = self.steady_state_dict.copy()
#
#         if not model_is_linear:
#             # We need shocks to be zero in A, B, C, D but 1 in T; can abuse the T_dummies to accomplish that.
#             if not_loglin_variables is None:
#                 not_loglin_variables = []
#
#             not_loglin_variables += [x.base_name for x in shocks]
#
#             # Validate that all user-supplied variables are in the model
#             for variable in not_loglin_variables:
#                 if variable not in valid_names:
#                     raise VariableNotFoundException(variable)
#
#             # Variables that are zero at the SS can't be log-linearized, check for these here.
#             close_to_zero_warnings = []
#             for variable in variables_and_shocks:
#                 if variable.base_name in not_loglin_variables:
#                     continue
#
#                 if abs(steady_state_dict[variable.to_ss().name]) < tol:
#                     not_loglin_variables.append(variable.base_name)
#                     close_to_zero_warnings.append(variable)
#
#             if len(close_to_zero_warnings) > 0 and verbose:
#                 warn(
#                     "The following variables have steady state values close to zero and will not be log linearized: "
#                     + ", ".join(x.base_name for x in close_to_zero_warnings)
#                 )
#
#         if order != 1:
#             raise NotImplementedError
#
#         if not self.steady_state_solved:
#             raise SteadyStateNotSolvedError()
#
#         if model_is_linear:
#             Fs = self.perturbation_solver.convert_linear_system_to_matrices()
#
#         else:
#             Fs = self.perturbation_solver.log_linearize_model(
#                 not_loglin_variables=not_loglin_variables
#             )
#
#         Fs_subbed = [F.subs(shock_ss_dict) for F in Fs]
#         self.build_perturbation_matrices = numba_lambdify(
#             exog_vars=all_params, endog_vars=variables, expr=Fs_subbed
#         )
#
#         if return_F_matrices:
#             return Fs_subbed
#
#     def check_bk_condition(
#         self,
#         free_param_dict: Optional[dict[str, float]] = None,
#         system_matrices: Optional[list[ArrayLike]] = None,
#         verbose: Optional[bool] = True,
#         return_value: Optional[str] = "df",
#         tol=1e-8,
#     ) -> Union[bool, pd.DataFrame]:
#         """
#         Compute the generalized eigenvalues of system in the form presented in [1]. Per [2], the number of
#         unstable eigenvalues (|v| > 1) should not be greater than the number of forward-looking variables. Failing
#         this test suggests timing problems in the definition of the model.
#
#         Parameters
#         ----------
#         free_param_dict: dict, optional
#             A dictionary of parameter values. If None, the current stored values are used.
#         system_matrices: list, optional
#             A list of matrices A, B, C, D to be used to compute the bk_condition. If none, the current
#             stored values are used.
#         verbose: bool, default: True
#             Flag to print the results of the test, otherwise the eigenvalues are returned without comment.
#         return_value: string, default: 'df'
#             Controls what is returned by the function. Valid values are 'df', 'bool', and 'none'.
#             If df, a dataframe containing eigenvalues is returned. If 'bool', a boolean indicating whether the BK
#             condition is satisfied. If None, nothing is returned.
#         tol: float, 1e-8
#             Convergence tolerance for the gensys solver
#
#         Returns
#         -------
#         None
#             If return_value is 'none'
#
#         condition_satisfied, bool
#             If return_value is 'bool', returns True if the Blanchard-Kahn condition is satisfied, False otherwise.
#
#         Eigenvalues, pd.DataFrame
#             If return_value is 'df', returns a dataframe containing the real and imaginary components of the system's
#             eigenvalues, along with their modulus.
#         """
#         if self.build_perturbation_matrices is None:
#             raise PerturbationSolutionNotFoundException()
#
#         if return_value not in ["df", "bool", "none"]:
#             raise ValueError(
#                 f'return_value must be one of "df", "bool", or "none". Found {return_value} '
#             )
#
#         if free_param_dict is not None:
#             results = self.f_ss(self.free_param_dict)
#             self.steady_state_dict = results["ss_dict"]
#             self.calib_param_dict = results["calib_dict"]
#
#         param_dict = self.free_param_dict | self.calib_param_dict
#         steady_state_dict = self.steady_state_dict
#
#         if system_matrices is not None:
#             A, B, C, D = system_matrices
#         else:
#             A, B, C, D = self.build_perturbation_matrices(
#                 np.array(list(param_dict.values())),
#                 np.array(list(steady_state_dict.values())),
#             )
#
#         n_forward = (C.sum(axis=0) > 0).sum().astype(int)
#         n_eq, n_vars = A.shape
#
#         # TODO: Compute system eigenvalues -- avoids calling the whole Gensys routine, but there is code duplication
#         #   building Gamma_0 and Gamma_1
#         lead_var_idx = np.where(np.sum(np.abs(C), axis=0) > tol)[0]
#
#         eqs_and_leads_idx = np.r_[np.arange(n_vars), lead_var_idx + n_vars].tolist()
#
#         Gamma_0 = np.vstack([np.hstack([B, C]), np.hstack([-np.eye(n_eq), np.zeros((n_eq, n_eq))])])
#
#         Gamma_1 = np.vstack(
#             [
#                 np.hstack([A, np.zeros((n_eq, n_eq))]),
#                 np.hstack([np.zeros((n_eq, n_eq)), np.eye(n_eq)]),
#             ]
#         )
#         Gamma_0 = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
#         Gamma_1 = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
#
#         # A, B, Q, Z = qzdiv(1.01, *linalg.qz(-Gamma_0, Gamma_1, 'complex'))
#
#         # Using scipy instead of qzdiv appears to offer a huge speedup for nearly the same answer; some eigenvalues
#         # have sign flip relative to qzdiv -- does it matter?
#         A, B, alpha, beta, Q, Z = linalg.ordqz(-Gamma_0, Gamma_1, sort="ouc", output="complex")
#
#         gev = np.c_[np.diagonal(A), np.diagonal(B)]
#
#         eigenval = gev[:, 1] / (gev[:, 0] + tol)
#         pos_idx = np.where(np.abs(eigenval) > 0)
#         eig = np.zeros(((np.abs(eigenval) > 0).sum(), 3))
#         eig[:, 0] = np.abs(eigenval)[pos_idx]
#         eig[:, 1] = np.real(eigenval)[pos_idx]
#         eig[:, 2] = np.imag(eigenval)[pos_idx]
#
#         sorted_idx = np.argsort(eig[:, 0])
#         eig = pd.DataFrame(eig[sorted_idx, :], columns=["Modulus", "Real", "Imaginary"])
#
#         n_g_one = (eig["Modulus"] > 1).sum()
#         condition_not_satisfied = n_forward > n_g_one
#         if verbose:
#             print(
#                 f"Model solution has {n_g_one} eigenvalues greater than one in modulus and {n_forward} "
#                 f"forward-looking variables."
#                 f'\nBlanchard-Kahn condition is{" NOT" if condition_not_satisfied else ""} satisfied.'
#             )
#
#         if return_value == "none":
#             return
#         if return_value == "df":
#             return eig
#         elif return_value == "bool":
#             return ~condition_not_satisfied
#
#     def compute_stationary_covariance_matrix(
#         self,
#         shock_dict: Optional[dict[str, float]] = None,
#         shock_cov_matrix: Optional[ArrayLike] = None,
#     ):
#         """
#         Compute the stationary covariance matrix of the solved system by solving the associated discrete lyapunov
#         equation. In order to construct the shock covariance matrix, exactly one or zero of shock_dict or
#         shock_cov_matrix should be provided. If neither is provided, the prior means on the shocks will be used. If no
#         information about a shock is available, the standard deviation will be set to 0.01.
#
#         Parameters
#         ----------
#         shock_dict, dict of str: float, optional
#             A dictionary of shock sizes to be used to compute the stationary covariance matrix.
#         shock_cov_matrix: array, optional
#             An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
#
#         Returns
#         -------
#         sigma: DataFrame
#         """
#         if not self.perturbation_solved:
#             raise PerturbationSolutionNotFoundException()
#         shock_std_priors = get_shock_std_priors_from_hyperpriors(
#             self.shocks, self.hyper_priors, out_keys="parent"
#         )
#
#         if (
#             shock_dict is None
#             and shock_cov_matrix is None
#             and len(shock_std_priors) < self.n_shocks
#         ):
#             unknown_shocks_list = [
#                 shock.base_name
#                 for shock in self.shocks
#                 if shock not in self.shock_priors.to_sympy()
#             ]
#             unknown_shocks = ", ".join(unknown_shocks_list)
#             warn(
#                 f"No standard deviation provided for shocks {unknown_shocks}. Using default of std = 0.01. Explicity"
#                 f"pass variance information for these shocks or set their priors to silence this warning."
#             )
#
#         Q = build_Q_matrix(
#             model_shocks=[x.base_name for x in self.shocks],
#             shock_dict=shock_dict,
#             shock_cov_matrix=shock_cov_matrix,
#             shock_std_priors=shock_std_priors,
#         )
#
#         T, R = self.T, self.R
#         sigma = linalg.solve_discrete_lyapunov(T.values, R.values @ Q @ R.values.T)
#
#         return pd.DataFrame(sigma, index=T.index, columns=T.index)
#
#     def compute_autocorrelation_matrix(
#         self,
#         shock_dict: Optional[dict[str, float]] = None,
#         shock_cov_matrix: Optional[ArrayLike] = None,
#         n_lags=10,
#     ):
#         """
#         Computes autocorrelations for each model variable using the stationary covariance matrix. See doc string for
#         compute_stationary_covariance_matrix for more information.
#
#         In order to construct the shock covariance matrix, exactly one or zero of shock_dict or
#         shock_cov_matrix should be provided. If neither is provided, the prior means on the shocks will be used. If no
#         information about a shock is available, the standard deviation will be set to 0.01.
#
#         Parameters
#         ----------
#         shock_dict, dict of str: float, optional
#             A dictionary of shock sizes to be used to compute the stationary covariance matrix.
#         shock_cov_matrix: array, optional
#             An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
#         n_lags: int
#             Number of lags over which to compute the autocorrelation
#
#         Returns
#         -------
#         acorr_mat: DataFrame
#         """
#         if not self.perturbation_solved:
#             raise PerturbationSolutionNotFoundException()
#
#         T, R = self.T, self.R
#
#         Sigma = self.compute_stationary_covariance_matrix(
#             shock_dict=shock_dict, shock_cov_matrix=shock_cov_matrix
#         )
#         acorr_mat = compute_autocorrelation_matrix(T.values, Sigma.values, n_lags=n_lags)
#
#         return pd.DataFrame(acorr_mat, index=T.index, columns=np.arange(n_lags))
#
#     # def fit(
#     #     self,
#     #     data,
#     #     estimate_a0=False,
#     #     estimate_P0=False,
#     #     a0_prior=None,
#     #     P0_prior=None,
#     #     filter_type="univariate",
#     #     draws=5000,
#     #     n_walkers=36,
#     #     moves=None,
#     #     emcee_x0=None,
#     #     verbose=True,
#     #     return_inferencedata=True,
#     #     burn_in=None,
#     #     thin=None,
#     #     skip_initial_state_check=False,
#     #     compute_sampler_stats=True,
#     #     **sampler_kwargs,
#     # ):
#     #     """
#     #     Estimate model parameters via Bayesian inference. Parameter likelihood is computed using the Kalman filter.
#     #     Posterior distributions are estimated using Markov Chain Monte Carlo (MCMC), specifically the Affine-Invariant
#     #     Ensemble Sampler algorithm of [1].
#     #
#     #     A "traditional" Random Walk Metropolis can be achieved using the moves argument, but by default this function
#     #     will use a mix of two Differential Evolution (DE) proposal algorithms that have been shown to work well on
#     #     weakly multi-modal problems. DSGE estimation can be multi-modal in the sense that regions of the posterior
#     #     space are separated by the constraints on the ability to solve the perturbation problem.
#     #
#     #     This function will start all MCMC chains around random draws from the prior distribution. This is in contrast
#     #     to Dynare and gEcon.estimate, which start MCMC chains around the Maximum Likelihood estimate for parameter
#     #     values.
#     #
#     #     Parameters
#     #     ----------
#     #     data: dataframe
#     #         A pandas dataframe of observed values, with column names corresponding to DSGE model states names.
#     #     estimate_a0: bool, default: False
#     #         Whether to estimate the initial values of the DSGE process. If False, x0 will be deterministically set to
#     #         a vector of zeros, corresponding to the steady state. If True, you must provide a
#     #     estimate_P0: bool, default: False
#     #         Whether to estimate the intial covariance matrix of the DSGE process. If False, P0 will be set to the
#     #         Kalman Filter steady state value by solving the associated discrete Lyapunov equation.
#     #     a0_prior: dict, optional
#     #         A dictionary with (variable name, scipy distribution) key-value pairs. If a key "initial_vector" is found,
#     #         all other keys will be ignored, and the single distribution over all initial states will be used. Otherwise,
#     #         n_states independent distributions should be included in the dictionary.
#     #         If estimate_a0 is False, this will be ignored.
#     #     P0_prior: dict, optional
#     #         A dictionary with (variable name, scipy distribution) key-value pairs. If a key "initial_covariance" is
#     #         found, all other keys will be ignored, and this distribution will be taken as over the entire covariance
#     #         matrix. Otherwise, n_states independent distributions are expected, and are used to construct a diagonal
#     #         initial covariance matrix.
#     #     filter_type: string, default: "standard"
#     #         Select a kalman filter implementation to use. Currently "standard" and "univariate" are supported. Try
#     #         univariate if you run into errors inverting the P matrix during filtering.
#     #     draws: integer
#     #         Number of draws from each MCMC chain, or "walker" in the jargon of emcee.
#     #     n_walkers: integer
#     #         The number of "walkers", which roughly correspond to chains in other MCMC packages. Note that one needs
#     #         many more walkers than chains; [1] recommends as many as possible.
#     #     cores: integer
#     #         The number of processing cores, which is passed to Multiprocessing.Pool to do parallel inference. To
#     #         maintain detailed balance, the pool of walkers must be split, resulting in n_walkers / cores sub-ensembles.
#     #         Be sure to raise the number of walkers to compensate.
#     #     moves: List of emcee.moves objects
#     #         Moves tell emcee how to generate MCMC proposals. See the emcee docs for details.
#     #     emcee_x0: array
#     #         An (n_walkers, k_parameters) array of initial values. Emcee will check the condition number of the matrix
#     #         to ensure all walkers begin in different regions of the parameter space. If MLE estimates are used, they
#     #         should be jittered to start walkers in a ball around the desired initial point.
#     #     return_inferencedata: bool, default: True
#     #         If true, return an Arviz InferenceData object containing posterior samples. If False, the fitted Emcee
#     #         sampler is returned.
#     #     burn_in: int, optional
#     #         Number of initial samples to discard from all chains. This is ignored if return_inferencedata is False.
#     #     thin: int, optional
#     #         Return only every n-th sample from each chain. This is done to reduce storage requirements in highly
#     #         autocorrelated chains by discarding redundant information. Ignored if return_inferencedata is False.
#     #
#     #     Returns
#     #     -------
#     #     sampler, emcee.Sampler object
#     #         An emcee.Sampler object with the estimated posterior over model parameters, as well as other diagnotic
#     #         information.
#     #
#     #     References
#     #     ----------
#     #     ..[1] Foreman-Mackey, Daniel, et al. “Emcee: The MCMC Hammer.” Publications of the Astronomical Society of the
#     #           Pacific, vol. 125, no. 925, Mar. 2013, pp. 306-12. arXiv.org, https://doi.org/10.1086/670067.
#     #     """
#     #     observed_vars = data.columns.tolist()
#     #     n_obs = len(observed_vars)
#     #     n_shocks = self.n_shocks
#     #     model_var_names = [x.base_name for x in self.variables]
#     #     n_noise_priors = len(self.observation_noise_priors)
#     #
#     #     if n_obs > (n_noise_priors + n_shocks):
#     #         raise ValueError(
#     #             f"Number of observed parameters in data ({n_obs}) is greater than the number of sources "
#     #             f"of stochastic variance - shocks ({n_shocks}) and observation noise ({n_noise_priors}). "
#     #             f"The model cannot be fit due to stochastic singularity."
#     #         )
#     #
#     #     if burn_in is None:
#     #         burn_in = 0
#     #
#     #     if not all([x in model_var_names for x in observed_vars]):
#     #         orphans = [x for x in observed_vars if x not in model_var_names]
#     #         raise ValueError(
#     #             f"Columns of data must correspond to states of the DSGE model. Found the following columns"
#     #             f'with no associated model state: {", ".join(orphans)}'
#     #         )
#     #
#     #     # sparse_data = extract_sparse_data_from_model(self)
#     #     prior_dict = extract_prior_dict(self)
#     #
#     #     if estimate_a0 is False:
#     #         a0 = None
#     #     else:
#     #         if a0_prior is None:
#     #             raise ValueError(
#     #                 "If estimate_a0 is True, you must provide a dictionary of prior distributions for"
#     #                 "the initial values of all individual states"
#     #             )
#     #         if not all([var in a0_prior.keys() for var in model_var_names]):
#     #             missing_keys = set(model_var_names) - set(list(a0_prior.keys()))
#     #             raise ValueError(
#     #                 "You must provide one key for each state in the model. "
#     #                 f'No keys found for: {", ".join(missing_keys)}'
#     #             )
#     #         for var in model_var_names:
#     #             prior_dict[f"{var}__initial"] = a0_prior[var]
#     #
#     #     moves = moves or [
#     #         (emcee.moves.DEMove(), 0.6),
#     #         (emcee.moves.DESnookerMove(), 0.4),
#     #     ]
#     #
#     #     shock_names = [x.base_name for x in self.shocks]
#     #
#     #     k_params = len(prior_dict)
#     #     Z = build_Z_matrix(observed_vars, model_var_names)
#     #
#     #     args = [
#     #         data.values,
#     #         self.f_ss,
#     #         self.build_perturbation_matrices,
#     #         self.free_param_dict,
#     #         Z,
#     #         prior_dict,
#     #         shock_names,
#     #         observed_vars,
#     #         filter_type,
#     #     ]
#     #
#     #     arg_names = [
#     #         "observed_data",
#     #         "f_ss",
#     #         "f_pert",
#     #         "free_params",
#     #         "Z",
#     #         "prior_dict",
#     #         "shock_names",
#     #         "observed_vars",
#     #         "filter_type",
#     #     ]
#     #
#     #     if emcee_x0:
#     #         x0 = emcee_x0
#     #     else:
#     #         x0 = np.stack([x.rvs(n_walkers) for x in prior_dict.values()]).T
#     #
#     #     param_names = list(prior_dict.keys())
#     #
#     #     sampler = emcee.EnsembleSampler(
#     #         n_walkers,
#     #         k_params,
#     #         evaluate_logp2,
#     #         args=args,
#     #         moves=moves,
#     #         parameter_names=param_names,
#     #         **sampler_kwargs,
#     #     )
#     #
#     #     with catch_warnings():
#     #         simplefilter("ignore")
#     #         _ = sampler.run_mcmc(
#     #             x0,
#     #             draws + burn_in,
#     #             progress=verbose,
#     #             skip_initial_state_check=skip_initial_state_check,
#     #         )
#     #
#     #     if return_inferencedata:
#     #         idata = az.from_emcee(
#     #             sampler,
#     #             var_names=param_names,
#     #             blob_names=["log_likelihood"],
#     #             arg_names=arg_names,
#     #         )
#     #
#     #         if compute_sampler_stats:
#     #             sampler_stats = xr.Dataset(
#     #                 data_vars=dict(
#     #                     acceptance_fraction=(["chain"], sampler.acceptance_fraction),
#     #                     autocorrelation_time=(
#     #                         ["parameters"],
#     #                         sampler.get_autocorr_time(discard=burn_in or 0, quiet=True),
#     #                     ),
#     #                 ),
#     #                 coords=dict(chain=np.arange(n_walkers), parameters=param_names),
#     #             )
#     #
#     #             idata["sample_stats"].update(sampler_stats)
#     #         idata.observed_data = idata.observed_data.drop_vars(["prior_dict"])
#     #
#     #         return idata.sel(draw=slice(burn_in, None, thin))
#     #
#     #     return sampler
#
#     def sample_param_dict_from_prior(self, n_samples=1, seed=None, param_subset=None):
#         """
#         Sample parameters from the parameter prior distributions.
#
#         Parameters
#         ----------
#         n_samples: int, default: 1
#             Number of samples to draw from the prior distributions.
#         seed: int, default: None
#             Seed for the random number generator.
#         param_subset: list, default: None
#             List of parameter names to sample. If None, all parameters are sampled.
#
#         Returns
#         -------
#         new_param_dict: dict
#             Dictionary of sampled parameters.
#         """
#         shock_std_priors = get_shock_std_priors_from_hyperpriors(self.shocks, self.hyper_priors)
#
#         all_priors = (
#             self.param_priors.to_sympy()
#             | shock_std_priors
#             | self.observation_noise_priors.to_sympy()
#         )
#
#         if len(all_priors) == 0:
#             raise ValueError("No model priors found, cannot sample.")
#
#         if param_subset is None:
#             n_variables = len(all_priors)
#             priors_to_sample = all_priors
#         else:
#             n_variables = len(param_subset)
#             priors_to_sample = SymbolDictionary(
#                 {k: v for k, v in all_priors.items() if k.name in param_subset}
#             )
#
#         if seed is not None:
#             seed_sequence = np.random.SeedSequence(seed)
#             child_seeds = seed_sequence.spawn(n_variables)
#             streams = [np.random.default_rng(s) for s in child_seeds]
#         else:
#             streams = [None] * n_variables
#
#         new_param_dict = {}
#         for i, (key, d) in enumerate(priors_to_sample.items()):
#             new_param_dict[key] = d.rvs(size=n_samples, random_state=streams[i])
#
#         free_param_dict, shock_dict, obs_dict = split_random_variables(
#             new_param_dict, self.shocks, self.variables
#         )
#
#         return free_param_dict.to_string(), shock_dict.to_string(), obs_dict.to_string()
#
#     def impulse_response_function(self, simulation_length: int = 40, shock_size: float = 1.0):
#         """
#         Compute the impulse response functions of the model.
#
#         Parameters
#         ----------
#         simulation_length : int, optional
#             The number of periods to compute the IRFs over. The default is 40.
#         shock_size : float, optional
#             The size of the shock. The default is 1.0.
#
#         Returns
#         -------
#         pandas.DataFrame
#             The IRFs for each variable in the model. The DataFrame has a multi-index
#             with the variable names as the first level and the timestep as the second.
#             The columns are the shocks.
#
#         Raises
#         ------
#         PerturbationSolutionNotFoundException
#             If a perturbation solution has not been found.
#         """
#
#         if not self.perturbation_solved:
#             raise PerturbationSolutionNotFoundException()
#
#         T, R = self.T, self.R
#
#         timesteps = simulation_length
#
#         data = np.zeros((self.n_variables, timesteps, self.n_shocks))
#
#         for i in range(self.n_shocks):
#             shock_path = np.zeros((self.n_shocks, timesteps))
#             shock_path[i, 0] = shock_size
#
#             for t in range(1, timesteps):
#                 stochastic = R.values @ shock_path[:, t - 1]
#                 deterministic = T.values @ data[:, t - 1, i]
#                 data[:, t, i] = deterministic + stochastic
#
#         index = pd.MultiIndex.from_product(
#             [R.index, np.arange(timesteps), R.columns],
#             names=["Variables", "Time", "Shocks"],
#         )
#
#         df = (
#             pd.DataFrame(data.ravel(), index=index, columns=["Values"])
#             .unstack([1, 2])
#             .droplevel(axis=1, level=0)
#             .sort_index(axis=1)
#         )
#
#         return df
#
#     def simulate(
#         self,
#         simulation_length: int = 40,
#         n_simulations: int = 100,
#         shock_dict: Optional[dict[str, float]] = None,
#         shock_cov_matrix: Optional[ArrayLike] = None,
#         show_progress_bar: bool = False,
#     ):
#         """
#         Simulate the model over a certain number of time periods.
#
#         Parameters
#         ----------
#         simulation_length : int, optional(default=40)
#             The number of time periods to simulate.
#         n_simulations : int, optional(default=100)
#             The number of simulations to run.
#         shock_dict : dict, optional(default=None)
#             Dictionary of shocks to use.
#         shock_cov_matrix : arraylike, optional(default=None)
#             Covariance matrix of shocks to use.
#         show_progress_bar : bool, optional(default=False)
#             Whether to show a progress bar for the simulation.
#
#         Returns
#         -------
#         df : pandas.DataFrame
#             The simulated data.
#         """
#
#         if not self.perturbation_solved:
#             raise PerturbationSolutionNotFoundException()
#
#         T, R = self.T, self.R
#         timesteps = simulation_length
#
#         n_shocks = R.shape[1]
#         shock_std_priors = get_shock_std_priors_from_hyperpriors(
#             self.shocks, self.hyper_priors, out_keys="parent"
#         )
#
#         if (
#             shock_dict is None
#             and shock_cov_matrix is None
#             and len(shock_std_priors) < self.n_shocks
#         ):
#             unknown_shocks_list = [
#                 shock.base_name
#                 for shock in self.shocks
#                 if shock not in self.shock_priors.to_sympy()
#             ]
#             unknown_shocks = ", ".join(unknown_shocks_list)
#             warn(
#                 f"No standard deviation provided for shocks {unknown_shocks}. Using default of std = 0.01. Explicity"
#                 f"pass variance information for these shocks or set their priors to silence this warning."
#             )
#
#         Q = build_Q_matrix(
#             model_shocks=[x.base_name for x in self.shocks],
#             shock_dict=shock_dict,
#             shock_cov_matrix=shock_cov_matrix,
#             shock_std_priors=shock_std_priors,
#         )
#
#         d_epsilon = stats.multivariate_normal(mean=np.zeros(n_shocks), cov=Q)
#         epsilons = np.r_[[d_epsilon.rvs(timesteps) for _ in range(n_simulations)]]
#
#         data = np.zeros((self.n_variables, timesteps, n_simulations))
#         if epsilons.ndim == 2:
#             epsilons = epsilons[:, :, None]
#
#         progress_bar = ProgressBar(timesteps - 1, verb="Sampling")
#
#         for t in range(1, timesteps):
#             progress_bar.start()
#             stochastic = np.einsum("ij,sj", R.values, epsilons[:, t - 1, :])
#             deterministic = T.values @ data[:, t - 1, :]
#             data[:, t, :] = deterministic + stochastic
#
#             if show_progress_bar:
#                 progress_bar.stop()
#
#         index = pd.MultiIndex.from_product(
#             [R.index, np.arange(timesteps), np.arange(n_simulations)],
#             names=["Variables", "Time", "Simulation"],
#         )
#         df = (
#             pd.DataFrame(data.ravel(), index=index, columns=["Values"])
#             .unstack([1, 2])
#             .droplevel(axis=1, level=0)
#         )
#
#         return df
