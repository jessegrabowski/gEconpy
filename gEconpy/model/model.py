import functools as ft
import logging

from typing import Union, Literal
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
    GensysFailedException,
)
from gEconpy.model.compile import BACKENDS
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.perturbation import (
    override_dummy_wrapper,
    solve_policy_function_with_gensys,
    solve_policy_function_with_cycle_reduction,
    statespace_to_gEcon_representation,
    residual_norms,
    check_perturbation_solution,
    check_bk_condition,
)
from gEconpy.model.steady_state import (
    ERROR_FUNCTIONS,
    compile_known_ss,
    compile_ss_resid_and_sq_err,
    steady_state_error_function,
)
from gEconpy.solvers.gensys import interpret_gensys_output
import pandas as pd

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

            if isinstance(res, float | int):
                return res
            elif res.ndim == 1:
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


def infer_variable_bounds(variable):
    assumptions = variable.assumptions0
    is_positive = assumptions.get("positive", False)
    is_negative = assumptions.get("negative", False)
    lhs = 1e-8 if is_positive else None
    rhs = -1e-8 if is_negative else None

    return (lhs, rhs)


def validate_policy_function(
    A, B, C, D, T, R, variables, tol: float = 1e-8, verbose: bool = True
) -> bool:
    gEcon_matrices = statespace_to_gEcon_representation(A, T, R, variables, tol)

    P, Q, _, _, A_prime, R_prime, S_prime = gEcon_matrices

    resid_norms = residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime)
    norm_deterministic, norm_stochastic = resid_norms

    if verbose:
        print(f"Norm of deterministic part: {norm_deterministic:0.9f}")
        print(f"Norm of stochastic part:    {norm_deterministic:0.9f}")


class Model:
    def __init__(
        self,
        variables: list[TimeAwareSymbol],
        shocks: list[TimeAwareSymbol],
        equations: list[sp.Expr],
        param_dict: SymbolDictionary,
        deterministic_dict: SymbolDictionary,
        calib_dict: SymbolDictionary,
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
        A Dynamic Stochastic General Equlibrium (DSGE) Model

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

        self.deterministic_params = list(deterministic_dict.to_sympy().keys())
        self.calibrated_params = list(calib_dict.to_sympy().keys())

        self._default_params = param_dict.copy()
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

        # self.param_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        # self.shock_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        # self.hyper_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        # self.observation_noise_priors: SymbolDictionary[str, Any] = SymbolDictionary()

        # self.n_variables: int = 0
        # self.n_shocks: int = 0
        # self.n_equations: int = 0
        # self.n_calibrating_equations: int = 0

        # # Steady state information
        # self.steady_state_solved: bool = False
        # self.steady_state_system: list[sp.Add] = []
        # self.steady_state_dict: SymbolDictionary[sp.Symbol, float] = SymbolDictionary()
        # self.residuals: list[float] = []

        # Linear representation
        self.A: pd.DataFrame | None = None
        self.B: pd.DataFrame | None = None
        self.C: pd.DataFrame | None = None
        self.D: pd.DataFrame | None = None

        # Perturbation solution information
        self.T: pd.DataFrame | None = None
        self.R: pd.DataFrame | None = None
        self.P: pd.DataFrame | None = None
        self.Q: pd.DataFrame | None = None
        self.R: pd.DataFrame | None = None
        self.S: pd.DataFrame | None = None

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
        bounds: dict[str, tuple[float, float]] | None = None,
        **updates: float,
    ):
        param_dict = self.parameters(**updates)

        if how == "analytic" and self.f_ss is None:
            how = "minimize"
        else:
            ss_dict = self.f_ss(**param_dict) if self.f_ss is not None else {}
            if len(ss_dict) != 0 and len(ss_dict) != len(self.variables):
                how = "minimize"
            elif len(ss_dict) == len(self.variables):
                return ss_dict

        ss_variables = [x.to_ss() for x in self.variables] + list(
            self.calibrated_params
        )

        known_variables = (
            []
            if self.f_ss is None
            else list(self.f_ss(**self.parameters()).to_sympy().keys())
        )
        vars_to_solve = [var for var in ss_variables if var not in known_variables]
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
                use_jac=use_jac,
                vars_to_solve=var_names_to_solve,
                unknown_var_idx=unknown_var_idx,
                progressbar=progressbar,
                optimizer_kwargs=optimizer_kwargs,
                bounds=bounds,
                **updates,
            )
        elif how == "minimize":
            res = self._solve_steady_state_with_minimize(
                use_jac=use_jac,
                use_hess=use_hess,
                vars_to_solve=var_names_to_solve,
                unknown_var_idx=unknown_var_idx,
                progressbar=progressbar,
                bounds=bounds,
                optimizer_kwargs=optimizer_kwargs,
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
        bounds: dict[str, tuple[float, float]] | None = None,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        n_variables = len(vars_to_solve)

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        method = optimizer_kwargs.pop("method", "hybr")

        if "options" not in optimizer_kwargs:
            optimizer_kwargs["options"] = {}

        if method in ["hybr", "df-sane"]:
            optimizer_kwargs["options"].update({"maxfev": maxiter})
        else:
            optimizer_kwargs["options"].update({"maxiter": maxiter})

        x0 = optimizer_kwargs.pop("x0", np.full(n_variables, 0.8))
        if jitter_x0:
            x0 += np.random.normal(scale=0.01, size=n_variables)

        param_dict = self.parameters(**param_updates)

        objective = CostFuncWrapper(
            maxeval=maxiter,
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
        with np.errstate(all="ignore"):
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
        bounds: dict[str, tuple[float, float]] | None = None,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        n_variables = len(vars_to_solve)

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        if "options" not in optimizer_kwargs:
            optimizer_kwargs["options"] = {}
        optimizer_kwargs["options"].update({"maxiter": maxiter})

        x0 = optimizer_kwargs.pop("x0", np.full(n_variables, 0.8))
        if jitter_x0:
            x0 += np.random.normal(scale=0.01, size=n_variables)
        tol = optimizer_kwargs.pop("tol", 1e-30)

        # kwarg_bounds = optimizer_kwargs.pop("bounds", None)
        user_bounds = {} if bounds is None else bounds

        ss_variables = [x.to_ss() for x in self.variables] + self.calibrated_params
        vars_to_solve_sp = [x for x in ss_variables if x.name in vars_to_solve]

        bound_dict = {x.name: infer_variable_bounds(x) for x in vars_to_solve_sp}
        bound_dict.update(user_bounds)

        bounds = [bound_dict[x] for x in vars_to_solve]

        has_bounds = any([x != (None, None) for x in bounds])
        method = optimizer_kwargs.pop(
            "method", "trust-krylov" if not has_bounds else "trust-constr"
        )
        if method not in ["trust-constr", "L-BFGS-B"]:
            has_bounds = False

        param_dict = self.parameters(**param_updates)

        objective = CostFuncWrapper(
            maxeval=maxiter,
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

        equation_names = [f"Equation {i}" for i in range(A.shape[0])]
        self.A = pd.DataFrame(
            A, index=equation_names, columns=[x.set_t(-1) for x in self.variables]
        )
        self.B = pd.DataFrame(
            B, index=equation_names, columns=[x.set_t(0) for x in self.variables]
        )
        self.C = pd.DataFrame(
            C, index=equation_names, columns=[x.set_t(1) for x in self.variables]
        )
        self.D = pd.DataFrame(D, index=equation_names, columns=self.shocks)

        return A, B, C, D

    def solve_model(
        self,
        solver="cycle_reduction",
        not_loglin_variables: list[str] | None = None,
        order: int = 1,
        loglin_negative_ss: bool = False,
        steady_state_kwargs: dict | None = None,
        tol: float = 1e-8,
        max_iter: int = 1000,
        verbose: bool = True,
        on_failure="error",
        **parameter_updates,
    ) -> None:
        """
        Solve for the linear approximation to the policy function via perturbation. Adapted from R code in the gEcon
        package by Grzegorz Klima, Karol Podemski, and Kaja Retkiewicz-Wijtiwiak., http://gecon.r-forge.r-project.org/.

        Parameters
        ----------
        solver: str, default: 'cycle_reduction'
            Name of the algorithm to solve the linear solution. Currently "cycle_reduction" and "gensys" are supported.
            Following Dynare, cycle_reduction is the default, but note that gEcon uses gensys.
        not_loglin_variables: list of strings, optional
            Variables to not log linearize when solving the model. Variables with steady state values close to zero
            will be automatically selected to not log linearize.
        order: int, default: 1
            Order of taylor expansion to use to solve the model. Currently only 1st order approximation is supported.
        loglin_negative_ss: bool, default is False
            Whether to force log-linearization of variable with negative steady-state. This is impossible in principle
            (how can :math:`exp(x_ss)` be negative?), but can still be done; see the docstring for
            :fun:`perturbation.linearize_model` for details. Use with caution, as results will not correct.
        tol: float, default 1e-8
            Desired level of floating point accuracy in the solution
        max_iter: int, default: 1000
            Maximum number of cycle_reduction iterations. Not used if solver is 'gensys'.
        verbose: bool, default: True
            Flag indicating whether to print solver results to the terminal
        on_failure: str, one of ['error', 'ignore'], default: 'error'
            Instructions on what to do if the algorithm to find a linearized policy matrix. "Error" will raise an error,
            while "ignore" will return None. "ignore" is useful when repeatedly solving the model, e.g. when sampling.
        steady_state_kwargs: dict, optional
            Keyword arguments passed to the `steady_state` method. Default is None.
        parameter_updates: dict
            New parameter values at which to solve the model. Unspecified values will be taken from the initial values
            set in the GCN file.
        Returns
        -------
        None
        """

        if on_failure not in ["error", "ignore"]:
            raise ValueError(
                f'Parameter on_failure must be one of "error" or "ignore", found {on_failure}'
            )
        if steady_state_kwargs is None:
            steady_state_kwargs = {}

        ss_dict = self.steady_state(
            **self.parameters(**parameter_updates), **steady_state_kwargs
        )
        n_variables = len(self.variables)

        A, B, C, D = self.linearize_model(
            order=order,
            not_loglin_variables=not_loglin_variables,
            steady_state_dict=ss_dict.to_string(),
            loglin_negative_ss=loglin_negative_ss,
        )

        if solver == "gensys":
            gensys_results = solve_policy_function_with_gensys(A, B, C, D, tol)
            G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = gensys_results

            success = all([x == 1 for x in eu[:2]])

            if not success:
                if on_failure == "error":
                    raise GensysFailedException(eu)
                elif on_failure == "ignore":
                    if verbose:
                        message = interpret_gensys_output(eu)
                        print(message)
                    self.P = None
                    self.Q = None
                    self.R = None
                    self.S = None

                    return

            if verbose:
                message = interpret_gensys_output(eu)
                print(message)
                print(
                    "Policy matrices have been stored in attributes model.P, model.Q, model.R, and model.S"
                )

            T = G_1[:n_variables, :][:, :n_variables]
            R = impact[:n_variables, :]

        elif solver == "cycle_reduction":
            (
                T,
                R,
                result,
                log_norm,
            ) = solve_policy_function_with_cycle_reduction(
                A, B, C, D, max_iter, tol, verbose
            )
            if T is None:
                if on_failure == "error":
                    raise GensysFailedException(result)
        else:
            raise NotImplementedError(
                'Only "cycle_reduction" and "gensys" are valid values for solver'
            )

        self.T = pd.DataFrame(
            T,
            index=[
                x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)
            ],
            columns=[
                x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)
            ],
        )
        self.R = pd.DataFrame(
            R,
            index=[
                x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)
            ],
            columns=[
                x.base_name for x in sorted(self.shocks, key=lambda x: x.base_name)
            ],
        )

        if verbose:
            check_perturbation_solution(A, B, C, D, T, R, tol=tol)

    def check_bk_condition(
        self,
        tol=1e-8,
        verbose=True,
        return_value: Literal["dataframe", "bool", None] = "dataframe",
    ):
        """
        Compute the generalized eigenvalues of system in the form presented in [1]. Per [2], the number of
        unstable eigenvalues (|v| > 1) should not be greater than the number of forward-looking variables. Failing
        this test suggests timing problems in the definition of the model.

        Parameters
        ----------
        verbose: bool, default: True
            Flag to print the results of the test, otherwise the eigenvalues are returned without comment.

        return_value: string, default: 'dataframe'
            Controls what is returned by the function. Valid values are 'dataframe', 'bool', and 'none'.
            If df, a dataframe containing eigenvalues is returned. If 'bool', a boolean indicating whether the BK
            condition is satisfied. If None, nothing is returned.

        tol: float, 1e-8
            Tolerance below which numerical values are considered zero

        Returns
        -------
        None
            If return_value is 'none'

        condition_satisfied, bool
            If return_value is 'bool', returns True if the Blanchard-Kahn condition is satisfied, False otherwise.

        Eigenvalues, pd.DataFrame
            If return_value is 'df', returns a dataframe containing the real and imaginary components of the system's
            eigenvalues, along with their modulus.
        """

        A, B, C, D = self.A, self.B, self.C, self.D
        if any([x is None for x in [A, B, C, D]]):
            raise ValueError("Model has not been linearized, cannot check BK condition")
        return check_bk_condition(
            A.values,
            B.values,
            C.values,
            D.values,
            tol=tol,
            return_value=return_value,
            verbose=verbose,
        )


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