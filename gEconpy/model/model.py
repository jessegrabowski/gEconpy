import functools as ft
import logging

from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Literal, cast

import numba as nb
import numpy as np
import sympy as sp
import xarray as xr

from scipy import linalg, optimize

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.optimize_wrapper import (
    CostFuncWrapper,
    optimzer_early_stopping_wrapper,
    postprocess_optimizer_res,
)
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions.exceptions import (
    GensysFailedException,
    ModelUnknownParameterError,
    PerturbationSolutionNotFoundException,
)
from gEconpy.model.compile import BACKENDS
from gEconpy.model.perturbation import (
    check_perturbation_solution,
    override_dummy_wrapper,
    residual_norms,
    solve_policy_function_with_cycle_reduction,
    solve_policy_function_with_gensys,
    statespace_to_gEcon_representation,
)
from gEconpy.model.steady_state import system_to_steady_state
from gEconpy.solvers.gensys import interpret_gensys_output

VariableType = sp.Symbol | TimeAwareSymbol
_log = logging.getLogger(__name__)


def scipy_wrapper(
    f: Callable,
    variables: list[str],
    unknown_var_idxs: np.ndarray[int | bool],
    unknown_eq_idxs: np.ndarray[int | bool],
    f_ss: Callable | None = None,
) -> Callable:
    if f_ss is not None:

        @ft.wraps(f)
        def inner(ss_values, param_dict):
            given_ss = f_ss(**param_dict)
            ss_dict = SymbolDictionary(zip(variables, ss_values)).to_string()
            ss_dict.update(given_ss)
            res = f(**ss_dict, **param_dict)

            if isinstance(res, float | int):
                return res
            elif res.ndim == 1:
                res = res[unknown_eq_idxs]
            elif res.ndim == 2:
                res = res[unknown_eq_idxs, :][:, unknown_var_idxs]
            return res

    else:

        @ft.wraps(f)
        def inner(ss_values, param_dict):
            ss_dict = SymbolDictionary(zip(variables, ss_values)).to_string()
            return f(**ss_dict, **param_dict)

    return inner


def add_more_ss_values_wrapper(
    f_ss: Callable | None, known_variables: SymbolDictionary
) -> Callable:
    """
    Inject user-provided constant steady state values to the return of the steady state function.

    Parameters
    ----------
    f_ss: Callable, Optional
        Compiled function that maps models parameters to numerical steady state values for variables.

    known_variables: SymbolDictionary
        Numerical values for model variables in the steady state provided by the user. Keys are expected to be string
        variable names, and values floats.

    Returns
    -------
    Callable
        A new version of f_ss whose returns always includes the contents of known_variables.
    """

    @ft.wraps(f_ss)
    def inner(**parameters):
        if f_ss is None:
            return known_variables

        ss_dict = f_ss(**parameters)
        ss_dict.update(known_variables)
        return ss_dict

    return inner


def infer_variable_bounds(variable):
    assumptions = variable.assumptions0
    is_positive = assumptions.get("positive", False)
    is_negative = assumptions.get("negative", False)
    lhs = 1e-8 if is_positive else None
    rhs = -1e-8 if is_negative else None

    return lhs, rhs


def _initialize_x0(optimizer_kwargs, variables, jitter_x0):
    n_variables = len(variables)

    use_default_x0 = "x0" not in optimizer_kwargs
    x0 = optimizer_kwargs.pop("x0", np.full(n_variables, 0.8))

    if use_default_x0:
        negative_idx = [x.assumptions0.get("negative", False) for x in variables]
        x0[negative_idx] = -x0[negative_idx]

    if jitter_x0:
        x0 += np.random.normal(scale=1e-4, size=n_variables)

    return x0


def validate_policy_function(
    A, B, C, D, T, R, tol: float = 1e-8, verbose: bool = True
) -> None:
    gEcon_matrices = statespace_to_gEcon_representation(A, T, R, tol)

    P, Q, _, _, A_prime, R_prime, S_prime = gEcon_matrices

    resid_norms = residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime)
    norm_deterministic, norm_stochastic = resid_norms

    if verbose:
        _log.info(f"Norm of deterministic part: {norm_deterministic:0.9f}")
        _log.info(f"Norm of stochastic part:    {norm_deterministic:0.9f}")


def get_known_equation_mask(
    steady_state_system: list[sp.Expr],
    ss_dict: SymbolDictionary[sp.Symbol, float],
    param_dict: SymbolDictionary[sp.Symbol, float],
    tol: float = 1e-8,
) -> np.ndarray:
    sub_dict = ss_dict.copy() | param_dict.copy()
    subbed_system = [eq.subs(sub_dict.to_sympy()) for eq in steady_state_system]

    eq_is_zero_mask = [
        (sp.Abs(subbed_eq) < tol) == True  # noqa: E712
        for eq, subbed_eq in zip(steady_state_system, subbed_system)
    ]

    return np.array(eq_is_zero_mask)


def validate_user_steady_state_simple(
    steady_state_system: list[sp.Expr],
    ss_dict: SymbolDictionary[sp.Symbol, float],
    param_dict: SymbolDictionary[sp.Symbol, float],
    tol: float = 1e-8,
) -> None:
    r"""
    Perform a "shallow" validation of user-provided steady-state values.

    Insert provided numeric values into the systesm of steady state equations and check for non-zero residuals. This
    is a "shallow" check in the sense that no effort is made to check dependencies between equations (that is,
    sp.solve is not called). Partial steady states are allowed -- the function simply looks for numeric, non-zero values
    after the provided values are substituted. Therefore, passing an incorrect value that would later cause a numeric
    solver to fail is also not detected.

    For example, the following system would be detected as having an incorrect steady-state: for :math:`x_1 = 0.5` :

    .. math::

        \begin{align}
            x_1 - 1 &= 0 \\
            x_2^ - 3 = 0
        \end{align}

    Because the first equation will reduce to :math:`-0.5` after simple substitution. On the other hand, this system
    would not be marked at :math:`x_1 = 0.5`:

    ..math::

        \begin{align}
            x_1 - x_2 &= 0 \\
            x_2 - x_3 &= 0 \\
            x_3 - 1 &= 0
        \end{align}

    Clearly this can be reduced to :math:`x_1 = 1$`, but no effort is made to perform these substitutions, so the error
    will not be flagged. In general, these substitutions are non-trivial, and attempting to solve results in significant
    time cost.

    Parameters
    ----------
    steady_state_system: list of sp.Expr
        System of model equations with all time indices set to the steady state
    ss_dict: SymbolDictionary
        Dictionary of user-provided steady state values. Expected to have TimeAwareSymbol variables as keys and numeric
        values as values.
    param_dict: SymbolDictionary
        Dictionary of parameter values at which to solve for the steady state. Expected to have Symbol variables as
         keys and numeric values as values.
    tol: float
        Radius around zero within which to consider values as zero. Default is 1e-8.
    """
    sub_dict = ss_dict.copy() | param_dict.copy()
    subbed_system = [eq.subs(sub_dict.to_sympy()) for eq in steady_state_system]

    # This has to use equality to check True -- sympy doesn't know the truth value of e.g. |x - 3| < 1e-8. But it does
    # know that this is NOT the same as True.
    invalid_equation_strings = [
        str(eq)
        for eq, subbed_eq in zip(steady_state_system, subbed_system)
        if (sp.Abs(subbed_eq) < tol) is False
    ]

    if len(invalid_equation_strings) > 0:
        msg = (
            "User-provide steady state is not valid. The following equations had non-zero residuals "
            "after subsitution:\n"
        )
        msg += "\n".join(invalid_equation_strings)
        raise ValueError(msg)


class Model:
    def __init__(
        self,
        variables: list[TimeAwareSymbol],
        shocks: list[TimeAwareSymbol],
        equations: list[sp.Expr],
        param_dict: SymbolDictionary,
        deterministic_dict: SymbolDictionary,
        calib_dict: SymbolDictionary,
        f_params: Callable[[np.ndarray, ...], SymbolDictionary],
        f_ss_resid: Callable[[np.ndarray, ...], float],
        f_ss: Callable[[np.ndarray, ...], SymbolDictionary],
        f_ss_error: Callable[[np.ndarray, ...], np.ndarray],
        f_ss_jac: Callable[[np.ndarray, ...], np.ndarray],
        f_ss_error_grad: Callable[[np.ndarray, ...], np.ndarray],
        f_ss_error_hess: Callable[[np.ndarray, ...], np.ndarray],
        f_linearize: Callable,
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
            Function that takes a dictionary of parameter values theta and steady-state variable values x_ss and
            evaluates the system of model equations f(x_ss, theta) = 0.
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
        self.f_params = f_params
        self.f_ss_resid = f_ss_resid

        self.f_ss_error = f_ss_error
        self.f_ss_error_grad = f_ss_error_grad
        self.f_ss_error_hess = f_ss_error_hess

        self.f_ss = f_ss
        self.f_ss_jac = f_ss_jac

        if backend == "numpy":
            f_linearize = override_dummy_wrapper(f_linearize, "not_loglin_variable")
        self.f_linearize = f_linearize

    def parameters(self, **updates: float):
        # Remove deterministic parameters for updates. These can appear **self.parameters() into a fitting function
        deterministic_names = [x.name for x in self.deterministic_params]
        updates = {k: v for k, v in updates.items() if k not in deterministic_names}

        # Check for unknown updates (typos, etc)
        param_dict = self._default_params.copy()
        unknown_updates = set(updates.keys()) - set(param_dict.keys())
        if unknown_updates:
            raise ModelUnknownParameterError(list(unknown_updates))
        param_dict.update(updates)

        return self.f_params(**param_dict).to_string()

    def steady_state(
        self,
        how: Literal["analytic", "root", "minimize"] = "analytic",
        use_jac=True,
        use_hess=True,
        progressbar=True,
        optimizer_kwargs: dict | None = None,
        verbose=True,
        bounds: dict[str, tuple[float, float]] | None = None,
        fixed_values: dict[str, float] | None = None,
        jitter_x0: bool = False,
        **updates: float,
    ) -> tuple[SymbolDictionary[str, float], bool]:
        """
        Solve for the deterministic steady state of the DSGE model


        Parameters
        ----------
        how: str, one of ['analytic', 'root', 'minimize'], default: 'analytic'
            Method to use to solve for the steady state. If ``'analytic'``, the model is solved analytically using
            user-provided steady-state equations. This is only possible if the steady-state equations are fully
            defined. If ``'root'``, the steady state is solved using a root-finding algorithm. If ``'minimize'``, the
            steady state is solved by minimizing a squared error loss function.

        use_jac: bool, default: True
            Flag indicating whether to use the Jacobian of the error function when solving for the steady state. Ignored
            if ``how`` is 'analytic'.

        use_hess: bool, default: True
            Flag indicating whether to use the Hessian of the error function when solving for the steady state. Ignored
            if ``how`` is 'analytic'.

        progressbar: bool, default: True
            Flag indicating whether to display a progress bar when solving for the steady state.

        optimizer_kwargs: dict, optional
            Keyword arguments passed to either scipy.optimize.root or scipy.optimize.minimize, depending on the value of
            ``how``. Common argments include:

            - 'method': str,
                The optimization method to use. Default is ``'hybr'`` for ``how = 'root'`` and ``trust-krylov`` for
                ``how = 'minimize'``
            - 'maxiter': int,
                The maximum number of iterations to use. Default is 5000. This argument will be automatically renamed
                to match the argument expected by different optimizers (for example, the ``'hybr'`` method uses
                ``maxfev``).

        verbose: bool, default True
            If true, print a message about convergence (or not) to the console .

        bounds: dict, optional
            Dictionary of bounds for the steady-state variables. The keys are the variable names and the values are
            tuples of the form (lower_bound, upper_bound). These are passed to the scipy.optimize.minimize function,
            see that docstring for more information.

        fixed_values: dict, optional
            Dictionary of fixed values for the steady-state variables. The keys are the variable names and the values
            are the fixed values. These are not check for validity, and passing an inaccurate value may result in the
            system becoming unsolvable.

        jitter_x0: bool
            Whether to apply some small N(0, 1e-4) jitter to the initial point

        **updates: float, optional
            Parameter values at which to solve the steady state. Passed to self.parameters. If not provided, the default
            parameter values (those originally defined during model construction) are used.

        Returns
        -------
        steady_state: SymbolDictionary
            Dictionary of steady-state values
        success: bool
            Flag indicating whether the steady state was successfully solved
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if fixed_values is None:
            f_ss = self.f_ss
        else:
            fixed_vars = [
                x for x in self.variables if x.to_ss().name in fixed_values.keys()
            ]
            fixed_dict = SymbolDictionary(
                {x.to_ss(): fixed_values[x.to_ss().name] for x in fixed_vars}
            ).to_string()
            f_ss = add_more_ss_values_wrapper(self.f_ss, fixed_dict)

        # This logic could be made a lot of complex by looking into solver-specific arguments passed via
        # "options"
        tol = optimizer_kwargs.get("tol", 1e-8)

        param_dict = self.parameters(**updates)
        ss_dict = SymbolDictionary()
        ss_system = system_to_steady_state(self.equations, self.shocks)
        unknown_eq_idx = np.full(len(ss_system), True)

        # The default value is analytic, because that's best if the user gave everything we need to proceed. If he gave
        # nothing though, use minimize as a fallback default.
        if how == "analytic" and f_ss is None:
            how = "minimize"
        else:
            # If we have at least some user information, check if its is complete. If it's not, we will minimize
            # with the user-provided values fixed.
            ss_dict = f_ss(**param_dict) if f_ss is not None else ss_dict
            if len(ss_dict) != 0 and len(ss_dict) != len(self.variables):
                if how == "root":
                    zero_eq_mask = get_known_equation_mask(
                        steady_state_system=ss_system,
                        ss_dict=ss_dict,
                        param_dict=param_dict,
                        tol=tol,
                    )
                    if sum(zero_eq_mask) != len(ss_dict):
                        n_eliminated = sum(zero_eq_mask)
                        raise ValueError(
                            'Solving a partially provided steady state with how = "root" is only allowed if applying '
                            f'the given values results in a new square system.\n'
                            f'Found: {len(ss_dict)} provided steady state value{"s" if len(ss_dict) != 1 else ""}\n'
                            f'Eliminated: {n_eliminated} equation{"s" if n_eliminated != 1 else ""}.'
                        )
                    unknown_eq_idx = ~zero_eq_mask
                else:
                    how = "minimize"

            # Or, if we have everything, we're done.
            elif len(ss_dict) == len(self.variables):
                resid = self.f_ss_resid(**param_dict, **ss_dict)
                success = np.allclose(resid, 0.0, atol=1e-8)
                return ss_dict, success

        # Quick and dirty check of user-provided steady-state validity. This is NOT robust at all.
        validate_user_steady_state_simple(
            steady_state_system=ss_system,
            ss_dict=ss_dict,
            param_dict=param_dict,
            tol=tol,
        )

        ss_variables = [x.to_ss() for x in self.variables] + list(
            self.calibrated_params
        )

        known_variables = (
            [] if f_ss is None else list(f_ss(**self.parameters()).to_sympy().keys())
        )

        vars_to_solve = [var for var in ss_variables if var not in known_variables]
        unknown_var_idx = np.array(
            [x in vars_to_solve for x in ss_variables], dtype="bool"
        )

        if how == "root":
            res = self._solve_steady_state_with_root(
                f_ss=f_ss,
                use_jac=use_jac,
                vars_to_solve=vars_to_solve,
                unknown_var_idx=unknown_var_idx,
                unknown_eq_idx=unknown_eq_idx,
                progressbar=progressbar,
                optimizer_kwargs=optimizer_kwargs,
                jitter_x0=jitter_x0,
                **updates,
            )

        elif how == "minimize":
            res = self._solve_steady_state_with_minimize(
                f_ss=f_ss,
                use_jac=use_jac,
                use_hess=use_hess,
                vars_to_solve=vars_to_solve,
                unknown_var_idx=unknown_var_idx,
                unknown_eq_idx=unknown_var_idx,
                progressbar=progressbar,
                bounds=bounds,
                optimizer_kwargs=optimizer_kwargs,
                jitter_x0=jitter_x0,
                **updates,
            )
        else:
            raise NotImplementedError()

        provided_ss_values = f_ss(**param_dict).to_sympy() if f_ss is not None else {}
        optimizer_results = SymbolDictionary(
            {var: res.x[i] for i, var in enumerate(vars_to_solve)}
        )
        res_dict = optimizer_results | provided_ss_values
        res_dict = SymbolDictionary({x: res_dict[x] for x in ss_variables}).to_string()

        return postprocess_optimizer_res(
            res=res,
            res_dict=res_dict,
            f_resid=ft.partial(self.f_ss_resid, **param_dict),
            f_jac=ft.partial(self.f_ss_error_grad, **param_dict),
            tol=tol,
            verbose=verbose,
        )

    def _evaluate_steady_state(self, **updates: float):
        param_dict = self.parameters(**updates)
        ss_dict = self.f_ss(**param_dict)

        return self.f_ss_resid(**param_dict, **ss_dict)

    def _solve_steady_state_with_root(
        self,
        f_ss,
        use_jac: bool = True,
        vars_to_solve: list[TimeAwareSymbol] | None = None,
        unknown_var_idx: np.ndarray | None = None,
        unknown_eq_idx: np.ndarray | None = None,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer_kwargs = deepcopy(optimizer_kwargs)

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        method = optimizer_kwargs.pop("method", "hybr")

        if "options" not in optimizer_kwargs:
            optimizer_kwargs["options"] = {}

        if method in ["hybr", "df-sane"]:
            optimizer_kwargs["options"].update({"maxfev": maxiter})
        else:
            optimizer_kwargs["options"].update({"maxiter": maxiter})

        x0 = _initialize_x0(optimizer_kwargs, vars_to_solve, jitter_x0)

        param_dict = self.parameters(**param_updates)
        wrapper = ft.partial(
            scipy_wrapper,
            variables=vars_to_solve,
            unknown_var_idxs=unknown_var_idx,
            unknown_eq_idxs=unknown_eq_idx,
            f_ss=f_ss,
        )

        f = wrapper(self.f_ss_resid)
        f_jac = wrapper(self.f_ss_jac) if use_jac else None

        objective = CostFuncWrapper(
            maxeval=maxiter,
            f=f,
            f_jac=f_jac,
            progressbar=progressbar,
        )

        f_optim = ft.partial(
            optimize.root,
            fun=objective,
            x0=x0,
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
        f_ss,
        use_jac: bool = True,
        use_hess: bool = True,
        vars_to_solve: list[str] | None = None,
        unknown_var_idx: np.ndarray | None = None,
        unknown_eq_idx: np.ndarray | None = None,
        progressbar: bool = True,
        optimizer_kwargs: dict | None = None,
        jitter_x0: bool = False,
        bounds: dict[str, tuple[float, float]] | None = None,
        **param_updates,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer_kwargs = deepcopy(optimizer_kwargs)

        x0 = _initialize_x0(optimizer_kwargs, vars_to_solve, jitter_x0)
        tol = optimizer_kwargs.pop("tol", 1e-30)

        user_bounds = {} if bounds is None else bounds
        bound_dict = {x.name: infer_variable_bounds(x) for x in vars_to_solve}
        bound_dict.update(user_bounds)

        bounds = [bound_dict[x.name] for x in vars_to_solve]
        has_bounds = any([x != (None, None) for x in bounds])

        method = optimizer_kwargs.pop(
            "method", "trust-ncg" if not has_bounds else "trust-constr"
        )
        if method not in ["trust-constr", "L-BFGS-B", "powell"]:
            has_bounds = False

        maxiter = optimizer_kwargs.pop("maxiter", 5000)
        if "options" not in optimizer_kwargs:
            optimizer_kwargs["options"] = {}
        optimizer_kwargs["options"].update({"maxiter": maxiter})
        if method == "L-BFGS-B":
            optimizer_kwargs["options"].update({"maxfun": maxiter})

        param_dict = self.parameters(**param_updates)

        wrapper = ft.partial(
            scipy_wrapper,
            variables=vars_to_solve,
            unknown_var_idxs=unknown_var_idx,
            unknown_eq_idxs=unknown_eq_idx,
            f_ss=f_ss,
        )

        f = wrapper(self.f_ss_error)
        f_jac = wrapper(self.f_ss_error_grad) if use_jac else None
        f_hess = wrapper(self.f_ss_error_hess) if use_hess else None

        objective = CostFuncWrapper(
            maxeval=maxiter,
            f=f,
            f_jac=f_jac,
            f_hess=f_hess,
            progressbar=progressbar,
        )

        f_optim = ft.partial(
            optimize.minimize,
            fun=objective,
            x0=x0,
            jac=use_jac,
            hess=f_hess,
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
        order: Literal[1] = 1,
        not_loglin_variables: list[str] | None = None,
        steady_state_dict: dict | None = None,
        loglin_negative_ss: bool = False,
        steady_state_kwargs: dict | None = None,
        verbose=True,
        **parameter_updates,
    ):
        if order != 1:
            raise NotImplementedError(
                "Only first order linearization is currently supported."
            )
        if steady_state_kwargs is None:
            steady_state_kwargs = {}

        param_dict = self.parameters(**parameter_updates)

        if steady_state_dict is None:
            steady_state_dict, success = self.steady_state(
                **self.parameters(**param_dict), **steady_state_kwargs
            )

        if not_loglin_variables is None:
            not_loglin_variables = []

        n_variables = len(self.variables)
        not_loglin_flags = np.zeros(n_variables)
        for i, var in enumerate(self.variables):
            not_loglin_flags[i] = var.base_name in not_loglin_variables

        ss_values = np.array(list(steady_state_dict.values()))
        ss_zeros = np.abs(ss_values) < 1e-8
        ss_negative = ss_values < 0.0

        if np.any(ss_zeros):
            zero_idxs = np.flatnonzero(ss_zeros)
            zero_vars = [self.variables[i] for i in zero_idxs]
            if verbose:
                _log.warning(
                    f"The following variables had steady-state values close to zero and will not be log-linearized:"
                    f"{[x.base_name for x in zero_vars]}"
                )

            not_loglin_flags[ss_zeros] = 1

        if np.any(ss_negative) and not loglin_negative_ss:
            neg_idxs = np.flatnonzero(ss_negative)
            neg_vars = [self.variables[i] for i in neg_idxs]
            if verbose:
                _log.warning(
                    f"The following variables had negative steady-state values and will not be log-linearized:"
                    f"{[x.base_name for x in neg_vars]}"
                )

            not_loglin_flags[neg_idxs] = 1

        A, B, C, D = self.f_linearize(
            **param_dict, **steady_state_dict, not_loglin_variable=not_loglin_flags
        )

        return A, B, C, D

    def solve_model(
        self,
        solver="cycle_reduction",
        not_loglin_variables: list[str] | None = None,
        order: Literal[1] = 1,
        loglin_negative_ss: bool = False,
        steady_state_kwargs: dict | None = None,
        tol: float = 1e-8,
        max_iter: int = 1000,
        verbose: bool = True,
        on_failure="error",
        **parameter_updates,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
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
        T: np.ndarray, optional
            Transition matrix, approximated to the requested order. Represents the policy function, governing agent's
            optimal state-conditional actions. If the solver fails, None is returned instead.

        R: np.ndarray, optional
            Selection matrix, approximated to the requested order. Represents the state- and agent-conditional
            transmission of stochastic shocks through the economy. If the solver fails, None is returned instead.
        """
        if on_failure not in ["error", "ignore"]:
            raise ValueError(
                f'Parameter on_failure must be one of "error" or "ignore", found {on_failure}'
            )
        if steady_state_kwargs is None:
            steady_state_kwargs = {}

        ss_dict, success = self.steady_state(
            **self.parameters(**parameter_updates), **steady_state_kwargs
        )
        n_variables = len(self.variables)

        A, B, C, D = self.linearize_model(
            order=order,
            not_loglin_variables=not_loglin_variables,
            steady_state_dict=ss_dict.to_string(),
            loglin_negative_ss=loglin_negative_ss,
            verbose=verbose,
            **parameter_updates,
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
                        _log.info(message)

                    return None, None

            if verbose:
                message = interpret_gensys_output(eu)
                _log.info(message)
                _log.info(
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
                elif on_failure == "ignore":
                    if verbose:
                        _log.info(result)
                    return None, None
        else:
            raise NotImplementedError(
                'Only "cycle_reduction" and "gensys" are valid values for solver'
            )

        if verbose:
            check_perturbation_solution(A, B, C, D, T, R, tol=tol)

        return np.ascontiguousarray(T), np.ascontiguousarray(R)


def _maybe_solve_model(
    model: Model, T: np.ndarray | None, R: np.ndarray | None, **solve_model_kwargs
):
    """
    Solve for the linearized policy matrix of a model if required, or return the provided T and R

    Parameters
    ----------
    model: Model
        DSGE Model assoicated with T and R
    T: np.ndarray, optional
        Transition matrix of the solved system. If None, this will be computed using the model's ``solve_model``
        method.
    R: np.ndarray
        Selection matrix of the solved system. If None, this will be computed using the model's ``solve_model`` method.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    T: np.ndarray, optional
        Transition matrix, approximated to the requested order. Represents the policy function, governing agent's
        optimal state-conditional actions. If the solver fails, None is returned instead.

    R: np.ndarray, optional
        Selection matrix, approximated to the requested order. Represents the state- and agent-conditional
        transmission of stochastic shocks through the economy. If the solver fails, None is returned instead.
    """
    n_matrices = sum(x is not None for x in [T, R])
    if n_matrices == 1:
        _log.warning(
            "Passing only one of T or R will still trigger ``model.solve_model`` (which might be expensive). "
            "Pass both to avoid this, or None to silence this warning."
        )
        T = None
        R = None

    if T is None and R is None:
        T, R = model.solve_model(**solve_model_kwargs)

    return T, R


def _validate_shock_options(
    shock_std_dict: dict[str, float] | None,
    shock_cov_matrix: np.ndarray | None,
    shock_std: float | np.ndarray | list | None,
    shocks: list[TimeAwareSymbol],
):
    n_shocks = len(shocks)
    n_provided = sum(
        x is not None for x in [shock_std_dict, shock_cov_matrix, shock_std]
    )
    if n_provided > 1 or n_provided == 0:
        raise ValueError(
            "Exactly one of shock_std_dict, shock_cov_matrix, or shock_std should be provided. You passed "
            f"{n_provided}."
        )

    if shock_cov_matrix is not None:
        if any(s != n_shocks for s in shock_cov_matrix.shape):
            raise ValueError(
                f"Incorrect covariance matrix shape. Expected ({n_shocks}, {n_shocks}), "
                f"found {shock_cov_matrix.shape}"
            )

    if shock_std_dict is not None:
        shock_names = [x.base_name for x in shocks]
        missing = [x for x in shock_std_dict.keys() if x not in shock_names]
        extra = [x for x in shock_names if x not in shock_std_dict.keys()]
        if len(missing) > 0:
            raise ValueError(
                f"If shock_std_dict is specified, it must give values for all shocks. The follow shocks were not found"
                f" among the provided keys: {', '.join(missing)}"
            )
        if len(extra) > 0:
            raise ValueError(
                f"Unexpected shocks in shock_std_dict. The following names were not found among the model shocks: "
                f"{', '.join(extra)}"
            )

    if shock_std is not None:
        if isinstance(shock_std, np.ndarray | list):
            shock_std = cast(np.ndarray | list, shock_std)
            if len(shock_std) != n_shocks:
                raise ValueError(
                    f"Length of shock_std ({len(shock_std)}) does not match the number of shocks ({n_shocks})"
                )
            if not np.all(shock_std > 0):
                raise ValueError("Shock standard deviations must be positive")
        elif isinstance(shock_std, (int, float)):
            if shock_std < 0:
                raise ValueError("Shock standard deviation must be positive")


def _validate_simulation_options(shock_size, shock_cov, shock_trajectory):
    options = [shock_size, shock_cov, shock_trajectory]
    n_options = sum(x is not None for x in options)

    if n_options > 1:
        raise ValueError(
            "Specify exactly 0 or 1 of shock_size, shock_cov, or shock_trajectory"
        )
    elif n_options == 0:
        # Default is a unit shock on everything
        shock_size = 1.0

    return shock_size, shock_cov, shock_trajectory


def build_Q_matrix(
    model_shocks: list[TimeAwareSymbol],
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
) -> np.array:
    """
    Take different options for user input and reconcile them into a covariance matrix. Exactly one or zero of shock_dict
    or shock_cov_matrix should be provided. Then, proceed according to the following logic:

    - If `shock_cov_matrix` is provided, it is Q. Return it.
    - If `shock_dict` is provided, insert these into a diagonal matrix at locations according to `model_shocks`.

    For values missing from `shock_dict`, or if neither `shock_dict` nor `shock_cov_matrix` are provided:

    - Fill missing values using the mean of the prior defined in `shock_priors`
    - If no prior is set, fill the value with `default_value`.

    Note that the only way to get off-diagonal elements is to explicitly pass the entire covariance matrix.

    Parameters
    ----------
    model_shocks: list of str
        List of model shock names, used to infer positions in the covariance matrix
    shock_std_dict: dict, optional
        Dictionary of shock names and standard deviations to be used to build Q
    shock_cov_matrix: array, optional
        An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
    shock_std: float or sequence of float, optional
        Standard deviation of all model shocks. If float, the same value will be used for all shocks. If sequence, the
        length must match the number of shocks.

    Raises
    ------
    LinalgError
        If the provided Q is not positive semi-definite
    ValueError
        If both model_shocks and shock_dict are provided

    Returns
    -------
    Q: ndarray
        Shock variance-covariance matrix
    """

    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=model_shocks,
    )

    if shock_cov_matrix is not None:
        return shock_cov_matrix

    elif shock_std_dict is not None:
        shock_names = [x.base_name for x in model_shocks]
        indices = [shock_names.index(x) for x in shock_std_dict.keys()]
        Q = np.zeros((len(model_shocks), len(model_shocks)))
        for i, (key, value) in enumerate(shock_std_dict.items()):
            Q[indices[i], indices[i]] = value**2
        return Q

    else:
        return np.eye(len(model_shocks)) * shock_std**2


def stationary_covariance_matrix(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    **solve_model_kwargs,
):
    """
    Compute the stationary covariance matrix of the solved system by solving the associated discrete lyapunov
    equation.

    In order to construct the shock covariance matrix, exactly one of shock_dict, shock_cov_matrix, or shock_std should
    be provided.

    Parameters
    ----------
    model: Model
        DSGE Model assoicated with T and R
    T: np.ndarray, optional
        Transition matrix of the solved system. If None, this will be computed using the model's ``solve_model``
        method.
    R: np.ndarray
        Selection matrix of the solved system. If None, this will be computed using the model's ``solve_model`` method.
    shock_std_dict: dict, optional
        A dictionary of shock sizes to be used to compute the stationary covariance matrix.
    shock_cov_matrix: array, optional
        An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
    shock_std: float, optional
        Standard deviation of all model shocks.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    Sigma: np.ndarray
        Stationary covariance matrix of the linearized model
    """
    shocks = model.shocks
    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=shocks,
    )

    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
    )

    RQRT = np.linalg.multi_dot([R, Q, R.T])
    Sigma = linalg.solve_discrete_lyapunov(T, RQRT)

    return Sigma


@nb.njit(cache=True)
def _compute_autocovariance_matrix(T, Sigma, n_lags=5, correlation=True):
    """Compute the autocorrelation matrix for the given state-space model.

    Parameters
    ----------
    T: np.ndarray, optional
        Transition matrix of the solved system.
    Sigma: np.ndarray
        Stationary covariance matrix of the linearized model
    n_lags : int, optional
        The number of lags for which to compute the autocorrelation matrices.
    correlation: bool
        If True, return the autocorrelation matrices instead of the autocovariance matrices.

    Returns
    -------
    acov : ndarray
        An array of shape (n_lags, n_variables, n_variables) whose (i, j, k)-th entry gives the autocorrelation
        (or autocovaraince) between variables j and k at lag i.
    """

    n_vars = T.shape[0]
    auto_coors = np.empty((n_lags, n_vars, n_vars))
    std_vec = np.sqrt(np.diag(Sigma))

    if correlation:
        normalization_factor = np.outer(std_vec, std_vec)
    else:
        normalization_factor = np.ones_like(Sigma)

    for i in range(n_lags):
        auto_coors[i] = np.linalg.matrix_power(T, i) @ Sigma / normalization_factor

    return auto_coors


def autocovariance_matrix(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    n_lags: int = 10,
    correlation=True,
    **solve_model_kwargs,
):
    """
    Computes the model's autocorrelation matricesusing the stationary covariance matrix.

    In order to construct the shock covariance matrix, exactly one of shock_dict, shock_cov_matrix, or shock_std should
    be provided.

    Parameters
    ----------
    model: Model
        DSGE Model assoicated with T and R
    T: np.ndarray, optional
        Transition matrix of the solved system. If None, this will be computed using the model's ``solve_model``
        method.
    R: np.ndarray
        Selection matrix of the solved system. If None, this will be computed using the model's ``solve_model`` method.
    shock_std_dict: dict, optional
        A dictionary of shock sizes to be used to compute the stationary covariance matrix.
    shock_cov_matrix: array, optional
        An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
    shock_std: float, optional
        Standard deviation of all model shocks.
    n_lags: int
        Number of lags of autocorrelation and cross-autocorrelation to compute. Default is 10.
    correlation: bool
        If True, return the autocorrelation matrices instead of the autocovariance matrices.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    acorr_mat: DataFrame
    """
    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    Sigma = stationary_covariance_matrix(
        model,
        T=T,
        R=R,
        shock_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
    )
    result = _compute_autocovariance_matrix(
        T, Sigma, n_lags=n_lags, correlation=correlation
    )

    return result


def summarize_perturbation_solution(
    linear_system: Sequence[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    perturbation_solution: Sequence[np.ndarray | None, np.ndarray | None],
    model: Model,
):
    A, B, C, D = linear_system
    T, R = perturbation_solution
    if T is None or R is None:
        raise PerturbationSolutionNotFoundException()

    coords = {
        "equation": np.arange(A.shape[0]).astype(int),
        "variable": [x.base_name for x in model.variables],
        "shock": [x.base_name for x in model.shocks],
    }

    return xr.Dataset(
        data_vars={
            "A": (("equation", "variable"), A),
            "B": (("equation", "variable"), B),
            "C": (("equation", "variable"), C),
            "D": (("equation", "shock"), D),
            "T": (("equation", "variable"), T),
            "R": (("equation", "shock"), R),
        },
        coords=coords,
    )


def impulse_response_function(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    simulation_length: int = 40,
    shock_size: float | np.ndarray | dict[str, float] | None = None,
    shock_cov: np.ndarray | None = None,
    shock_trajectory: np.ndarray | None = None,
    orthogonalize_shocks: bool = False,
    random_seed: int | np.random.RandomState | None = None,
    **solve_model_kwargs,
) -> xr.DataArray:
    """
    Generate impulse response functions (IRF) from state space model dynamics.

    An impulse response function represents the dynamic response of the state space model
    to an instantaneous shock applied to the system. This function calculates the IRF
    based on either provided shock specifications or the posterior state covariance matrix.

    Parameters
    ----------
    model: Model
        DSGE Model object
    T: np.ndarray, optional
        Transition matrix of the solved system. If None, this will be computed using the model's ``solve_model``
        method.
    R: np.ndarray, optional
        Selection matrix of the solved system. If None, this will be computed using the model's ``solve_model`` method.
    simulation_length : int, optional
        The number of periods to compute the IRFs over. The default is 40.
    shock_size : Optional[Union[float, np.ndarray]], default=None
        The size of the shock applied to the system. If specified, it will create a covariance
        matrix for the shock with diagonal elements equal to `shock_size`. If float, the diagonal will be filled
        with `shock_size`. If an array, `shock_size` must match the number of shocks in the state space model.

        Only one of `use_stationary_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.
    shock_cov : Optional[np.ndarray], default=None
        A user-specified covariance matrix for the shocks. It should be a 2D numpy array with
        dimensions (n_shocks, n_shocks), where n_shocks is the number of shocks in the state space model.

        Only one of `use_stationary_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.
    shock_trajectory : Optional[np.ndarray], default=None
        A pre-defined trajectory of shocks applied to the system. It should be a 2D numpy array
        with dimensions (n, n_shocks), where n is the number of time steps and k_posdef is the
        number of shocks in the state space model.

        Only one of `use_stationary_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.
    orthogonalize_shocks : bool, default=False
        If True, orthogonalize the shocks using Cholesky decomposition when generating the impulse
        response. This option is ignored if `shock_trajectory` or `shock_size` are used.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    xr.DataArray
        The IRFs for each variable in the model.
    """
    n_variables = len(model.variables)
    n_shocks = len(model.shocks)
    rng = np.random.default_rng(random_seed)
    Q = None  # No covariance matrix needed if a trajectory is provided. Will be overwritten later if needed.

    shock_size, shock_cov, shock_trajectory = _validate_simulation_options(
        shock_size, shock_cov, shock_trajectory
    )

    if shock_trajectory is not None:
        # Validate the shock trajectory
        n, k = shock_trajectory.shape

        if k != n_shocks:
            raise ValueError(
                "If shock_trajectory is provided, there must be a trajectory provided for each shock. "
                f"Model has {n_shocks} shocks, but shock_trajectory has only {k} columns"
            )
        if simulation_length is not None and simulation_length != n:
            _log.warning(
                "Both steps and shock_trajectory were provided but do not agree. Length of "
                "shock_trajectory will take priority, and steps will be ignored."
            )
        simulation_length = n  # Overwrite steps with the length of the shock trajectory
        shock_trajectory = np.array(shock_trajectory)

    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    data = np.zeros((n_variables, simulation_length))

    if shock_cov is not None:
        Q = np.array(shock_cov)
        if orthogonalize_shocks:
            Q = linalg.cholesky(Q) / np.diag(Q)[:, None]

    if shock_trajectory is None:
        shock_trajectory = np.zeros((simulation_length, n_shocks))
        if Q is not None:
            init_shock = rng.multivariate_normal(mean=np.zeros(n_shocks), cov=Q)
        else:
            # Last remaining possibility is that shock_size was provided
            if isinstance(shock_size, dict):
                shock_size = [shock_size.get(x.base_name, 0.0) for x in model.shocks]
            init_shock = np.array(shock_size)
        shock_trajectory[0] = init_shock

    else:
        shock_trajectory = np.array(shock_trajectory)

    for t in range(1, simulation_length):
        stochastic = R @ shock_trajectory[t - 1]
        deterministic = T @ data[:, t - 1]
        data[:, t] = deterministic + stochastic

    variable_names = [x.base_name for x in model.variables]

    irf = xr.DataArray(
        data.T,
        dims=["time", "variable"],
        coords={"time": np.arange(simulation_length), "variable": variable_names},
    )

    return irf


def simulate(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    n_simulations: int = 1,
    simulation_length: int = 40,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | np.ndarray = None,
    random_seed: int | np.random.RandomState | None = None,
    **solve_model_kwargs,
) -> xr.DataArray:
    """
    Simulate the model over a certain number of time periods.

    Parameters
    ----------
    model: Model
        DSGE Model object
    T: np.ndarray, optional
        Transition matrix of the solved system. If None, this will be computed using the model's ``solve_model``
        method.
    R: np.ndarray, optional
        Selection matrix of the solved system. If None, this will be computed using the model's ``solve_model`` method.
    n_simulations : int, optional
        Number of trajectories to simulate. Default is 1.
    simulation_length : int, optional
        Length of each simulated trajectory. Default is 40.
    shock_std_dict: dict, optional
        Dictionary of shock names and standard deviations to be used to build Q
    shock_cov_matrix: array, optional
        An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
    shock_std: float or sequence, optional
        Standard deviation of all model shocks.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    xr.DataArray
        Simulated trajectories.
    """
    rng = np.random.default_rng(random_seed)
    shocks = model.shocks
    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    n_variables, n_shocks = R.shape

    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=shocks,
    )

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
    )

    epsilons = rng.multivariate_normal(
        mean=np.zeros(n_shocks),
        cov=Q,
        size=(n_simulations, simulation_length),
        method="svd",
    )

    data = np.zeros((n_simulations, simulation_length, n_variables))

    for t in range(1, simulation_length):
        stochastic = np.einsum("nk,sk->sn", R, epsilons[:, t - 1, :])
        deterministic = np.einsum("nm,sm->sn", T, data[:, t - 1, :])
        data[:, t, :] = deterministic + stochastic

    data = xr.DataArray(
        data,
        dims=["simulation", "time", "variable"],
        coords={
            "variable": [x.base_name for x in model.variables],
            "simulation": np.arange(n_simulations),
            "time": np.arange(simulation_length),
        },
    )

    return data
