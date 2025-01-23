import difflib
import functools as ft
import logging

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Literal, cast

import numba as nb
import numpy as np
import pandas as pd
import sympy as sp
import xarray as xr

from better_optimize import minimize, root
from scipy import linalg

from gEconpy.classes.containers import SteadyStateResults, SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import (
    GensysFailedException,
    ModelUnknownParameterError,
    PerturbationSolutionNotFoundException,
    SteadyStateNotFoundError,
)
from gEconpy.model.compile import BACKENDS
from gEconpy.model.perturbation import check_bk_condition as _check_bk_condition
from gEconpy.model.perturbation import (
    check_perturbation_solution,
    make_not_loglin_flags,
    override_dummy_wrapper,
    residual_norms,
    statespace_to_gEcon_representation,
)
from gEconpy.model.steady_state import system_to_steady_state
from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction
from gEconpy.solvers.gensys import (
    interpret_gensys_output,
    solve_policy_function_with_gensys,
)
from gEconpy.utilities import get_name, postprocess_optimizer_res, safe_to_ss

VariableType = sp.Symbol | TimeAwareSymbol
_log = logging.getLogger(__name__)


def scipy_wrapper(
    f: Callable,
    variables: list[str],
    unknown_var_idxs: np.ndarray[int | bool],
    unknown_eq_idxs: np.ndarray[int | bool],
    f_ss: Callable | None = None,
    include_p=False,
) -> Callable:
    if f_ss is not None:
        if not include_p:

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
            def inner(ss_values, p, param_dict):
                given_ss = f_ss(**param_dict)
                ss_dict = SymbolDictionary(zip(variables, ss_values)).to_string()
                ss_dict.update(given_ss)

                p_full = np.zeros(unknown_eq_idxs.shape[0])
                p_full[unknown_var_idxs] = p

                res = f(p_full, **ss_dict, **param_dict)

                if isinstance(res, float | int):
                    return res
                elif res.ndim == 1:
                    res = res[unknown_eq_idxs]
                elif res.ndim == 2:
                    res = res[unknown_eq_idxs, :][:, unknown_var_idxs]
                return res

    else:
        if not include_p:

            @ft.wraps(f)
            def inner(ss_values, param_dict):
                ss_dict = SymbolDictionary(zip(variables, ss_values)).to_string()
                return f(**ss_dict, **param_dict)
        else:

            @ft.wraps(f)
            def inner(ss_values, p, param_dict):
                ss_dict = SymbolDictionary(zip(variables, ss_values)).to_string()
                return f(p, **ss_dict, **param_dict)

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
        if (sp.Abs(subbed_eq) < tol) == False  #  noqa
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
        steady_state_relationships: list[sp.Eq],
        param_dict: SymbolDictionary,
        deterministic_dict: SymbolDictionary,
        calib_dict: SymbolDictionary,
        priors: tuple | None,
        f_params: Callable[[np.ndarray, ...], SymbolDictionary],
        f_ss_resid: Callable[[np.ndarray, ...], float],
        f_ss: Callable[[np.ndarray, ...], SymbolDictionary],
        f_ss_error: Callable[[np.ndarray, ...], np.ndarray],
        f_ss_jac: Callable[[np.ndarray, ...], np.ndarray],
        f_ss_error_grad: Callable[[np.ndarray, ...], np.ndarray],
        f_ss_error_hess: Callable[[np.ndarray, ...], np.ndarray],
        f_ss_error_hessp: Callable[[np.ndarray, ...], np.ndarray],
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

        f_ss_error_hessp: Callable, optional
            Function that takes a dictionary of parameter values theta and steady-state variable values x_ss and returns
            the Hessian-vector product of the error function f_ss_error with respect to the steady-state variable values x_ss


        f_ss_jac: Callable, optional

        f_linearize: Callable, optional

        """

        self.variables = variables
        self.shocks = shocks
        self.equations = equations
        self.params = list(param_dict.to_sympy().keys())

        self.deterministic_params = list(deterministic_dict.to_sympy().keys())
        self.calibrated_params = list(calib_dict.to_sympy().keys())

        self.steady_state_relationships = steady_state_relationships

        self._all_names_to_symbols = {
            get_name(x, base_name=True): x
            for x in (
                self.variables
                + self.params
                + self.calibrated_params
                + self.deterministic_params
                + self.shocks
            )
        }

        self.priors = priors

        self._default_params = param_dict.copy()
        self.f_params = f_params
        self.f_ss_resid = f_ss_resid

        self.f_ss_error = f_ss_error
        self.f_ss_error_grad = f_ss_error_grad
        self.f_ss_error_hess = f_ss_error_hess
        self.f_ss_error_hessp = f_ss_error_hessp

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

    def get(self, name: str) -> sp.Symbol:
        """
        Get a variable or parameter by name
        """
        ss_requested = name.endswith("_ss")
        name = name.removesuffix("_ss")

        result = self._all_names_to_symbols.get(name)
        if result is None:
            close_match = difflib.get_close_matches(
                name, [get_name(x) for x in self._all_names_to_symbols.keys()], n=1
            )[0]
            raise IndexError(
                f"Did not find {name} among model objects. Did you mean {close_match}?"
            )
        if ss_requested:
            return result.to_ss()
        return result

    def _validate_provided_steady_state_variables(
        self, user_fixed_variables: Sequence[str]
    ):
        # User is allowed to pass the variable name either with or without the _ss suffix. Begin by normalizing the
        # inputs
        fixed_variables_normed = [x.removesuffix("_ss") for x in user_fixed_variables]

        # Check for duplicated values. This should only be possible if the user passed both `x` and `x_ss`.
        counts = [fixed_variables_normed.count(x) for x in fixed_variables_normed]
        duplicates = [x for x, c in zip(fixed_variables_normed, counts) if c > 1]
        if len(duplicates) > 0:
            raise ValueError(
                'The following variables were provided twice (once with a _ss prefix and once without):\n'
                f'{", ".join(duplicates)}'
            )

        # Check that all variables are in the model
        model_variable_names = [x.base_name for x in self.variables]
        unknown_fixed = set(fixed_variables_normed) - set(model_variable_names)

        if len(unknown_fixed) > 0:
            raise ValueError(
                f"The following variables or calibrated parameters were given fixed steady state values but are "
                f"unknown to the model: {', '.join(unknown_fixed)}"
            )

    def steady_state(
        self,
        how: Literal["analytic", "root", "minimize"] = "analytic",
        use_jac=True,
        use_hess=True,
        use_hessp=False,
        progressbar=True,
        optimizer_kwargs: dict | None = None,
        verbose=True,
        bounds: dict[str, tuple[float, float]] | None = None,
        fixed_values: dict[str, float] | None = None,
        jitter_x0: bool = False,
        **updates: float,
    ) -> SteadyStateResults:
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

        use_hess: bool, default: False
            Flag indicating whether to use the Hessian of the error function when solving for the steady state. Ignored
            if ``how`` is not 'minimize'

        use_hessp: bool, default: True
            Flag indicating whether to use the Hessian-vector product of the error function when solving for the
            steady state. This should be preferred over ``use_hess`` if your chosen method supports it. For larger
            problems it is substantially more performant.
            Ignored if ``how`` not "minimize".

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
        steady_state: SteadyStateResults
            Dictionary of steady-state values

        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if fixed_values is None:
            f_ss = self.f_ss

        else:
            self._validate_provided_steady_state_variables(list(fixed_values.keys()))
            fixed_symbols = [safe_to_ss(self.get(x)) for x in fixed_values.keys()]

            fixed_dict = SymbolDictionary(
                {
                    symbol: value
                    for symbol, value in zip(fixed_symbols, fixed_values.values())
                },
            ).to_string()

            f_ss = add_more_ss_values_wrapper(self.f_ss, fixed_dict)

        # This logic could be made a lot of complex by looking into solver-specific arguments passed via
        # "options"
        tol = optimizer_kwargs.get("tol", 1e-8)

        param_dict = self.parameters(**updates)
        ss_dict = SteadyStateResults()
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
                ss_dict.success = success
                if not success:
                    _log.warning(
                        f"Steady State was not found. Sum of square residuals: {np.square(resid).sum()}"
                    )
                return ss_dict

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
                use_hessp=use_hessp,
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
        res_dict = SteadyStateResults(
            {x: res_dict[x] for x in ss_variables}
        ).to_string()

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

        with np.errstate(all="ignore"):
            res = root(
                f=f,
                x0=x0,
                args=(param_dict,),
                jac=f_jac,
                method=method,
                progressbar=progressbar,
                **optimizer_kwargs,
            )

        return res

    def _solve_steady_state_with_minimize(
        self,
        f_ss,
        use_jac: bool = True,
        use_hess: bool = False,
        use_hessp: bool = True,
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

        if use_hess and use_hessp:
            _log.warning(
                "Both use_hess and use_hessp are set to True. use_hessp will be used."
            )
            use_hess = False

        f = wrapper(self.f_ss_error)
        f_jac = wrapper(self.f_ss_error_grad) if use_jac else None
        f_hess = wrapper(self.f_ss_error_hess) if use_hess else None
        f_hessp = wrapper(self.f_ss_error_hessp, include_p=True) if use_hessp else None

        res = minimize(
            f=f,
            x0=x0,
            args=(param_dict,),
            jac=f_jac,
            hess=f_hess,
            hessp=f_hessp,
            method=method,
            bounds=bounds if has_bounds else None,
            tol=tol,
            progressbar=progressbar,
            **optimizer_kwargs,
        )

        return res

    def linearize_model(
        self,
        order: Literal[1] = 1,
        log_linearize: bool = True,
        not_loglin_variables: list[str] | None = None,
        steady_state: dict | None = None,
        loglin_negative_ss: bool = False,
        steady_state_kwargs: dict | None = None,
        verbose: bool = True,
        **parameter_updates,
    ):
        if order != 1:
            raise NotImplementedError(
                "Only first order linearization is currently supported."
            )
        if steady_state_kwargs is None:
            steady_state_kwargs = {}

        param_dict = self.parameters(**parameter_updates)

        if steady_state is None:
            steady_state = self.steady_state(
                **self.parameters(**param_dict), **steady_state_kwargs
            )

        not_loglin_flags = make_not_loglin_flags(
            variables=self.variables,
            calibrated_params=self.calibrated_params,
            steady_state=steady_state,
            log_linearize=log_linearize,
            not_loglin_variables=not_loglin_variables,
            loglin_negative_ss=loglin_negative_ss,
            verbose=verbose,
        )

        A, B, C, D = self.f_linearize(
            **param_dict, **steady_state, not_loglin_variable=not_loglin_flags
        )

        return A, B, C, D

    def solve_model(
        self,
        solver="cycle_reduction",
        log_linearize: bool = True,
        not_loglin_variables: list[str] | None = None,
        order: Literal[1] = 1,
        loglin_negative_ss: bool = False,
        steady_state: dict | None = None,
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
        log_linearize: bool, default: True
            Whether to log-linearize the model. If False, the model will be solved in levels.
        not_loglin_variables: list of strings, optional
            Variables to not log linearize when solving the model. Variables with steady state values close to zero
            (or negative) will be automatically selected to not log linearize. Ignored if log_linearize is False.
        order: int, default: 1
            Order of taylor expansion to use to solve the model. Currently only 1st order approximation is supported.
        steady_state: dict, optional
            Dictionary of steady-state solutions. If not provided, the steady state will be solved for using the
            ``steady_state`` method.
        steady_state_kwargs: dict, optional
            Keyword arguments passed to the `steady_state` method. Ignored if a steady-state solution is provided
            via the steady_state argument, Default is None.
        loglin_negative_ss: bool, default is False
            Whether to force log-linearization of variable with negative steady-state. This is impossible in principle
            (how can :math:`exp(x_ss)` be negative?), but can still be done; see the docstring for
            :fun:`perturbation.linearize_model` for details. Use with caution, as results will not correct. Ignored if
            log_linearize is False.
        tol: float, default 1e-8
            Desired level of floating point accuracy in the solution
        max_iter: int, default: 1000
            Maximum number of cycle_reduction iterations. Not used if solver is 'gensys'.
        verbose: bool, default: True
            Flag indicating whether to print solver results to the terminal
        on_failure: str, one of ['error', 'ignore'], default: 'error'
            Instructions on what to do if the algorithm to find a linearized policy matrix. "Error" will raise an error,
            while "ignore" will return None. "ignore" is useful when repeatedly solving the model, e.g. when sampling.
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

        ss_dict = _maybe_solve_steady_state(
            self, steady_state, steady_state_kwargs, parameter_updates
        )
        n_variables = len(self.variables)

        A, B, C, D = self.linearize_model(
            order=order,
            log_linearize=log_linearize,
            not_loglin_variables=not_loglin_variables,
            steady_state=ss_dict.to_string(),
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


def _maybe_solve_steady_state(
    model: Model,
    steady_state: dict | None,
    steady_state_kwargs: dict | None,
    parameter_updates: dict | None,
):
    if steady_state is None:
        return model.steady_state(
            **model.parameters(**parameter_updates), **steady_state_kwargs
        )

    ss_resid = model.f_ss_resid(**steady_state, **model.parameters(**parameter_updates))
    unsatisfied_flags = np.abs(ss_resid) > 1e-8
    unsatisfied_eqs = [
        f"Equation {i}" for i, flag in enumerate(unsatisfied_flags) if flag
    ]

    if np.any(unsatisfied_flags):
        raise SteadyStateNotFoundError(unsatisfied_eqs)
    steady_state.success = True

    return steady_state


def _maybe_linearize_model(
    model: Model,
    A: np.ndarray | None,
    B: np.ndarray | None,
    C: np.ndarray | None,
    D: np.ndarray | None,
    verbose: bool = True,
    **linearize_model_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearize a model if required, or return the provided matrices

    Parameters
    ----------
    model: Model
        DSGE model
    A: np.ndarray, optional
        Matrix of partial derivatives of model equations with respect to variables at time t-1, evaluated at the
        steady-state
    B: np.ndarray, optional
        Matrix of partial derivatives of model equations with respect to variables at time t, evaluated at the
        steady-state
    C: np.ndarray, optional
        Matrix of partial derivatives of model equations with respect to variables at time t+1, evaluated at the
        steady-state
    D: np.ndarray, optional
        Matrix of partial derivatives of model equations with respect to stochastic innovations, evaluated at the
        steady-state
    verbose: bool, default: True
        Flag indicating whether to print details about the linearization process to the console
    linearize_model_kwargs
        Arguments forwarded to the ``model.linearize_model`` method. Ignored if all of A, B, C, and D are provided.

    Returns
    -------
    linear_system: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """

    n_matrices = sum(x is not None for x in [A, B, C, D])
    if n_matrices < 4 and n_matrices > 0 and verbose:
        _log.warning(
            f"Passing an incomplete subset of A, B, C, and D (you passed {n_matrices}) will still trigger "
            f"``model.linearize_model`` (which might be expensive). Pass all to avoid this, or None to silence "
            f"this warning."
        )
        A = None
        B = None
        C = None
        D = None

    if all(x is None for x in [A, B, C, D]):
        A, B, C, D = model.linearize_model(verbose=verbose, **linearize_model_kwargs)

    return A, B, C, D


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
                f"If shock_std_dict is specified, it must give values for all shocks. The following shocks were not "
                f"found among the provided keys: {', '.join(missing)}"
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
        elif isinstance(shock_std, int | float):
            if shock_std < 0:
                raise ValueError("Shock standard deviation must be positive")


def _validate_simulation_options(shock_size, shock_cov, shock_trajectory) -> None:
    options = [shock_size, shock_cov, shock_trajectory]
    n_options = sum(x is not None for x in options)

    if n_options != 1:
        raise ValueError(
            "Specify exactly 1 of shock_size, shock_cov, or shock_trajectory"
        )


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
    return_df: bool = True,
    **solve_model_kwargs,
) -> np.ndarray | pd.DataFrame:
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
    return_df: bool
        If True, return the covariance matrix as a DataFrame
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    Sigma: np.ndarray | pd.DataFrame
        Stationary covariance matrix of the linearized model. Datatype depends on the variable of the ``return_df``
        argument.
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

    if return_df:
        variables = [x.base_name for x in model.variables]
        Sigma = pd.DataFrame(Sigma, index=variables, columns=variables)

    return Sigma


def check_bk_condition(
    model: Model,
    *,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    C: np.ndarray | None = None,
    D: np.ndarray | None = None,
    tol=1e-8,
    verbose=True,
    on_failure: Literal["raise", "ignore"] = "ignore",
    return_value: Literal["dataframe", "bool", None] = "dataframe",
    **linearize_model_kwargs,
) -> bool | pd.DataFrame | None:
    """
    Compute the generalized eigenvalues of system in the form presented in [1]. Per [2], the number of
    unstable eigenvalues (|v| > 1) should not be greater than the number of forward-looking variables. Failing
    this test suggests timing problems in the definition of the model.

    Parameters
    ----------
    model: Model
        DSGE model
    A: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to past variables
        values that are known when decision-making: those with t-1 subscripts.
    B: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        are observed when decision-making: those with t subscripts.
    C: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        enter in expectation when decision-making: those with t+1 subscripts.
    D: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to exogenous shocks.
    verbose: bool, default: True
        Flag to print the results of the test, otherwise the eigenvalues are returned without comment.
    on_failure: str, default: 'ignore'
        Action to take if the Blanchard-Kahn condition is not satisfied. Valid values are 'ignore' and 'raise'.
    return_value: string, default: 'dataframe'
        Controls what is returned by the function. Valid values are 'dataframe', 'bool', and 'none'.
        If df, a dataframe containing eigenvalues is returned. If 'bool', a boolean indicating whether the BK
        condition is satisfied. If None, nothing is returned.
    tol: float, 1e-8
        Tolerance below which numerical values are considered zero

    Returns
    -------
    bk_result, bool or pd.DataFrame, optional.
        Return value requested. Datatype corresponds to what was requested in the ``return_value`` argument:
        - None, If return_value is 'none'
        - condition_satisfied, bool, if return_value is 'bool', returns True if the Blanchard-Kahn condition is
          satisfied, False otherwise.
        - Eigenvalues, pd.DataFrame, if return_value is 'df', returns a dataframe containing the real and imaginary
          components of the system's, eigenvalues, along with their modulus.
    """

    A, B, C, D = _maybe_linearize_model(
        model, A, B, C, D, verbose=verbose, **linearize_model_kwargs
    )
    bk_result = _check_bk_condition(
        A,
        B,
        C,
        D,
        tol=tol,
        verbose=verbose,
        on_failure=on_failure,
        return_value=return_value,
    )
    return bk_result


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
    correlation=False,
    return_xr=True,
    **solve_model_kwargs,
):
    """
    Computes the model's autocovariance matrix using the stationary covariance matrix. Alteratively, the autocorrelation
    matrix can be returned by specifying ``correlation = True``.

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
        Number of lags of auto-covariance and cross-covariance to compute. Default is 10.
    correlation: bool
        If True, return the autocorrelation matrices instead of the autocovariance matrices.
    return_xr: bool
        If True, return the covariance matrices as a DataArray with dimensions ["variable", "variable_aux", and "lag"].
        Otherwise returns a 3d numpy array with shape (lag, variable, variable).
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
        return_df=False,
    )
    result = _compute_autocovariance_matrix(
        T, Sigma, n_lags=n_lags, correlation=correlation
    )

    if return_xr:
        variables = [x.base_name for x in model.variables]
        result = xr.DataArray(
            result,
            dims=["lag", "variable", "variable_aux"],
            coords={
                "lag": range(n_lags),
                "variable": variables,
                "variable_aux": variables,
            },
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


autocorrelation_matrix = ft.partial(autocovariance_matrix, correlation=True)
autocorrelation_matrix.__doc__ = autocovariance_matrix.__doc__


def impulse_response_function(
    model: Model,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    simulation_length: int = 40,
    shock_size: float | np.ndarray | dict[str, float] | None = None,
    shock_cov: np.ndarray | None = None,
    shock_trajectory: np.ndarray | None = None,
    return_individual_shocks: bool | None = None,
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
    shock_size : float, array, or dict; default=None
        The size of the shock applied to the system. If specified, it will create a covariance
        matrix for the shock with diagonal elements equal to `shock_size`:
            -  If float, the covariance matrix will be the identity matrix, scaled by `shock_size`.
            -  If array, the covariance matrix will be ``diag(shock_size)``. In this case, the length of the provided array
            must match the number of shocks in the state space model.
            -  If dictionary, a diagonal matrix will be created with entries corresponding to the keys in the dictionary.
               Shocks which are not specified will be set to zero.

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
    return_individual_shocks: bool, optional
        If True, an IRF will be computed separately for each shock in the model. An additional dimension will be added
        to the output DataArray to show each shock. This is only valid if `shock_size` is a scalar, dictionary, or if
        the covariance matrix is diagonal.

        If not specified, this will be set to True if ``shock_size`` if the above conditions are met.
    orthogonalize_shocks : bool, default=False
        If True, orthogonalize the shocks using Cholesky decomposition when generating the impulse
        response. This option is ignored if `shock_trajectory` or `shock_size` are used, or if the covariance matrix is
        diagonal.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    **solve_model_kwargs
        Arguments forwarded to the ``solve_model`` method. Ignored if T and R are provided.

    Returns
    -------
    xr.DataArray
        The IRFs for each variable in the model.
    """
    variable_names = [x.base_name for x in model.variables]
    model_shock_names = [x.base_name for x in model.shocks]

    n_variables = len(model.variables)
    n_model_shocks = len(model.shocks)

    rng = np.random.default_rng(random_seed)
    Q = None  # No covariance matrix needed if a trajectory is provided. Will be overwritten later if needed.

    _validate_simulation_options(shock_size, shock_cov, shock_trajectory)

    return_individual_shocks = (
        True if return_individual_shocks is None else return_individual_shocks
    )

    if shock_trajectory is not None:
        n, k = shock_trajectory.shape

        # Validate the shock trajectory
        if k != n_model_shocks:
            raise ValueError(
                "If shock_trajectory is provided, there must be a trajectory provided for each shock. "
                f"Model has {n_model_shocks} shocks, but shock_trajectory has only {k} columns"
            )
        if simulation_length is not None and simulation_length != n:
            _log.warning(
                "Both steps and shock_trajectory were provided but do not agree. Length of "
                "shock_trajectory will take priority, and steps will be ignored."
            )
        simulation_length = n  # Overwrite steps with the length of the shock trajectory
        shock_trajectory = np.array(shock_trajectory)

    if shock_cov is not None:
        Q = np.array(shock_cov)
        is_diag = np.all(Q == np.diag(np.diagonal(Q)))
        return_individual_shocks = is_diag

        if orthogonalize_shocks:
            Q = linalg.cholesky(Q) / np.diag(Q)[:, None]

    T, R = _maybe_solve_model(model, T, R, **solve_model_kwargs)

    def _simulate(shock_trajectory):
        data = np.zeros((simulation_length, n_variables))

        for t in range(1, simulation_length):
            stochastic = R @ shock_trajectory[t - 1]
            deterministic = T @ data[t - 1]
            data[t] = deterministic + stochastic

        return data

    def _create_shock_trajectory(
        n_shocks, shock_names, Q=None, shock_size=None, shock_trajectory=None
    ):
        if shock_trajectory is not None:
            return np.array(shock_trajectory)

        shock_trajectory = np.zeros((simulation_length, n_shocks))

        if Q is not None:
            shock_size = rng.multivariate_normal(
                mean=np.zeros(n_shocks), cov=Q, size=simulation_length
            )

        else:
            if isinstance(shock_size, int | float):
                shock_size = np.ones(n_shocks) * shock_size
            if isinstance(shock_size, dict):
                shock_dict = shock_size.copy()
                shock_size = np.zeros(n_shocks)
                for i, name in enumerate(shock_names):
                    if name in shock_dict:
                        shock_size[i] = shock_dict[name]

        shock_trajectory[0] = shock_size

        return shock_trajectory

    def _make_shock_dict(shocks, shock_size=None, Q=None):
        if Q is not None:
            return {x.base_name: np.sqrt(Q[i, i]) for i, x in enumerate(shocks)}
        if isinstance(shock_size, dict):
            return shock_size
        if isinstance(shock_size, int | float):
            return {x.base_name: shock_size for x in shocks}
        if isinstance(shock_size, np.ndarray | list):
            return {x.base_name: shock_size[i] for i, x in enumerate(shocks)}

    shock_dict = _make_shock_dict(model.shocks, shock_size, Q)
    shock_names = (
        list(shock_dict.keys()) if shock_dict is not None else model_shock_names
    )

    # Sort the shock names to match the order of the model shocks
    shock_names = [x for x in model_shock_names if x in shock_names]
    n_shocks = len(shock_names)

    data_shape = (simulation_length, n_variables)

    coords = {"time": np.arange(simulation_length), "variable": variable_names}
    dims = ["time", "variable"]

    if return_individual_shocks:
        data_shape = (n_shocks, *data_shape)
        coords.update({"shock": shock_names})
        dims = ["shock", "time", "variable"]

    data = np.zeros(data_shape)

    if return_individual_shocks and shock_dict is not None:
        for i, (shock_name, init_shock) in enumerate(shock_dict.items()):
            step_dict = {
                k: shock_dict[k] if k == shock_name else 0.0 for k in shock_dict
            }
            traj = _create_shock_trajectory(
                shock_names=model_shock_names,
                n_shocks=n_model_shocks,
                shock_size=step_dict,
            )

            data[i] = _simulate(traj)

    elif return_individual_shocks and shock_trajectory is not None:
        for i, shock_name in enumerate(shock_names):
            traj = np.zeros_like(shock_trajectory)
            traj[i] = shock_trajectory[i]
            data[i] = _simulate(traj)

    else:
        traj = _create_shock_trajectory(
            shock_names=model_shock_names,
            n_shocks=n_model_shocks,
            Q=Q,
            shock_trajectory=shock_trajectory,
            shock_size=shock_size,
        )

        data = _simulate(traj)

    irf = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
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


def matrix_to_dataframe(
    matrix,
    model,
    dim1: str | None = None,
    dim2: str | None = None,
    round: None | int = None,
) -> pd.DataFrame:
    """
    Convert a matrix to a DataFrame with variable names as columns and rows.


    Parameters
    ----------
    matrix: np.ndarray
        DSGE matrix to convert to a DataFrame. Each dimension should have shape n_variables or n_shocks.
    model: Model
        DSGE model object
    dim1: str, Optional
        Name of the first dimension of the matrix. Must be one of "variable", "equation",  or "shock". If None, the
        function will guess based on the shape of the matrix. In the event that the model has exactly as many
        variables as shocks, it will guess "variable", so be careful!
    dim2: str, Optional
        Name of the second dimension of the matrix. Must be one of "variable", "equation", or "shock". If None, the
        function will guess based on the shape of the matrix.
    round: int, Optional
        Number of decimal places to round the values in the DataFrame. If None, values will not be rounded.

    Returns
    -------
    pd.DataFrame
        DataFrame with variable names as columns and rows.
    """
    var_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]
    equation_names = [f"Equation {i}" for i in range(len(model.equations))]

    coords = {"variable": var_names, "shock": shock_names, "equation": equation_names}

    n_variables = len(var_names)
    n_shocks = len(shock_names)

    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional")

    for i, ordinal in enumerate(["First", "Secoond"]):
        if matrix.shape[i] not in [n_variables, n_shocks]:
            raise ValueError(
                f"{ordinal} dimension of the matrix must match the number of variables or shocks "
                f"in the model"
            )

    if dim1 is None:
        dim1 = "variable" if matrix.shape[0] == n_variables else "shock"
    if dim2 is None:
        dim2 = "variable" if matrix.shape[1] == n_variables else "shock"

    df = pd.DataFrame(
        matrix,
        index=coords[dim1],
        columns=coords[dim2],
    )

    if round is not None:
        return df.round(round)

    return df


def check_steady_state(
    model: Model,
    stead_state: SteadyStateResults | None = None,
    steady_state_kwargs: dict | None = None,
    **parameter_updates,
) -> None:
    if steady_state_kwargs is None:
        steady_state_kwargs = {}

    ss_dict = _maybe_solve_steady_state(
        model, stead_state, steady_state_kwargs, parameter_updates
    )
    if ss_dict.success:
        _log.warning("Steady state successfully found!")
        return

    parameters = model.parameters(**parameter_updates)
    residuals = model.f_ss_resid(**ss_dict, **parameters)
    _log.warning(
        "Steady state NOT successful. The following equations have non-zero residuals:"
    )

    for resid, eq in zip(residuals, model.equations):
        if np.abs(resid) > 1e-6:
            _log.warning(eq)
            _log.warning(f"Residual: {resid:0.4f}")
