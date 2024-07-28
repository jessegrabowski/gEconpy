from typing import Any, Literal
from collections.abc import Callable
from warnings import catch_warnings, simplefilter

import numpy as np
import sympy as sp

from scipy import optimize

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import (
    BACKENDS,
    compile_function,
    dictionary_return_wrapper,
)
from gEconpy.numba_tools.utilities import numba_lambdify
from gEconpy.shared.utilities import eq_to_ss, substitute_all_equations

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


def system_to_steady_state(system, shock_dict):
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
        stack_return=backend != "pytensor",
        pop_return=False,
        **kwargs,
    )

    f_ss_jac, cache = compile_function(
        ss_variables + parameters,
        resid_jac,
        backend=backend,
        cache=cache,
        return_symbolic=return_symbolic,
        stack_return=backend != "pytensor",
        pop_return=backend == "pytensor",
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
        stack_return=backend
        != "pytensor",  # error_hess is a list of one element; don't stack into a (1,n,n) array
        pop_return=backend
        == "pytensor",  # error_hess is a list of one element; need to remove the list wrapper
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

    f_ss, cache = compile_function(
        parameters,
        output_exprs,
        backend=backend,
        cache=cache,
        stack_return=True if stack_return is None else stack_return,
        return_symbolic=return_symbolic,
        **kwargs,
    )
    if return_symbolic and backend == "pytensor":
        return f_ss, cache

    return dictionary_return_wrapper(f_ss, output_vars), cache


class SteadyStateSolver:
    def __init__(self, model):
        self.variables: list[TimeAwareSymbol] = model.variables
        self.shocks: list[sp.Add] = model.shocks

        self.n_variables: int = model.n_variables

        self.free_param_dict: SymbolDictionary[str, float] = model.free_param_dict
        self.params_to_calibrate: list[sp.Symbol] = model.params_to_calibrate
        self.calibrating_equations: list[sp.Add] = model.calibrating_equations
        self.shock_dict: SymbolDictionary[str, float] | None = None

        self.system_equations: list[sp.Add] = model.system_equations
        self.steady_state_relationships: SymbolDictionary[str, float | sp.Add] = (
            model.steady_state_relationships
        )

        ss_vars = make_steady_state_variables(self.variables)
        self.steady_state_dict: SymbolDictionary[str, float] = (
            SymbolDictionary.fromkeys(ss_vars, None).to_string().sort_keys()
        )

        self.shock_dict: SymbolDictionary[str, float] = make_steady_state_shock_dict(
            self.shocks
        )

        steady_state_system = system_to_steady_state(
            self.system_equations, self.shock_dict
        )
        self.steady_state_system: list[sp.Expr] = steady_state_system

        self.steady_state_solved: bool = False

    def apply_user_simplifications(self) -> list[sp.Add]:
        """
        Check if the system is analytically solvable without resorting to an optimizer. Currently, this is true only
        if it is a linear model, or if the user has provided the complete steady state.

        Returns
        -------
        is_presolved: bool
        """
        param_dict = self.free_param_dict.copy().to_sympy()
        user_provided = (
            self.steady_state_relationships.copy().to_sympy().float_to_values()
        )
        ss_eqs = self.steady_state_system.copy()
        calib_eqs = self.calibrating_equations.copy()
        all_eqs = ss_eqs + calib_eqs

        all_vars_sym = list(self.steady_state_dict.to_sympy().keys())
        all_vars_and_calib_sym = all_vars_sym + self.params_to_calibrate

        zeros = np.full_like(all_eqs, False)
        simplified_eqs = substitute_all_equations(all_eqs, user_provided)

        for i, eq in enumerate(simplified_eqs):
            subbed_eq = eq.subs(param_dict)

            # Janky, but many expressions won't reduce to zero even if they ought to -> test numerically
            atoms = [x for x in subbed_eq.atoms() if x in all_vars_and_calib_sym]
            test_values = {x: np.random.uniform(1e-2, 0.99) for x in atoms}
            eq_is_zero = sp.Abs(subbed_eq.subs(test_values)) < 1e-8
            zeros[i] = eq_is_zero

            if isinstance(subbed_eq, sp.Float) and not eq_is_zero:
                raise ValueError(
                    f"Applying user steady state definitions to equation {i}:\n"
                    f"\t{all_eqs[i]}\n"
                    f"resulted in non-zero residuals: {subbed_eq}.\n"
                    f"Please verify the provided steady state relationships are correct."
                )

        try:
            eqs_to_solve = [eq for i, eq in enumerate(simplified_eqs) if not zeros[i]]
        except TypeError:
            msg = "Found the following loose symbols during simplification:\n"
            # Something didn't reduce, figure out what and show the user
            for i, eq in enumerate(zeros):
                loose_symbols = [
                    x for x in eq.atoms() if isinstance(x, (sp.Symbol, TimeAwareSymbol))
                ]
                if len(loose_symbols) > 0:
                    msg += (
                        f"Equation {i}: "
                        + ", ".join([x.name for x in loose_symbols])
                        + "\n"
                    )
            raise ValueError(msg)

        return eqs_to_solve

    def solve_steady_state(
        self,
        apply_user_simplifications: bool | None = True,
        model_is_linear: bool | None = True,
        optimizer_kwargs: dict[str, Any] | None = None,
        method: str | None = "root",
        use_jac: bool | None = True,
        use_hess: bool | None = True,
    ) -> Callable:
        """
        Solving of the steady state proceeds in three steps: solve calibrating equations (if any), gather user provided
        equations into a function, then solve the remaining equations.

        Calibrating equations are handled first because if the user passed a complete steady state solution, it is
        unlikely to include solutions for calibrating equations. Calibrating equations are then combined with
        user supplied equations, and we check if everything necessary to solve the model is now present. If not,
        a final optimizer step runs to solve for the remaining variables.

        Note that no checks are done in this function to validate the steady state solution. If a user supplies an
        incorrect steady state, this function will not catch it. It will, however, still fail if an optimizer fails
        to find a solution.

        Parameters
        ----------
        apply_user_simplifications: bool
            If true, substitute all equations using the steady-state equations provided in the steady_state block
            of the GCN file.
        model_is_linear: bool
            A flag indicating that the model has already been linearized by the user. In this case, the steady state
            can be obtained simply by forming an augmented matrix and finding its reduced row-echelon form. If True,
            all other arguments to this function have no effect. Default is False.
        optimizer_kwargs: dict
            A dictionary of keyword arguments to pass to the scipy optimizer, either root or minimize. See the docstring
            for scipy.optimize.root or scipy.optimize.minimize for more information.
        method: str, default: "root"
            Whether to seek the steady state via root finding algorithm or via minimization of squared errors. "root"
            requires that the number of unknowns be equal to the number of equations; this assumption can be violated
            if the user provides only a subset of steady-state relationship (and this subset does not result in
            elimination of model equations via substitution).
            One of "root" or "minimize".
        use_jac: bool
            A flag indicating whether to use the Jacobian of the steady-state system when solving. Can help the
            solver on complex problems, but symbolic computation may be slow on large problems. Default is True.
        use_hess: bool
            A flag indicating whether to use the Hessian of the loss function of the steady-state system when solving.
            Ignored if method is "root", as these routines do not use Hessian information.

        Returns
        -------
        f_ss: Callable
            A function that maps a dictionary of parameters to steady state values for all system variables and
            calibrated parameters.
        """

        param_dict = self.free_param_dict.copy().to_sympy()
        params = list(param_dict.keys())
        calib_params = self.params_to_calibrate
        user_provided = (
            self.steady_state_relationships.copy().to_sympy().float_to_values()
        )
        ss_eqs = self.steady_state_system.copy()
        calib_eqs = self.calibrating_equations.copy()
        all_eqs = ss_eqs + calib_eqs

        all_vars_sym = list(self.steady_state_dict.to_sympy().keys())
        all_vars_and_calib_sym = all_vars_sym + self.params_to_calibrate

        # This can be skipped if we're working on a linear model (there should be no user simplifications)
        if apply_user_simplifications and not model_is_linear:
            eqs_to_solve = self.apply_user_simplifications()
        else:
            eqs_to_solve = all_eqs

        vars_sym = sorted(
            list(
                {
                    x
                    for eq in eqs_to_solve
                    for x in eq.atoms()
                    if isinstance(x, TimeAwareSymbol)
                }
            ),
            key=lambda x: x.name,
        )

        vars_and_calib_sym = vars_sym + calib_params

        k_vars = len(vars_sym)
        k_calib = len(calib_params)
        n_eq = len(eqs_to_solve)
        n_loose = len(vars_and_calib_sym)

        if (n_eq != n_loose) and (n_eq > 0) and (method == "root"):
            raise ValueError(
                'method = "root" is only possible when the number of equations (after substitution of '
                "user-provided steady-state relationships) is equal to the number of (remaining) "
                f"variables.\nFound {n_eq} equations and {k_vars + k_calib} variables. This can happen if "
                f"user-provided steady-state relationships do not result in elimination of model "
                f"equations after substitution. \nCheck the provided steady state relationships, or "
                f'use method = "minimize" to attempt to solve via minimization of squared errors.'
            )

        # Get residuals for all equations, regardless of how much simplification was done
        f_ss_resid = numba_lambdify(
            exog_vars=all_vars_and_calib_sym, endog_vars=params, expr=[all_eqs]
        )

        if model_is_linear:
            steady_state_values = self._solve_linear_steady_state()
            f_ss = numba_lambdify(exog_vars=params, expr=steady_state_values)

            def ss_func(param_dict):
                success = True
                params = np.array(list(param_dict.values()))

                # Need to ravel because the result of Ab.rref() is a column vector
                ss_values = f_ss(params).ravel()
                result_dict = SymbolDictionary(
                    dict(zip(all_vars_and_calib_sym, ss_values))
                )

                ss_dict = self.steady_state_dict.float_to_values().to_sympy().copy()
                calib_dict = SymbolDictionary(
                    dict(zip(self.params_to_calibrate, [np.inf] * k_calib))
                )

                for k in ss_dict.keys():
                    ss_dict[k] = result_dict[k]
                for k in calib_dict.keys():
                    calib_dict[k] = result_dict[k]

                return {
                    "ss_dict": ss_dict.to_string(),
                    "calib_dict": calib_dict.to_string(),
                    "resids": np.array(
                        f_ss_resid(
                            np.array(
                                list(ss_dict.values()) + list(calib_dict.values())
                            ),
                            params,
                        )
                    ),
                    "success": success,
                }

            return ss_func

        f_user = numba_lambdify(
            exog_vars=vars_and_calib_sym,
            endog_vars=params,
            expr=[list(user_provided.values())],
        )

        optimizer_required = True
        f_jac_ss = None
        f_hess_ss = None

        if n_eq == 0:
            optimizer_required = False

        elif method == "root":
            f_ss = numba_lambdify(
                exog_vars=vars_and_calib_sym, endog_vars=params, expr=[eqs_to_solve]
            )

            if use_jac:
                jac = sp.Matrix(
                    [[eq.diff(x) for x in vars_and_calib_sym] for eq in eqs_to_solve]
                )
                f_jac_ss = numba_lambdify(
                    exog_vars=vars_and_calib_sym, endog_vars=params, expr=jac
                )

        elif method == "minimize":
            # For minimization, need to form a loss function (use L2 norm -- better options?).
            loss = sum([eq**2 for eq in eqs_to_solve])
            f_loss = numba_lambdify(
                exog_vars=vars_and_calib_sym, endog_vars=params, expr=[loss]
            )
            if use_jac:
                jac = [loss.diff(x) for x in vars_and_calib_sym]

                f_jac_ss = numba_lambdify(
                    exog_vars=vars_and_calib_sym, endog_vars=params, expr=[jac]
                )

            if use_hess:
                hess = sp.hessian(loss, vars_and_calib_sym)
                f_hess_ss = numba_lambdify(
                    exog_vars=vars_and_calib_sym, endog_vars=params, expr=hess
                )

        optimizer_kwargs = _validate_optimizer_kwargs(
            optimizer_kwargs, n_eq, method, use_jac, use_hess
        )

        def ss_func(param_dict):
            params = np.array(list(param_dict.values()))

            if optimizer_required:
                if "x0" not in optimizer_kwargs.keys():
                    optimizer_kwargs["x0"] = np.full(k_vars + k_calib, 0.8)
                with catch_warnings():
                    simplefilter("ignore")
                    if method == "root":
                        optim = optimize.root(
                            f_ss, jac=f_jac_ss, args=params, **optimizer_kwargs
                        )
                    elif method == "minimize":
                        optim = optimize.minimize(
                            f_loss,
                            jac=f_jac_ss,
                            hess=f_hess_ss,
                            args=params,
                            **optimizer_kwargs,
                        )

                optim_dict = SymbolDictionary(dict(zip(vars_and_calib_sym, optim.x)))
                success = optim.success
            else:
                optim_dict = SymbolDictionary()
                success = True

            ss_dict = self.steady_state_dict.float_to_values().to_sympy().copy()
            calib_dict = SymbolDictionary(
                dict(zip(self.params_to_calibrate, [np.inf] * k_calib))
            )
            user_dict = SymbolDictionary(
                dict(
                    zip(
                        user_provided.keys(),
                        f_user(np.array(list(optim_dict.values())), params),
                    )
                )
            )

            for k in all_vars_sym:
                if k in optim_dict.keys():
                    ss_dict[k] = optim_dict[k]
                elif k in user_provided.keys():
                    ss_dict[k] = user_dict[k]
                else:
                    raise ValueError(
                        f"Could not find {k} among either optimizer or user provided solutions"
                    )

            for k in calib_params:
                if k in optim_dict.keys():
                    calib_dict[k] = optim_dict[k]
                elif k in user_provided.keys():
                    calib_dict[k] = user_dict[k]
                else:
                    raise ValueError(
                        f"Could not find {k} among either optimizer or user provided solutions"
                    )

            ss_dict.sort_keys(inplace=True)
            calib_dict.sort_keys(inplace=True)

            return {
                "ss_dict": ss_dict.to_string(),
                "calib_dict": calib_dict.to_string(),
                "resids": np.array(
                    f_ss_resid(
                        np.array(list(ss_dict.values()) + list(calib_dict.values())),
                        params,
                    )
                ),
                "success": success,
            }

        return ss_func

    def _solve_linear_steady_state(self) -> list[sp.Add]:
        """
        If the model is linear, we can quickly solve for the steady state by putting everything into a matrix and
        getting the reduced row-echelon form.

        # TODO: Potentially save a "reverse deterministic sub" dict for use here.

        Returns
        -------
        steady_state_values, list
            A list of closed-form solutions to the steady-state, one per
        """

        shock_subs = {shock.to_ss(): 0 for shock in self.shocks}

        all_vars_sym = list(self.steady_state_dict.to_sympy().keys())

        all_vars_and_calib_sym = self.variables + self.params_to_calibrate
        all_vars_and_calib_sym_ss = all_vars_sym + self.params_to_calibrate
        all_eqs = self.system_equations + self.calibrating_equations

        # simplifications make the next few steps a lot faster
        sub_dict, simplified_system = sp.cse(all_eqs, ignore=all_vars_and_calib_sym)
        A, b = sp.linear_eq_to_matrix(
            [eq_to_ss(eq).subs(shock_subs) for eq in simplified_system],
            all_vars_and_calib_sym_ss,
        )
        Ab = sp.Matrix([[A, b]])
        A_rref, _ = Ab.rref()

        # Recursive substitution to undo any simplifications
        steady_state_values = A_rref[:, -1].subs(sub_dict * len(sub_dict))

        return steady_state_values

    def _steady_state_fast(self, model_is_linear: bool | None = True):
        param_dict = self.free_param_dict.copy().to_sympy()
        params = list(param_dict.keys())

        if model_is_linear:
            steady_state_values = self._solve_linear_steady_state()
        elif self._system_is_presolved():
            steady_state_values = self.get_presolved_system()
        else:
            raise ValueError(
                "Cannot get a fast steady state solution unless the model is linear or a full closed-form"
                "solution is provided"
            )

        f_ss = numba_lambdify(exog_vars=params, expr=steady_state_values)
        return f_ss
