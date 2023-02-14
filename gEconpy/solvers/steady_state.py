from functools import wraps
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import catch_warnings, simplefilter

import numpy as np
import sympy as sp
from joblib import Parallel, delayed
from scipy import optimize

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.shared.typing import VariableType
from gEconpy.shared.utilities import eq_to_ss, substitute_all_equations

# def convert_to_numba_function(expr: sp.Expr, vars: List[str]) -> Callable:
#     """
#     Convert a sympy expression into a Numba-compiled function.
#
#     Parameters
#     ----------
#     expr : sympy.Expr
#         The sympy expression to be converted.
#     vars : List[str]
#         A list of strings containing the names of the variables in the expression.
#
#     Returns
#     -------
#     numba.types.function
#         A Numba-compiled function equivalent to the input expression.
#
#     Notes
#     -----
#     The function returned by this function is pickleable.
#     """
#     code = sp.printing.ccode(expr)
#     # The code string will contain a single line, so we add line breaks to make it a valid block of code
#     code = "@nb.njit\ndef f({}):\n{}\n    return np.array({})".format(",".join(vars), " " * 4, code)
#
#     # Compile the code and return the resulting function
#     exec(code)
#     return locals()["f"]


def sympy_inputs_to_scipy(f):
    @wraps(f)
    def new_f(x0, *args):
        d, *_ = args
        return f(*x0, **d)

    return new_f


def postprocess_jac(f, shape):
    @wraps(f)
    def new_f(*args, **kwargs):
        return np.array(f(*args, **kwargs)).reshape(shape)

    return new_f


class SteadyStateSolver:
    def __init__(self, model):

        self.variables: List[VariableType] = model.variables
        self.shocks: List[sp.Add] = model.shocks

        self.n_variables: int = model.n_variables

        self.free_param_dict: SymbolDictionary[str, float] = model.free_param_dict
        self.params_to_calibrate: List[VariableType] = model.params_to_calibrate
        self.calibrating_equations: List[sp.Add] = model.calibrating_equations
        self.shock_dict: Optional[SymbolDictionary[str, float]] = None

        self.system_equations: List[sp.Add] = model.system_equations
        self.steady_state_relationships: SymbolDictionary[
            str, Union[float, sp.Add]
        ] = model.steady_state_relationships

        self.steady_state_system: List[sp.Add] = []
        self.steady_state_dict: SymbolDictionary[str, float] = SymbolDictionary()
        self.steady_state_solved: bool = False

        #         self.f_calib_params: Callable = lambda *args, **kwargs: {}
        #         self.f_ss_resid: Callable = lambda *args, **kwargs: np.inf
        #         self.f_ss: Callable = lambda *args, **kwargs: np.inf

        self.build_steady_state_system()

    def build_steady_state_system(self):

        ss_vars = map(lambda x: x.to_ss(), self.variables)
        self.steady_state_dict = SymbolDictionary.fromkeys(ss_vars, None).to_string().sort_keys()

        self.shock_dict = SymbolDictionary.fromkeys(self.shocks, 0.0).to_ss()
        self.steady_state_system = [
            eq_to_ss(eq).subs(self.shock_dict).simplify() for eq in self.system_equations
        ]

    def _validate_optimizer_kwargs(
        self, optimizer_kwargs: dict, n_eq: int, method: str, use_jac: bool, use_hess: bool
    ) -> dict:
        """
        Validate user-provided keyword arguments to either scipy.optimize.root or scipy.optimize.minimize, and insert
        good defaults where not provided.

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
        method_given = hasattr(optimizer_kwargs, "method")

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

        if not hasattr(optimizer_kwargs, "tol"):
            optimizer_kwargs["tol"] = 1e-9

        return optimizer_kwargs

    def solve_steady_state(
        self,
        apply_user_simplifications=True,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        method: Optional[str] = "root",
        use_jac: Optional[bool] = True,
        use_hess: Optional[bool] = True,
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

        param_dict = self.free_param_dict.to_sympy()
        params = list(param_dict.to_string().keys())
        calib_params = self.params_to_calibrate
        user_provided = self.steady_state_relationships.to_sympy().float_to_values()

        ss_eqs = self.steady_state_system
        calib_eqs = self.calibrating_equations
        all_eqs = ss_eqs + calib_eqs

        all_vars_sym = list(self.steady_state_dict.to_sympy().keys())
        all_vars_and_calib_sym = all_vars_sym + self.params_to_calibrate

        if apply_user_simplifications:

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
            eqs_to_solve = [eq for i, eq in enumerate(simplified_eqs) if not zeros[i]]
        else:
            eqs_to_solve = all_eqs

        vars_sym = sorted(
            list({x for eq in eqs_to_solve for x in eq.atoms() if isinstance(x, TimeAwareSymbol)}),
            key=lambda x: x.name,
        )

        vars_and_calib_sym = vars_sym + calib_params
        vars_and_calib_str = [x.name for x in vars_and_calib_sym]

        k_vars = len(vars_sym)
        k_calib = len(calib_params)
        n_eq = len(eqs_to_solve)

        if (n_eq != (k_vars + k_calib)) and (n_eq > 0) and (method == "root"):
            raise ValueError(
                'method = "root" is only possible when the number of equations (after substitution of '
                "user-provided steady-state relationships) is equal to the number of (remaining) "
                f"variables.\nFound {n_eq} equations and {k_vars} variables. This can happen if "
                f"user-provided steady-state relationships do not result in elimination of model "
                f"equations after substitution. \nCheck the provided steady state relationships, or "
                f'use method = "minimize" to attempt to solve via minimization of squared errors.'
            )

        # Get residuals for all equations, regardless of how much simplification was done
        f_ss_resid = sp.lambdify([x.name for x in all_vars_and_calib_sym] + params, all_eqs)
        f_user = sp.lambdify(vars_and_calib_sym + params, list(user_provided.values()))

        optimizer_required = True
        f_jac_ss = None
        f_hess_ss = None

        if n_eq == 0:
            optimizer_required = False
        if method == "root":
            if n_eq > 0:
                # The ccode printer complains about nested lists; make a flat jacobian and reshape it later
                ss_jac_flat = [eq.diff(x) for eq in eqs_to_solve for x in vars_and_calib_sym]

                f_ss = sympy_inputs_to_scipy(sp.lambdify(vars_and_calib_str + params, eqs_to_solve))

                if use_jac:
                    f_jac_ss = postprocess_jac(
                        sympy_inputs_to_scipy(
                            sp.lambdify(vars_and_calib_str + params, ss_jac_flat)
                        ),
                        (n_eq, k_vars + k_calib),
                    )

        elif method == "minimize":
            # For minimization, need to form a loss function (use L2 norm -- better options?).
            loss = sum([eq**2 for eq in eqs_to_solve])
            f_loss = sympy_inputs_to_scipy(sp.lambdify(vars_and_calib_str + params, loss))
            if use_jac:
                f_jac_ss = sympy_inputs_to_scipy(
                    sp.lambdify(
                        vars_and_calib_str + params, [loss.diff(x) for x in vars_and_calib_sym]
                    )
                )
            if use_hess:
                hess_flat = [
                    loss.diff(x, y) for x in vars_and_calib_sym for y in vars_and_calib_sym
                ]
                f_hess_ss = postprocess_jac(
                    sympy_inputs_to_scipy(sp.lambdify(vars_and_calib_sym + params, hess_flat)),
                    (k_vars + k_calib, k_vars + k_calib),
                )

        optimizer_kwargs = self._validate_optimizer_kwargs(
            optimizer_kwargs, n_eq, method, use_jac, use_hess
        )

        def ss_func(param_dict):
            if optimizer_required:
                x0 = np.full(k_vars + k_calib, 0.8)

                with catch_warnings():
                    simplefilter("ignore")
                    if method == "root":
                        optim = optimize.root(
                            f_ss, jac=f_jac_ss, x0=x0, args=(param_dict,), **optimizer_kwargs
                        )
                    elif method == "minimize":
                        optim = optimize.minimize(
                            f_loss,
                            jac=f_jac_ss,
                            hess=f_hess_ss,
                            x0=x0,
                            args=(param_dict,),
                            **optimizer_kwargs,
                        )

                optim_dict = SymbolDictionary(dict(zip(vars_and_calib_sym, optim.x)))
                success = optim.success
            else:
                optim_dict = SymbolDictionary()
                success = True

            ss_dict = self.steady_state_dict.float_to_values().to_sympy().copy()
            calib_dict = SymbolDictionary(dict(zip(self.params_to_calibrate, [np.inf] * k_calib)))
            user_dict = SymbolDictionary(
                dict(
                    zip(
                        user_provided.keys(),
                        f_user(**optim_dict.to_string(), **param_dict.to_string()),
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
                    f_ss_resid(**ss_dict.to_string(), **calib_dict.to_string(), **param_dict)
                ),
                "success": success,
            }

        return ss_func


class SymbolicSteadyStateSolver:
    def __init__(self):
        pass

    @staticmethod
    def score_eq(
        eq: sp.Expr,
        var_list: List[sp.Symbol],
        state_vars: List[sp.Symbol],
        var_penalty_factor: float = 25,
        state_var_penalty_factor: float = 5,
        length_penalty_factor: float = 1,
    ) -> float:

        """
        Compute an "unfitness" score for an equation using three simple heuristics:
            1. The number of jumper variables in the expression
            2. The number of state variables in the expression
            3. The total length of the expression

        Expressions with the lowest unfitness will be selected. Setting a lower penalty for state variables will
        push the system towards finding solutions expressed in state variables if a steady state is parameters only
        cannot be found.

        Parameters
        ----------
        eq: sp.Expr
            A sympy expression representing a steady-state equation
        var_list: list of sp.Symbol
            A list of sympy symbols representing all variables in the model (state and jumper)
        state_vars: list of sp.Symbol
            A list of symbol symbols representing all state variables in the model
        var_penalty_factor: float, default: 25
            A penalty factor applied to unfitness for each jumper variable in the expression.
        state_var_penalty_factor: float, default: 5
            A penalty factor applied to unfitness for each control variable in the expression.
        length_penalty_factor: float, default: 1
            A penalty factor applied to each term in the expression

        Returns
        -------
        unfitness: float
            An unfitness score used to select potential substitutions between system equations
        """

        # If the equation is length zero, it's been reduced away and should never be selected.
        if eq == 0:
            return 10000

        var_list = list(set(var_list) - set(state_vars))

        # The equation with the LOWEST score will be chosen to substitute, so punishing state variables less
        # ensures that equations that have only state variables will be chosen more often.
        var_penalty = len([x for x in eq.atoms() if x in var_list]) * var_penalty_factor
        state_var_penalty = (
            len([x for x in eq.atoms() if x in state_vars]) * state_var_penalty_factor
        )

        # Prefer shorter equations
        length_penalty = eq.count_ops() * length_penalty_factor

        return var_penalty + state_var_penalty + length_penalty

    @staticmethod
    def solve_and_return(eq: sp.Expr, v: sp.Symbol) -> sp.Expr:
        """
        Attempt to solve an expression for a given variable. Returns 0 if the expression is not solvable or if the
        given variable does not appear in the expression. If multiple solutions are found, only the first one is
        returned.

        Parameters
        ----------
        eq: sp.Expr
            A sympy expression
        v: sp.Symbol
            A sympy symbol

        Returns
        -------
        solution: sp.Expr
            Given f(x, ...) =  0, returns x = g(...) if possible, or 0 if not.
        """

        if v not in eq.atoms():
            return sp.Float(0)
        try:
            solution = sp.solve(eq, v)
        except Exception:
            return sp.Float(0)

        if len(solution) > 0:
            return solution[0]

        return sp.Float(0)

    @staticmethod
    def clean_substitutions(sub_dict: Dict[sp.Symbol, sp.Expr]) -> Dict[sp.Symbol, sp.Expr]:
        """
        "Cleans" a dictionary of substitutions by:
            1. Delete substitutions in the form of x=x or x=0 (x=0 implies the substitution is redundant with other
                substitutions in sub_dict)
            2. If a substitution is of the form x = f(x, ...), attempts to solve the expression x - f(x, ...) = 0 for x,
                and deletes the substitution if no solution exists.
            3. Apply all substitutions in sub_dict to expressions in sub_dict to ensure older solutions remain up to
                date with newly found solutions.

        Parameters
        ----------
        sub_dict: dict
            Dictionary of sp.Symbol keys and sp.Expr values to be passed to the subs method of sympy expressions.

        Returns
        -------
        sub_dict: dict
            Cleaned dictionary of sympy substitutions
        """
        result = sub_dict.copy()

        for k, eq in sub_dict.items():
            # Remove invalid or useless substitutions
            if eq == 0 or k == eq:
                del result[k]
                continue

            # Solve for the sub variable if necessary
            elif k in eq.atoms():
                try:
                    eq = sp.solve(k - eq, k)[0]
                except Exception:
                    del result[k]
                    continue
            result[k] = eq

        # Substitute subs into the sub dict
        result = {k: v.subs(result) for k, v in result.items()}
        return result

    def get_candidates(
        self,
        system: List[sp.Expr],
        variables: List[sp.Symbol],
        state_variables: List[sp.Symbol],
        var_penalty_factor: float = 25,
        state_var_penalty_factor: float = 5,
        length_penalty_factor: float = 1,
        cores: int = -1,
    ) -> Dict[sp.Symbol, Tuple[sp.Expr, float]]:
        """
        Attempt to solve every equation in the system for every variable. Scores the results using the score_eq
        function, and returns (solution, score) pairs with the highest fitness (lowest unfitness).

        Solving equations is parallelized using joblib.

        Parameters
        ----------
        system: list of sp.Expr
            List of steady state equations to be scored
        variables: list of sp.Symbol
            List of all variables among all steady state equations
        state_variables: list of Sp.Symbol
            List of all state variables among all steady state equations
        var_penalty_factor: float, default: 25
            A penalty factor applied to unfitness for each jumper variable in the expression.
        state_var_penalty_factor: float, default: 5
            A penalty factor applied to unfitness for each control variable in the expression.
        length_penalty_factor: float, default: 1
            A penalty factor applied to each term in the expression
        cores: int, default -1
            Number of cores over which to parallelize computation. Passed to joblib.Parallel. -1 for all available
            cores.

        Returns
        -------
        candidates: dict
            A dictionary of candidate substitutions to simplify the steady state system. One candidate is produced
            for each variable in the system. Keys are sp.Symbol, and values are (sp.Expr, float) tuples with the
            candidate substitution and its fitness.
        """
        eq_vars = product(system, variables)

        n = len(system)
        k = len(variables)
        args = (
            variables,
            state_variables,
            var_penalty_factor,
            state_var_penalty_factor,
            length_penalty_factor,
        )

        with Parallel(cores) as pool:
            solutions = pool(delayed(self.solve_and_return)(eq, v) for eq, v in eq_vars)
            scores = np.array(pool(delayed(self.score_eq)(eq, *args) for eq in solutions))

        score_matrix = scores.reshape(n, k)
        idx_matrix = np.arange(n * k).reshape(n, k)
        best_idx = idx_matrix[score_matrix.argmin(axis=0), np.arange(k)]

        return dict(zip(variables, [(solutions[idx], scores[idx]) for idx in best_idx]))

    @staticmethod
    def make_solved_subs(sub_dict, assumptions):
        res = {}
        for k, v in sub_dict.items():
            if not any([isinstance(x, TimeAwareSymbol) for x in v.atoms()]):
                if v == 1:
                    continue
                res[v] = sp.Symbol(k.name + r"^\star", **assumptions[k.base_name])

        return res

    def solve_symbolic_steady_state(
        self,
        mod,
        top_k=3,
        var_penalty_factor=25,
        state_var_penalty_factor=5,
        length_penalty_factor=1,
        cores=-1,
        zero_tol=12,
    ):

        ss_vars = [x.to_ss() for x in mod.variables]
        state_vars = [x for x in mod.variables if x.base_name == "Y"]
        ss_system = mod.steady_state_system

        system = ss_system.copy()
        calib_eqs = [
            var - eq for var, eq in zip(mod.params_to_calibrate, mod.calibrating_equations)
        ]
        system.extend(calib_eqs)

        params = list(mod.free_param_dict.to_sympy().keys())
        sub_dict = {}
        unsolved_dict = {}

        while True:
            candidates = self.get_candidates(
                system,
                ss_vars,
                state_vars,
                var_penalty_factor=var_penalty_factor,
                state_var_penalty_factor=state_var_penalty_factor,
                length_penalty_factor=length_penalty_factor,
                cores=cores,
            )

            scores = np.array([score for eq, score in candidates.values()])
            print(scores)
            top_k_score_idxs = scores.argsort()[:top_k]
            for idx in top_k_score_idxs:
                key = list(candidates.keys())[idx]
                if candidates[key][0] == 0:
                    continue
                sub_dict[key] = candidates[key][0]

            sub_dict = self.clean_substitutions(sub_dict)

            system = [eq.subs(sub_dict) for eq in system]
            system = [
                eq
                for eq in system
                if not self.test_expr_is_zero(eq.subs(unsolved_dict), params, tol=zero_tol)
            ]
            solved_dict = self.make_solved_subs(sub_dict, mod.assumptions)
            unsolved_dict = {v: k.subs(unsolved_dict) for k, v in solved_dict.items()}
            system = [eq.subs(solved_dict) for eq in system]

            if len(system) == 0:
                break

            if min(scores) > 100:
                break

        to_solve = {x for eq in system for x in eq.atoms() if isinstance(x, TimeAwareSymbol)}
        system = [eq.simplify() for eq in system]
        try:
            final_solutions = sp.solve(system, to_solve, dict=True)
        except NotImplementedError:
            final_solutions = [{}]

        return [sub_dict.update(d) for d in final_solutions]


# from functools import partial
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# from warnings import catch_warnings, simplefilter
#
# import numpy as np
# import sympy as sp
# from numpy.typing import ArrayLike
# from scipy import optimize
#
# from gEconpy.classes.containers import SymbolDictionary
#
# from gEconpy.shared.typing import VariableType
# from gEconpy.shared.utilities import (
#     float_values_to_sympy_float,
#     is_variable,
#     merge_dictionaries,
#     merge_functions,
#     safe_string_to_sympy,
#     sequential,
#     sort_dictionary,
#     string_keys_to_sympy,
#     substitute_all_equations,
#     symbol_to_string,
#     sympy_keys_to_strings,
#     sympy_number_values_to_floats,
# )
#
#
# class SteadyStateSolver:
#     def __init__(self, model):
#
#         self.variables: List[VariableType] = model.variables
#         self.shocks: List[sp.Add] = model.shocks
#
#         self.n_variables: int = model.n_variables
#
#         self.free_param_dict: SymbolDictionary[str, float] = model.free_param_dict
#         self.params_to_calibrate: List[VariableType] = model.params_to_calibrate
#         self.calibrating_equations: List[sp.Add] = model.calibrating_equations
#         self.system_equations: List[sp.Add] = model.system_equations
#         self.steady_state_relationships: SymbolDictionary[
#             str, Union[float, sp.Add]
#         ] = model.steady_state_relationships
#
#         self.steady_state_system: List[sp.Add] = []
#         self.steady_state_dict: SymbolDictionary[str, float] = SymbolDictionary()
#         self.steady_state_solved: bool = False
#
#         self.f_calib_params: Callable = lambda *args, **kwargs: {}
#         self.f_ss_resid: Callable = lambda *args, **kwargs: np.inf
#         self.f_ss: Callable = lambda *args, **kwargs: np.inf
#
#         self.build_steady_state_system()
#
#     def build_steady_state_system(self):
#         self.steady_state_system = []
#
#         all_atoms = [
#             x for eq in self.system_equations for x in eq.atoms() if is_variable(x)
#         ]
#         all_variables = set(all_atoms) - set(self.shocks)
#         ss_sub_dict = {variable: variable.to_ss() for variable in set(all_variables)}
#         unique_ss_variables = list(set(list(ss_sub_dict.values())))
#
#         steady_state_dict = dict.fromkeys(unique_ss_variables, None)
#         steady_state_dict = (SymbolDictionary(steady_state_dict)
#                              .to_string()
#                              .sort_keys())
#
#         self.steady_state_dict = steady_state_dict
#
#         for shock in self.shocks:
#             ss_sub_dict[shock] = 0
#
#         for eq in self.system_equations:
#             self.steady_state_system.append(eq.subs(ss_sub_dict))
#
#     def solve_steady_state(
#         self,
#         param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#         use_jac: Optional[bool] = False,
#     ) -> Callable:
#         """
#
#         Parameters
#         ----------
#         param_bounds: dict
#             A dictionary of string, tuple(float, float) pairs, giving bounds for each variable or parameter to be
#             solved for. Only used by certain optimizers; check the scipy docs. Pass it here instead of in
#             optimizer_kwargs to make sure the correct variables have the correct bounds.
#         optimizer_kwargs: dict
#             A dictionary of keyword arguments to pass to the scipy optimizer, either root or root_scalar.
#         use_jac: bool
#             A flag to symbolically compute the Jacobain function of the model before optimization, can help the solver
#             on complex problems.
#
#         Returns
#         -------
#         f_ss: Callable
#             A function that maps a dictionary of parameters to steady state values for all system variables and
#             calibrated parameters.
#
#         Solving of the steady state proceeds in three steps: solve calibrating equations (if any), gather user provided
#         equations into a function, then solve the remaining equations.
#
#         Calibrating equations are handled first because if the user passed a complete steady state solution, it is
#         unlikely to include solutions for calibrating equations. Calibrating equations are then combined with
#         user supplied equations, and we check if everything necessary to solve the model is now present. If not,
#         a final optimizer step runs to solve for the remaining variables.
#
#         Note that no checks are done in this function to validate the steady state solution. If a user supplies an
#         incorrect steady state, this function will not catch it. It will, however, still fail if an optimizer fails
#         to find a solution.
#         """
#         free_param_dict = self.free_param_dict.copy()
#         parameters = list(free_param_dict.keys())
#         variables = list(self.steady_state_dict.keys())
#
#         params_to_calibrate = [symbol_to_string(x) for x in self.params_to_calibrate]
#
#         n_to_calibrate = len(params_to_calibrate)
#         has_calibrating_equations = n_to_calibrate > 0
#
#         params_and_variables = parameters + params_to_calibrate + variables
#         steady_state_system = self.steady_state_system
#
#         # TODO: Move the creation of this residual function somewhere more logical
#         self.f_ss_resid = sp.lambdify(params_and_variables, steady_state_system)
#
#         # Solve calibrating equations, if any.
#         if has_calibrating_equations:
#             f_calib, additional_solutions = self._solve_calibrating_equations(
#                 param_bounds=param_bounds,
#                 optimizer_kwargs=optimizer_kwargs,
#                 use_jac=use_jac,
#             )
#         else:
#             f_calib = lambda *args, **kwargs: {}
#             additional_solutions = {}
#
#         solved_calib_params = list(f_calib(free_param_dict).keys())
#
#         # Gather user provided steady state solutions
#         f_provided = self._gather_provided_solutions(solved_calib_params)
#
#         calib_dict = f_calib(free_param_dict)
#         var_dict = f_provided(free_param_dict, calib_dict)
#
#         # If we have everything we're done. We don't need to use final_f, set it to return an empty dictionary.
#         if (
#             set(params_and_variables) - set(var_dict.keys()).union(calib_dict.keys())
#         ) == set(free_param_dict.keys()):
#             f_ss = self._create_final_function(
#                 final_f=lambda x: {}, f_calib=f_calib, f_provided=f_provided
#             )
#
#         else:
#             final_f = self._solve_remaining_equations(
#                 calib_dict=calib_dict,
#                 var_dict=var_dict,
#                 additional_solutions=additional_solutions,
#                 param_bounds=param_bounds,
#                 optimizer_kwargs=optimizer_kwargs,
#                 use_jac=use_jac,
#             )
#             f_ss = self._create_final_function(
#                 final_f=final_f, f_calib=f_calib, f_provided=f_provided
#             )
#
#         return f_ss
#
#
#     def _solve_calibrating_equations(
#         self,
#         param_bounds: Optional[Dict[str, Tuple[float, float]]],
#         optimizer_kwargs: Optional[Dict[str, Any]],
#         use_jac: bool = False,
#     ) -> Tuple[Callable, Dict]:
#         """
#         Parameters
#         ----------
#         param_bounds: dict
#             See docstring of solve_steady_state for details
#         optimizer_kwargs: dict
#             See docstring of solve_steady_state for details
#         use_jac: bool
#             See docstring of solve_steady_state for details
#
#         Returns
#         -------
#         f_calib: callable
#             A function that maps param_dict to values of calibrated parameteres
#         additional_solutions: dict
#             A dictionary of symbolic solutions to non-calibrating parameters that were solved en passant and can be
#             reused later
#         """
#         calibrating_equations = self.calibrating_equations
#         symbolic_solutions = self.steady_state_relationships.copy()
#         free_param_dict = self.free_param_dict.copy()
#         steady_state_system = self.steady_state_system
#
#         parameters = list(free_param_dict.keys())
#         variables = list(self.steady_state_dict.keys())
#         params_to_calibrate = [symbol_to_string(x) for x in self.params_to_calibrate]
#         params_and_variables = parameters + params_to_calibrate + variables
#
#         unknown_variables = set(variables).union(set(params_to_calibrate)) - set(
#             symbolic_solutions.keys()
#         )
#
#         n_to_calibrate = len(params_to_calibrate)
#
#         additional_solutions = {}
#
#         # Make substitutions
#         calib_with_user_solutions = substitute_all_equations(
#             calibrating_equations, symbolic_solutions
#         )
#
#         # Try the heuristic solver
#         calib_solutions, solved_mask = self.heuristic_solver(
#             {},
#             calib_with_user_solutions,
#             calib_with_user_solutions,
#             [safe_string_to_sympy(x) for x in params_and_variables],
#         )
#
#         # Case 1: We found something! Refine the solution.
#         if solved_mask.sum() > 0:
#             # If the heuristic solver worked, we got solutions for variables that will allow us to go back and solve for
#             # the calibrating parameters.
#
#             sub_dict = merge_dictionaries(free_param_dict, calib_solutions)
#             more_solutions, solved_mask = self.heuristic_solver(
#                 sub_dict,
#                 substitute_all_equations(steady_state_system, sub_dict),
#                 steady_state_system,
#                 [safe_string_to_sympy(x) for x in params_and_variables],
#             )
#
#             calib_solutions = {
#                 key: value
#                 for key, value in more_solutions.items()
#                 if (key in params_to_calibrate)
#             }
#
#             # We potentially pick up additional solutions from this heuristic pass, we can save them and use them later
#             # to help the heuristic solver later.
#             additional_solutions = {
#                 key: value
#                 for key, value in more_solutions.items()
#                 if (key not in params_to_calibrate) and (key not in free_param_dict)
#             }
#
#             calib_solutions = SymbolDictionary(calib_solutions).to_string().sort_keys().values_to_float()
#             f_calib = lambda *args, **kwargs: calib_solutions
#
#         # Case 2: Found nothing, try to use an optimizer
#         else:
#             # Here we check how many equations are remaining to solve after accounting for the user's SS info.
#             # We're looking for the case when all information is given EXCEPT the calibrating parameters.
#             # If there is more than that, we handle it in the final pass.
#             calib_remaining_to_solve = list(
#                 set(unknown_variables) - set(symbolic_solutions.keys())
#             )
#             calib_n_eqs = len(calib_remaining_to_solve)
#             if calib_n_eqs > len(calibrating_equations):
#
#                 def f_calib(*args, **kwargs):
#                     return SymbolDictionary()
#
#                 return f_calib, SymbolDictionary()
#
#             # TODO: Is there a more elegant way to handle one equation vs many equations here?
#             if calib_n_eqs == 1:
#                 calib_with_user_solutions = calib_with_user_solutions[0]
#
#                 _f_calib = sp.lambdify(
#                     calib_remaining_to_solve + parameters, calib_with_user_solutions
#                 )
#
#                 def f_calib(x, kwargs):
#                     return _f_calib(x, **kwargs)
#
#             else:
#                 _f_calib = sp.lambdify(
#                     calib_remaining_to_solve + parameters, calib_with_user_solutions
#                 )
#
#                 def f_calib(args, kwargs):
#                     return _f_calib(*args, **kwargs)
#
#             f_jac = None
#             if use_jac:
#                 f_jac = self._build_jacobian(
#                     diff_variables=calib_remaining_to_solve,
#                     additional_inputs=parameters,
#                     equations=calib_with_user_solutions,
#                 )
#
#             f_calib = self._bundle_symbolic_solutions_with_optimizer_solutions(
#                 unknowns=calib_remaining_to_solve,
#                 f=f_calib,
#                 f_jac=f_jac,
#                 param_dict=free_param_dict,
#                 symbolic_solutions=calib_solutions,
#                 n_eqs=calib_n_eqs,
#                 output_names=calib_remaining_to_solve,
#                 param_bounds=param_bounds,
#                 optimizer_kwargs=optimizer_kwargs,
#             )
#
#         return f_calib, additional_solutions
#
#     def _gather_provided_solutions(self, solved_calib_params) -> Callable:
#         """
#         Returns
#         -------
#         f_provided: Callable
#             A function that takes model parameters, both calibrated and otherwise, as keywork arguments, and returns
#             a dictionary of variable values according to steady state equations supplied by the user
#         """
#
#         free_param_dict = self.free_param_dict.copy()
#         symbolic_solutions = self.steady_state_relationships.copy()
#         parameters = list(free_param_dict.keys())
#
#         _provided_lambda = sp.lambdify(
#             parameters + solved_calib_params, [eq for eq in symbolic_solutions.values()]
#         )
#
#         def f_provided(param_dict, calib_dict):
#             return SymbolDictionary(dict(
#                 zip(
#                     symbolic_solutions.keys(),
#                     _provided_lambda(**param_dict, **calib_dict),
#                 )
#             ))
#
#         return f_provided
#
#     def _solve_remaining_equations(
#         self,
#         calib_dict: Dict[str, float],
#         var_dict: Dict[str, float],
#         additional_solutions: Dict[str, float],
#         param_bounds: Optional[Dict[str, Tuple[float, float]]],
#         optimizer_kwargs: Optional[Dict[str, Any]],
#         use_jac: bool,
#     ) -> Callable:
#         """
#         Parameters
#         ----------
#         calib_dict: Dict
#             A dictionary of solved calibrating parameters, if any.
#         var_dict: Dict
#             A dictionary of user-provided steady-state relationships, if any.
#         additional_solutions:
#             A dictionary of variable solutions found en passant by the heuristic solver while solving for the
#             calibrated parameters, if any.
#         param_bounds:
#             See docstring of solve_steady_state for details
#         optimizer_kwargs:
#             See docstring of solve_steady_state for details
#         use_jac:
#             See docstring of solve_steady_state for details
#
#         Returns
#         -------
#         f_final: Callable
#             A function that takes model parameters as keyword arguments and returns steady-state values for each
#             model variable without an explicit symbolic solution.
#         """
#         free_param_dict = self.free_param_dict
#         steady_state_system = self.steady_state_system
#         calibrating_equations = self.calibrating_equations
#
#         parameters = list(free_param_dict.keys())
#         variables = list(self.steady_state_dict.keys())
#         params_to_calibrate = [symbol_to_string(x) for x in self.params_to_calibrate]
#
#         sub_dict = merge_dictionaries(calib_dict, var_dict, additional_solutions)
#         params_and_variables = parameters + params_to_calibrate + variables
#
#         ss_solutions, solved_mask = self.heuristic_solver(
#             sub_dict,
#             substitute_all_equations(
#                 steady_state_system + calibrating_equations, sub_dict, free_param_dict
#             ),
#             steady_state_system + calibrating_equations,
#             [safe_string_to_sympy(x) for x in params_and_variables],
#         )
#
#         ss_solutions = {
#             key: value
#             for key, value in ss_solutions.items()
#             if key not in calib_dict.keys()
#         }
#         sub_dict.update(ss_solutions)
#
#         ss_remaining_to_solve = sorted(
#             list(
#                 set(variables + params_to_calibrate)
#                 - set(ss_solutions.keys())
#                 - set(calib_dict.keys())
#             )
#         )
#
#         unsolved_eqs = substitute_all_equations(
#             [
#                 eq
#                 for idx, eq in enumerate(steady_state_system + calibrating_equations)
#                 if not solved_mask[idx]
#             ],
#             sub_dict,
#         )
#
#         n_eqs = len(unsolved_eqs)
#
#         _f_unsolved_ss = sp.lambdify(ss_remaining_to_solve + parameters, unsolved_eqs)
#
#         def f_unsolved_ss(args, kwargs):
#             return _f_unsolved_ss(*args, **kwargs)
#
#         f_jac = None
#         if use_jac:
#             f_jac = self._build_jacobian(
#                 diff_variables=ss_remaining_to_solve,
#                 additional_inputs=parameters,
#                 equations=unsolved_eqs,
#             )
#
#         f_final = self._bundle_symbolic_solutions_with_optimizer_solutions(
#             unknowns=ss_remaining_to_solve,
#             f=f_unsolved_ss,
#             f_jac=f_jac,
#             param_dict=free_param_dict,
#             symbolic_solutions=ss_solutions,
#             n_eqs=n_eqs,
#             output_names=ss_remaining_to_solve,
#             param_bounds=param_bounds,
#             optimizer_kwargs=optimizer_kwargs,
#         )
#
#         return f_final
#
#     def _create_final_function(self, final_f, f_calib, f_provided):
#         """
#
#         Parameters
#         ----------
#         final_f: Callable
#             Function generated by solve_remaining_equations
#         f_calib: Callable
#             Function generated by _solve_calibrating_equations
#         f_provided: Callable
#             Function generated by _gather_provided_solutions
#
#         Returns
#         -------
#         f_ss: Callable
#             A single function wrapping the three steady state functions, that returns a complete solution to the
#             model's steady state as two dictionaries: one with variable values, and one with calibrated parameter
#             values.
#         """
#         calib_params = [x.name for x in self.params_to_calibrate]
#         ss_vars = [x.to_ss().name for x in self.variables]
#
#         def combined_function(param_dict):
#             ss_out = SymbolDictionary()
#
#             calib_dict = f_calib(param_dict).copy()
#             var_dict = f_provided(param_dict, calib_dict).copy()
#             final_dict = final_f(param_dict).copy()
#
#             for param in calib_params:
#                 if param in final_dict.keys():
#                     calib_dict[param] = final_dict[param]
#                     del final_dict[param]
#
#             var_dict_final = {}
#             for key in var_dict:
#                 if key in ss_vars:
#                     var_dict_final[key] = var_dict[key]
#
#             ss_out = ss_out | var_dict_final | final_dict
#
#             return ss_out.sort_keys(), calib_dict.sort_keys()
#
#         return combined_function
#
#     def _bundle_symbolic_solutions_with_optimizer_solutions(
#         self,
#         unknowns: List[str],
#         f: Callable,
#         f_jac: Optional[Callable],
#         param_dict: Dict[str, float],
#         symbolic_solutions: Optional[Dict[str, float]],
#         n_eqs: int,
#         output_names: List[str],
#         param_bounds: Optional[Dict[str, Tuple[float, float]]],
#         optimizer_kwargs: Optional[Dict[str, Any]],
#     ) -> Callable:
#
#         parameters = list(param_dict.keys())
#
#         optimize_wrapper = partial(
#             self._optimize_dispatcher,
#             unknowns=unknowns,
#             f=f,
#             f_jac=f_jac,
#             n_eqs=n_eqs,
#             param_bounds=param_bounds,
#             optimizer_kwargs=optimizer_kwargs,
#         )
#         _symbolic_lambda = sp.lambdify(parameters, list(symbolic_solutions.values()))
#
#         def solve_optimizer_variables(param_dict):
#             return SymbolDictionary(dict(zip(output_names, optimize_wrapper(param_dict))))
#
#         def solve_symbolic_variables(param_dict):
#             return SymbolDictionary(dict(zip(symbolic_solutions.keys(), _symbolic_lambda(**param_dict))))
#
#         wrapped_f = merge_functions(
#             [solve_optimizer_variables, solve_symbolic_variables], param_dict
#         )
#
#         return wrapped_f
#
#     def _optimize_dispatcher(
#         self, param_dict, unknowns, f, f_jac, n_eqs, param_bounds, optimizer_kwargs
#     ):
#         if n_eqs == 1:
#             optimize_fun = optimize.root_scalar
#             if param_bounds is None:
#                 param_bounds = self._prepare_param_bounds(None, 1)[0]
#             optimizer_kwargs = self._prepare_optimizer_kwargs(optimizer_kwargs, n_eqs)
#             optimizer_kwargs.update(
#                 dict(args=param_dict, method="brentq", bracket=param_bounds)
#             )
#
#         else:
#             optimize_fun = optimize.root
#
#             optimizer_kwargs = self._prepare_optimizer_kwargs(optimizer_kwargs, n_eqs)
#             optimizer_kwargs.update(dict(args=param_dict, jac=f_jac))
#
#         with catch_warnings():
#             simplefilter("ignore")
#             result = optimize_fun(f, **optimizer_kwargs)
#
#         if hasattr(result, "converged") and result.converged:
#             return np.atleast_1d(result.root)
#         elif hasattr(result, "converged") and not result.converged:
#             raise ValueError(
#                 f"Optimization failed while solving for steady state solution of the following "
#                 f'variables: {", ".join([symbol_to_string(x) for x in unknowns])}\n\n {result}'
#             )
#
#         if hasattr(result, "success") and result.success:
#             return result.x
#
#         elif hasattr(result, "success") and not result.success:
#             raise ValueError(
#                 f"Optimization failed while solving for steady state solution of the following "
#                 f'variables: {", ".join([symbol_to_string(x) for x in unknowns])}\n\n {result}'
#             )
#
#     @staticmethod
#     def _build_jacobian(
#         diff_variables: List[Union[str, VariableType]],
#         additional_inputs: List[Union[str, VariableType]],
#         equations: List[sp.Add],
#     ) -> Callable:
#         """
#         Parameters
#         ----------
#         diff_variables: list
#             A list of variables, as either TimeAwareSymbols or strings that the equations will be differentiated with
#             respect to.
#         additional_inputs: list
#             A list of variables or parameters that will be arguments to the Jacobian function, but that will NOT
#             be used in differentiation (i.e. the model parameters)
#         equations: list
#             A list of equations to be differentiated
#
#         Returns
#         -------
#         f_jac: Callable
#             A function that takes diff_variables + additional_inputs as keyword arguments and returns an
#             len(equations) x len(diff_variables) matrix of derivatives.
#         """
#         equations = np.atleast_1d(equations)
#         sp_variables = [safe_string_to_sympy(x) for x in diff_variables]
#         _f_jac = sp.lambdify(
#             diff_variables + additional_inputs,
#             [[eq.diff(x) for x in sp_variables] for eq in equations],
#         )
#
#         def f_jac(args, kwargs):
#             return np.array(_f_jac(*args, **kwargs))
#
#         return f_jac
#
#     @staticmethod
#     def _prepare_optimizer_kwargs(
#         optimizer_kwargs: Optional[Dict[str, Any]], n_unknowns: int
#     ) -> Dict[str, Any]:
#         if optimizer_kwargs is None:
#             optimizer_kwargs = {}
#
#         arg_names = list(optimizer_kwargs.keys())
#         if "x0" not in arg_names:
#             optimizer_kwargs["x0"] = np.full(n_unknowns, 0.8)
#         if "method" not in arg_names:
#             optimizer_kwargs["method"] = "hybr"
#
#         return optimizer_kwargs
#
#     @staticmethod
#     def _prepare_param_bounds(
#         param_bounds: Optional[List[Tuple[float, float]]], n_params
#     ) -> List[Tuple[float, float]]:
#         if param_bounds is None:
#             bounds = [(1e-4, 0.999) for _ in range(n_params)]
#         else:
#             bounds = [(lower + 1e-4, upper - 1e-4) for lower, upper in param_bounds]
#
#         return bounds
#
#     def _get_n_unknowns_in_eq(self, eq: sp.Add) -> int:
#         params_to_calibrate = (
#             [] if self.params_to_calibrate is None else self.params_to_calibrate
#         )
#         unknown_atoms = [
#             x for x in eq.atoms() if is_variable(x) or x in params_to_calibrate
#         ]
#         n_unknowns = len(list(set(unknown_atoms)))
#
#         return n_unknowns
#
#     def heuristic_solver(
#         self,
#         solution_dict: Dict[str, float],
#         subbed_ss_system: List[Any],
#         steady_state_system: List[Any],
#         unknowns: List[str],
#     ) -> Tuple[Dict[str, float], ArrayLike]:
#         """
#         Parameters
#         ----------
#         solution_dict: dict
#             A dictionary of TimeAwareSymbol: float pairs, giving steady-state values that have already been determined
#
#         subbed_ss_system: list
#             A list containing all unsolved steady state equations, pre-substituted with parameter values and known
#             steady-state values.
#
#         steady_state_system: list
#             A list containing all steady state equations, without substitution
#
#         unknowns: list
#             A list of sympy variables containing unknown values to solve for; variables plus any unsolved calibrated
#             parameters.
#
#         Returns
#         -------
#         It is likely that the GCN model will contain simple equations that amount to little more than parameters, for
#         example declaring that P = 1 in a perfect competition setup. These types of simple expressions can be "solved"
#         and removed from the system to reduce the dimensionality of the problem given to the numerical solver.
#
#         This function performs this simplification in a heuristic way in the following manner. We first look for
#         "simple" equations, defined as those with only a single unknown variable. Solutions are then substituted back
#         into the system, equations that have reduced to 0=0 as a result of substitution are removed, then we repeat
#         the procedure to see if any additional equations have become heuristically solvable as a result of substitution.
#
#         The process terminates when no "simple" equations remain.
#         """
#
#         solved_mask = np.array([eq == 0 for eq in subbed_ss_system])
#         eq_to_var_dict = {}
#         check_again_mask = np.full_like(solved_mask, True)
#         solution_dict = sequential(
#             solution_dict, [float_values_to_sympy_float, string_keys_to_sympy]
#         )
#
#         numeric_solutions = solution_dict.copy()
#
#         while True:
#             solution_dict = {
#                 key: eq.subs(solution_dict) for key, eq in solution_dict.items()
#             }
#             subbed_ss_system = [
#                 eq.subs(numeric_solutions).simplify() for eq in subbed_ss_system
#             ]
#
#             n_unknowns = np.array(
#                 [self._get_n_unknowns_in_eq(eq) for eq in subbed_ss_system]
#             )
#             eq_len = np.array([len(eq.atoms()) for eq in subbed_ss_system])
#
#             solvable_mask = (n_unknowns < 2) & (~solved_mask) & check_again_mask
#
#             # Sympy struggles with solving complicated functions inside powers, just avoid them. 5 is a magic number
#             # for the maximum number of variable in a function to be considered "complicated", needs tuning.
#             has_power_argument = np.array(
#                 [
#                     any([isinstance(arg, sp.core.power.Pow)] for arg in eq.args)
#                     for eq in subbed_ss_system
#                 ]
#             )
#             solvable_mask &= ~(has_power_argument & (eq_len > 5))
#
#             if sum(solvable_mask) == 0:
#                 break
#
#             for idx in np.flatnonzero(solvable_mask):
#                 # Putting the solved = True flag here is ugly, but it catches equations
#                 # that are 0 = 0 after substitution
#                 solved_mask[idx] = True
#
#                 eq = subbed_ss_system[idx]
#
#                 variables = list({x for x in eq.atoms() if x in unknowns})
#                 if len(variables) > 0:
#                     eq_to_var_dict[variables[0]] = idx
#
#                     try:
#                         symbolic_solution = sp.solve(
#                             steady_state_system[idx], variables[0]
#                         )
#                     except NotImplementedError:
#                         # There are functional forms sympy can't handle;  mark the equation as unsolvable and continue.
#                         check_again_mask[idx] = False
#                         solved_mask[idx] = False
#                         continue
#
#                     # The solution should only ever be length 0 or 1, if it's more than 1 something went wrong. Haven't
#                     # hit this case yet in testing.
#                     if len(symbolic_solution) == 1:
#                         solution_dict[variables[0]] = symbolic_solution[0]
#                         numeric_solutions[variables[0]] = (
#                             symbolic_solution[0]
#                             .subs(self.free_param_dict)
#                             .subs(numeric_solutions)
#                         )
#                         check_again_mask[:] = True
#                         solved_mask[idx] = True
#
#                     else:
#                         # Solver failed; something went wrong. Skip this equation.
#                         solved_mask[idx] = False
#                         check_again_mask[idx] = False
#
#                 else:
#                     check_again_mask[idx] = False
#
#         numeric_solutions = sympy_number_values_to_floats(numeric_solutions)
#         for key, eq in numeric_solutions.items():
#             if not isinstance(eq, float):
#                 del solution_dict[key]
#                 solved_mask[eq_to_var_dict[key]] = False
#
#         solution_dict = sequential(
#             solution_dict, [sympy_keys_to_strings, sympy_number_values_to_floats]
#         )
#
#         return solution_dict, solved_mask
