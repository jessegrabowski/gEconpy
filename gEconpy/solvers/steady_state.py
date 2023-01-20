from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import catch_warnings, simplefilter

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike
from scipy import optimize

from gEconpy.shared.typing import VariableType
from gEconpy.shared.utilities import (
    float_values_to_sympy_float,
    is_variable,
    merge_dictionaries,
    merge_functions,
    safe_string_to_sympy,
    sequential,
    sort_dictionary,
    string_keys_to_sympy,
    substitute_all_equations,
    symbol_to_string,
    sympy_keys_to_strings,
    sympy_number_values_to_floats,
)


class SteadyStateSolver:
    def __init__(self, model):

        self.variables: List[VariableType] = model.variables
        self.shocks: List[sp.Add] = model.shocks

        self.n_variables: int = model.n_variables

        self.free_param_dict: Dict[str, float] = model.free_param_dict
        self.params_to_calibrate: List[VariableType] = model.params_to_calibrate
        self.calibrating_equations: List[sp.Add] = model.calibrating_equations
        self.system_equations: List[sp.Add] = model.system_equations
        self.steady_state_relationships: Dict[
            str, Union[float, sp.Add]
        ] = model.steady_state_relationships

        self.steady_state_system: List[sp.Add] = []
        self.steady_state_dict: Dict[str, float] = {}
        self.steady_state_solved: bool = False

        self.f_calib_params: Callable = lambda *args, **kwargs: {}
        self.f_ss_resid: Callable = lambda *args, **kwargs: np.inf
        self.f_ss: Callable = lambda *args, **kwargs: np.inf

        self.build_steady_state_system()

    def build_steady_state_system(self):
        self.steady_state_system = []

        all_atoms = [
            x for eq in self.system_equations for x in eq.atoms() if is_variable(x)
        ]
        all_variables = set(all_atoms) - set(self.shocks)
        ss_sub_dict = {variable: variable.to_ss() for variable in set(all_variables)}
        unique_ss_variables = list(set(list(ss_sub_dict.values())))

        steady_state_dict = sequential(
            dict(zip(unique_ss_variables, [None] * self.n_variables)),
            [sympy_keys_to_strings, sort_dictionary],
        )

        self.steady_state_dict = steady_state_dict

        for shock in self.shocks:
            ss_sub_dict[shock] = 0

        for eq in self.system_equations:
            self.steady_state_system.append(eq.subs(ss_sub_dict))

    def solve_steady_state(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_jac: Optional[bool] = False,
    ) -> Callable:
        """

        Parameters
        ----------
        param_bounds: dict
            A dictionary of string, tuple(float, float) pairs, giving bounds for each variable or parameter to be
            solved for. Only used by certain optimizers; check the scipy docs. Pass it here instead of in
            optimizer_kwargs to make sure the correct variables have the correct bounds.
        optimizer_kwargs: dict
            A dictionary of keyword arguments to pass to the scipy optimizer, either root or root_scalar.
        use_jac: bool
            A flag to symbolically compute the Jacobain function of the model before optimization, can help the solver
            on complex problems.

        Returns
        -------
        f_ss: Callable
            A function that maps a dictionary of parameters to steady state values for all system variables and
            calibrated parameters.

        Solving of the steady state proceeds in three steps: solve calibrating equations (if any), gather user provided
        equations into a function, then solve the remaining equations.

        Calibrating equations are handled first because if the user passed a complete steady state solution, it is
        unlikely to include solutions for calibrating equations. Calibrating equations are then combined with
        user supplied equations, and we check if everything necessary to solve the model is now present. If not,
        a final optimizer step runs to solve for the remaining variables.

        Note that no checks are done in this function to validate the steady state solution. If a user supplies an
        incorrect steady state, this function will not catch it. It will, however, still fail if an optimizer fails
        to find a solution.
        """
        free_param_dict = self.free_param_dict.copy()
        parameters = list(free_param_dict.keys())
        variables = list(self.steady_state_dict.keys())

        params_to_calibrate = [symbol_to_string(x) for x in self.params_to_calibrate]

        n_to_calibrate = len(params_to_calibrate)
        has_calibrating_equations = n_to_calibrate > 0

        params_and_variables = parameters + params_to_calibrate + variables
        steady_state_system = self.steady_state_system

        # TODO: Move the creation of this residual function somewhere more logical
        self.f_ss_resid = sp.lambdify(params_and_variables, steady_state_system)

        # Solve calibrating equations, if any.
        if has_calibrating_equations:
            f_calib, additional_solutions = self._solve_calibrating_equations(
                param_bounds=param_bounds,
                optimizer_kwargs=optimizer_kwargs,
                use_jac=use_jac,
            )
        else:
            f_calib = lambda *args, **kwargs: {}
            additional_solutions = {}

        solved_calib_params = list(f_calib(free_param_dict).keys())

        # Gather user provided steady state solutions
        f_provided = self._gather_provided_solutions(solved_calib_params)

        calib_dict = f_calib(free_param_dict)
        var_dict = f_provided(free_param_dict, calib_dict)

        # If we have everything we're done. We don't need to use final_f, set it to return an empty dictionary.
        if (
            set(params_and_variables) - set(var_dict.keys()).union(calib_dict.keys())
        ) == set(free_param_dict.keys()):
            f_ss = self._create_final_function(
                final_f=lambda x: {}, f_calib=f_calib, f_provided=f_provided
            )

        else:
            final_f = self._solve_remaining_equations(
                calib_dict=calib_dict,
                var_dict=var_dict,
                additional_solutions=additional_solutions,
                param_bounds=param_bounds,
                optimizer_kwargs=optimizer_kwargs,
                use_jac=use_jac,
            )
            f_ss = self._create_final_function(
                final_f=final_f, f_calib=f_calib, f_provided=f_provided
            )

        return f_ss

    def _solve_calibrating_equations(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]],
        optimizer_kwargs: Optional[Dict[str, Any]],
        use_jac: bool = False,
    ) -> Tuple[Callable, Dict]:
        """
        Parameters
        ----------
        param_bounds: dict
            See docstring of solve_steady_state for details
        optimizer_kwargs: dict
            See docstring of solve_steady_state for details
        use_jac: bool
            See docstring of solve_steady_state for details

        Returns
        -------
        f_calib: callable
            A function that maps param_dict to values of calibrated parameteres
        additional_solutions: dict
            A dictionary of symbolic solutions to non-calibrating parameters that were solved en passant and can be
            reused later
        """
        calibrating_equations = self.calibrating_equations
        symbolic_solutions = self.steady_state_relationships.copy()
        free_param_dict = self.free_param_dict.copy()
        steady_state_system = self.steady_state_system

        parameters = list(free_param_dict.keys())
        variables = list(self.steady_state_dict.keys())
        params_to_calibrate = [symbol_to_string(x) for x in self.params_to_calibrate]
        params_and_variables = parameters + params_to_calibrate + variables

        unknown_variables = set(variables).union(set(params_to_calibrate)) - set(
            symbolic_solutions.keys()
        )

        n_to_calibrate = len(params_to_calibrate)

        additional_solutions = {}

        # Make substitutions
        calib_with_user_solutions = substitute_all_equations(
            calibrating_equations, symbolic_solutions
        )

        # Try the heuristic solver
        calib_solutions, solved_mask = self.heuristic_solver(
            {},
            calib_with_user_solutions,
            calib_with_user_solutions,
            [safe_string_to_sympy(x) for x in params_and_variables],
        )

        # Case 1: We found something! Refine the solution.
        if solved_mask.sum() > 0:
            # If the heuristic solver worked, we got solutions for variables that will allow us to go back and solve for
            # the calibrating parameters.

            sub_dict = merge_dictionaries(free_param_dict, calib_solutions)
            more_solutions, solved_mask = self.heuristic_solver(
                sub_dict,
                substitute_all_equations(steady_state_system, sub_dict),
                steady_state_system,
                [safe_string_to_sympy(x) for x in params_and_variables],
            )

            calib_solutions = {
                key: value
                for key, value in more_solutions.items()
                if (key in params_to_calibrate)
            }

            # We potentially pick up additional solutions from this heuristic pass, we can save them and use them later
            # to help the heuristic solver later.
            additional_solutions = {
                key: value
                for key, value in more_solutions.items()
                if (key not in params_to_calibrate) and (key not in free_param_dict)
            }

            calib_solutions = sequential(
                calib_solutions,
                [sympy_number_values_to_floats, sympy_keys_to_strings, sort_dictionary],
            )
            f_calib = lambda *args, **kwargs: calib_solutions

        # Case 2: Found nothing, try to use an optimizer
        else:
            # Here we check how many equations are remaining to solve after accounting for the user's SS info.
            # We're looking for the case when all information is given EXCEPT the calibrating parameters.
            # If there is more than that, we handle it in the final pass.
            calib_remaining_to_solve = list(
                set(unknown_variables) - set(symbolic_solutions.keys())
            )
            calib_n_eqs = len(calib_remaining_to_solve)
            if calib_n_eqs > len(calibrating_equations):

                def f_calib(*args, **kwargs):
                    return {}

                return f_calib, {}

            # TODO: Is there a more elegant way to handle one equation vs many equations here?
            if calib_n_eqs == 1:
                calib_with_user_solutions = calib_with_user_solutions[0]

                _f_calib = sp.lambdify(
                    calib_remaining_to_solve + parameters, calib_with_user_solutions
                )

                def f_calib(x, kwargs):
                    return _f_calib(x, **kwargs)

            else:
                _f_calib = sp.lambdify(
                    calib_remaining_to_solve + parameters, calib_with_user_solutions
                )

                def f_calib(args, kwargs):
                    return _f_calib(*args, **kwargs)

            f_jac = None
            if use_jac:
                f_jac = self._build_jacobian(
                    diff_variables=calib_remaining_to_solve,
                    additional_inputs=parameters,
                    equations=calib_with_user_solutions,
                )

            f_calib = self._bundle_symbolic_solutions_with_optimizer_solutions(
                unknowns=calib_remaining_to_solve,
                f=f_calib,
                f_jac=f_jac,
                param_dict=free_param_dict,
                symbolic_solutions=calib_solutions,
                n_eqs=calib_n_eqs,
                output_names=calib_remaining_to_solve,
                param_bounds=param_bounds,
                optimizer_kwargs=optimizer_kwargs,
            )

        return f_calib, additional_solutions

    def _gather_provided_solutions(self, solved_calib_params) -> Callable:
        """
        Returns
        -------
        f_provided: Callable
            A function that takes model parameters, both calibrated and otherwise, as keywork arguments, and returns
            a dictionary of variable values according to steady state equations supplied by the user
        """

        free_param_dict = self.free_param_dict.copy()
        symbolic_solutions = self.steady_state_relationships.copy()
        parameters = list(free_param_dict.keys())

        _provided_lambda = sp.lambdify(
            parameters + solved_calib_params, [eq for eq in symbolic_solutions.values()]
        )

        def f_provided(param_dict, calib_dict):
            return dict(
                zip(
                    symbolic_solutions.keys(),
                    _provided_lambda(**param_dict, **calib_dict),
                )
            )

        return f_provided

    def _solve_remaining_equations(
        self,
        calib_dict: Dict[str, float],
        var_dict: Dict[str, float],
        additional_solutions: Dict[str, float],
        param_bounds: Optional[Dict[str, Tuple[float, float]]],
        optimizer_kwargs: Optional[Dict[str, Any]],
        use_jac: bool,
    ) -> Callable:
        """
        Parameters
        ----------
        calib_dict: Dict
            A dictionary of solved calibrating parameters, if any.
        var_dict: Dict
            A dictionary of user-provided steady-state relationships, if any.
        additional_solutions:
            A dictionary of variable solutions found en passant by the heuristic solver while solving for the
            calibrated parameters, if any.
        param_bounds:
            See docstring of solve_steady_state for details
        optimizer_kwargs:
            See docstring of solve_steady_state for details
        use_jac:
            See docstring of solve_steady_state for details

        Returns
        -------
        f_final: Callable
            A function that takes model parameters as keyword arguments and returns steady-state values for each
            model variable without an explicit symbolic solution.
        """
        free_param_dict = self.free_param_dict
        steady_state_system = self.steady_state_system
        calibrating_equations = self.calibrating_equations

        parameters = list(free_param_dict.keys())
        variables = list(self.steady_state_dict.keys())
        params_to_calibrate = [symbol_to_string(x) for x in self.params_to_calibrate]

        sub_dict = merge_dictionaries(calib_dict, var_dict, additional_solutions)
        params_and_variables = parameters + params_to_calibrate + variables

        ss_solutions, solved_mask = self.heuristic_solver(
            sub_dict,
            substitute_all_equations(
                steady_state_system + calibrating_equations, sub_dict, free_param_dict
            ),
            steady_state_system + calibrating_equations,
            [safe_string_to_sympy(x) for x in params_and_variables],
        )

        ss_solutions = {
            key: value
            for key, value in ss_solutions.items()
            if key not in calib_dict.keys()
        }
        sub_dict.update(ss_solutions)

        ss_remaining_to_solve = sorted(
            list(
                set(variables + params_to_calibrate)
                - set(ss_solutions.keys())
                - set(calib_dict.keys())
            )
        )

        unsolved_eqs = substitute_all_equations(
            [
                eq
                for idx, eq in enumerate(steady_state_system + calibrating_equations)
                if not solved_mask[idx]
            ],
            sub_dict,
        )

        n_eqs = len(unsolved_eqs)

        _f_unsolved_ss = sp.lambdify(ss_remaining_to_solve + parameters, unsolved_eqs)

        def f_unsolved_ss(args, kwargs):
            return _f_unsolved_ss(*args, **kwargs)

        f_jac = None
        if use_jac:
            f_jac = self._build_jacobian(
                diff_variables=ss_remaining_to_solve,
                additional_inputs=parameters,
                equations=unsolved_eqs,
            )

        f_final = self._bundle_symbolic_solutions_with_optimizer_solutions(
            unknowns=ss_remaining_to_solve,
            f=f_unsolved_ss,
            f_jac=f_jac,
            param_dict=free_param_dict,
            symbolic_solutions=ss_solutions,
            n_eqs=n_eqs,
            output_names=ss_remaining_to_solve,
            param_bounds=param_bounds,
            optimizer_kwargs=optimizer_kwargs,
        )

        return f_final

    def _create_final_function(self, final_f, f_calib, f_provided):
        """

        Parameters
        ----------
        final_f: Callable
            Function generated by solve_remaining_equations
        f_calib: Callable
            Function generated by _solve_calibrating_equations
        f_provided: Callable
            Function generated by _gather_provided_solutions

        Returns
        -------
        f_ss: Callable
            A single function wrapping the three steady state functions, that returns a complete solution to the
            model's steady state as two dictionaries: one with variable values, and one with calibrated parameter
            values.
        """
        calib_params = [x.name for x in self.params_to_calibrate]
        ss_vars = [x.to_ss().name for x in self.variables]

        def combined_function(param_dict):
            ss_out = {}

            calib_dict = f_calib(param_dict).copy()
            var_dict = f_provided(param_dict, calib_dict).copy()
            final_dict = final_f(param_dict).copy()

            for param in calib_params:
                if param in final_dict.keys():
                    calib_dict[param] = final_dict[param]
                    del final_dict[param]

            var_dict_final = {}
            for key in var_dict:
                if key in ss_vars:
                    var_dict_final[key] = var_dict[key]

            ss_out.update(var_dict_final)
            ss_out.update(final_dict)

            return sort_dictionary(ss_out), sort_dictionary(calib_dict)

        return combined_function

    def _bundle_symbolic_solutions_with_optimizer_solutions(
        self,
        unknowns: List[str],
        f: Callable,
        f_jac: Optional[Callable],
        param_dict: Dict[str, float],
        symbolic_solutions: Optional[Dict[str, float]],
        n_eqs: int,
        output_names: List[str],
        param_bounds: Optional[Dict[str, Tuple[float, float]]],
        optimizer_kwargs: Optional[Dict[str, Any]],
    ) -> Callable:

        parameters = list(param_dict.keys())

        optimize_wrapper = partial(
            self._optimize_dispatcher,
            unknowns=unknowns,
            f=f,
            f_jac=f_jac,
            n_eqs=n_eqs,
            param_bounds=param_bounds,
            optimizer_kwargs=optimizer_kwargs,
        )
        _symbolic_lambda = sp.lambdify(parameters, list(symbolic_solutions.values()))

        def solve_optimizer_variables(param_dict):
            return dict(zip(output_names, optimize_wrapper(param_dict)))

        def solve_symbolic_variables(param_dict):
            return dict(zip(symbolic_solutions.keys(), _symbolic_lambda(**param_dict)))

        wrapped_f = merge_functions(
            [solve_optimizer_variables, solve_symbolic_variables], param_dict
        )

        return wrapped_f

    def _optimize_dispatcher(
        self, param_dict, unknowns, f, f_jac, n_eqs, param_bounds, optimizer_kwargs
    ):
        if n_eqs == 1:
            optimize_fun = optimize.root_scalar
            if param_bounds is None:
                param_bounds = self._prepare_param_bounds(None, 1)[0]
            optimizer_kwargs = self._prepare_optimizer_kwargs(optimizer_kwargs, n_eqs)
            optimizer_kwargs.update(
                dict(args=param_dict, method="brentq", bracket=param_bounds)
            )

        else:
            optimize_fun = optimize.root

            optimizer_kwargs = self._prepare_optimizer_kwargs(optimizer_kwargs, n_eqs)
            optimizer_kwargs.update(dict(args=param_dict, jac=f_jac))

        with catch_warnings():
            simplefilter("ignore")
            result = optimize_fun(f, **optimizer_kwargs)

        if hasattr(result, "converged") and result.converged:
            return np.atleast_1d(result.root)
        elif hasattr(result, "converged") and not result.converged:
            raise ValueError(
                f"Optimization failed while solving for steady state solution of the following "
                f'variables: {", ".join([symbol_to_string(x) for x in unknowns])}\n\n {result}'
            )

        if hasattr(result, "success") and result.success:
            return result.x

        elif hasattr(result, "success") and not result.success:
            raise ValueError(
                f"Optimization failed while solving for steady state solution of the following "
                f'variables: {", ".join([symbol_to_string(x) for x in unknowns])}\n\n {result}'
            )

    @staticmethod
    def _build_jacobian(
        diff_variables: List[Union[str, VariableType]],
        additional_inputs: List[Union[str, VariableType]],
        equations: List[sp.Add],
    ) -> Callable:
        """
        Parameters
        ----------
        diff_variables: list
            A list of variables, as either TimeAwareSymbols or strings that the equations will be differentiated with
            respect to.
        additional_inputs: list
            A list of variables or parameters that will be arguments to the Jacobian function, but that will NOT
            be used in differentiation (i.e. the model parameters)
        equations: list
            A list of equations to be differentiated

        Returns
        -------
        f_jac: Callable
            A function that takes diff_variables + additional_inputs as keyword arguments and returns an
            len(equations) x len(diff_variables) matrix of derivatives.
        """
        equations = np.atleast_1d(equations)
        sp_variables = [safe_string_to_sympy(x) for x in diff_variables]
        _f_jac = sp.lambdify(
            diff_variables + additional_inputs,
            [[eq.diff(x) for x in sp_variables] for eq in equations],
        )

        def f_jac(args, kwargs):
            return np.array(_f_jac(*args, **kwargs))

        return f_jac

    @staticmethod
    def _prepare_optimizer_kwargs(
        optimizer_kwargs: Optional[Dict[str, Any]], n_unknowns: int
    ) -> Dict[str, Any]:
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        arg_names = list(optimizer_kwargs.keys())
        if "x0" not in arg_names:
            optimizer_kwargs["x0"] = np.full(n_unknowns, 0.8)
        if "method" not in arg_names:
            optimizer_kwargs["method"] = "hybr"

        return optimizer_kwargs

    @staticmethod
    def _prepare_param_bounds(
        param_bounds: Optional[List[Tuple[float, float]]], n_params
    ) -> List[Tuple[float, float]]:
        if param_bounds is None:
            bounds = [(1e-4, 0.999) for _ in range(n_params)]
        else:
            bounds = [(lower + 1e-4, upper - 1e-4) for lower, upper in param_bounds]

        return bounds

    def _get_n_unknowns_in_eq(self, eq: sp.Add) -> int:
        params_to_calibrate = (
            [] if self.params_to_calibrate is None else self.params_to_calibrate
        )
        unknown_atoms = [
            x for x in eq.atoms() if is_variable(x) or x in params_to_calibrate
        ]
        n_unknowns = len(list(set(unknown_atoms)))

        return n_unknowns

    def heuristic_solver(
        self,
        solution_dict: Dict[str, float],
        subbed_ss_system: List[Any],
        steady_state_system: List[Any],
        unknowns: List[str],
    ) -> Tuple[Dict[str, float], ArrayLike]:
        """
        Parameters
        ----------
        solution_dict: dict
            A dictionary of TimeAwareSymbol: float pairs, giving steady-state values that have already been determined

        subbed_ss_system: list
            A list containing all unsolved steady state equations, pre-substituted with parameter values and known
            steady-state values.

        steady_state_system: list
            A list containing all steady state equations, without substitution

        unknowns: list
            A list of sympy variables containing unknown values to solve for; variables plus any unsolved calibrated
            parameters.

        Returns
        -------
        It is likely that the GCN model will contain simple equations that amount to little more than parameters, for
        example declaring that P = 1 in a perfect competition setup. These types of simple expressions can be "solved"
        and removed from the system to reduce the dimensionality of the problem given to the numerical solver.

        This function performs this simplification in a heuristic way in the following manner. We first look for
        "simple" equations, defined as those with only a single unknown variable. Solutions are then substituted back
        into the system, equations that have reduced to 0=0 as a result of substitution are removed, then we repeat
        the procedure to see if any additional equations have become heuristically solvable as a result of substitution.

        The process terminates when no "simple" equations remain.
        """

        solved_mask = np.array([eq == 0 for eq in subbed_ss_system])
        eq_to_var_dict = {}
        check_again_mask = np.full_like(solved_mask, True)
        solution_dict = sequential(
            solution_dict, [float_values_to_sympy_float, string_keys_to_sympy]
        )

        numeric_solutions = solution_dict.copy()

        while True:
            solution_dict = {
                key: eq.subs(solution_dict) for key, eq in solution_dict.items()
            }
            subbed_ss_system = [
                eq.subs(numeric_solutions).simplify() for eq in subbed_ss_system
            ]

            n_unknowns = np.array(
                [self._get_n_unknowns_in_eq(eq) for eq in subbed_ss_system]
            )
            eq_len = np.array([len(eq.atoms()) for eq in subbed_ss_system])

            solvable_mask = (n_unknowns < 2) & (~solved_mask) & check_again_mask

            # Sympy struggles with solving complicated functions inside powers, just avoid them. 5 is a magic number
            # for the maximum number of variable in a function to be considered "complicated", needs tuning.
            has_power_argument = np.array(
                [
                    any([isinstance(arg, sp.core.power.Pow)] for arg in eq.args)
                    for eq in subbed_ss_system
                ]
            )
            solvable_mask &= ~(has_power_argument & (eq_len > 5))

            if sum(solvable_mask) == 0:
                break

            for idx in np.flatnonzero(solvable_mask):
                # Putting the solved = True flag here is ugly, but it catches equations
                # that are 0 = 0 after substitution
                solved_mask[idx] = True

                eq = subbed_ss_system[idx]

                variables = list({x for x in eq.atoms() if x in unknowns})
                if len(variables) > 0:
                    eq_to_var_dict[variables[0]] = idx

                    try:
                        symbolic_solution = sp.solve(
                            steady_state_system[idx], variables[0]
                        )
                    except NotImplementedError:
                        # There are functional forms sympy can't handle;  mark the equation as unsolvable and continue.
                        check_again_mask[idx] = False
                        solved_mask[idx] = False
                        continue

                    # The solution should only ever be length 0 or 1, if it's more than 1 something went wrong. Haven't
                    # hit this case yet in testing.
                    if len(symbolic_solution) == 1:
                        solution_dict[variables[0]] = symbolic_solution[0]
                        numeric_solutions[variables[0]] = (
                            symbolic_solution[0]
                            .subs(self.free_param_dict)
                            .subs(numeric_solutions)
                        )
                        check_again_mask[:] = True
                        solved_mask[idx] = True

                    else:
                        # Solver failed; something went wrong. Skip this equation.
                        solved_mask[idx] = False
                        check_again_mask[idx] = False

                else:
                    check_again_mask[idx] = False

        numeric_solutions = sympy_number_values_to_floats(numeric_solutions)
        for key, eq in numeric_solutions.items():
            if not isinstance(eq, float):
                del solution_dict[key]
                solved_mask[eq_to_var_dict[key]] = False

        solution_dict = sequential(
            solution_dict, [sympy_keys_to_strings, sympy_number_values_to_floats]
        )

        return solution_dict, solved_mask
