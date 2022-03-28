from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from gEcon.classes.time_aware_symbol import TimeAwareSymbol
from gEcon.shared.types import VariableType
from gEcon.shared.utilities import sympy_number_values_to_floats, is_variable, sort_dictionary, \
    sympy_keys_to_strings, sequential, safe_string_to_sympy, select_keys, string_keys_to_sympy, merge_dicts, \
    symbol_to_string

from scipy import optimize
from functools import partial
from warnings import catch_warnings, simplefilter
import sympy as sp
import numpy as np


class SteadyStateSolver:

    def __init__(self, model):

        self.variables: List[VariableType] = model.variables
        self.shocks: List[sp.Add] = model.shocks

        self.n_variables: int = model.n_variables

        self.param_dict: Dict[str, float] = model.param_dict
        self.params_to_calibrate: List[VariableType] = model.params_to_calibrate
        self.calibrating_equations: List[sp.Add] = model.calibrating_equations
        self.system_equations: List[sp.Add] = model.system_equations
        self.steady_state_relationships: Dict[str, Union[float, sp.Add]] = model.steady_state_relationships

        self.steady_state_system: List[sp.Add] = []
        self.steady_state_dict: Dict[str, float] = {}
        self.steady_state_solved: bool = False

        self.f_calib_params: Callable = lambda *args, **kwargs: {}
        self.f_ss_resid: Callable = lambda *args, **kwargs: np.inf
        self.f_ss: Callable = lambda *args, **kwargs: np.inf

        self.build_steady_state_system()

    def build_steady_state_system(self):
        self.steady_state_system = []

        ss_sub_dict = {}
        for variable in self.variables:
            ss_sub_dict[variable] = variable.to_ss()
            ss_sub_dict[variable.step_backward()] = variable.to_ss()
            ss_sub_dict[variable.step_forward()] = variable.to_ss()

        unique_ss_variables = list(set(list(ss_sub_dict.values())))
        steady_state_dict = sequential(dict(zip(unique_ss_variables, [None] * self.n_variables)),
                                       [sympy_keys_to_strings, sort_dictionary])

        self.steady_state_dict = steady_state_dict

        for shock in self.shocks:
            ss_sub_dict[shock] = 0

        for eq in self.system_equations:
            self.steady_state_system.append(eq.subs(ss_sub_dict))

    def solve_steady_state(self, param_bounds, optimizer_kwargs, use_jac=True) -> None:
        param_dict = self.param_dict.copy()
        params_to_calibrate = [] if self.params_to_calibrate is [] else [symbol_to_string(x) for x in
                                                                         self.params_to_calibrate]

        n_to_calibrate = len(params_to_calibrate)
        calib_eqs = self.calibrating_equations

        steady_state_system = self.steady_state_system
        user_supplied_dict = self.steady_state_relationships

        params_and_variables = list(param_dict.keys()) + params_to_calibrate + list(self.steady_state_dict.keys())

        self.f_ss_resid = sp.lambdify(params_and_variables, steady_state_system)

        unknowns = list(set(self.steady_state_dict.keys()) - set(user_supplied_dict.keys()))

        all_ss_relationships_provided = len(unknowns) == 0
        has_calibrating_equations = n_to_calibrate > 0

        # First off, if we have everything from the user, save it as a function.
        if all_ss_relationships_provided:
            f_variables = sp.lambdify(list(param_dict) + params_to_calibrate,
                                      [eq for eq in user_supplied_dict.values()])

            # If there are are no calibrating equations we're done
            if not has_calibrating_equations:
                self.f_ss = f_variables

        # We might have everything from the user but still need to solve for calibrating parameters. Or, we might have
        # enough from the user to solve the calibrating parameters separately and use them as inputs to the system
        # solution.

        # If there are calibrating equations, did the user give everything we need to solve them?
        if has_calibrating_equations:
            required_variables = [atom.safe_name for eq in calib_eqs for atom in eq.atoms() if
                                  isinstance(atom, TimeAwareSymbol)]
            can_solve_calib_apart = all([variable in user_supplied_dict.keys() for variable in required_variables])

            # If so, save a function that solves for the calibrating equations given the non-calibrated equations
            if can_solve_calib_apart:
                sub_dict = sequential(user_supplied_dict, [partial(select_keys, keys=required_variables),
                                                           string_keys_to_sympy])
                subbed_calib_eqs = [eq.subs(sub_dict) for eq in calib_eqs]
                calib_solution_dict, calib_solved_mask = self.find_heuristic_ss_solutions({}, subbed_calib_eqs,
                                                                                          calib_eqs)

                # Given user definitions, all calibrated parameters are one (ss values already equal)
                if np.all(calib_solved_mask) and len(calib_solution_dict) == 0:
                    self.f_calib_params = lambda *x: dict(zip(params_to_calibrate, np.ones(n_to_calibrate)))

                else:
                    f_calib_resid = sp.lambdify(params_to_calibrate + list(param_dict.keys()), subbed_calib_eqs)

                    def f_calib_params(param_dict, param_bounds):
                        bounds = self._prepare_param_bounds(param_bounds, n_to_calibrate)

                        if n_to_calibrate == 1:
                            f = lambda obj, kwargs: f_calib_resid(obj, **kwargs)[0]
                            with catch_warnings:
                                simplefilter('ignore')
                                result = optimize.root_scalar(f, args=param_dict, bracket=bounds[0])

                            if not result.converged:
                                raise ValueError(f'Optimization failed while solving for calibrating parameters: '
                                                 f'{", ".join(params_to_calibrate)}\n\n {result}')

                            solution = result.root

                        else:
                            f = lambda obj, kwargs: np.array(f_calib_resid(obj, **kwargs)) ** 2

                            with catch_warnings():
                                simplefilter('ignore')
                                result = optimize.minimize(f,
                                                           x0=[0.5 for _ in range(n_to_calibrate)],
                                                           args=param_dict,
                                                           method='nelder-mead')
                            if not result.success:
                                raise ValueError(f'Optimization failed while solving for calibrating parameters: '
                                                 f'{", ".join(params_to_calibrate)}\n\n {result}')
                            solution = result.x

                        return dict(zip(self.params_to_calibrate, np.atleast_1d(solution)))

                    self.f_calib_params = f_calib_params

        if all_ss_relationships_provided and has_calibrating_equations:
            def solve_ss(param_dict, param_bounds=None, *args, **kwargs):
                sanitized_input = sequential(param_dict, [sympy_keys_to_strings, sympy_number_values_to_floats])
                calib_param_dict = sequential(self.f_calib_params(sanitized_input, param_bounds),
                                              [sympy_keys_to_strings, sympy_number_values_to_floats])

                sanitized_input.update(calib_param_dict)
                solution = f_variables(**sanitized_input)

                solution_dict = dict(zip(user_supplied_dict.keys(), solution))
                solution_dict.update(calib_param_dict)

                return sequential(solution_dict,
                                  [sympy_keys_to_strings, sympy_number_values_to_floats, sort_dictionary])

            self.f_ss = solve_ss

        # If its not, try to reduce the problem as much as possible.
        else:
            # If we have a solution for the calibrated parameters, use it
            calib_param_dict = self.f_calib_params(param_dict, param_bounds)

            # If not, add the calibrating equations to the steady-state system
            for param, eq in zip(params_to_calibrate, calib_eqs):
                if param not in calib_param_dict.keys():
                    steady_state_system.append(eq)

            # Transform the provided steady-state equations into a sub_dict. We will store intermediate answers in this
            # dictionary.

            solution_dict = string_keys_to_sympy(user_supplied_dict)
            solution_dict.update(calib_param_dict)
            subbed_system = [eq.subs(param_dict).subs(calib_param_dict) for eq in steady_state_system]

            unknowns = set([atom for eq in subbed_system for atom in eq.atoms() if is_variable(atom)])
            unknowns = unknowns.union({sp.Symbol(x) for x in params_to_calibrate if x not in calib_param_dict.keys()})

            solution_dict, solved_mask = self.find_heuristic_ss_solutions(solution_dict,
                                                                          subbed_system,
                                                                          steady_state_system,
                                                                          unknowns)

            knowns = set(solution_dict.keys())
            unknowns = list(unknowns - knowns)
            knowns = list(knowns)

            _f_heuristic = sp.lambdify(list(param_dict.keys()), list(solution_dict.values()))
            f_heuristic = lambda *args, **kwargs: dict(zip(solution_dict.keys(), _f_heuristic(*args, **kwargs)))

            reduced_system = [eq for eq, solved in zip(steady_state_system, solved_mask) if not solved]

            _f = sp.lambdify(unknowns + list(param_dict.keys()) + knowns, reduced_system)
            f = lambda args, kwargs: _f(*args, **kwargs)

            if use_jac:
                _f_jac = sp.lambdify(unknowns + list(param_dict.keys()) + knowns,
                                     [[eq.diff(x) for x in unknowns] for eq in reduced_system])
                f_jac = lambda args, kwargs: _f_jac(*args, **kwargs)
            else:
                f_jac = None

            def solve_ss(param_dict, optimizer_kwargs, *args, **kwargs):
                calib_param_dict = self.f_calib_params(param_dict, param_bounds)
                joint_param_dict = merge_dicts(param_dict, calib_param_dict)
                heuristic_solution = sequential(f_heuristic(**param_dict),
                                                [sympy_keys_to_strings, sympy_number_values_to_floats])
                args = merge_dicts(joint_param_dict, heuristic_solution)

                optimizer_kwargs = self._prepare_optimizer_kwargs(optimizer_kwargs, len(unknowns))

                with catch_warnings():
                    simplefilter('ignore')
                    result = optimize.root(fun=f,
                                           jac=f_jac,
                                           args=args,
                                           **optimizer_kwargs)

                if not result.success:
                    raise ValueError(f'Optimization failed while solving for steady state solution of the following '
                                     f'variables: {", ".join([symbol_to_string(x) for x in unknowns])}\n\n {result}')

                solution = sequential(dict(zip(unknowns, result.x)),
                                      [sympy_keys_to_strings, sympy_number_values_to_floats])
                solution.update(heuristic_solution)

                return sort_dictionary(solution)

            self.f_ss = solve_ss

    @staticmethod
    def _prepare_optimizer_kwargs(optimizer_kwargs: Optional[Dict[str, Any]],
                                  n_unknowns: int) -> Dict[str, Any]:
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        arg_names = list(optimizer_kwargs.keys())
        if 'x0' not in arg_names:
            optimizer_kwargs['x0'] = np.full(n_unknowns, 0.8)
        if 'tol' not in arg_names:
            optimizer_kwargs['tol'] = 1e-12
        if 'method' not in arg_names:
            optimizer_kwargs['method'] = 'hybr'

        return optimizer_kwargs

    @staticmethod
    def _prepare_param_bounds(param_bounds: Optional[List[Tuple[float, float]]],
                              n_params) -> List[Tuple[float, float]]:
        if param_bounds is None:
            bounds = [(1e-4, 0.999) for _ in range(n_params)]
        else:
            bounds = [(lower + 1e-4, upper - 1e-4) for lower, upper in param_bounds]

        return bounds

    def _get_n_unknowns_in_eq(self, eq: sp.Add) -> int:
        params_to_calibrate = [] if self.params_to_calibrate is None else self.params_to_calibrate
        unknown_atoms = [x for x in eq.atoms() if is_variable(x) or x in params_to_calibrate]
        n_unknowns = len(list(set(unknown_atoms)))

        return n_unknowns

    def find_heuristic_ss_solutions(self,
                                    solution_dict: Dict[VariableType, float],
                                    subbed_ss_system: List[sp.Add],
                                    steady_state_system: List[sp.Add],
                                    unknowns: List[Union[VariableType, sp.Symbol]]) -> Tuple[Dict[VariableType, float],
                                                                                             List[sp.Add]]:
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
            parameteres.

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

        n_eqs = len(steady_state_system)
        solved_mask = np.array([eq == 0 for eq in subbed_ss_system])
        check_again_mask = np.full_like(solved_mask, True)
        numeric_solutions = solution_dict.copy()

        while True:
            solution_dict = {key: eq.subs(solution_dict) for key, eq in solution_dict.items()}
            subbed_ss_system = [eq.subs(numeric_solutions).simplify() for eq in subbed_ss_system]
            n_unknowns = np.array([self._get_n_unknowns_in_eq(eq) for eq in subbed_ss_system])
            eq_len = np.array([len(eq.atoms()) for eq in subbed_ss_system])

            # When there are calibrating equations these can get quite long after repeated substitution,
            # putting a length cap of 10 is just a way to prevent Sympy from grinding away forever on horrible messes
            solvable_mask = ((n_unknowns < 2) & (eq_len < 10) & (~solved_mask) & check_again_mask)

            if sum(solvable_mask) == 0:
                break

            for idx in np.flatnonzero(solvable_mask):
                eq = subbed_ss_system[idx]

                variables = list(set([x for x in eq.atoms() if x in unknowns]))
                if len(variables) > 0:
                    solved_mask[idx] = True
                    symbolic_solution = sp.solve(steady_state_system[idx], variables[0])[0]
                    solution_dict[variables[0]] = symbolic_solution
                    numeric_solutions[variables[0]] = symbolic_solution.subs(numeric_solutions)

                    check_again_mask[:] = True
                else:
                    check_again_mask[idx] = False

        return solution_dict, solved_mask
