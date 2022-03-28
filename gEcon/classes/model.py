from gEcon.parser import gEcon_parser, file_loaders
from gEcon.parser.parse_distributions import create_prior_distribution_dictionary
from gEcon.parser.parse_equations import single_symbol_to_sympy
from gEcon.classes.block import Block
from gEcon.shared.utilities import unpack_keys_and_values, is_variable, eq_to_ss, sort_dictionary, \
    sympy_keys_to_strings, sympy_number_values_to_floats, sequential, symbol_to_string, string_keys_to_sympy
from gEcon.classes.time_aware_symbol import TimeAwareSymbol
from gEcon.parser.constants import STEADY_STATE_NAMES
from gEcon.exceptions.exceptions import SteadyStateNotSolvedError, GensysFailedException, VariableNotFoundException, \
    MultipleSteadyStateBlocksException
from gEcon.solvers.gensys import gensys
from gEcon.solvers.steady_state import SteadyStateSolver

import numpy as np
import sympy as sp
import pandas as pd

from scipy import linalg

from warnings import warn
from functools import partial

from numpy.typing import ArrayLike
from typing import List, Dict, Tuple, Optional, Union, Callable, Any

VariableType = Union[sp.Symbol, TimeAwareSymbol]


class gEconModel:
    """
    Class to build, debug, and solve a DSGE model from a GCN file.
    """

    def __init__(self, model_filepath: str, verbose: bool = True) -> None:
        '''
        :param model_filepath: str, a string path to a GCN file
        :param verbose: bool, default = True, if true, prints a short diagnostic message after parsing the GCN file.
        '''

        self.model_filepath: str = model_filepath

        # Model metadata
        self.options: Optional[Dict[str, bool]] = None
        self.try_reduce_vars: Optional[List[TimeAwareSymbol]] = None

        self.blocks: Dict[str, Block] = {}
        self.n_blocks: int = 0

        # Model components
        self.variables: List[TimeAwareSymbol] = []
        self.shocks: List[TimeAwareSymbol] = []
        self.system_equations: List[sp.Add] = []
        self.calibrating_equations: List[sp.Add] = []
        self.params_to_calibrate: List[sp.Symbol] = []
        self.param_dict: Dict[sp.Symbol, float] = {}
        self.steady_state_relationships: Dict[VariableType, sp.Add] = {}

        self.priors: Dict[str, Any] = {}

        self.n_variables: int = 0
        self.n_shocks: int = 0
        self.n_equations: int = 0
        self.n_calibrating_equations: int = 0

        # Functional representations of the model
        self.f_ss: Union[Callable, None] = None
        self.f_ss_resid: Union[Callable, None] = None

        # Steady state information
        self.steady_state_solved: bool = False
        self.steady_state_system: List[sp.Add] = []
        self.steady_state_dict: Dict[sp.Symbol, float] = {}
        self.residuals: List[float] = []

        # Perturbation solution information
        self.perturbation_solved: bool = False
        self.P: pd.DataFrame = None
        self.Q: pd.DataFrame = None
        self.R: pd.DataFrame = None
        self.S: pd.DataFrame = None

        self.build(verbose=verbose)

        # Assign Solvers
        self.steady_state_solver = SteadyStateSolver(self)

    def build(self, verbose: bool = True) -> None:
        """
        :param verbose: bool, default: True. If true, print a short diagnostic message after successfully
                building the model.
        :return: None

        Main parsing function for the model. Build loads the GCN file, decomposes it into blocks, solves optimization
        problems contained in each block, then extracts parameters, equations, calibrating equations, calibrated
        parameters, and exogenous shocks into their respective class attributes.

        Gathering block information is done for convenience. For diagnostic purposes the block structure is retained
        as well.
        """
        raw_model = file_loaders.load_gcn(self.model_filepath)
        parsed_model, prior_dict = gEcon_parser.preprocess_gcn(raw_model)

        self._build_model_blocks(parsed_model)
        self._get_all_block_equations()
        self._get_all_block_parameters()
        self._get_all_block_params_to_calibrate()
        self._get_variables_and_shocks()

        self._build_prior_dict(prior_dict)
        self._validate_steady_state_block()

        if verbose:
            self.build_report()

    def build_report(self):
        """
        :return: None

        Write a disagostic message after building the model. Note that successfully building the model does not
        guarantee that the model is correctly specified. For example, it is possible to build a model with more
        equations than parameters. This message will warn the user in this case.
        """

        eq_str = "equation" if self.n_equations == 1 else "equations"
        var_str = "variable" if self.n_variables == 1 else "variables"
        shock_str = "shock" if self.n_shocks == 1 else "shocks"
        cal_eq_str = "equation" if self.n_calibrating_equations == 1 else "equations"
        par_str = "parameter" if self.n_params_to_calibrate == 1 else "parameters"

        n_params = len(self.param_dict)

        param_priors = set(self.param_dict.keys()).intersection(set(self.priors.keys()))
        shock_priors = set(self.shocks).intersection(set(self.priors.keys()))

        report = 'Model Building Complete.\nFound:\n'
        report += f'\t{self.n_equations} {eq_str}\n'
        report += f'\t{self.n_variables} {var_str}\n'
        report += f'\t{self.n_shocks} stochastic {shock_str}\n'
        report += f'\t\t {len(shock_priors)} / {self.n_shocks} have a defined prior. \n'

        report += f'\t{n_params} {par_str}\n'
        report += f'\t\t {len(param_priors)} / {n_params} have a defined prior. \n'
        report += f'\t{self.n_calibrating_equations} calibrating {cal_eq_str}\n'
        report += f'\t{self.n_params_to_calibrate} {par_str} to calibrate\n '

        if self.n_equations == self.n_variables:
            report += 'Model appears well defined and ready to proceed to solving.\n'
            print(report)
        else:
            print(report)
            message = f'The model does not appear correctly specified, there are {self.n_equations} {eq_str} but ' \
                      f'{self.n_variables} {var_str}. It will not be possible to solve this model. Please check the ' \
                      f'specification using available diagnostic tools, and check the GCN file for typos.'
            warn(message)

    def steady_state(self, verbose=True, optimizer_kwargs=None, param_bounds=None, use_jac=True):
        if not self.steady_state_solved:
            self.steady_state_solver.solve_steady_state(optimizer_kwargs, param_bounds, use_jac=use_jac)

            self.f_ss_resid = self.steady_state_solver.f_ss_resid
            self.f_ss = partial(self.steady_state_solver.f_ss,
                                optimizer_kwargs=optimizer_kwargs,
                                param_bounds=param_bounds)

        else:
            self._clear_calibrated_parameters_from_param_dict()

        self._process_steady_state_results(verbose)

    def _process_steady_state_results(self, verbose=True) -> None:
        self.steady_state_dict = self.f_ss(self.param_dict)

        self.residuals = np.array(self.f_ss_resid(**self.steady_state_dict, **self.param_dict))

        self.steady_state_solved = np.allclose(self.residuals, 0)

        if verbose:
            if self.steady_state_solved:
                print(f'Steady state found! Sum of squared residuals is {(self.residuals ** 2).sum()}')
            else:
                print(f'Steady state NOT found. Sum of squared residuals is {(self.residuals ** 2).sum()}')

        self._separate_param_results_and_variable_results()

    def _clear_calibrated_parameters_from_param_dict(self):
        params_to_calibrate = [] if self.params_to_calibrate is [] else [symbol_to_string(x) for x in
                                                                         self.params_to_calibrate]

        # If refitting a model with calibrated parameters, need to remove the old solution from the param dict.
        for param in params_to_calibrate:
            if param in self.param_dict.keys():
                del self.param_dict[param]

    def _separate_param_results_and_variable_results(self):

        params_to_calibrate = [] if self.params_to_calibrate is None else self.params_to_calibrate

        for param in params_to_calibrate:
            name = symbol_to_string(param)
            self.param_dict[name] = self.steady_state_dict[name]
            del self.steady_state_dict[name]

    def print_steady_state(self):
        if self.steady_state_dict is None:
            print('Run the steady_state method to find a steady state before calling this method.')
            return

        if not self.steady_state_solved:
            print('Values come from the latest solver iteration but are NOT a valid steady state.')

        max_var_name = max([len(x) for x in self.steady_state_dict.keys()]) + 5
        for key, value in self.steady_state_dict.items():
            print(f'{key:{max_var_name}}{value:>10.3f}')

        if self.params_to_calibrate is not None:
            print('\n')
            print('In addition, the following parameter values were calibrated:')
            for param in self.params_to_calibrate:
                print(f'{param.name:10}{self.param_dict[param.name]:>10.3f}')

    def solve_model(self, not_loglin_variable: Optional[List[str]] = None,
                    order: int = 1,
                    model_is_linear: bool = False,
                    tol: float = 1e-8,
                    verbose: bool = True) -> None:
        """
        Parameters
        ----------
        not_loglin_variable
        order
        model_is_linear
        tol
        verbose

        Returns
        -------

        Solve for the linear approximation to the policy function via perturbation. Adapted from R code in the gEcon
        package by Grzegorz Klima, Karol Podemski, and Kaja Retkiewicz-Wijtiwiak., http://gecon.r-forge.r-project.org/.
        """
        loglin_sub_dict = string_keys_to_sympy(self.steady_state_dict.copy())

        if not_loglin_variable is None:
            not_loglin_variable = []
        else:
            for variable in not_loglin_variable:
                new_var = TimeAwareSymbol(variable, 0).to_ss()
                if new_var not in self.steady_state_dict.keys():
                    raise VariableNotFoundException(new_var)
                loglin_sub_dict[new_var] = 1

        if order != 1:
            raise NotImplementedError

        if not self.steady_state_solved:
            raise SteadyStateNotSolvedError()

        if model_is_linear:
            warn('Model will be solved as though ALL system equations have already been linearized in the GCN file. No'
                 'checks are performed to ensure this is indeed the case. Proceed with caution.')
            Fs = self._convert_linear_system_to_matrices()

        else:
            Fs = self._log_linearize_model()

        shock_sub_dict = dict(zip([x.to_ss() for x in self.shocks], np.ones(self.n_shocks)))

        ss_values = np.array(list(loglin_sub_dict.values()))
        zero_mask = np.abs(ss_values) < 1e-4
        ss_values[zero_mask] = 1
        loglin_sub_dict = dict(zip(loglin_sub_dict.keys(), ss_values))

        # TODO: .subs() is way faster for a single substitution, but if we need to repeatedly get this jacobian
        #   (i.e. during MCMC), it will be better to lambdify F.
        A, B, C, D = [F.doit().subs(loglin_sub_dict).subs(self.param_dict).subs(shock_sub_dict) for F in Fs]

        G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = self._solve_policy_function_with_gensys(A, B, C, D,
                                                                                                           tol,
                                                                                                           verbose)
        if G_1 is None:
            raise GensysFailedException(eu)

        _, variables, _ = self._make_all_variable_time_combinations()

        P, Q, R, S, A_prime, R_prime, S_prime = self._extract_policy_matrices(A, G_1, impact, variables, tol)

        norm_deterministic, norm_stochastic = self._residual_norms(B, C, D,
                                                                   Q.values, P.values, A_prime, R_prime, S_prime)

        if verbose:
            print(f'Norm of deterministic part: {norm_deterministic:0.9f}')
            print(f'Norm of stochastic part:    {norm_deterministic:0.9f}')

        self.P = P
        self.Q = Q
        self.R = R
        self.S = S

    @staticmethod
    def _solve_policy_function_with_gensys(A: sp.Matrix, B: sp.Matrix, C: sp.Matrix, D: sp.Matrix,
                                           tol: float = 1e-8,
                                           verbose: bool = True) -> Tuple[Optional[ArrayLike], Optional[ArrayLike],
                                                                       Optional[ArrayLike], Optional[ArrayLike],
                                                                       Optional[ArrayLike], Optional[ArrayLike],
                                                                       Optional[ArrayLike], List[int],
                                                                       Optional[ArrayLike]]:
        n_eq, n_vars = A.shape
        _, n_shocks = D.shape

        lead_var_idx = np.where(np.sum(np.abs(C), axis=0) > tol)[0]
        eqs_and_leads_idx = np.r_[np.arange(n_vars), lead_var_idx + n_vars].tolist()

        n_leads = len(lead_var_idx)

        Gamma_0 = sp.Matrix.vstack(sp.Matrix.hstack(B, C),
                                   sp.Matrix.hstack(-sp.eye(n_eq), sp.zeros(n_eq)))

        Gamma_1 = sp.Matrix.vstack(sp.Matrix.hstack(A, sp.zeros(n_eq)),
                                   sp.Matrix.hstack(sp.zeros(n_eq), sp.eye(n_eq)))

        Pi = sp.Matrix.vstack(sp.zeros(n_eq), sp.eye(n_eq))

        Psi = sp.Matrix.vstack(sp.Matrix(D),
                               sp.zeros(n_eq, n_shocks))

        Gamma_0 = Gamma_0[eqs_and_leads_idx, eqs_and_leads_idx]
        Gamma_1 = Gamma_1[eqs_and_leads_idx, eqs_and_leads_idx]
        Psi = Psi[eqs_and_leads_idx, :]
        Pi = Pi[eqs_and_leads_idx, lead_var_idx.tolist()]

        # Is this necessary?
        g0 = np.ascontiguousarray(np.array(-Gamma_0).astype(np.float64))  # NOTE THE IMPORTANT MINUS SIGN LURKING
        g1 = np.ascontiguousarray(np.array(Gamma_1).astype(np.float64))
        c = np.ascontiguousarray(np.zeros(shape=(n_vars + n_leads, 1)))
        psi = np.ascontiguousarray(np.array(Psi).astype(np.float64))
        pi = np.ascontiguousarray(np.array(Pi).astype(np.float64))

        G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = gensys(g0, g1, c, psi, pi)

        if eu[0] == 1 and eu[1] == 1 and verbose:
            print('Gensys found a unique solution.\n'
                  'Policy matrices have been stored in attributes model.P, model.Q, model.R, and model.S')

        return G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose

    def _extract_policy_matrices(self, A, G_1, impact, variables, tol):
        n_vars = len(variables)

        g = G_1[:n_vars, :n_vars]
        state_var_idx = np.where(np.abs(g[np.argmax(np.abs(g), axis=0), np.arange(n_vars)]) >= tol)[0]
        state_var_mask = np.isin(np.arange(n_vars), state_var_idx)

        n_shocks = self.n_shocks
        shock_idx = np.arange(n_shocks)

        variables = np.atleast_1d(variables).squeeze()

        state_vars = variables[state_var_mask]
        L1_state_vars = np.array([x.step_backward() for x in state_vars])
        jumpers = np.atleast_1d(variables)[~state_var_mask]

        state_vars = [x.name for x in state_vars]
        L1_state_vars = [x.name for x in L1_state_vars]
        jumpers = [x.name for x in jumpers]
        shocks = [x.name for x in sorted(self.shocks, key=lambda x: x.base_name)]

        PP = g.copy()
        PP[np.where(np.abs(PP) < tol)] = 0
        QQ = impact.copy()
        QQ = QQ[:n_vars, :]
        QQ[np.where(np.abs(QQ) < tol)] = 0

        P = PP[state_var_mask, :][:, state_var_mask]
        Q = QQ[state_var_mask, :][:, shock_idx]
        R = PP[~state_var_mask, :][:, state_var_idx]
        S = QQ[~state_var_mask, :][:, shock_idx]

        A_prime = np.array(A).astype(np.float64)[:, state_var_mask]
        R_prime = PP[:, state_var_mask]
        S_prime = QQ[:, shock_idx]

        P = pd.DataFrame(P, index=state_vars, columns=L1_state_vars)
        Q = pd.DataFrame(Q, index=state_vars, columns=shocks)
        R = pd.DataFrame(R, index=jumpers, columns=L1_state_vars)
        S = pd.DataFrame(S, index=jumpers, columns=shocks)

        return P, Q, R, S, A_prime, R_prime, S_prime

    @staticmethod
    def _residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime):
        B_np = np.array(B).astype(np.float64)
        C_np = np.array(C).astype(np.float64)
        D_np = np.array(D).astype(np.float64)

        norm_deterministic = linalg.norm(A_prime + B_np @ R_prime +
                                         C_np @ R_prime @ P)

        norm_stochastic = linalg.norm(B_np @ S_prime +
                                      C_np @ R_prime @ Q + D_np)

        return norm_deterministic, norm_stochastic

    def _log_linearize_model(self) -> List[sp.Matrix]:
        """
        :return: List, a list of Sympy matrices comprised of parameters and steady-state values, see docstring.

        Convert the non-linear model to its log-linear approximation using a first-order Taylor expansion around the
        deterministic steady state. The specific method of log-linearization is taken from the gEcon User's Guide,
        page 54, equation 9.9.

            F1 @ T @ y_{t-1} + F2 @ T @ y_t + F3 @ T @ y_{t+1} + F4 @ epsilon_t = 0

        Where T is a diagonal matrix containing steady-state values on the diagonal. Evaluating the matrix
        multiplications in the expression above obtains:

            A @ y_{t-1} + B @ y_t + C @ y_{t+1} + D @ epsilon = 0

        Matrices A, B, C, and D are returned by this function.

        TODO: Presently, everything is done using sympy, which is extremely slow. This should all be re-written in a
            way that is Numba and CUDA compatible.
        """

        Fs = []
        lags, now, leads = self._make_all_variable_time_combinations()
        shocks = self.shocks
        for var_group in [lags, now, leads, shocks]:
            F = []
            T = sp.diag(*[x.to_ss() for x in var_group])

            for eq in self.system_equations:
                F_row = []
                for var in var_group:
                    dydx = sp.powsimp(eq_to_ss(eq.diff(var)))
                    F_row.append(dydx)
                F.append(F_row)
            F = sp.Matrix(F)
            Fs.append(sp.MatMul(F, T, evaluate=False))

        return Fs

    def _convert_linear_system_to_matrices(self) -> List[sp.Matrix]:
        """

        :return: List of sympy Matrices representing the linear system

        If the model has already been log-linearized by hand, this method is used to simplify the construction of the
        solution matrices. Following the gEcon user's guide, page 54, equation 9.10, the solution should be of the form:

            A @ y_{t-1} + B @ y_t + C @ y_{t+1} + D @ epsilon = 0

        This function organizes the model equations and returns matrices A, B, C, and D.

        TODO: Add some checks to ensure that the model is indeed linear so that this can't be erroneously called.
        """

        Fs = []
        lags, now, leads = self._make_all_variable_time_combinations()
        shocks = self.shocks
        model = self.system_equations

        for var_group, name in zip([lags, now, leads, shocks], ['lags', 'now', 'leads', 'shocks']):
            F = sp.zeros(len(var_group)) if name != 'shocks' else sp.zeros(rows=len(model), cols=len(var_group))
            for i, var in enumerate(var_group):
                for j, eq in enumerate(model):
                    args = eq.expand().args
                    for arg in args:
                        if var in arg.atoms():
                            F[j, i] = sp.simplify(arg / var)
            Fs.append(F)

        return Fs

    def _make_all_variable_time_combinations(self) -> Tuple[List[TimeAwareSymbol],
                                                            List[TimeAwareSymbol],
                                                            List[TimeAwareSymbol]]:
        """
        :return: Tuple of three lists, containing all model variables at time steps t-1, t, and t+1, respectively.
        """

        now = sorted(self.variables, key=lambda x: x.base_name)
        lags = [x.step_backward() for x in now]
        leads = [x.step_forward() for x in now]

        return lags, now, leads

    def _build_prior_dict(self, prior_dict: Dict[str, str], package='scipy') -> None:
        """
        Parameters
        ----------
        prior_dict: dict
            Dictionary of variable_name: distribution_string pairs, prepared by the parse_gcn function.
        package: str
            Which backend to put the distributions into. Just scipy for now, but PyMC support is high on the to-do list.

        Returns
        -------
        self.param_dict: dict
            Dictionary of variable:distribution pairs. Distributions are scipy rv_frozen objects, unless the
            distribution is parameterized by another distribution, in which case a "CompositeDistribution" object
            with methods .rvs, .pdf, and .logpdf is returned.
        """

        priors = create_prior_distribution_dictionary(prior_dict)
        sympy_priors = {}

        hyper_parameters = set(prior_dict.keys()) - set(priors.keys())

        # Clean up the hyper parameters from the model, they aren't needed anymore
        for parameter in hyper_parameters:
            del self.param_dict[single_symbol_to_sympy(parameter)]

        for key, value in priors.items():
            sympy_priors[single_symbol_to_sympy(key)] = value

        self.priors = sympy_priors

    def _build_model_blocks(self, parsed_model):
        raw_blocks = gEcon_parser.split_gcn_into_block_dictionary(parsed_model)

        self.options = raw_blocks['options']
        self.try_reduce_vars = raw_blocks['tryreduce']

        del raw_blocks['options']
        del raw_blocks['tryreduce']

        self._get_steady_state_equations(raw_blocks)

        for block_name, block_content in raw_blocks.items():
            block_dict = gEcon_parser.parsed_block_to_dict(block_content)
            block = Block(name=block_name, block_dict=block_dict)
            block.solve_optimization()

            self.blocks[block.name] = block

        self.n_blocks = len(self.blocks)

    def _get_all_block_equations(self) -> None:
        _, blocks = unpack_keys_and_values(self.blocks)
        for block in blocks:
            self.system_equations.extend(block.system_equations)
        self.n_equations = len(self.system_equations)

    def _get_all_block_parameters(self) -> None:
        _, blocks = unpack_keys_and_values(self.blocks)
        for block in blocks:
            self.param_dict.update(block.param_dict)

        self.param_dict = sequential(self.param_dict,
                                     [sympy_keys_to_strings, sympy_number_values_to_floats, sort_dictionary])

    def _get_all_block_params_to_calibrate(self) -> None:
        _, blocks = unpack_keys_and_values(self.blocks)
        for block in blocks:
            if block.params_to_calibrate is None:
                continue

            if len(self.params_to_calibrate) == 0:
                self.params_to_calibrate = block.params_to_calibrate
            else:
                self.params_to_calibrate.append(block.params_to_calibrate)

            if block.calibrating_equations is None:
                continue

            if len(self.calibrating_equations) == 0:
                self.calibrating_equations = block.calibrating_equations
            else:
                self.calibrating_equations.append(block.calibrating_equations)

        self.n_calibrating_equations = len(self.calibrating_equations)
        self.n_params_to_calibrate = len(self.params_to_calibrate)

    def _get_variables_and_shocks(self) -> None:
        all_shocks = []
        _, blocks = unpack_keys_and_values(self.blocks)

        for block in blocks:
            if block.shocks is not None:
                all_shocks.extend([x for x in block.shocks])
        self.shocks = all_shocks
        self.n_shocks = len(all_shocks)

        for eq in self.system_equations:
            atoms = eq.atoms()
            variables = [x for x in atoms if is_variable(x)]
            for variable in variables:
                if variable.set_t(0) not in self.variables and variable not in all_shocks:
                    self.variables.append(variable.set_t(0))
        self.n_variables = len(self.variables)

    def _get_steady_state_equations(self, raw_blocks: Dict[str, List[str]]):
        block_names = raw_blocks.keys()
        ss_block_names = [name for name in block_names if name in STEADY_STATE_NAMES]
        n_ss_blocks = len(ss_block_names)

        if n_ss_blocks == 0:
            return
        if n_ss_blocks > 1:
            raise MultipleSteadyStateBlocksException(ss_block_names)

        block_content = raw_blocks[ss_block_names[0]]
        block_dict = gEcon_parser.parsed_block_to_dict(block_content)
        block = Block(name='steady_state', block_dict=block_dict)

        sub_dict = dict()
        steady_state_dict = dict()

        if block.definitions is not None:
            _, definitions = unpack_keys_and_values(block.definitions)
            sub_dict = {eq.lhs: eq.rhs for eq in definitions}

        if block.identities is not None:
            _, identities = unpack_keys_and_values(block.identities)
            for eq in identities:
                subbed_rhs = eq.rhs.subs(sub_dict)
                steady_state_dict[eq.lhs] = subbed_rhs
                sub_dict[eq.lhs] = subbed_rhs

        self.steady_state_relationships = sequential(steady_state_dict,
                                                     [sympy_keys_to_strings, sympy_number_values_to_floats,
                                                      sort_dictionary])

        del raw_blocks[ss_block_names[0]]

    def _validate_steady_state_block(self):
        pass
