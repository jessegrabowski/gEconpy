from collections import defaultdict
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import arviz as az
import emcee
import numpy as np
import pandas as pd
import sympy as sp
import xarray as xr
from numpy.typing import ArrayLike
from scipy import linalg, stats

from gEconpy.classes.block import Block
from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.progress_bar import ProgressBar
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.estimation.estimate import build_Z_matrix, evaluate_logp
from gEconpy.estimation.estimation_utilities import (
    extract_prior_dict,
    extract_sparse_data_from_model,
)
from gEconpy.exceptions.exceptions import (
    GensysFailedException,
    MultipleSteadyStateBlocksException,
    PerturbationSolutionNotFoundException,
    SteadyStateNotSolvedError,
    VariableNotFoundException,
)
from gEconpy.numba_tools.utilities import numba_lambdify
from gEconpy.parser import file_loaders, gEcon_parser
from gEconpy.parser.constants import STEADY_STATE_NAMES
from gEconpy.parser.parse_distributions import create_prior_distribution_dictionary, CompositeDistribution
from gEconpy.parser.parse_equations import single_symbol_to_sympy
from gEconpy.shared.utilities import (
    expand_subs_for_all_times,
    is_variable,
    make_all_var_time_combos,
    merge_dictionaries,
    substitute_all_equations,
    unpack_keys_and_values, build_Q_matrix, compute_autocorrelation_matrix, get_shock_std_priors_from_hyperpriors,
    split_random_variables,
)
from gEconpy.solvers.gensys import interpret_gensys_output
from gEconpy.solvers.perturbation import PerturbationSolver
from gEconpy.solvers.steady_state import SteadyStateSolver

VariableType = Union[sp.Symbol, TimeAwareSymbol]


class gEconModel:
    def __init__(
            self,
            model_filepath: str,
            verbose: bool = True,
            simplify_blocks=True,
            simplify_constants=True,
            simplify_tryreduce=True,
    ) -> None:
        """
        Initialize a DSGE model object from a GCN file.

        Parameters
        ----------
        model_filepath : str
            Filepath to the GCN file
        verbose : bool, optional
            Flag for verbose output, by default True
        simplify_blocks : bool, optional
            Flag to simplify blocks, by default True
        simplify_constants : bool, optional
            Flag to simplify constants, by default True
        simplify_tryreduce : bool, optional
            Flag to simplify using `try_reduce_vars`, by default True
        """
        self.model_filepath: str = model_filepath

        # Model metadata
        self.options: Optional[Dict[str, bool]] = None
        self.try_reduce_vars: Optional[List[TimeAwareSymbol]] = None

        self.blocks: Dict[str, Block] = {}
        self.n_blocks: int = 0

        # Model components
        self.variables: List[TimeAwareSymbol] = []
        self.assumptions: Dict[str, dict] = defaultdict(SymbolDictionary)
        self.shocks: List[TimeAwareSymbol] = []
        self.system_equations: List[sp.Add] = []
        self.calibrating_equations: List[sp.Add] = []
        self.params_to_calibrate: List[sp.Symbol] = []

        self.deterministic_relationships: List[sp.Add] = []
        self.deterministic_params: List[sp.Symbol] = []

        self.free_param_dict: SymbolDictionary[sp.Symbol, float] = SymbolDictionary()
        self.calib_param_dict: SymbolDictionary[sp.Symbol, float] = SymbolDictionary()
        self.det_param_dict: SymbolDictionary[sp.Symbol, float] = SymbolDictionary()
        self.steady_state_relationships: SymbolDictionary[VariableType, sp.Add] = SymbolDictionary()

        self.param_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        self.shock_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        self.hyper_priors: SymbolDictionary[str, Any] = SymbolDictionary()
        self.observation_noise_priors: SymbolDictionary[str, Any] = SymbolDictionary()

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
        self.steady_state_dict: SymbolDictionary[sp.Symbol, float] = SymbolDictionary()
        self.residuals: List[float] = []

        # Functional representation of the perturbation system
        self.build_perturbation_matrices: Union[Callable, None] = None

        # Perturbation solution information
        self.perturbation_solved: bool = False
        self.T: pd.DataFrame = None
        self.R: pd.DataFrame = None
        self.P: pd.DataFrame = None
        self.Q: pd.DataFrame = None
        self.R: pd.DataFrame = None
        self.S: pd.DataFrame = None

        self.build(
            verbose=verbose,
            simplify_blocks=simplify_blocks,
            simplify_constants=simplify_constants,
            simplify_tryreduce=simplify_tryreduce,
        )

        # Assign Solvers
        self.steady_state_solver = SteadyStateSolver(self)
        self.perturbation_solver = PerturbationSolver(self)

        # TODO: Here I copy the assumptions from the model (which should be the only source of truth for assumptions)
        #  into every SymbolDictionary. This setup is really bad; if these dictionaries go out of sync there could be
        #  disagreements about what the assumptions for a variable should be.

        for d in [
            self.free_param_dict,
            self.calib_param_dict,
            self.steady_state_relationships,
            self.param_priors,
            self.shock_priors,
            self.observation_noise_priors,
        ]:
            d._assumptions.update(self.assumptions)

    def build(
            self,
            verbose: bool,
            simplify_blocks: bool,
            simplify_constants: bool,
            simplify_tryreduce: bool,
    ) -> None:
        """
        Main parsing function for the model. Build loads the GCN file, decomposes it into blocks, solves optimization
        problems contained in each block, then extracts parameters, equations, calibrating equations, calibrated
        parameters, and exogenous shocks into their respective class attributes.

        Priors declared in the GCN file are converted into scipy distribution objects and stored in two dictionaries:
        self.param_priors and self.shock_priors.

        Gathering block information is done for convenience. For diagnostic purposes the block structure is retained
        as well.

        Parameters
        ----------
        verbose : bool, optional
            When True, print a build report describing the model structure and warning the user if the number of
            variables does not match the number of equations.
        simplify_blocks : bool, optional
            If True, simplify equations in the model blocks.
        simplify_constants : bool, optional
            If True, simplify constants in the model equations.
        simplify_tryreduce : bool, optional
            If True, try to reduce the number of variables in the model by eliminating unnecessary equations.

        Returns
        -------
        None
        """

        raw_model = file_loaders.load_gcn(self.model_filepath)
        parsed_model, prior_dict = gEcon_parser.preprocess_gcn(raw_model)

        self._build_model_blocks(parsed_model, simplify_blocks)
        self._get_all_block_equations()
        self._get_all_block_parameters()
        self._get_all_block_params_to_calibrate()
        self._get_all_block_deterministic_parameters()
        self._get_variables_and_shocks()
        self._build_prior_dict(prior_dict)

        self._make_deterministic_substitutions()

        reduced_vars = None
        singletons = None

        if simplify_tryreduce:
            reduced_vars = self._try_reduce()
        if simplify_constants:
            singletons = self._simplify_singletons()

        self.build_report(reduced_vars, singletons, verbose=verbose)

    def build_report(self, reduced_vars: List[str],
                     singletons: List[str],
                     verbose: bool = True) -> None:
        """
        Write a diagnostic message after building the model. Note that successfully building the model does not
        guarantee that the model is correctly specified. For example, it is possible to build a model with more
        equations than parameters. This message will warn the user in this case.

        Parameters
        -------
        reduced_vars: list
            A list of variables reduced by the `try_reduce` method. Used to print the names of eliminated variables.

        singletons: list
            A list of "singleton" variables -- those defined as time-invariant constants. Used ot print the sames of
            eliminated variables.

        verbose: bool
            Flag to print the build report to the terminal. Default is True. Regardless of the flag, the function will
            always issue a warning to the user if the system is not fully defined.
        Returns
        -------
        None
        """

        if singletons and len(singletons) == 0:
            singletons = None

        eq_str = "equation" if self.n_equations == 1 else "equations"
        var_str = "variable" if self.n_variables == 1 else "variables"
        shock_str = "shock" if self.n_shocks == 1 else "shocks"
        cal_eq_str = "equation" if self.n_calibrating_equations == 1 else "equations"
        free_par_str = 'parameter' if len(self.free_param_dict) == 1 else 'parameters'
        calib_par_str = "parameter" if self.n_params_to_calibrate == 1 else "parameters"

        n_params = len(self.free_param_dict) + len(self.calib_param_dict)

        param_priors = self.param_priors.keys()
        shock_priors = self.shock_priors.keys()

        report = "Model Building Complete.\nFound:\n"
        report += f"\t{self.n_equations} {eq_str}\n"
        report += f"\t{self.n_variables} {var_str}\n"

        if reduced_vars:
            report += f"\tThe following variables were eliminated at user request:\n"
            report += f"\t\t" + ",".join(reduced_vars) + "\n"

        if singletons:
            report += f'\tThe following "variables" were defined as constants and have been substituted away:\n'
            report += f"\t\t" + ",".join(singletons) + "\n"

        report += f"\t{self.n_shocks} stochastic {shock_str}\n"
        report += (
            f'\t\t {len(shock_priors)} / {self.n_shocks} {"have" if len(shock_priors) == 1 else "has"}'
            f" a defined prior. \n"
        )

        report += f"\t{n_params} {free_par_str}\n"
        report += (
            f'\t\t {len(param_priors)} / {n_params} {"have" if len(param_priors) == 1 else "has"} '
            f"a defined prior. \n"
        )
        report += f"\t{self.n_calibrating_equations} calibrating {cal_eq_str}\n"
        report += f"\t{self.n_params_to_calibrate} {calib_par_str} to calibrate\n "

        if self.n_equations == self.n_variables:
            report += "Model appears well defined and ready to proceed to solving.\n"
        else:
            message = (
                f"The model does not appear correctly specified, there are {self.n_equations} {eq_str} but "
                f"{self.n_variables} {var_str}. It will not be possible to solve this model. Please check the "
                f"specification using available diagnostic tools, and check the GCN file for typos."
            )
            warn(message)

        if verbose:
            print(report)

    def steady_state(
            self,
            verbose: Optional[bool] = True,
            model_is_linear: Optional[bool] = False,
            apply_user_simplifications=True,
            method: Optional[str] = "root",
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            use_jac: Optional[bool] = True,
            use_hess: Optional[bool] = True,
            tol: Optional[float] = 1e-6,
    ) -> None:
        """
        Solves for a function f(params) that computes steady state values and calibrated parameter values given
        parameter values, stores results, and verifies that the residuals of the solution are zero.

        Parameters
        ----------
        verbose: bool
            Flag controlling whether to print results of the steady state solver. Default is True.
        model_is_linear: bool, optional
            If True, the model is assumed to have been linearized by the user. A specialized solving routine is used
            to find the steady state, which is likely all zeros. If True, all other arguments to this function
            have no effect (except verbose). Default is False.
        apply_user_simplifications: bool
            Whether to simplify system equations using the user-defined steady state relationships defined in the GCN
            before passing the system to the numerical solver. Default is True.
        method: str
            One of "root" or "minimize". Indicates which family of solution algorithms should be used to find a
            numerical steady state: direct root finding or minimization of squared error. Not that "root" is not
            suitable if the number of inputs is not equal to the number of outputs, for example if user-provided
            steady state relationships do not result in elimination of model equations. Default is "root".
        optimizer_kwargs: dict
            Dictionary of arguments to be passed to scipy.optimize.root or scipy.optimize.minimize, see those
            functions for more details.
        use_jac: bool
            Whether to symbolically compute the Jacobian matrix of the steady state system (when method is "root") or
            the Jacobian vector of the loss function (when method is "minimize"). Strongly recommended. Default is True
        use_hess: bool
            Whether to symbolically compute the Hessian matrix of the loss function. Ignored if method is "root".
            If "False", the default BFGS solver will compute a numerical approximation, so not necessarily required.
            Still recommended. Default is True.
        tol: float
            Numerical tolerance for declaring a steady-state solution valid. Default is 1e-6. Note that this only used
            by the gEconpy model to decide if a steady state has been found, and is **NOT** passed to the scipy
            solution algorithms. To adjust solution tolerance for these algorithms, use optimizer_kwargs.

        Returns
        -------
        None
        """
        if not self.steady_state_solved:
            self.f_ss = self.steady_state_solver.solve_steady_state(
                apply_user_simplifications=apply_user_simplifications,
                model_is_linear=model_is_linear,
                method=method,
                optimizer_kwargs=optimizer_kwargs,
                use_jac=use_jac,
                use_hess=use_hess,
            )

        self._process_steady_state_results(verbose, tol=tol)

    def _process_steady_state_results(self, verbose=True, tol=1e-6) -> None:
        """Process results from steady state solver.

        This function sets the steady state dictionary, calibrated parameter dictionary, and residuals attribute
        based on the results of the steady state solver. It also sets the `steady_state_solved` attribute to
        indicate whether the steady state was successfully found. If `verbose` is True, it prints a message
        indicating whether the steady state was found and the sum of squared residuals.

        Parameters
        ----------
        verbose : bool, optional
            If True, print a message indicating whether the steady state was found and the sum of squared residuals.
            Default is True.
        tol: float, optional
            Numerical tolerance for declaring a steady-state solution has been found. Default is 1e-6.

        Returns
        -------
        None
        """
        results = self.f_ss(self.free_param_dict)
        self.steady_state_dict = results["ss_dict"]
        self.calib_param_dict = results["calib_dict"]
        self.residuals = results["resids"]

        self.steady_state_system = self.steady_state_solver.steady_state_system
        self.steady_state_solved = np.allclose(self.residuals, 0, atol=tol) & results["success"]

        if verbose:
            if self.steady_state_solved:
                print(
                    f"Steady state found! Sum of squared residuals is {(self.residuals ** 2).sum()}"
                )
            else:
                print(
                    f"Steady state NOT found. Sum of squared residuals is {(self.residuals ** 2).sum()}"
                )

    def print_steady_state(self):
        """
        Prints the steady state values for the model's variables and calibrated parameters.

        Prints an error message if a valid steady state has not yet been found.
        """
        if len(self.steady_state_dict) == 0:
            print("Run the steady_state method to find a steady state before calling this method.")
            return

        output = []
        if not self.steady_state_solved:
            output.append("Values come from the latest solver iteration but are NOT a valid steady state.")

        max_var_name = (
                max(
                    len(x)
                    for x in list(self.steady_state_dict.keys()) + list(self.calib_param_dict.keys())
                )
                + 5
        )

        for key, value in self.steady_state_dict.items():
            output.append(f"{key:{max_var_name}}{value:>10.3f}")

        if len(self.params_to_calibrate) > 0:
            output.append("\n")
            output.append("In addition, the following parameter values were calibrated:")
            for key, value in self.calib_param_dict.items():
                output.append(f"{key:{max_var_name}}{value:>10.3f}")

        print('\n'.join(output))

    def solve_model(
            self,
            solver="cycle_reduction",
            not_loglin_variable: Optional[List[str]] = None,
            order: int = 1,
            model_is_linear: bool = False,
            tol: float = 1e-8,
            max_iter: int = 1000,
            verbose: bool = True,
            on_failure="error",
    ) -> None:
        """
        Solve for the linear approximation to the policy function via perturbation. Adapted from R code in the gEcon
        package by Grzegorz Klima, Karol Podemski, and Kaja Retkiewicz-Wijtiwiak., http://gecon.r-forge.r-project.org/.

        Parameters
        ----------
        solver: str, default: 'cycle_reduction'
            Name of the algorithm to solve the linear solution. Currently "cycle_reduction" and "gensys" are supported.
            Following Dynare, cycle_reduction is the default, but note that gEcon uses gensys.
        not_loglin_variable: List, default: None
            Variables to not log linearize when solving the model. Variables with steady state values close to zero
            will be automatically selected to not log linearize.
        order: int, default: 1
            Order of taylor expansion to use to solve the model. Currently only 1st order approximation is supported.
        model_is_linear: bool, default: False
            Flag indicating whether a model has already been linearized by the user.
        tol: float, default 1e-8
            Desired level of floating point accuracy in the solution
        max_iter: int, default: 1000
            Maximum number of cycle_reduction iterations. Not used if solver is 'gensys'.
        verbose: bool, default: True
            Flag indicating whether to print solver results to the terminal
        on_failure: str, one of ['error', 'ignore'], default: 'error'
            Instructions on what to do if the algorithm to find a linearized policy matrix. "Error" will raise an error,
            while "ignore" will return None. "ignore" is useful when repeatedly solving the model, e.g. when sampling.

        Returns
        -------
        None
        """

        if on_failure not in ['error', 'ignore']:
            raise ValueError(f'Parameter on_failure must be one of "error" or "ignore", found {on_failure}')

        param_dict = self.free_param_dict | self.calib_param_dict
        steady_state_dict = self.steady_state_dict

        if self.build_perturbation_matrices is None:
            self._perturbation_setup(not_loglin_variable, order, model_is_linear, verbose, bool)

        A, B, C, D = self.build_perturbation_matrices(
            np.array(list(param_dict.values())), np.array(list(steady_state_dict.values()))
        )
        _, variables, _ = self.perturbation_solver.make_all_variable_time_combinations()

        if solver == "gensys":
            gensys_results = self.perturbation_solver.solve_policy_function_with_gensys(
                A, B, C, D, tol, verbose
            )
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

                    self.perturbation_solved = False

                    return

            if verbose:
                message = interpret_gensys_output(eu)
                print(message)
                print("Policy matrices have been stored in attributes model.P, model.Q, model.R, and model.S")

            T = G_1[: self.n_variables, :][:, : self.n_variables]
            R = impact[: self.n_variables, :]

        elif solver == "cycle_reduction":
            (
                T,
                R,
                result,
                log_norm,
            ) = self.perturbation_solver.solve_policy_function_with_cycle_reduction(
                A, B, C, D, max_iter, tol, verbose
            )
            if T is None:
                if on_failure == "error":
                    raise GensysFailedException(result)
        else:
            raise NotImplementedError(
                'Only "cycle_reduction" and "gensys" are valid values for solver'
            )

        gEcon_matrices = self.perturbation_solver.statespace_to_gEcon_representation(
            A, T, R, variables, tol
        )
        P, Q, _, _, A_prime, R_prime, S_prime = gEcon_matrices

        resid_norms = self.perturbation_solver.residual_norms(
            B, C, D, Q, P, A_prime, R_prime, S_prime
        )
        norm_deterministic, norm_stochastic = resid_norms

        if verbose:
            print(f"Norm of deterministic part: {norm_deterministic:0.9f}")
            print(f"Norm of stochastic part:    {norm_deterministic:0.9f}")

        self.T = pd.DataFrame(
            T,
            index=[x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)],
            columns=[x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)],
        )
        self.R = pd.DataFrame(
            R,
            index=[x.base_name for x in sorted(self.variables, key=lambda x: x.base_name)],
            columns=[x.base_name for x in sorted(self.shocks, key=lambda x: x.base_name)],
        )

        self.perturbation_solved = True

    def _perturbation_setup(
            self,
            not_loglin_variables=None,
            order=1,
            model_is_linear=False,
            verbose=True,
            return_F_matrices=False,
            tol=1e-8,
    ):
        """
        This function is used to set up the perturbation matrices needed to simulate the model. It linearizes the model
        around the steady state and constructs matrices A, B, C, and D needed to solve the system.

        Parameters
        ----------
        not_loglin_variables: list of str
            List of variables that should not be log-linearized. This is useful when a variable has a zero or negative
             steady state value and cannot be log-linearized.
        order: int
            The order of the approximation. Currently only order 1 is implemented.
        model_is_linear: bool
            If True, assumes that the model is already linearized in the GCN file and directly
             returns the matrices A, B, C, D.
        verbose: bool
            If True, prints warning messages.
        return_F_matrices: bool
            If True, returns the matrices A, B, C, D.
        tol: float
            The tolerance used to determine if a steady state value is close to zero.

        Returns
        -------
        None or list of sympy matrices
            If return_F_matrices is True, returns the F matrices. Otherwise, does not return anything.

        """

        free_param_dict = self.free_param_dict.copy()

        parameters = list(free_param_dict.to_sympy().keys())
        variables = list(self.steady_state_dict.to_sympy().keys())
        params_to_calibrate = list(self.calib_param_dict.to_sympy().keys())

        all_params = parameters + params_to_calibrate

        shocks = self.shocks
        shock_ss_dict = dict(zip([x.to_ss() for x in shocks], np.zeros(self.n_shocks)))
        variables_and_shocks = self.variables + shocks
        valid_names = [x.base_name for x in variables_and_shocks]

        steady_state_dict = self.steady_state_dict.copy()

        if not model_is_linear:
            # We need shocks to be zero in A, B, C, D but 1 in T; can abuse the T_dummies to accomplish that.
            if not_loglin_variables is None:
                not_loglin_variables = []

            not_loglin_variables += [x.base_name for x in shocks]

            # Validate that all user-supplied variables are in the model
            for variable in not_loglin_variables:
                if variable not in valid_names:
                    raise VariableNotFoundException(variable)

            # Variables that are zero at the SS can't be log-linearized, check for these here.
            close_to_zero_warnings = []
            for variable in variables_and_shocks:
                if variable.base_name in not_loglin_variables:
                    continue

                if abs(steady_state_dict[variable.to_ss().name]) < tol:
                    not_loglin_variables.append(variable.base_name)
                    close_to_zero_warnings.append(variable)

            if len(close_to_zero_warnings) > 0 and verbose:
                warn(
                    "The following variables have steady state values close to zero and will not be log linearized: "
                    + ", ".join(x.base_name for x in close_to_zero_warnings)
                )

        if order != 1:
            raise NotImplementedError

        if not self.steady_state_solved:
            raise SteadyStateNotSolvedError()

        if model_is_linear:
            Fs = self.perturbation_solver.convert_linear_system_to_matrices()

        else:
            Fs = self.perturbation_solver.log_linearize_model(
                not_loglin_variables=not_loglin_variables
            )

        Fs_subbed = [F.subs(shock_ss_dict) for F in Fs]
        self.build_perturbation_matrices = numba_lambdify(
            exog_vars=all_params, endog_vars=variables, expr=Fs_subbed
        )

        if return_F_matrices:
            return Fs_subbed

    def check_bk_condition(
            self,
            free_param_dict: Optional[Dict[str, float]] = None,
            system_matrices: Optional[List[ArrayLike]] = None,
            verbose: Optional[bool] = True,
            return_value: Optional[str] = "df",
            tol=1e-8,
    ) -> Optional[ArrayLike]:
        """
        Compute the generalized eigenvalues of system in the form presented in [1]. Per [2], the number of
        unstable eigenvalues (|v| > 1) should not be greater than the number of forward-looking variables. Failing
        this test suggests timing problems in the definition of the model.

        Parameters
        ----------
        free_param_dict: dict, optional
            A dictionary of parameter values. If None, the current stored values are used.
        system_matrices: list, optional
            A list of matrices A, B, C, D to be used to compute the bk_condition. If none, the current
            stored values are used.
        verbose: bool, default: True
            Flag to print the results of the test, otherwise the eigenvalues are returned without comment.
        return_value: string, default: 'df'
            Controls what is returned by the function. Valid values are 'df', 'bool', and 'none'.
            If df, a dataframe containing eigenvalues is returned. If 'bool', a boolean indicating whether the BK
            condition is satisfied. If None, nothing is returned.
        tol: float, 1e-8
            Convergence tolerance for the gensys solver

        Returns
        -------

        """
        if self.build_perturbation_matrices is None:
            raise PerturbationSolutionNotFoundException()

        if return_value not in ['df', 'bool', 'none']:
            raise ValueError(f'return_value must be one of "df", "bool", or "none". Found {return_value} ')

        if free_param_dict is not None:
            results = self.f_ss(self.free_param_dict)
            self.steady_state_dict = results["ss_dict"]
            self.calib_param_dict = results["calib_dict"]

        param_dict = self.free_param_dict | self.calib_param_dict
        steady_state_dict = self.steady_state_dict

        if system_matrices is not None:
            A, B, C, D = system_matrices
        else:
            A, B, C, D = self.build_perturbation_matrices(
                np.array(list(param_dict.values())), np.array(list(steady_state_dict.values()))
            )

        n_forward = (C.sum(axis=0) > 0).sum().astype(int)
        n_eq, n_vars = A.shape

        # TODO: Compute system eigenvalues -- avoids calling the whole Gensys routine, but there is code duplication
        #   building Gamma_0 and Gamma_1
        lead_var_idx = np.where(np.sum(np.abs(C), axis=0) > tol)[0]

        eqs_and_leads_idx = np.r_[np.arange(n_vars), lead_var_idx + n_vars].tolist()

        Gamma_0 = np.vstack([np.hstack([B, C]), np.hstack([-np.eye(n_eq), np.zeros((n_eq, n_eq))])])

        Gamma_1 = np.vstack(
            [
                np.hstack([A, np.zeros((n_eq, n_eq))]),
                np.hstack([np.zeros((n_eq, n_eq)), np.eye(n_eq)]),
            ]
        )
        Gamma_0 = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
        Gamma_1 = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]

        # A, B, Q, Z = qzdiv(1.01, *linalg.qz(-Gamma_0, Gamma_1, 'complex'))

        # Using scipy instead of qzdiv appears to offer a huge speedup for nearly the same answer; some eigenvalues
        # have sign flip relative to qzdiv -- does it matter?
        A, B, alpha, beta, Q, Z = linalg.ordqz(-Gamma_0, Gamma_1, sort="ouc", output="complex")

        gev = np.c_[np.diagonal(A), np.diagonal(B)]

        eigenval = gev[:, 1] / (gev[:, 0] + tol)
        pos_idx = np.where(np.abs(eigenval) > 0)
        eig = np.zeros(((np.abs(eigenval) > 0).sum(), 3))
        eig[:, 0] = np.abs(eigenval)[pos_idx]
        eig[:, 1] = np.real(eigenval)[pos_idx]
        eig[:, 2] = np.imag(eigenval)[pos_idx]

        sorted_idx = np.argsort(eig[:, 0])
        eig = pd.DataFrame(eig[sorted_idx, :], columns=["Modulus", "Real", "Imaginary"])

        n_g_one = (eig["Modulus"] > 1).sum()
        condition_not_satisfied = n_forward > n_g_one
        if verbose:
            print(
                f"Model solution has {n_g_one} eigenvalues greater than one in modulus and {n_forward} "
                f"forward-looking variables."
                f'\nBlanchard-Kahn condition is{" NOT" if condition_not_satisfied else ""} satisfied.'
            )

        if return_value == 'none':
            return
        if return_value == "df":
            return eig
        elif return_value == "bool":
            return ~condition_not_satisfied

    def compute_stationary_covariance_matrix(self,
                                             shock_dict: Optional[Dict[str, float]] = None,
                                             shock_cov_matrix: Optional[ArrayLike] = None):
        """
        Compute the stationary covariance matrix of the solved system by solving the associated discrete lyapunov
        equation. In order to construct the shock covariance matrix, exactly one or zero of shock_dict or
        shock_cov_matrix should be provided. If neither is provided, the prior means on the shocks will be used. If no
        information about a shock is available, the standard deviation will be set to 0.01.

        Parameters
        -------
        shock_dict, dict of str: float, optional
            A dictionary of shock sizes to be used to compute the stationary covariance matrix.
        shock_cov_matrix: array, optional
            An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks

        Returns
        -------
        sigma: DataFrame
        """
        if not self.perturbation_solved:
            raise PerturbationSolutionNotFoundException()
        shock_std_priors = get_shock_std_priors_from_hyperpriors(self.shocks, self.hyper_priors, out_keys='parent')

        if shock_dict is None and shock_cov_matrix is None and len(shock_std_priors) < self.n_shocks:
            unknown_shocks_list = [shock.base_name for shock in self.shocks if shock not in self.shock_priors.to_sympy()]
            unknown_shocks = ', '.join(unknown_shocks_list)
            warn(f'No standard deviation provided for shocks {unknown_shocks}. Using default of std = 0.01. Explicity'
                 f'pass variance information for these shocks or set their priors to silence this warning.')

        Q = build_Q_matrix(model_shocks=[x.base_name for x in self.shocks],
                           shock_dict=shock_dict,
                           shock_cov_matrix=shock_cov_matrix,
                           shock_std_priors=shock_std_priors)

        T, R = self.T, self.R
        sigma = linalg.solve_discrete_lyapunov(T.values, R.values @ Q @ R.values.T)

        return pd.DataFrame(sigma, index=T.index, columns=T.index)

    def compute_autocorrelation_matrix(self,
                                       shock_dict: Optional[Dict[str, float]] = None,
                                       shock_cov_matrix: Optional[ArrayLike] = None,
                                       n_lags=10):
        """
        Computes autocorrelations for each model variable using the stationary covariance matrix. See doc string for
        compute_stationary_covariance_matrix for more information.

        In order to construct the shock covariance matrix, exactly one or zero of shock_dict or
        shock_cov_matrix should be provided. If neither is provided, the prior means on the shocks will be used. If no
        information about a shock is available, the standard deviation will be set to 0.01.

        Parameters
        ----------
        shock_dict, dict of str: float, optional
            A dictionary of shock sizes to be used to compute the stationary covariance matrix.
        shock_cov_matrix: array, optional
            An (n_shocks, n_shocks) covariance matrix describing the exogenous shocks
        n_lags: int
            Number of lags over which to compute the autocorrelation

        Returns
        -------
        acorr_mat: DataFrame
        """
        if not self.perturbation_solved:
            raise PerturbationSolutionNotFoundException()

        T, R = self.T, self.R

        Sigma = self.compute_stationary_covariance_matrix(shock_dict=shock_dict, shock_cov_matrix=shock_cov_matrix)
        acorr_mat = compute_autocorrelation_matrix(T.values, Sigma.values, n_lags=n_lags)

        return pd.DataFrame(acorr_mat, index=T.index, columns=np.arange(n_lags))

    def fit(
            self,
            data,
            estimate_a0=False,
            estimate_P0=False,
            a0_prior=None,
            P0_prior=None,
            filter_type="univariate",
            draws=5000,
            n_walkers=36,
            moves=None,
            emcee_x0=None,
            verbose=True,
            return_inferencedata=True,
            burn_in=None,
            thin=None,
            skip_initial_state_check=False,
            **sampler_kwargs,
    ):
        """
        Estimate model parameters via Bayesian inference. Parameter likelihood is computed using the Kalman filter.
        Posterior distributions are estimated using Markov Chain Monte Carlo (MCMC), specifically the Affine-Invariant
        Ensemble Sampler algorithm of [1].

        A "traditional" Random Walk Metropolis can be achieved using the moves argument, but by default this function
        will use a mix of two Differential Evolution (DE) proposal algorithms that have been shown to work well on
        weakly multi-modal problems. DSGE estimation can be multi-modal in the sense that regions of the posterior
        space are separated by the constraints on the ability to solve the perturbation problem.

        This function will start all MCMC chains around random draws from the prior distribution. This is in contrast
        to Dynare and gEcon.estimate, which start MCMC chains around the Maximum Likelihood estimate for parameter
        values.

        Parameters
        ----------
        data: dataframe
            A pandas dataframe of observed values, with column names corresponding to DSGE model states names.
        estimate_a0: bool, default: False
            Whether to estimate the initial values of the DSGE process. If False, x0 will be deterministically set to
            a vector of zeros, corresponding to the steady state. If True, you must provide a
        estimate_P0: bool, default: False
            Whether to estimate the intial covariance matrix of the DSGE process. If False, P0 will be set to the
            Kalman Filter steady state value by solving the associated discrete Lyapunov equation.
        a0_prior: dict, optional
            A dictionary with (variable name, scipy distribution) key-value pairs. If a key "initial_vector" is found,
            all other keys will be ignored, and the single distribution over all initial states will be used. Otherwise,
            n_states independent distributions should be included in the dictionary.
            If estimate_a0 is False, this will be ignored.
        P0_prior: dict, optional
            A dictionary with (variable name, scipy distribution) key-value pairs. If a key "initial_covariance" is
            found, all other keys will be ignored, and this distribution will be taken as over the entire covariance
            matrix. Otherwise, n_states independent distributions are expected, and are used to construct a diagonal
            initial covariance matrix.
        filter_type: string, default: "standard"
            Select a kalman filter implementation to use. Currently "standard" and "univariate" are supported. Try
            univariate if you run into errors inverting the P matrix during filtering.
        draws: integer
            Number of draws from each MCMC chain, or "walker" in the jargon of emcee.
        n_walkers: integer
            The number of "walkers", which roughly correspond to chains in other MCMC packages. Note that one needs
            many more walkers than chains; [1] recommends as many as possible.
        cores: integer
            The number of processing cores, which is passed to Multiprocessing.Pool to do parallel inference. To
            maintain detailed balance, the pool of walkers must be split, resulting in n_walkers / cores sub-ensembles.
            Be sure to raise the number of walkers to compensate.
        moves: List of emcee.moves objects
            Moves tell emcee how to generate MCMC proposals. See the emcee docs for details.
        emcee_x0: array
            An (n_walkers, k_parameters) array of initial values. Emcee will check the condition number of the matrix
            to ensure all walkers begin in different regions of the parameter space. If MLE estimates are used, they
            should be jittered to start walkers in a ball around the desired initial point.
        return_inferencedata: bool, default: True
            If true, return an Arviz InferenceData object containing posterior samples. If False, the fitted Emcee
            sampler is returned.
        burn_in: int, optional
            Number of initial samples to discard from all chains. This is ignored if return_inferencedata is False.
        thin: int, optional
            Return only every n-th sample from each chain. This is done to reduce storage requirements in highly
            autocorrelated chains by discarding redundant information. Ignored if return_inferencedata is False.

        Returns
        -------
        sampler, emcee.Sampler object
            An emcee.Sampler object with the estimated posterior over model parameters, as well as other diagnotic
            information.

        References
        -------
        ..[1] Foreman-Mackey, Daniel, et al. “Emcee: The MCMC Hammer.” Publications of the Astronomical Society of the
              Pacific, vol. 125, no. 925, Mar. 2013, pp. 306–12. arXiv.org, https://doi.org/10.1086/670067.
        """
        observed_vars = data.columns.tolist()
        model_var_names = [x.base_name for x in self.variables]
        if not all([x in model_var_names for x in observed_vars]):
            orphans = [x for x in observed_vars if x not in model_var_names]
            raise ValueError(
                f"Columns of data must correspond to states of the DSGE model. Found the following columns"
                f'with no associated model state: {", ".join(orphans)}'
            )

        sparse_data = extract_sparse_data_from_model(self)
        prior_dict = extract_prior_dict(self)

        if estimate_a0 is False:
            a0 = None
        else:
            if a0_prior is None:
                raise ValueError(
                    "If estimate_a0 is True, you must provide a dictionary of prior distributions for"
                    "the initial values of all individual states"
                )
            if not all([var in a0_prior.keys() for var in model_var_names]):
                missing_keys = set(model_var_names) - set(list(a0_prior.keys()))
                raise ValueError(
                    "You must provide one key for each state in the model. "
                    f'No keys found for: {", ".join(missing_keys)}'
                )
            for var in model_var_names:
                prior_dict[f"{var}__initial"] = a0_prior[var]

        moves = moves or [
            (emcee.moves.DEMove(), 0.6),
            (emcee.moves.DESnookerMove(), 0.4),
        ]

        shock_names = [x.base_name for x in self.shocks]

        k_params = len(prior_dict)
        Z = build_Z_matrix(observed_vars, model_var_names)

        args = [
            data,
            sparse_data,
            Z,
            prior_dict,
            shock_names,
            observed_vars,
            filter_type,
        ]
        arg_names = [
            "observed_data",
            "sparse_data",
            "Z",
            "prior_dict",
            "shock_names",
            "observed_vars",
            "filter_type",
        ]

        if emcee_x0:
            x0 = emcee_x0
        else:
            x0 = np.stack([x.rvs(n_walkers) for x in prior_dict.values()]).T

        param_names = list(prior_dict.keys())

        sampler = emcee.EnsembleSampler(
            n_walkers,
            k_params,
            evaluate_logp,
            args=args,
            moves=moves,
            parameter_names=param_names,
            **sampler_kwargs,
        )

        _ = sampler.run_mcmc(
            x0,
            draws,
            progress=verbose,
            skip_initial_state_check=skip_initial_state_check,
        )

        if return_inferencedata:
            sampler_stats = xr.Dataset(
                data_vars=dict(
                    acceptance_fraction=(["chain"], sampler.acceptance_fraction),
                    autocorrelation_time=(
                        ["parameters"],
                        sampler.get_autocorr_time(discard=burn_in or 0, quiet=True),
                    ),
                ),
                coords=dict(chain=np.arange(n_walkers), parameters=param_names),
            )

            idata = az.from_emcee(
                sampler,
                var_names=param_names,
                blob_names=["log_likelihood"],
                arg_names=arg_names,
            )

            idata["sample_stats"].update(sampler_stats)
            idata.observed_data = idata.observed_data.drop(["sparse_data", "prior_dict"])
            idata.observed_data = idata.observed_data.drop_dims(
                ["sparse_data_dim_0", "sparse_data_dim_1", "prior_dict_dim_0"]
            )

            return idata.sel(draw=slice(burn_in, None, thin))

        return sampler

    def sample_param_dict_from_prior(self, n_samples=1, seed=None, param_subset=None):

        """
        Sample parameters from the parameter prior distributions.

        Parameters
        ----------
        n_samples: int, default: 1
            Number of samples to draw from the prior distributions.
        seed: int, default: None
            Seed for the random number generator.
        param_subset: list, default: None
            List of parameter names to sample. If None, all parameters are sampled.

        Returns
        -------
        new_param_dict: dict
            Dictionary of sampled parameters.
        """
        shock_std_priors = get_shock_std_priors_from_hyperpriors(self.shocks, self.hyper_priors)

        all_priors = self.param_priors.to_sympy() | shock_std_priors | self.observation_noise_priors.to_sympy()

        if param_subset is None:
            n_variables = len(all_priors)
            priors_to_sample = all_priors
        else:
            n_variables = len(param_subset)
            priors_to_sample = SymbolDictionary({k: v for k, v in all_priors.items() if k.name in param_subset})

        if seed is not None:
            seed_sequence = np.random.SeedSequence(seed)
            child_seeds = seed_sequence.spawn(n_variables)
            streams = [np.random.default_rng(s) for s in child_seeds]
        else:
            streams = [None] * n_variables

        new_param_dict = {}
        for i, (key, d) in enumerate(priors_to_sample.items()):
            new_param_dict[key] = d.rvs(size=n_samples, random_state=streams[i])

        free_param_dict, shock_dict, obs_dict = split_random_variables(new_param_dict,
                                                                       self.shocks,
                                                                       self.variables)

        return free_param_dict, shock_dict, obs_dict

    def impulse_response_function(self, simulation_length: int = 40, shock_size: float = 1.0):
        """
        Compute the impulse response functions of the model.

        Parameters
        ----------
        simulation_length : int, optional
            The number of periods to compute the IRFs over. The default is 40.
        shock_size : float, optional
            The size of the shock. The default is 1.0.

        Returns
        -------
        pandas.DataFrame
            The IRFs for each variable in the model. The DataFrame has a multi-index
            with the variable names as the first level and the timestep as the second.
            The columns are the shocks.

        Raises
        ------
        PerturbationSolutionNotFoundException
            If a perturbation solution has not been found.
        """

        if not self.perturbation_solved:
            raise PerturbationSolutionNotFoundException()

        T, R = self.T, self.R

        timesteps = simulation_length

        data = np.zeros((self.n_variables, timesteps, self.n_shocks))

        for i in range(self.n_shocks):
            shock_path = np.zeros((self.n_shocks, timesteps))
            shock_path[i, 0] = shock_size

            for t in range(1, timesteps):
                stochastic = R.values @ shock_path[:, t - 1]
                deterministic = T.values @ data[:, t - 1, i]
                data[:, t, i] = deterministic + stochastic

        index = pd.MultiIndex.from_product(
            [R.index, np.arange(timesteps), R.columns],
            names=["Variables", "Time", "Shocks"],
        )

        df = (
            pd.DataFrame(data.ravel(), index=index, columns=["Values"])
            .unstack([1, 2])
            .droplevel(axis=1, level=0)
            .sort_index(axis=1)
        )

        return df

    def simulate(
            self,
            simulation_length: int = 40,
            n_simulations: int = 100,
            shock_dict: Optional[Dict[str, float]] = None,
            shock_cov_matrix: Optional[ArrayLike] = None,
            show_progress_bar: bool = False,
    ):

        """
        Simulate the model over a certain number of time periods.

        Parameters
        ----------
        simulation_length : int, optional(default=40)
            The number of time periods to simulate.
        n_simulations : int, optional(default=100)
            The number of simulations to run.
        shock_dict : dict, optional(default=None)
            Dictionary of shocks to use.
        shock_cov_matrix : arraylike, optional(default=None)
            Covariance matrix of shocks to use.
        show_progress_bar : bool, optional(default=False)
            Whether to show a progress bar for the simulation.

        Returns
        -------
        df : pandas.DataFrame
            The simulated data.
        """

        if not self.perturbation_solved:
            raise PerturbationSolutionNotFoundException()

        T, R = self.T, self.R
        timesteps = simulation_length

        n_shocks = R.shape[1]
        shock_std_priors = get_shock_std_priors_from_hyperpriors(self.shocks, self.hyper_priors, out_keys='parent')

        if shock_dict is None and shock_cov_matrix is None and len(shock_std_priors) < self.n_shocks:
            unknown_shocks_list = [shock.base_name for shock in self.shocks if shock not in self.shock_priors.to_sympy()]
            unknown_shocks = ', '.join(unknown_shocks_list)
            warn(f'No standard deviation provided for shocks {unknown_shocks}. Using default of std = 0.01. Explicity'
                 f'pass variance information for these shocks or set their priors to silence this warning.')

        Q = build_Q_matrix(model_shocks=[x.base_name for x in self.shocks],
                           shock_dict=shock_dict,
                           shock_cov_matrix=shock_cov_matrix,
                           shock_std_priors=shock_std_priors)

        d_epsilon = stats.multivariate_normal(mean=np.zeros(n_shocks), cov=Q)
        epsilons = np.r_[[d_epsilon.rvs(timesteps) for _ in range(n_simulations)]]

        data = np.zeros((self.n_variables, timesteps, n_simulations))
        if epsilons.ndim == 2:
            epsilons = epsilons[:, :, None]

        progress_bar = ProgressBar(timesteps - 1, verb="Sampling")

        for t in range(1, timesteps):
            progress_bar.start()
            stochastic = np.einsum("ij,sj", R.values, epsilons[:, t - 1, :])
            deterministic = T.values @ data[:, t - 1, :]
            data[:, t, :] = deterministic + stochastic

            if show_progress_bar:
                progress_bar.stop()

        index = pd.MultiIndex.from_product(
            [R.index, np.arange(timesteps), np.arange(n_simulations)],
            names=["Variables", "Time", "Simulation"],
        )
        df = (
            pd.DataFrame(data.ravel(), index=index, columns=["Values"])
            .unstack([1, 2])
            .droplevel(axis=1, level=0)
        )

        return df

    def _build_prior_dict(self, prior_dict: Dict[str, str]) -> None:
        """
        Parameters
        ----------
        prior_dict: dict
            Dictionary of variable_name: distribution_string pairs, prepared by the parse_gcn function.

        Returns
        -------
        self.param_dict: dict
            Dictionary of variable:distribution pairs. Distributions are scipy rv_frozen objects, unless the
            distribution is parameterized by another distribution, in which case a "CompositeDistribution" object
            with methods .rvs, .pdf, and .logpdf is returned.
        """

        priors, hyper_priors = create_prior_distribution_dictionary(prior_dict)
        hyper_parameters = set(prior_dict.keys()) - set(priors.keys())

        # Remove hyperparameters from the free parameters
        for parameter in hyper_parameters:
            del self.free_param_dict[parameter]

        param_priors = SymbolDictionary()
        shock_priors = SymbolDictionary()
        hyper_priors_final = SymbolDictionary()

        for key, value in priors.items():
            sympy_key = single_symbol_to_sympy(key, assumptions=self.assumptions)
            if isinstance(sympy_key, TimeAwareSymbol):
                shock_priors[sympy_key.base_name] = value
            else:
                param_priors[sympy_key.name] = value

        for key, value in hyper_priors.items():
            parent_rv, param_type, dist = value
            parent_key = single_symbol_to_sympy(parent_rv, assumptions=self.assumptions)
            param_key = single_symbol_to_sympy(key, assumptions=self.assumptions)

            hyper_priors_final[param_key] = (parent_key, param_type, dist)

        self.param_priors = param_priors
        self.shock_priors = shock_priors
        self.hyper_priors = hyper_priors_final

    def _build_model_blocks(self, parsed_model, simplify_blocks: bool):
        """
        Builds blocks of the gEconpy model using strings parsed from the GCN file.

        Parameters
        ----------
        parsed_model : str
            The GCN model as a string.
        simplify_blocks : bool
            Whether to try to simplify equations or not.
        """

        raw_blocks = gEcon_parser.split_gcn_into_block_dictionary(parsed_model)

        self.options = raw_blocks["options"]
        self.try_reduce_vars = raw_blocks["tryreduce"]
        self.assumptions = raw_blocks["assumptions"]

        del raw_blocks["options"]
        del raw_blocks["tryreduce"]
        del raw_blocks["assumptions"]

        self._get_steady_state_equations(raw_blocks)

        for block_name, block_content in raw_blocks.items():
            block_dict = gEcon_parser.parsed_block_to_dict(block_content)
            block = Block(name=block_name, block_dict=block_dict, assumptions=self.assumptions)
            block.solve_optimization(try_simplify=simplify_blocks)

            self.blocks[block.name] = block

        self.n_blocks = len(self.blocks)

    def _get_all_block_equations(self) -> None:
        """
        Extract all equations from the blocks in the model.

        Parameters
        ----------
        self : `Model`
            The model object whose block system equations will be extracted.

        Returns
        -------
        None

        Notes
        -----
        Updates the `system_equations` attribute of `self` with the extracted equations.
        Also updates the `n_equations` attribute of `self` with the number of extracted equations.
        """

        _, blocks = unpack_keys_and_values(self.blocks)
        for block in blocks:
            self.system_equations.extend(block.system_equations)
        self.n_equations = len(self.system_equations)

    def _get_all_block_parameters(self) -> None:
        """
        Extract all parameters from all blocks and store them in the model's free_param_dict attribute. The
        `free_param_dict` attribute is updated in place.
        """

        _, blocks = unpack_keys_and_values(self.blocks)
        for block in blocks:
            self.free_param_dict = self.free_param_dict | block.param_dict

        self.free_param_dict = self.free_param_dict.sort_keys().to_string().values_to_float()

    def _get_all_block_params_to_calibrate(self) -> None:
        """
        Retrieve the list of parameters to calibrate and the list of
        equations used to calibrate the parameters from each block of
        the model.
        """
        _, blocks = unpack_keys_and_values(self.blocks)
        for block in blocks:
            if block.params_to_calibrate is None:
                continue

            if len(self.params_to_calibrate) == 0:
                self.params_to_calibrate = block.params_to_calibrate
            else:
                self.params_to_calibrate.extend(block.params_to_calibrate)

            if block.calibrating_equations is None:
                continue

            if len(self.calibrating_equations) == 0:
                self.calibrating_equations = block.calibrating_equations
            else:
                self.calibrating_equations.extend(block.calibrating_equations)

        alpha_sort_idx = np.argsort([x.name for x in self.params_to_calibrate])
        self.params_to_calibrate = [self.params_to_calibrate[i] for i in alpha_sort_idx]
        self.calibrating_equations = [self.calibrating_equations[i] for i in alpha_sort_idx]

        self.n_calibrating_equations = len(self.calibrating_equations)
        self.n_params_to_calibrate = len(self.params_to_calibrate)

    def _get_all_block_deterministic_parameters(self) -> None:
        _, blocks = unpack_keys_and_values(self.blocks)
        for block in blocks:
            if block.deterministic_params is None:
                continue

            if len(self.deterministic_params) == 0:
                self.deterministic_params = block.deterministic_params
            else:
                self.deterministic_params.extend(block.deterministic_params)

            if block.deterministic_relationships is None:
                continue

            if len(self.deterministic_relationships) == 0:
                self.deterministic_relationships = block.deterministic_relationships
            else:
                self.deterministic_relationships.extend(block.deterministic_relationships)

        alpha_sort_idx = np.argsort([x.name for x in self.deterministic_params])
        self.deterministic_params = [self.deterministic_params[i] for i in alpha_sort_idx]
        self.deterministic_relationships = [
            self.deterministic_relationships[i] for i in alpha_sort_idx
        ]

    def _get_variables_and_shocks(self) -> None:
        """
        Collect all variables and shocks from the blocks and set their counts.

        This method is called after the blocks have been processed. It collects all the shocks and variables from the
        blocks, sorts them, and sets the n_shocks and n_variables properties.
        """

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

        self.variables = sorted(self.variables, key=lambda x: x.name)
        self.shocks = sorted(self.shocks, key=lambda x: x.name)

    def _get_steady_state_equations(self, raw_blocks: Dict[str, List[str]]):
        """
        Extract user-provided steady state equations from the `raw_blocks` dictionary and store the resulting
        relationships in self.steady_state_relationships.

        Parameters
        ----------
        raw_blocks : dict
            Dictionary of block names and block contents extracted from a gEcon model.

        Raises
        ------
        MultipleSteadyStateBlocksException
            If there is more than one block in `raw_blocks` with a name from `STEADY_STATE_NAMES`.
        """

        block_names = raw_blocks.keys()
        ss_block_names = [name for name in block_names if name in STEADY_STATE_NAMES]
        n_ss_blocks = len(ss_block_names)

        if n_ss_blocks == 0:
            return
        if n_ss_blocks > 1:
            raise MultipleSteadyStateBlocksException(ss_block_names)

        block_content = raw_blocks[ss_block_names[0]]
        block_dict = gEcon_parser.parsed_block_to_dict(block_content)
        block = Block(name="steady_state", block_dict=block_dict, assumptions=self.assumptions)

        sub_dict = SymbolDictionary()
        steady_state_dict = SymbolDictionary()

        if block.definitions is not None:
            _, definitions = unpack_keys_and_values(block.definitions)
            sub_dict = SymbolDictionary({eq.lhs: eq.rhs for eq in definitions})

        if block.identities is not None:
            _, identities = unpack_keys_and_values(block.identities)
            for eq in identities:
                subbed_rhs = eq.rhs.subs(sub_dict)
                steady_state_dict[eq.lhs] = subbed_rhs
                sub_dict[eq.lhs] = subbed_rhs

        for k, eq in steady_state_dict.items():
            steady_state_dict[k] = eq.subs(steady_state_dict)

        self.steady_state_relationships = (
            steady_state_dict.sort_keys().to_string().values_to_float()
        )

        del raw_blocks[ss_block_names[0]]

    def _try_reduce(self):
        """
        Attempt to reduce the number of equations in the system by removing equations requested in the `tryreduce`
        block of the GCN file. Equations are considered safe to remove if they are "self-contained" that is, if
        no other variables depend on their values.

        Returns
        -------
        list
            The names of the variables that were removed. If reduction was not possible, None is returned.
        """
        if self.n_equations != self.n_variables:
            warn("Simplification via try_reduce was requested but not possible because the system is not well defined.")
            return

        if self.try_reduce_vars is None:
            return

        self.try_reduce_vars = [
            single_symbol_to_sympy(x, self.assumptions) for x in self.try_reduce_vars
        ]

        variables = self.variables
        n_variables = self.n_variables

        occurrence_matrix = np.zeros((n_variables, n_variables))
        reduced_system = []

        for i, eq in enumerate(self.system_equations):
            for j, var in enumerate(self.variables):
                if any([x in eq.atoms() for x in make_all_var_time_combos([var])]):
                    occurrence_matrix[i, j] += 1

        # Columns with a sum of 1 are variables that appear only in a single equations; these equations can be deleted
        # without consequence w.r.t solving the system.

        isolated_variables = np.array(variables)[occurrence_matrix.sum(axis=0) == 1]
        to_remove = set(isolated_variables).intersection(set(self.try_reduce_vars))

        for eq in self.system_equations:
            if not any([var in eq.atoms() for var in to_remove]):
                reduced_system.append(eq)

        self.system_equations = reduced_system
        self.n_equations = len(self.system_equations)

        self.variables = {
            atom.set_t(0) for eq in reduced_system for atom in eq.atoms() if is_variable(atom)
        }
        self.variables -= set(self.shocks)
        self.variables = sorted(list(self.variables), key=lambda x: x.name)
        self.n_variables = len(self.variables)

        eliminated_vars = [var.name for var in variables if var not in self.variables]

        return eliminated_vars

    def _simplify_singletons(self):
        """
        Simplify the system by removing variables that are deterministically defined as a known value. Common examples
        include P[] = 1, setting the price level of the economy as the numeraire, or B[] = 0, putting the bond market
        in net-zero supply.

        In these cases, the variable can be replaced by the deterministic value after all FoC
        have been computed.

        Returns
        -------
        eliminated_vars : List[str]
            The names of the variables that were removed.
        """

        if self.n_equations != self.n_variables:
            warn(
                "Removal of constant variables was requested but not possible because the system is not well defined."
            )
            return

        system = self.system_equations

        variables = self.variables
        reduce_dict = {}

        for eq in system:
            if len(eq.atoms()) < 4:
                var = [x for x in eq.atoms() if is_variable(x)]
                if len(var) != 1:
                    continue
                var = var[0]
                sub_dict = expand_subs_for_all_times(sp.solve(eq, var, dict=True)[0])
                reduce_dict.update(sub_dict)

        reduced_system = substitute_all_equations(system, reduce_dict)
        reduced_system = [eq for eq in reduced_system if eq != 0]

        self.system_equations = reduced_system
        self.n_equations = len(reduced_system)

        self.variables = {
            atom.set_t(0) for eq in reduced_system for atom in eq.atoms() if is_variable(atom)
        }
        self.variables -= set(self.shocks)
        self.variables = sorted(list(self.variables), key=lambda x: x.name)
        self.n_variables = len(self.variables)

        eliminated_vars = [var.name for var in variables if var not in self.variables]

        return eliminated_vars

    def _make_deterministic_substitutions(self):
        if self.deterministic_params is None:
            return

        all_atoms = reduce(
            lambda left, right: left.union(right), [eq.atoms() for eq in self.system_equations]
        )

        if not any([det_var in all_atoms for det_var in self.deterministic_params]):
            return

        det_sub_dict = dict(zip(self.deterministic_params, self.deterministic_relationships))

        # recursively substitute the dictionary on itself, in case there are any relationships between the relationships
        for i in range(5):
            all_atoms = reduce(
                lambda left, right: left.union(right), [eq.atoms() for eq in det_sub_dict.values()]
            )
            if any([det_param in all_atoms for det_param in self.deterministic_params]):
                det_sub_dict = substitute_all_equations(det_sub_dict, det_sub_dict)
            else:
                break
        if i == 5:
            raise ValueError(
                "Could not reduce deterministic relationships to functions of only free parameters after"
                "five recursive substitutions. Check that there are not circular definitions among the"
                "deterministic parameters."
            )

        self.system_equations = [eq.subs(det_sub_dict) for eq in self.system_equations]


# #@njit
# def _compute_stationary_covariance_matrix(A, C, tol=1e-9, max_iter=10_000):
#     sigma = np.eye(A.shape[0])
#     for _ in range(max_iter):
#         new_sigma = A @ sigma @ A.T + C @ C.T
#         if ((sigma - new_sigma) ** 2).mean() < tol:
#             return sigma
#         else:
#             sigma = new_sigma


