from typing import List, Tuple

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike
from scipy import linalg

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.shared.utilities import eq_to_ss
from gEconpy.solvers.cycle_reduction import cycle_reduction, solve_shock_matrix
from gEconpy.solvers.gensys import gensys


def print_gensys_results(eu):
    if eu[0] == 1 and eu[1] == 1:
        print(
            "Gensys found a unique solution.\n"
            "Policy matrices have been stored in attributes model.P, model.Q, model.R, and model.S"
        )

    else:
        print(eu)


class PerturbationSolver:
    def __init__(self, model):
        self.steady_state_dict = model.steady_state_dict
        self.steady_state_solved = model.steady_state_solved
        self.param_dict = model.free_param_dict
        self.system_equations = model.system_equations
        self.variables = model.variables

        self.shocks = model.shocks
        self.n_shocks = model.n_shocks

    @staticmethod
    def solve_policy_function_with_gensys(
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        tol: float = 1e-8,
        verbose: bool = True,
    ) -> Tuple:
        n_eq, n_vars = A.shape
        _, n_shocks = D.shape

        lead_var_idx = np.where(np.sum(np.abs(C), axis=0) > tol)[0]
        eqs_and_leads_idx = np.r_[np.arange(n_vars), lead_var_idx + n_vars].tolist()

        n_leads = len(lead_var_idx)

        Gamma_0 = np.vstack(
            [np.hstack([B, C]), np.hstack([-np.eye(n_eq), np.zeros((n_eq, n_eq))])]
        )

        Gamma_1 = np.vstack(
            [
                np.hstack([A, np.zeros((n_eq, n_eq))]),
                np.hstack([np.zeros((n_eq, n_eq)), np.eye(n_eq)]),
            ]
        )

        Pi = np.vstack([np.zeros((n_eq, n_eq)), np.eye(n_eq)])

        Psi = np.vstack([D, np.zeros((n_eq, n_shocks))])

        Gamma_0 = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
        Gamma_1 = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
        Psi = Psi[eqs_and_leads_idx, :]
        Pi = Pi[eqs_and_leads_idx, :][:, lead_var_idx]

        # Is this necessary?
        g0 = -np.ascontiguousarray(Gamma_0)  # NOTE THE IMPORTANT MINUS SIGN LURKING
        g1 = np.ascontiguousarray(Gamma_1)
        c = np.ascontiguousarray(np.zeros(shape=(n_vars + n_leads, 1)))
        psi = np.ascontiguousarray(Psi)
        pi = np.ascontiguousarray(Pi)

        G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = gensys(
            g0, g1, c, psi, pi
        )
        if verbose:
            print_gensys_results(eu)

        return G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose

    @staticmethod
    def solve_policy_function_with_cycle_reduction(
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        max_iter: int = 1000,
        tol: float = 1e-8,
        verbose: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike, str, float]:
        """
        Solve quadratic matrix equation of the form $A0x^2 + A1x + A2 = 0$ via cycle reduction algorithm of [1] to
        obtain the first-order linear approxiate policy matrices T and R.

        Parameters
        ----------
        A: Arraylike
            Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to past variables
            values that are known when decision-making: those with t-1 subscripts.
        B: ArrayLike
            Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
            are observed when decision-making: those with t subscripts.
        C: ArrayLike
            Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
            enter in expectation when decision-making: those with t+1 subscripts.
        D: ArrayLike
            Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to exogenous shocks.
        max_iter: int, default: 1000
            Maximum number of iterations to perform before giving up.
        tol: float, default: 1e-7
            Floating point tolerance used to detect algorithmic convergence
        verbose: bool, default: True
            If true, prints the sum of squared residuals that result when the system is computed used the solution.

        Returns
        -------
        T: ArrayLike
            Transition matrix T in state space jargon. Gives the effect of variable values at time t on the
            values of the variables at time t+1.
        R: ArrayLike
            Selection matrix R in state space jargon. Gives the effect of exogenous shocks at the t on the values of
            variables at time t+1.
        result: str
            String describing result of the cycle reduction algorithm
        log_norm: float
            Log L1 matrix norm of the first matrix (A2 -> A1 -> A0) that did not converge.
        """

        # Sympy gives back integers in the case of x/dx = 1, which can screw up the dtypes when passing to numba if
        # a Jacobian matrix is all constants (i.e. dF/d_shocks) -- cast everything to float64 here to avoid
        # a numba warning.
        T, R = None, None

        # A, B, C, D = A.astype('float64'), B.astype('float64'), C.astype('float64'), D.astype('float64')

        T, result, log_norm = cycle_reduction(A, B, C, max_iter, tol, verbose)

        if T is not None:
            R = solve_shock_matrix(B, C, D, T)

        return T, R, result, log_norm

    def statespace_to_gEcon_representation(self, A, T, R, variables, tol):
        n_vars = len(variables)

        state_var_idx = np.where(
            np.abs(T[np.argmax(np.abs(T), axis=0), np.arange(n_vars)]) >= tol
        )[0]
        state_var_mask = np.isin(np.arange(n_vars), state_var_idx)

        n_shocks = self.n_shocks
        shock_idx = np.arange(n_shocks)

        # variables = np.atleast_1d(variables).squeeze()

        # state_vars = variables[state_var_mask]
        # L1_state_vars = np.array([x.step_backward() for x in state_vars])
        # jumpers = np.atleast_1d(variables)[~state_var_mask]

        PP = T.copy()
        PP[np.where(np.abs(PP) < tol)] = 0
        QQ = R.copy()
        QQ = QQ[:n_vars, :]
        QQ[np.where(np.abs(QQ) < tol)] = 0

        P = PP[state_var_mask, :][:, state_var_mask]
        Q = QQ[state_var_mask, :][:, shock_idx]
        R = PP[~state_var_mask, :][:, state_var_idx]
        S = QQ[~state_var_mask, :][:, shock_idx]

        A_prime = A[:, state_var_mask]
        R_prime = PP[:, state_var_mask]
        S_prime = QQ[:, shock_idx]

        return P, Q, R, S, A_prime, R_prime, S_prime

    @staticmethod
    def residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime):
        norm_deterministic = linalg.norm(A_prime + B @ R_prime + C @ R_prime @ P)

        norm_stochastic = linalg.norm(B @ S_prime + C @ R_prime @ Q + D)

        return norm_deterministic, norm_stochastic

    def log_linearize_model(self, not_loglin_variables=None) -> List[sp.Matrix]:
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
            way that is Numba and/or CUDA compatible.
        """

        Fs = []
        lags, now, leads = self.make_all_variable_time_combinations()
        shocks = self.shocks
        for var_group in [lags, now, leads, shocks]:
            F = []

            # If the user selects a variable to not be log linearized, we need to set the value in T to be one, but
            # still replace all SS values in A, B, C, D as usual. These dummies facilitate that.
            # T = sp.diag(*[TimeAwareSymbol(x.base_name + '_T', 'ss') for x in var_group])

            for eq in self.system_equations:
                F_row = []
                for var in var_group:
                    dydx = sp.powsimp(eq_to_ss(eq.diff(var)))
                    dydx *= (
                        1.0 if var.base_name in not_loglin_variables else var.to_ss()
                    )
                    atoms = dydx.atoms()
                    if len(atoms) == 1:
                        x = list(atoms)[0]
                        if isinstance(x, sp.core.numbers.Number) and x != 0:
                            dydx = sp.Float(x)
                    F_row.append(dydx)

                F.append(F_row)
            F = sp.Matrix(F)
            # Fs.append(sp.MatMul(F, T, evaluate=False))
            Fs.append(F)

        return Fs

    def convert_linear_system_to_matrices(self) -> List[sp.Matrix]:
        """

        :return: List of sympy Matrices representing the linear system

        If the model has already been log-linearized by hand, this method is used to simplify the construction of the
        solution matrices. Following the gEcon user's guide, page 54, equation 9.10, the solution should be of the form:

            A @ y_{t-1} + B @ y_t + C @ y_{t+1} + D @ epsilon = 0

        This function organizes the model equations and returns matrices A, B, C, and D.

        TODO: Add some checks to ensure that the model is indeed linear so that this can't be erroneously called.
        """

        Fs = []
        lags, now, leads = self.make_all_variable_time_combinations()
        shocks = self.shocks
        model = self.system_equations

        for var_group, name in zip(
            [lags, now, leads, shocks], ["lags", "now", "leads", "shocks"]
        ):
            F = (
                sp.zeros(len(var_group))
                if name != "shocks"
                else sp.zeros(rows=len(model), cols=len(var_group))
            )
            for i, var in enumerate(var_group):
                for j, eq in enumerate(model):
                    args = eq.expand().args
                    for arg in args:
                        if var in arg.atoms():
                            F[j, i] = sp.simplify(arg / var)
            Fs.append(F)

        return Fs

    def make_all_variable_time_combinations(
        self,
    ) -> Tuple[List[TimeAwareSymbol], List[TimeAwareSymbol], List[TimeAwareSymbol]]:
        """
        :return: Tuple of three lists, containing all model variables at time steps t-1, t, and t+1, respectively.
        """

        now = sorted(self.variables, key=lambda x: x.base_name)
        lags = [x.step_backward() for x in now]
        leads = [x.step_forward() for x in now]

        return lags, now, leads
