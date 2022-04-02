from typing import Optional, List, Tuple
from numpy.typing import ArrayLike

from gEcon.shared.utilities import string_keys_to_sympy, eq_to_ss
from gEcon.classes.time_aware_symbol import TimeAwareSymbol
from gEcon.exceptions.exceptions import VariableNotFoundException, SteadyStateNotSolvedError, GensysFailedException
from gEcon.solvers.gensys import gensys

from warnings import warn
import numpy as np
import sympy as sp
from scipy import linalg
import pandas as pd


class PerturbationSolver:

    def __init__(self, model):
        self.steady_state_dict = model.steady_state_dict
        self.steady_state_solved = model.steady_state_solved
        self.param_dict = model.param_dict
        self.system_equations = model.system_equations
        self.variables = model.variables

        self.shocks = model.shocks
        self.n_shocks = model.n_shocks

    @staticmethod
    def solve_policy_function_with_gensys(A: sp.Matrix, B: sp.Matrix, C: sp.Matrix, D: sp.Matrix,
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

    def extract_policy_matrices(self, A, G_1, impact, variables, tol):
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
    def residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime):
        B_np = np.array(B).astype(np.float64)
        C_np = np.array(C).astype(np.float64)
        D_np = np.array(D).astype(np.float64)

        norm_deterministic = linalg.norm(A_prime + B_np @ R_prime +
                                         C_np @ R_prime @ P)

        norm_stochastic = linalg.norm(B_np @ S_prime +
                                      C_np @ R_prime @ Q + D_np)

        return norm_deterministic, norm_stochastic

    def log_linearize_model(self) -> List[sp.Matrix]:
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
        lags, now, leads = self.make_all_variable_time_combinations()
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

    def make_all_variable_time_combinations(self) -> Tuple[List[TimeAwareSymbol],
                                                           List[TimeAwareSymbol],
                                                           List[TimeAwareSymbol]]:
        """
        :return: Tuple of three lists, containing all model variables at time steps t-1, t, and t+1, respectively.
        """

        now = sorted(self.variables, key=lambda x: x.base_name)
        lags = [x.step_backward() for x in now]
        leads = [x.step_forward() for x in now]

        return lags, now, leads
