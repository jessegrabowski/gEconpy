from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from scipy.sparse.linalg import spsolve

from gEconpy.solvers.sparse_root.base import RootFunction, SolverState, StepInfo, initial_state, merit
from gEconpy.solvers.sparse_root.direction import try_linear_solve


@dataclass
class LevenbergMarquardt:
    r"""Levenberg-Marquardt solver for sparse nonlinear systems.

    Solves :math:`(J^T J + \lambda D) p = -J^T r`, where :math:`D` is a diagonal scaling matrix. A step is accepted
    when the actual-to-predicted reduction ratio exceeds ``eta``, and the damping :math:`\lambda` falls on acceptance
    and rises on rejection.

    Parameters
    ----------
    lam0 : float, optional
        Initial damping parameter. Defaults to 1e-3.
    lam_up : float, optional
        Factor applied to the damping on a rejected step. Defaults to 10.0.
    lam_down : float, optional
        Factor applied to the damping on an accepted step. Defaults to 0.3.
    eta : float, optional
        Minimum actual-to-predicted reduction ratio to accept a step. Defaults to 1e-4.
    min_lam : float, optional
        Floor for the damping parameter. Defaults to 1e-15.
    max_lam : float, optional
        Ceiling for the damping parameter. The solver reports a fatal failure when the damping reaches it. Defaults
        to 1e15.
    max_reject : int, optional
        Maximum consecutive rejected steps before reporting failure. Defaults to 50.
    linear_solver : callable, optional
        Sparse linear solver called as ``linear_solver(A, b)``. Defaults to :func:`scipy.sparse.linalg.spsolve`.
    """

    lam0: float = 1e-3
    lam_up: float = 10.0
    lam_down: float = 0.3
    eta: float = 1e-4
    min_lam: float = 1e-15
    max_lam: float = 1e15
    max_reject: int = 50
    linear_solver: Callable = spsolve

    _lam: float = field(init=False, repr=False, default=0.0)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        """Evaluate ``fun`` at ``x0``, reset the damping, and build the initial solver state.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        x0 : ndarray
            Initial guess for the root.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        state : ~gEconpy.solvers.sparse_root.base.SolverState
            State holding the initial point, residuals, Jacobian, merit value and evaluation counts.
        """
        self._lam = self.lam0
        return initial_state(fun, x0, args)

    def step(self, fun: RootFunction, state: SolverState, args: tuple) -> tuple[SolverState, StepInfo]:
        """Take one Levenberg-Marquardt iteration from ``state``.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        state : ~gEconpy.solvers.sparse_root.base.SolverState
            Current solver state.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        new_state : ~gEconpy.solvers.sparse_root.base.SolverState
            State after the accepted step, or the unchanged input state when no step is accepted.
        info : ~gEconpy.solvers.sparse_root.base.StepInfo
            Whether a step was accepted, the step taken, and a failure message when it was not.
        """
        jac = state.jac
        normal_matrix = (jac.T @ jac).tocsc()
        grad = jac.T @ state.res

        scaling = np.maximum(1.0, np.asarray(normal_matrix.diagonal()).ravel())
        diag_indices = np.arange(normal_matrix.shape[0])
        nfev = 0

        for n_rejected in range(self.max_reject):
            damped_matrix = normal_matrix.copy()
            damped_matrix[diag_indices, diag_indices] += self._lam * scaling
            p = try_linear_solve(self.linear_solver, damped_matrix, -grad)

            if p is not None:
                jac_p = jac @ p
                damping_term = self._lam * float(scaling @ (p * p))
                predicted = -(float(grad @ p) + 0.5 * (float(jac_p @ jac_p) + damping_term))

                if predicted > 0:
                    x_trial = state.x + p
                    res_trial, jac_trial = fun(x_trial, *args)
                    phi_trial = merit(res_trial)
                    nfev += 1

                    rho = (state.phi - phi_trial) / predicted
                    if rho > self.eta:
                        self._lam = max(self._lam * self.lam_down, self.min_lam)
                        stats = state.stats.update(nit=1, nfev=nfev, njev=nfev, nsolve=1, nreject=n_rejected)
                        new_state = SolverState(x=x_trial, res=res_trial, jac=jac_trial, phi=phi_trial, stats=stats)
                        return new_state, StepInfo(accepted=True, step=p)

            self._lam = min(self._lam * self.lam_up, self.max_lam)
            if self._lam >= self.max_lam:
                break

        return state, StepInfo(
            accepted=False,
            step=np.zeros_like(state.x),
            message=f"fatal: Levenberg-Marquardt rejected every step, damping reached {self._lam:.1e}",
        )
