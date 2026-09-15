from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve

from gEconpy.solvers.sparse_root.base import (
    TRUST_REGION_BOUNDARY_FRACTION,
    TRUST_REGION_GROW_RHO,
    RootFunction,
    SolverState,
    StepInfo,
    initial_state,
    merit,
)
from gEconpy.solvers.sparse_root.direction import try_linear_solve

ZERO_CURVATURE_TOL = 1e-30


@dataclass
class SparseDogleg:
    r"""Powell dogleg solver within a trust region.

    Computes both the Cauchy point (steepest descent with optimal step length) and the Newton point. When the Newton
    step lies inside the trust region it is taken as is. When even the Cauchy point lies outside, the steepest
    descent direction is scaled to the boundary. Otherwise the step is the point on the segment from the Cauchy point
    to the Newton point that meets the boundary.

    Parameters
    ----------
    delta0 : float, optional
        Initial trust region radius. Defaults to 1.0.
    delta_max : float, optional
        Maximum trust region radius. Defaults to 100.0.
    eta : float, optional
        Minimum actual-to-predicted reduction ratio to accept a step. Defaults to 0.1.
    shrink_factor : float, optional
        Factor applied to the radius on a rejected step. Defaults to 0.25.
    grow_factor : float, optional
        Factor applied to the radius when the ratio exceeds 0.75 and the step reached the boundary. Defaults to 2.0.
    max_reject : int, optional
        Maximum consecutive rejected steps before reporting failure. Defaults to 50.
    linear_solver : callable, optional
        Sparse linear solver for the Newton step, called as ``linear_solver(A, b)``. Defaults to
        :func:`scipy.sparse.linalg.spsolve`.
    """

    delta0: float = 1.0
    delta_max: float = 100.0
    eta: float = 0.1
    shrink_factor: float = 0.25
    grow_factor: float = 2.0
    max_reject: int = 50
    linear_solver: Callable = spsolve

    _delta: float = field(init=False, repr=False, default=0.0)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        """Evaluate ``fun`` at ``x0``, reset the trust region radius, and build the initial solver state.

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
        self._delta = self.delta0
        return initial_state(fun, x0, args)

    def step(self, fun: RootFunction, state: SolverState, args: tuple) -> tuple[SolverState, StepInfo]:
        """Take one dogleg iteration from ``state``.

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
        res = state.res
        grad = jac.T @ res
        nfev = 0

        for n_rejected in range(self.max_reject):
            p = self._compute_dogleg_step(jac, res, grad, self._delta)
            jac_p = jac @ p
            predicted = -(float(grad @ p) + 0.5 * float(jac_p @ jac_p))

            if predicted > 0:
                x_trial = state.x + p
                res_trial, jac_trial = fun(x_trial, *args)
                phi_trial = merit(res_trial)
                nfev += 1

                rho = (state.phi - phi_trial) / predicted
                if rho > self.eta:
                    if rho > TRUST_REGION_GROW_RHO and np.linalg.norm(p) > TRUST_REGION_BOUNDARY_FRACTION * self._delta:
                        self._delta = min(self.grow_factor * self._delta, self.delta_max)
                    stats = state.stats.update(nit=1, nfev=nfev, njev=nfev, nsolve=1, nreject=n_rejected)
                    new_state = SolverState(x=x_trial, res=res_trial, jac=jac_trial, phi=phi_trial, stats=stats)
                    return new_state, StepInfo(accepted=True, step=p)

            self._delta *= self.shrink_factor

        return state, StepInfo(
            accepted=False,
            step=np.zeros_like(state.x),
            message=f"fatal: dogleg rejected {self.max_reject} consecutive steps",
        )

    def _compute_dogleg_step(self, jac: sp.spmatrix, res: np.ndarray, grad: np.ndarray, delta: float) -> np.ndarray:
        jac_grad = jac @ grad
        grad_sq = float(np.dot(grad, grad))
        curvature = float(np.dot(jac_grad, jac_grad))

        if curvature < ZERO_CURVATURE_TOL:
            grad_norm = np.linalg.norm(grad)
            if grad_norm < ZERO_CURVATURE_TOL:
                return np.zeros_like(res)
            return -(delta / grad_norm) * grad

        cauchy = -(grad_sq / curvature) * grad
        cauchy_norm = np.linalg.norm(cauchy)

        newton = try_linear_solve(self.linear_solver, jac, -res)
        if newton is None:
            return cauchy if cauchy_norm <= delta else (delta / cauchy_norm) * cauchy

        if np.linalg.norm(newton) <= delta:
            return newton
        if cauchy_norm >= delta:
            return (delta / cauchy_norm) * cauchy
        return _dogleg_boundary_point(cauchy, newton, delta)


def _dogleg_boundary_point(cauchy: np.ndarray, newton: np.ndarray, delta: float) -> np.ndarray:
    """Return the point on the segment from ``cauchy`` to ``newton`` whose norm equals ``delta``."""
    diff = newton - cauchy
    dd = np.dot(diff, diff)
    cd = np.dot(cauchy, diff)
    cc = np.dot(cauchy, cauchy)
    discriminant = cd * cd - dd * (cc - delta * delta)
    tau = (-cd + np.sqrt(max(discriminant, 0.0))) / dd
    return cauchy + tau * diff
