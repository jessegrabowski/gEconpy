from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.base import RootFunction, SolverState, StepInfo, initial_state, merit


@dataclass
class GaussNewtonTrustRegion:
    r"""Gauss-Newton solver with trust region globalization via Steihaug-CG.

    Solves :math:`\min_p \lVert J p + r \rVert^2` subject to :math:`\lVert p \rVert \le \Delta` with the Steihaug-CG
    method on the normal equations, and adjusts the radius :math:`\Delta` from the actual-to-predicted reduction
    ratio.

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
    """

    delta0: float = 1.0
    delta_max: float = 100.0
    eta: float = 0.1
    shrink_factor: float = 0.25
    grow_factor: float = 2.0
    max_reject: int = 50

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
        """Take one trust region iteration from ``state``.

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
        normal_matrix = jac.T @ jac
        grad = jac.T @ state.res
        nfev = 0

        for n_rejected in range(self.max_reject):
            p = _steihaug_cg(normal_matrix, grad, self._delta)
            jac_p = jac @ p
            predicted = -(float(grad @ p) + 0.5 * float(jac_p @ jac_p))

            if predicted > 0:
                x_trial = state.x + p
                res_trial, jac_trial = fun(x_trial, *args)
                phi_trial = merit(res_trial)
                nfev += 1

                rho = (state.phi - phi_trial) / predicted
                if rho > self.eta:
                    if rho > 0.75 and np.linalg.norm(p) > 0.9 * self._delta:
                        self._delta = min(self.grow_factor * self._delta, self.delta_max)
                    stats = state.stats.update(nit=1, nfev=nfev, njev=nfev, nsolve=1, nreject=n_rejected)
                    new_state = SolverState(x=x_trial, res=res_trial, jac=jac_trial, phi=phi_trial, stats=stats)
                    return new_state, StepInfo(accepted=True, step=p)

            self._delta *= self.shrink_factor

        return state, StepInfo(
            accepted=False,
            step=np.zeros_like(state.x),
            message=f"fatal: trust region rejected {self.max_reject} consecutive steps",
        )


def _steihaug_cg(hessian: sp.spmatrix, grad: np.ndarray, delta: float, max_cg_iter: int = 0) -> np.ndarray:
    r"""Approximately minimize :math:`\frac{1}{2} p^T H p + g^T p` subject to :math:`\lVert p \rVert \le \delta`.

    Parameters
    ----------
    hessian : sparse matrix
        Model Hessian :math:`H`, the Gauss-Newton normal matrix :math:`J^T J`.
    grad : ndarray
        Model gradient :math:`g`.
    delta : float
        Trust region radius.
    max_cg_iter : int, optional
        Cap on conjugate gradient iterations. Defaults to 0, meaning twice the problem dimension.

    Returns
    -------
    p : ndarray
        Step. Lies on the boundary when the iteration met it or found negative curvature.
    """
    n = len(grad)
    if max_cg_iter <= 0:
        max_cg_iter = 2 * n

    p = np.zeros(n)
    residual = grad.copy()
    direction = -residual
    residual_sq = np.dot(residual, residual)

    if np.sqrt(residual_sq) < 1e-15:
        return p

    for _ in range(max_cg_iter):
        hessian_direction = hessian @ direction
        curvature = np.dot(direction, hessian_direction)

        if curvature <= 0:
            return _boundary_step(p, direction, delta)

        alpha = residual_sq / curvature
        p_next = p + alpha * direction

        if np.linalg.norm(p_next) >= delta:
            return _boundary_step(p, direction, delta)

        p = p_next
        # residual and direction are owned by this function, so in-place updates save an allocation per iteration.
        residual += alpha * hessian_direction
        residual_sq_next = np.dot(residual, residual)

        if np.sqrt(residual_sq_next) < 1e-10 * np.linalg.norm(grad):
            return p

        direction *= residual_sq_next / residual_sq
        direction -= residual
        residual_sq = residual_sq_next

    return p


def _boundary_step(p: np.ndarray, d: np.ndarray, delta: float) -> np.ndarray:
    """Return ``p + tau * d`` for the ``tau >= 0`` that puts it on the sphere of radius ``delta``."""
    pp = np.dot(p, p)
    pd = np.dot(p, d)
    dd = np.dot(d, d)
    discriminant = pd * pd - dd * (pp - delta * delta)
    tau = (-pd + np.sqrt(max(discriminant, 0.0))) / dd
    return p + tau * d
