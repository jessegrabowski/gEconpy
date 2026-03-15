from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.base import IterationStats, RootFunction, SolverState, StepInfo, merit


def _steihaug_cg(JtJ: sp.spmatrix, g: np.ndarray, delta: float, max_cg_iter: int = 0) -> np.ndarray:
    """Steihaug-CG: truncated conjugate gradient on the trust region subproblem.

    Approximately minimizes ``0.5 p^T H p + g^T p`` subject to ``||p|| ≤ delta``
    where ``H = J^T J``.

    Returns the step ``p``.
    """
    n = len(g)
    if max_cg_iter <= 0:
        max_cg_iter = 2 * n

    p = np.zeros(n)
    r = g.copy()
    d = -r.copy()
    r_dot_r = np.dot(r, r)

    if np.sqrt(r_dot_r) < 1e-15:
        return p

    for _ in range(max_cg_iter):
        Hd = JtJ @ d
        dHd = np.dot(d, Hd)

        # Negative curvature — go to trust region boundary along d
        if dHd <= 0:
            return _boundary_step(p, d, delta)

        alpha = r_dot_r / dHd
        p_next = p + alpha * d

        # If step exceeds trust region, go to boundary
        if np.linalg.norm(p_next) >= delta:
            return _boundary_step(p, d, delta)

        p = p_next
        r = r + alpha * Hd
        r_dot_r_new = np.dot(r, r)

        if np.sqrt(r_dot_r_new) < 1e-10 * np.linalg.norm(g):
            return p

        beta = r_dot_r_new / r_dot_r
        d = -r + beta * d
        r_dot_r = r_dot_r_new

    return p


def _boundary_step(p: np.ndarray, d: np.ndarray, delta: float) -> np.ndarray:
    """Find τ ≥ 0 such that ||p + τ d|| = delta."""
    pp = np.dot(p, p)
    pd = np.dot(p, d)
    dd = np.dot(d, d)
    # Solve ||p + τ d||^2 = delta^2
    # dd τ^2 + 2 pd τ + (pp - delta^2) = 0
    discriminant = pd * pd - dd * (pp - delta * delta)
    tau = (-pd + np.sqrt(max(discriminant, 0.0))) / dd
    return p + tau * d


@dataclass
class GaussNewtonTrustRegion:
    """Gauss-Newton solver with trust region globalization via Steihaug-CG.

    Solves ``min_p ||J p + r||^2`` subject to ``||p|| ≤ Δ`` using the
    Steihaug-CG method on the normal equations. Adjusts the trust region
    radius ``Δ`` based on the actual-to-predicted reduction ratio.

    Parameters
    ----------
    delta0 : float
        Initial trust region radius.
    delta_max : float
        Maximum trust region radius.
    eta : float
        Minimum actual/predicted reduction ratio to accept a step.
    shrink_factor : float
        Factor to shrink ``Δ`` on a rejected step.
    grow_factor : float
        Factor to grow ``Δ`` on a very good step (``ρ > 0.75``).
    max_reject : int
        Maximum consecutive rejected steps before reporting failure.
    """

    delta0: float = 1.0
    delta_max: float = 100.0
    eta: float = 0.1
    shrink_factor: float = 0.25
    grow_factor: float = 2.0
    max_reject: int = 50

    _delta: float = field(init=False, repr=False, default=0.0)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        self._delta = self.delta0
        x = np.asarray(x0, dtype=np.float64).copy()
        res, jac = fun(x, *args)
        return SolverState(x=x, res=res, jac=jac, phi=merit(res), stats=IterationStats(nfev=1, njev=1))

    def step(self, fun: RootFunction, state: SolverState, args: tuple) -> tuple[SolverState, StepInfo]:
        J = state.jac
        r = state.res
        JtJ = J.T @ J
        g = J.T @ r

        consecutive_rejects = 0
        nfev = 0

        while True:
            p = _steihaug_cg(JtJ, g, self._delta)

            x_trial = state.x + p
            res_trial, jac_trial = fun(x_trial, *args)
            phi_trial = merit(res_trial)
            nfev += 1

            # Predicted reduction from the quadratic model
            Jp = J @ p
            pred = -(float(g @ p) + 0.5 * float(Jp @ Jp))

            if pred <= 0:
                # Model predicts no improvement — shrink trust region
                self._delta *= self.shrink_factor
                consecutive_rejects += 1
                if consecutive_rejects >= self.max_reject:
                    return state, StepInfo(
                        accepted=False,
                        step=np.zeros_like(state.x),
                        message="fatal: trust region predicted reduction non-positive",
                    )
                continue

            actual = state.phi - phi_trial
            rho = actual / pred

            if rho > self.eta:
                # Accept
                if rho > 0.75 and np.linalg.norm(p) > 0.9 * self._delta:
                    self._delta = min(self.grow_factor * self._delta, self.delta_max)
                new_state = SolverState(
                    x=x_trial,
                    res=res_trial,
                    jac=jac_trial,
                    phi=phi_trial,
                    stats=state.stats.update(nit=1, nfev=nfev, njev=nfev, nsolve=1, nreject=consecutive_rejects),
                )
                return new_state, StepInfo(accepted=True, step=p)
            # Reject — shrink
            self._delta *= self.shrink_factor
            consecutive_rejects += 1
            if consecutive_rejects >= self.max_reject:
                return state, StepInfo(
                    accepted=False,
                    step=np.zeros_like(state.x),
                    message="fatal: trust region exceeded max rejected steps",
                )
