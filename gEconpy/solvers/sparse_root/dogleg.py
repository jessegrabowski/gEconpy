from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve

from gEconpy.solvers.sparse_root.base import IterationStats, RootFunction, SolverState, StepInfo, merit


@dataclass
class SparseDogleg:
    """Powell dogleg solver within a trust region.

    Computes both the Cauchy point (steepest descent with optimal step length)
    and the Newton point. If the Newton step is within the trust region, take it.
    If the Cauchy point is outside, scale the steepest descent direction to the
    boundary. Otherwise, interpolate along the dogleg path.

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
    linear_solver : callable or None
        Sparse linear solver for the Newton step. Defaults to ``spsolve``.
    """

    delta0: float = 1.0
    delta_max: float = 100.0
    eta: float = 0.1
    shrink_factor: float = 0.25
    grow_factor: float = 2.0
    max_reject: int = 50
    linear_solver: Callable = field(default=None)

    _delta: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        if self.linear_solver is None:
            self.linear_solver = spsolve

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        self._delta = self.delta0
        x = np.asarray(x0, dtype=np.float64).copy()
        res, jac = fun(x, *args)
        return SolverState(x=x, res=res, jac=jac, phi=merit(res), stats=IterationStats(nfev=1, njev=1))

    def _compute_dogleg_step(self, J: sp.spmatrix, r: np.ndarray, delta: float) -> np.ndarray:  # noqa: PLR0911
        """Compute the dogleg step within the trust region of radius ``delta``."""
        g = J.T @ r  # gradient of merit function
        Jg = J @ g

        # Cauchy step: steepest descent with optimal step length
        gTg = float(np.dot(g, g))
        JgTJg = float(np.dot(Jg, Jg))

        if JgTJg < 1e-30:
            # Zero curvature along gradient
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-30:
                return np.zeros_like(r)
            return -(delta / g_norm) * g

        alpha_c = gTg / JgTJg
        p_c = -alpha_c * g  # Cauchy point

        # Newton step
        try:
            p_n = self.linear_solver(J, -r)
            if not np.all(np.isfinite(p_n)):
                raise ValueError("non-finite Newton step")  # noqa: TRY301
        except Exception:
            # Fall back to Cauchy point (clipped to trust region)
            p_c_norm = np.linalg.norm(p_c)
            if p_c_norm <= delta:
                return p_c
            return (delta / p_c_norm) * p_c

        p_n_norm = np.linalg.norm(p_n)

        # If Newton step is inside trust region, take it
        if p_n_norm <= delta:
            return p_n

        p_c_norm = np.linalg.norm(p_c)

        # If Cauchy point is outside trust region, scale gradient to boundary
        if p_c_norm >= delta:
            return (delta / p_c_norm) * p_c

        # Dogleg: interpolate between Cauchy and Newton
        # Find τ such that ||p_c + τ (p_n - p_c)|| = delta
        diff = p_n - p_c
        dd = np.dot(diff, diff)
        cd = np.dot(p_c, diff)
        cc = np.dot(p_c, p_c)
        discriminant = cd * cd - dd * (cc - delta * delta)
        tau = (-cd + np.sqrt(max(discriminant, 0.0))) / dd
        return p_c + tau * diff

    def step(self, fun: RootFunction, state: SolverState, args: tuple) -> tuple[SolverState, StepInfo]:
        J = state.jac
        r = state.res
        g = J.T @ r

        consecutive_rejects = 0
        nfev = 0

        while True:
            p = self._compute_dogleg_step(J, r, self._delta)

            x_trial = state.x + p
            res_trial, jac_trial = fun(x_trial, *args)
            phi_trial = merit(res_trial)
            nfev += 1

            # Predicted reduction from the linear model
            Jp = J @ p
            pred = -(float(g @ p) + 0.5 * float(Jp @ Jp))

            if pred <= 0:
                self._delta *= self.shrink_factor
                consecutive_rejects += 1
                if consecutive_rejects >= self.max_reject:
                    return state, StepInfo(
                        accepted=False,
                        step=np.zeros_like(state.x),
                        message="fatal: dogleg predicted reduction non-positive",
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
            self._delta *= self.shrink_factor
            consecutive_rejects += 1
            if consecutive_rejects >= self.max_reject:
                return state, StepInfo(
                    accepted=False,
                    step=np.zeros_like(state.x),
                    message="fatal: dogleg exceeded max rejected steps",
                )
