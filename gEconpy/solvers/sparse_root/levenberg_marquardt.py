from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from scipy.sparse.linalg import spsolve

from gEconpy.solvers.sparse_root.base import IterationStats, RootFunction, SolverState, StepInfo, merit


@dataclass
class LevenbergMarquardt:
    """Levenberg-Marquardt solver for sparse nonlinear systems.

    Solves ``(J^T J + λ D) p = -J^T r`` where ``D`` is a diagonal scaling matrix.
    Accepts a step when the actual-to-predicted reduction ratio ``ρ > η``, and
    adjusts the damping parameter ``λ`` accordingly.

    Parameters
    ----------
    lam0 : float
        Initial damping parameter.
    lam_up : float
        Factor by which to increase ``λ`` on a rejected step.
    lam_down : float
        Factor by which to decrease ``λ`` on an accepted step.
    eta : float
        Minimum actual/predicted reduction ratio to accept a step.
    min_lam : float
        Floor for the damping parameter.
    max_lam : float
        Ceiling for the damping parameter. If ``λ`` exceeds this, the solver
        reports a fatal failure.
    max_reject : int
        Maximum consecutive rejected steps before reporting failure.
    linear_solver : callable or None
        Sparse linear solver. Defaults to ``spsolve``.
    """

    lam0: float = 1e-3
    lam_up: float = 10.0
    lam_down: float = 0.3
    eta: float = 1e-4
    min_lam: float = 1e-15
    max_lam: float = 1e15
    max_reject: int = 50
    linear_solver: Callable = field(default=None)

    # Mutable state — reset in init
    _lam: float = field(init=False, repr=False, default=0.0)
    _nreject: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if self.linear_solver is None:
            self.linear_solver = spsolve

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        self._lam = self.lam0
        self._nreject = 0
        x = np.asarray(x0, dtype=np.float64).copy()
        res, jac = fun(x, *args)
        return SolverState(x=x, res=res, jac=jac, phi=merit(res), stats=IterationStats(nfev=1, njev=1))

    def step(self, fun: RootFunction, state: SolverState, args: tuple) -> tuple[SolverState, StepInfo]:
        J = state.jac
        r = state.res
        JtJ = J.T @ J
        g = J.T @ r

        # Diagonal scaling for scale invariance
        diag_JtJ = np.asarray(JtJ.diagonal()).ravel()
        D_diag = np.maximum(1.0, diag_JtJ)

        # Pre-build the LHS sparsity pattern once; update diagonal per lambda
        lhs_base = JtJ.tocsc().copy()
        diag_indices = np.arange(lhs_base.shape[0])

        consecutive_rejects = 0
        nfev = 0

        while True:
            # Solve the damped normal equations — update diagonal in-place
            lhs = lhs_base.copy()
            lhs[diag_indices, diag_indices] += self._lam * D_diag

            try:
                p = self.linear_solver(lhs, -g)
                if not np.all(np.isfinite(p)):
                    raise ValueError("non-finite direction")  # noqa: TRY301
            except Exception:
                # If solve fails, increase damping and retry
                self._lam = min(self._lam * self.lam_up, self.max_lam)
                consecutive_rejects += 1
                if consecutive_rejects >= self.max_reject or self._lam >= self.max_lam:
                    return state, StepInfo(
                        accepted=False,
                        step=np.zeros_like(state.x),
                        message="fatal: LM linear solve failed repeatedly",
                    )
                continue

            x_trial = state.x + p
            res_trial, jac_trial = fun(x_trial, *args)
            phi_trial = merit(res_trial)
            nfev += 1

            # Predicted reduction
            Jp = J @ p
            pred = -(float(g @ p) + 0.5 * (float(Jp @ Jp) + self._lam * float(D_diag @ (p * p))))

            if pred <= 0:
                # Model predicts no improvement — increase damping
                self._lam = min(self._lam * self.lam_up, self.max_lam)
                consecutive_rejects += 1
                self._nreject += 1
                if consecutive_rejects >= self.max_reject or self._lam >= self.max_lam:
                    return state, StepInfo(
                        accepted=False,
                        step=np.zeros_like(state.x),
                        message="fatal: LM predicted reduction non-positive",
                    )
                continue

            actual = state.phi - phi_trial
            rho = actual / pred

            if rho > self.eta:
                # Accept the step
                self._lam = max(self._lam * self.lam_down, self.min_lam)
                new_state = SolverState(
                    x=x_trial,
                    res=res_trial,
                    jac=jac_trial,
                    phi=phi_trial,
                    stats=state.stats.update(nit=1, nfev=nfev, njev=nfev, nsolve=1, nreject=consecutive_rejects),
                )
                return new_state, StepInfo(accepted=True, step=p)
            # Reject — increase damping and retry
            self._lam = min(self._lam * self.lam_up, self.max_lam)
            consecutive_rejects += 1
            self._nreject += 1
            if consecutive_rejects >= self.max_reject or self._lam >= self.max_lam:
                return state, StepInfo(
                    accepted=False,
                    step=np.zeros_like(state.x),
                    message="fatal: LM exceeded max rejected steps",
                )
