from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import bicgstab, gmres, spsolve


@dataclass(frozen=True, slots=True)
class DirectionProposal:
    direction: np.ndarray
    slope: float
    kind: str


class DirectionStrategy(Protocol):
    def compute(self, x: np.ndarray, res: np.ndarray, jac: sp.spmatrix) -> DirectionProposal: ...


@dataclass
class NewtonDirection:
    linear_solver: callable = field(default=None)

    def __post_init__(self):
        if self.linear_solver is None:
            self.linear_solver = spsolve

    def compute(
        self,
        x: np.ndarray,  # noqa: ARG002
        res: np.ndarray,
        jac: sp.spmatrix,
    ) -> DirectionProposal:
        try:
            dx = self.linear_solver(jac, -res)
            if not np.all(np.isfinite(dx)):
                msg = "non-finite"
                raise ValueError(msg)  # noqa: TRY301
            kind = "newton"
        except Exception:
            dx = -(jac.T @ res)
            kind = "gradient_fallback"

        slope = float(np.dot(res, jac @ dx))
        if slope >= 0:
            dx = -dx
            slope = -slope
            kind = f"{kind}_flipped"

        return DirectionProposal(direction=dx, slope=slope, kind=kind)


@dataclass
class ChordDirection:
    """Direction strategy that caches and reuses the Jacobian.

    Instead of solving with the current Jacobian every step, the Chord method
    solves with a cached Jacobian, refreshing it every ``recompute_every`` calls.
    The residual is always current — only the linear solve reuses old data.

    Parameters
    ----------
    linear_solver : callable or None
        Sparse linear solver. Defaults to ``spsolve``.
    recompute_every : int
        Number of direction computations between Jacobian refreshes.
    """

    linear_solver: callable = field(default=None)
    recompute_every: int = 5
    _cached_jac: sp.spmatrix = field(init=False, repr=False, default=None)
    _call_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if self.linear_solver is None:
            self.linear_solver = spsolve

    def reset(self):
        """Clear cached Jacobian. Called by the solver's ``init``."""
        self._cached_jac = None
        self._call_count = 0

    def compute(
        self,
        x: np.ndarray,  # noqa: ARG002
        res: np.ndarray,
        jac: sp.spmatrix,
    ) -> DirectionProposal:
        # Refresh cache when needed
        if self._cached_jac is None or self._call_count % self.recompute_every == 0:
            self._cached_jac = jac.copy()
        self._call_count += 1

        J = self._cached_jac

        try:
            dx = self.linear_solver(J, -res)
            if not np.all(np.isfinite(dx)):
                msg = "non-finite"
                raise ValueError(msg)  # noqa: TRY301
            kind = "chord"
        except Exception:
            dx = -(J.T @ res)
            kind = "chord_gradient_fallback"

        # Slope is computed with the *current* Jacobian for correct Armijo test
        slope = float(np.dot(res, jac @ dx))
        if slope >= 0:
            dx = -dx
            slope = -slope
            kind = f"{kind}_flipped"

        return DirectionProposal(direction=dx, slope=slope, kind=kind)


_KRYLOV_METHODS = {
    "gmres": gmres,
    "bicgstab": bicgstab,
}


@dataclass
class KrylovDirection:
    """Direction via iterative Krylov solve with Eisenstat-Walker forcing.

    Solves the Newton system ``J dx = -r`` approximately using GMRES or BiCGSTAB.
    The tolerance for the iterative solve is ``η_k ||r_k||``, where ``η_k``
    is adapted via the Eisenstat-Walker formula to tighten as the solver
    approaches the solution.

    Parameters
    ----------
    krylov_method : str
        Name of the Krylov method: ``"gmres"`` or ``"bicgstab"``.
    eta_max : float
        Upper bound on the forcing term.
    eta_min : float
        Lower bound on the forcing term.
    eisenstat_walker : bool
        Whether to adapt the forcing term. If ``False``, ``eta_max`` is used
        throughout.
    """

    krylov_method: str = "gmres"
    eta_max: float = 0.9
    eta_min: float = 1e-6
    eisenstat_walker: bool = True

    _eta: float = field(init=False, repr=False, default=0.9)
    _prev_res_norm: float = field(init=False, repr=False, default=0.0)
    _prev_pred_norm: float = field(init=False, repr=False, default=0.0)

    def reset(self):
        """Clear adaptive state. Called by the solver's ``init``."""
        self._eta = self.eta_max
        self._prev_res_norm = 0.0
        self._prev_pred_norm = 0.0

    def compute(
        self,
        x: np.ndarray,  # noqa: ARG002
        res: np.ndarray,
        jac: sp.spmatrix,
    ) -> DirectionProposal:
        res_norm = np.linalg.norm(res)

        # Eisenstat-Walker forcing term adaptation
        if self.eisenstat_walker and self._prev_res_norm > 0:
            eta_new = abs(res_norm - self._prev_pred_norm) / self._prev_res_norm
            # Safeguard: don't let η decrease too fast
            eta_safe = self._eta**2
            self._eta = float(np.clip(max(eta_new, eta_safe), self.eta_min, self.eta_max))

        tol = self._eta * res_norm

        solver_fn = _KRYLOV_METHODS.get(self.krylov_method)
        if solver_fn is None:
            raise ValueError(f"Unknown Krylov method: {self.krylov_method!r}. Choose from {list(_KRYLOV_METHODS)}")

        try:
            dx, info = solver_fn(jac, -res, atol=tol)
            if info != 0 or not np.all(np.isfinite(dx)):
                raise ValueError("Krylov solve did not converge")  # noqa: TRY301
            kind = f"krylov_{self.krylov_method}"
        except Exception:
            dx = -(jac.T @ res)
            kind = f"krylov_{self.krylov_method}_gradient_fallback"

        # Track for Eisenstat-Walker
        self._prev_res_norm = res_norm
        self._prev_pred_norm = np.linalg.norm(res + jac @ dx)

        slope = float(np.dot(res, jac @ dx))
        if slope >= 0:
            dx = -dx
            slope = -slope
            kind = f"{kind}_flipped"

        return DirectionProposal(direction=dx, slope=slope, kind=kind)
