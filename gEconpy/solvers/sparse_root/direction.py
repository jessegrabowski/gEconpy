from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve


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
