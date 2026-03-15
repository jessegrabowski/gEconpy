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
