from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.base import (
    DEFAULT_ARMIJO_BETA,
    DEFAULT_ARMIJO_C1,
    DEFAULT_ARMIJO_MAX_ITER,
    RootFunction,
    merit,
)
from gEconpy.solvers.sparse_root.direction import DirectionProposal


@dataclass(frozen=True, slots=True)
class LineSearchResult:
    x_new: np.ndarray
    res_new: np.ndarray
    jac_new: sp.spmatrix
    phi_new: float
    alpha: float
    n_evals: int


class GlobalizationStrategy(Protocol):
    def search(
        self,
        fun: RootFunction,
        x: np.ndarray,
        phi_current: float,
        proposal: DirectionProposal,
        args: tuple[Any, ...],
    ) -> LineSearchResult: ...


@dataclass
class ArmijoBacktracking:
    c1: float = DEFAULT_ARMIJO_C1
    beta: float = DEFAULT_ARMIJO_BETA
    max_iter: int = DEFAULT_ARMIJO_MAX_ITER

    def search(
        self,
        fun: RootFunction,
        x: np.ndarray,
        phi_current: float,
        proposal: DirectionProposal,
        args: tuple[Any, ...],
    ) -> LineSearchResult:
        alpha = 1.0
        dx, slope = proposal.direction, proposal.slope

        for n_evals in range(1, self.max_iter + 1):
            x_trial = x + alpha * dx
            res_trial, jac_trial = fun(x_trial, *args)
            phi_trial = merit(res_trial)

            if phi_trial <= phi_current + self.c1 * alpha * slope:
                return LineSearchResult(x_trial, res_trial, jac_trial, phi_trial, alpha, n_evals)

            alpha *= self.beta

        raise RuntimeError(f"Line search failed after {self.max_iter} reductions")
