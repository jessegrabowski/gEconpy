from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.base import (
    DEFAULT_ARMIJO_BETA,
    DEFAULT_ARMIJO_C1,
    DEFAULT_ARMIJO_MAX_ITER,
    MeritFunction,
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
    """Backtracking line search with the Armijo sufficient decrease condition.

    Parameters
    ----------
    c1 : float
        Sufficient decrease parameter.
    beta : float
        Step-size reduction factor.
    max_iter : int
        Maximum number of backtracking reductions.
    merit_fun : callable or None
        Cheap function ``merit_fun(x, *args) -> residuals`` used to evaluate the merit function during
        backtracking. When provided, trial points are evaluated with this function instead of the full
        ``fun`` (which also computes the Jacobian). The full ``fun`` is called once at the accepted point
        to obtain the Jacobian for the next iteration. This can dramatically reduce cost when the Jacobian
        is expensive relative to the residuals (e.g. large stacked perfect-foresight systems).

        When ``None`` (default), the full ``fun`` is used for all evaluations.
    """

    c1: float = DEFAULT_ARMIJO_C1
    beta: float = DEFAULT_ARMIJO_BETA
    max_iter: int = DEFAULT_ARMIJO_MAX_ITER
    merit_fun: MeritFunction | None = None

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

        if self.merit_fun is None:
            for n_evals in range(1, self.max_iter + 1):
                x_trial = x + alpha * dx
                res_trial, jac_trial = fun(x_trial, *args)
                phi_trial = merit(res_trial)

                if phi_trial <= phi_current + self.c1 * alpha * slope:
                    return LineSearchResult(x_trial, res_trial, jac_trial, phi_trial, alpha, n_evals)

                alpha *= self.beta
        else:
            for n_evals in range(1, self.max_iter + 1):
                x_trial = x + alpha * dx
                res_trial = self.merit_fun(x_trial, *args)
                phi_trial = merit(res_trial)

                if phi_trial <= phi_current + self.c1 * alpha * slope:
                    # Accepted: evaluate the full function once to get the Jacobian
                    res_trial, jac_trial = fun(x_trial, *args)
                    return LineSearchResult(x_trial, res_trial, jac_trial, phi_trial, alpha, n_evals + 1)

                alpha *= self.beta

        raise RuntimeError(f"Line search failed after {self.max_iter} reductions")


@dataclass
class NonmonotoneBacktracking:
    """Grippo-Lampariello-Lucidi nonmonotone backtracking line search.

    Instead of requiring ``φ(x + α d) ≤ φ(x) + c₁ α slope`` (monotone Armijo),
    this strategy compares against the maximum merit over the last ``memory``
    iterates::

        φ(x + α d) ≤ max(φ_{k}, ..., φ_{k-M+1}) + c₁ α slope

    This allows occasional increases in the merit function, helping the solver
    escape narrow valleys where monotone line search takes tiny steps.

    Parameters
    ----------
    c1 : float
        Sufficient decrease parameter.
    beta : float
        Step-size reduction factor.
    max_iter : int
        Maximum number of backtracking reductions.
    memory : int
        Number of past merit values to keep. ``memory=1`` recovers standard Armijo.
    merit_fun : callable or None
        Cheap function ``merit_fun(x, *args) -> residuals`` used to evaluate the merit function during
        backtracking. See :class:`ArmijoBacktracking` for details.
    """

    c1: float = DEFAULT_ARMIJO_C1
    beta: float = DEFAULT_ARMIJO_BETA
    max_iter: int = DEFAULT_ARMIJO_MAX_ITER
    memory: int = 10
    merit_fun: MeritFunction | None = None
    _phi_history: deque = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._phi_history = deque(maxlen=self.memory)

    def search(
        self,
        fun: RootFunction,
        x: np.ndarray,
        phi_current: float,
        proposal: DirectionProposal,
        args: tuple[Any, ...],
    ) -> LineSearchResult:
        self._phi_history.append(phi_current)
        phi_ref = max(self._phi_history)

        alpha = 1.0
        dx, slope = proposal.direction, proposal.slope

        if self.merit_fun is None:
            for n_evals in range(1, self.max_iter + 1):
                x_trial = x + alpha * dx
                res_trial, jac_trial = fun(x_trial, *args)
                phi_trial = merit(res_trial)

                if phi_trial <= phi_ref + self.c1 * alpha * slope:
                    return LineSearchResult(x_trial, res_trial, jac_trial, phi_trial, alpha, n_evals)

                alpha *= self.beta
        else:
            for n_evals in range(1, self.max_iter + 1):
                x_trial = x + alpha * dx
                res_trial = self.merit_fun(x_trial, *args)
                phi_trial = merit(res_trial)

                if phi_trial <= phi_ref + self.c1 * alpha * slope:
                    res_trial, jac_trial = fun(x_trial, *args)
                    return LineSearchResult(x_trial, res_trial, jac_trial, phi_trial, alpha, n_evals + 1)

                alpha *= self.beta

        raise RuntimeError(f"Nonmonotone line search failed after {self.max_iter} reductions")
