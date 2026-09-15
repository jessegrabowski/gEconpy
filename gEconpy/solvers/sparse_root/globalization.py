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
    """Point accepted by a line search, with everything the solver needs to continue from it.

    Attributes
    ----------
    x_new : ndarray
        Accepted iterate.
    res_new : ndarray
        Residuals at ``x_new``.
    jac_new : sparse matrix
        Jacobian at ``x_new``.
    phi_new : float
        Merit value at ``x_new``.
    alpha : float
        Step length along the proposed direction.
    n_evals : int
        Number of function evaluations the search used.
    """

    x_new: np.ndarray
    res_new: np.ndarray
    jac_new: sp.spmatrix
    phi_new: float
    alpha: float
    n_evals: int


class GlobalizationStrategy(Protocol):
    """Strategy that picks a step length along a proposed direction."""

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
    r"""Backtracking line search with the Armijo sufficient decrease condition.

    Accepts the first step length :math:`\alpha` in the sequence :math:`1, \beta, \beta^2, \dots` satisfying

    .. math::

        \phi(x + \alpha d) \le \phi(x) + c_1 \alpha \, \nabla\phi(x)^T d

    Parameters
    ----------
    c1 : float, optional
        Sufficient decrease parameter :math:`c_1`. Defaults to 1e-4.
    beta : float, optional
        Step-size reduction factor :math:`\beta`. Defaults to 0.5.
    max_iter : int, optional
        Maximum number of trial step lengths. Defaults to 50.
    merit_fun : callable, optional
        Residual-only function ``merit_fun(x, *args)`` used to evaluate trial points. When given, ``fun`` is called
        once at the accepted point to obtain the Jacobian, which saves work when the Jacobian costs far more than the
        residuals. Defaults to ``None``, meaning ``fun`` evaluates every trial point.
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
        """Backtrack along the proposed direction until the Armijo sufficient decrease condition is satisfied.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        x : ndarray
            Current iterate.
        phi_current : float
            Merit value at ``x``.
        proposal : ~gEconpy.solvers.sparse_root.direction.DirectionProposal
            Search direction and its slope.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        result : LineSearchResult
            Accepted point, its residuals and Jacobian, the merit value, the step length, and the number of
            function evaluations used.
        """
        return _backtrack(fun, self.merit_fun, x, phi_current, proposal, args, self.c1, self.beta, self.max_iter)


@dataclass
class NonmonotoneBacktracking:
    r"""Grippo-Lampariello-Lucidi nonmonotone backtracking line search.

    Compares each trial point against the largest merit value among the last ``memory`` iterates:

    .. math::

        \phi(x + \alpha d) \le \max_{0 \le j < M} \phi_{k-j} + c_1 \alpha \, \nabla\phi(x)^T d

    Occasional increases in the merit function are therefore allowed, which helps the solver leave narrow valleys
    where a monotone line search takes tiny steps.

    Parameters
    ----------
    c1 : float, optional
        Sufficient decrease parameter :math:`c_1`. Defaults to 1e-4.
    beta : float, optional
        Step-size reduction factor :math:`\beta`. Defaults to 0.5.
    max_iter : int, optional
        Maximum number of trial step lengths. Defaults to 50.
    memory : int, optional
        Number of past merit values :math:`M` to keep. ``memory=1`` recovers standard Armijo. Defaults to 10.
    merit_fun : callable, optional
        Residual-only function ``merit_fun(x, *args)`` used to evaluate trial points. See
        :class:`~gEconpy.solvers.sparse_root.globalization.ArmijoBacktracking`. Defaults to ``None``.
    """

    c1: float = DEFAULT_ARMIJO_C1
    beta: float = DEFAULT_ARMIJO_BETA
    max_iter: int = DEFAULT_ARMIJO_MAX_ITER
    memory: int = 10
    merit_fun: MeritFunction | None = None
    _phi_history: deque[float] = field(init=False, repr=False)

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
        """Backtrack along the proposed direction until the nonmonotone decrease condition is satisfied.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        x : ndarray
            Current iterate.
        phi_current : float
            Merit value at ``x``.
        proposal : ~gEconpy.solvers.sparse_root.direction.DirectionProposal
            Search direction and its slope.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        result : LineSearchResult
            Accepted point, its residuals and Jacobian, the merit value, the step length, and the number of
            function evaluations used.
        """
        self._phi_history.append(phi_current)
        phi_reference = max(self._phi_history)

        return _backtrack(fun, self.merit_fun, x, phi_reference, proposal, args, self.c1, self.beta, self.max_iter)


def _backtrack(
    fun: RootFunction,
    merit_fun: MeritFunction | None,
    x: np.ndarray,
    phi_reference: float,
    proposal: DirectionProposal,
    args: tuple[Any, ...],
    c1: float,
    beta: float,
    max_iter: int,
) -> LineSearchResult:
    """Shrink the step length by ``beta`` until the decrease test against ``phi_reference`` passes."""
    alpha = 1.0
    dx, slope = proposal.direction, proposal.slope

    for n_evals in range(1, max_iter + 1):
        x_trial = x + alpha * dx

        if merit_fun is None:
            res_trial, jac_trial = fun(x_trial, *args)
            n_total_evals = n_evals
        else:
            res_trial = merit_fun(x_trial, *args)
            jac_trial = None
            n_total_evals = n_evals + 1

        phi_trial = merit(res_trial)
        if phi_trial <= phi_reference + c1 * alpha * slope:
            if jac_trial is None:
                res_trial, jac_trial = fun(x_trial, *args)
            return LineSearchResult(x_trial, res_trial, jac_trial, phi_trial, alpha, n_total_evals)

        alpha *= beta

    raise RuntimeError(
        f"Line search failed after {max_iter} reductions. Increase max_iter, loosen c1, or start closer to the root."
    )
