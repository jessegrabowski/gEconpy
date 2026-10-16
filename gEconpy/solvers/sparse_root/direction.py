import math
import warnings

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, SuperLU, bicgstab, gmres, splu, spsolve

from gEconpy.solvers.sparse_root.base import DEFAULT_TOL, check_step_convergence

_KRYLOV_METHODS = {
    "gmres": gmres,
    "bicgstab": bicgstab,
}


@dataclass(frozen=True, slots=True)
class DirectionProposal:
    """Search direction proposed by a direction strategy.

    Attributes
    ----------
    direction : ndarray
        Proposed step direction.
    slope : float
        Directional derivative of the merit function along ``direction``. Negative for a usable direction and
        zero only at a stationary point, which is how a caller tells the two apart.
    kind : str
        Label describing how the direction was computed, such as ``"newton"`` or ``"gradient_fallback"``. The
        ``_inexact``, ``_flipped`` and ``_stationary`` suffixes record which branch produced it.
    """

    direction: np.ndarray
    slope: float
    kind: str


class DirectionStrategy(Protocol):
    """Strategy that proposes a search direction from the current residuals and Jacobian.

    ``reset`` discards whatever the strategy carried over from a previous solve, and a solver calls it before the
    first step so one instance can be reused across solves.
    """

    def compute(self, x: np.ndarray, res: np.ndarray, jac: sp.spmatrix) -> DirectionProposal: ...

    def reset(self) -> None: ...


@dataclass
class NewtonDirection:
    r"""Direction strategy that solves the Newton system with the current Jacobian.

    Solves :math:`J \, dx = -r` at every call. When the solve fails or returns non-finite values, the strategy falls
    back to the steepest descent direction :math:`-J^T r`.

    Parameters
    ----------
    linear_solver : callable, optional
        Sparse linear solver called as ``linear_solver(A, b)``. Defaults to :func:`scipy.sparse.linalg.spsolve`.
    """

    # A function stored directly as the class attribute becomes the attribute's docstring under autodoc, and
    # scipy's spsolve docstring uses roles Sphinx does not define, so the default lives in a factory.
    linear_solver: Callable = field(default_factory=lambda: spsolve)

    def reset(self) -> None:
        """Do nothing: this strategy carries no state between solves."""

    def compute(
        self,
        x: np.ndarray,  # noqa: ARG002
        res: np.ndarray,
        jac: sp.spmatrix,
    ) -> DirectionProposal:
        """Compute a search direction at the current iterate.

        Parameters
        ----------
        x : ndarray
            Current iterate. Unused, accepted for interface compatibility.
        res : ndarray
            Residuals at ``x``.
        jac : sparse matrix
            Jacobian at ``x``.

        Returns
        -------
        proposal : ~gEconpy.solvers.sparse_root.direction.DirectionProposal
            Direction, its slope against the current residuals, and a label describing how it was computed.
        """
        dx = try_linear_solve(self.linear_solver, jac, -res)
        if dx is None:
            return _descent_proposal(-(jac.T @ res), res, jac, kind="gradient_fallback")
        return _descent_proposal(dx, res, jac, kind="newton")


@dataclass
class _ScheduledFactorization:
    """A Jacobian and its factorization, refreshed every ``recompute_every`` updates or on demand."""

    matrix: sp.spmatrix | None = None
    lu: SuperLU | None = None
    refreshed: bool = False
    _updates_since_refresh: int = 0

    def reset(self) -> None:
        self.matrix = None
        self.lu = None
        self.refreshed = False
        self._updates_since_refresh = 0

    def update(
        self, jac: sp.spmatrix, recompute_every: int | None, permc_spec: str = "COLAMD", force: bool = False
    ) -> tuple[sp.spmatrix, SuperLU | None]:
        # On a schedule, a failed factorization waits for the next window, so a singular Jacobian is not retried on
        # every call. Without one, no window ever comes, so it is retried on the next call.
        due = self.lu is None if recompute_every is None else self._updates_since_refresh >= recompute_every
        self.refreshed = self.matrix is None or self.matrix.shape != jac.shape or force or due
        if self.refreshed:
            self.matrix = jac
            self.lu = _try_factorize(jac, permc_spec)
            self._updates_since_refresh = 0
        self._updates_since_refresh += 1
        return self.matrix, self.lu


@dataclass
class ChordDirection:
    """Direction strategy that solves with a cached factorization, refreshed on a schedule or when progress slows.

    The residual is always current, so only the linear solve reuses old data. One :func:`scipy.sparse.linalg.splu`
    serves every call until the next refresh, which is what makes a chord step cheaper than the Newton step it
    approximates.

    Parameters
    ----------
    recompute_every : int or None, optional
        Number of direction computations between scheduled refreshes, at least 1. ``None`` turns the schedule off.
        Defaults to 5.
    refresh_ratio : float or None, optional
        Refresh when the residual norm falls by less than this factor from the previous call of the same solve, that
        is when ``||r_k|| > refresh_ratio * ||r_{k-1}||``. ``None`` turns the rule off. Defaults to None.
    refresh_step_tol : float or None, optional
        Refresh a factorization from an earlier call when its step is below this tolerance relative to the iterate,
        as the solver's step tolerance measures it. Matching the solver's ``x_tol`` keeps a short step from an old
        factorization from ending the solve as a stall. ``None`` turns the rule off. Defaults to 1e-10, the default
        ``x_tol`` of :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root`.
    keep_across_solves : bool, optional
        Whether :meth:`reset` keeps the factorization, so the next solve starts with the last one. Defaults to False.
    permc_spec : str, optional
        Column ordering :func:`scipy.sparse.linalg.splu` factorizes in. Defaults to ``"COLAMD"``.
    """

    recompute_every: int | None = 5
    refresh_ratio: float | None = None
    refresh_step_tol: float | None = DEFAULT_TOL
    keep_across_solves: bool = False
    permc_spec: str = "COLAMD"
    _previous_res_norm: float | None = field(init=False, repr=False, default=None)
    _factorization: _ScheduledFactorization = field(init=False, repr=False, default_factory=_ScheduledFactorization)

    def __post_init__(self):
        if self.recompute_every is not None and self.recompute_every < 1:
            raise ValueError(
                f"recompute_every must be at least 1, got {self.recompute_every}. Pass None to turn it off."
            )

    def reset(self) -> None:
        """Forget the previous residual, and discard the cached factorization unless ``keep_across_solves``."""
        self._previous_res_norm = None
        if not self.keep_across_solves:
            self._factorization.reset()

    def compute(
        self,
        x: np.ndarray,
        res: np.ndarray,
        jac: sp.spmatrix,
    ) -> DirectionProposal:
        """Compute a search direction at the current iterate.

        Parameters
        ----------
        x : ndarray
            Current iterate.
        res : ndarray
            Residuals at ``x``.
        jac : sparse matrix
            Jacobian at ``x``. Used for the slope even when the cached factorization is used for the solve.

        Returns
        -------
        proposal : ~gEconpy.solvers.sparse_root.direction.DirectionProposal
            Direction, its slope against the current residuals, and a label describing how it was computed.
        """
        res_norm = float(np.linalg.norm(res))
        slowed = (
            self.refresh_ratio is not None
            and self._previous_res_norm is not None
            and res_norm > self.refresh_ratio * self._previous_res_norm
        )
        self._previous_res_norm = res_norm

        cached_jac, lu = self._factorization.update(jac, self.recompute_every, self.permc_spec, force=slowed)
        dx = None if lu is None else try_linear_solve(SuperLU.solve, lu, -res)
        if self._is_short_step_from_an_old_factorization(dx, x):
            cached_jac, lu = self._factorization.update(jac, self.recompute_every, self.permc_spec, force=True)
            dx = None if lu is None else try_linear_solve(SuperLU.solve, lu, -res)

        if dx is None:
            return _descent_proposal(-(cached_jac.T @ res), res, jac, kind="chord_gradient_fallback")
        return _descent_proposal(dx, res, jac, kind="chord")

    def _is_short_step_from_an_old_factorization(self, dx: np.ndarray | None, x: np.ndarray) -> bool:
        return (
            self.refresh_step_tol is not None
            and dx is not None
            and not self._factorization.refreshed
            and check_step_convergence(dx, x + dx, self.refresh_step_tol)
        )


@dataclass
class KrylovDirection:
    r"""Direction strategy that solves the Newton system inexactly with a Krylov method.

    Solves :math:`J \, dx = -r` approximately using GMRES or BiCGSTAB with absolute tolerance
    :math:`\eta_k \lVert r_k \rVert`. With Eisenstat-Walker forcing, :math:`\eta_k` tightens as the residual shrinks.
    By default each solve is preconditioned with an LU factorization of the Jacobian, refreshed every
    ``recompute_every`` calls.

    Parameters
    ----------
    krylov_method : str, optional
        Name of the Krylov method, ``"gmres"`` or ``"bicgstab"``. Defaults to ``"gmres"``.
    eta_max : float, optional
        Upper bound on the forcing term. Defaults to 0.9.
    eta_min : float, optional
        Lower bound on the forcing term. Defaults to 1e-6.
    eisenstat_walker : bool, optional
        Whether to adapt the forcing term. When ``False``, ``eta_max`` is used throughout. Defaults to ``True``.
    max_krylov_iter : int, optional
        Cap on inner Krylov iterations per direction, at least 1. For GMRES it is rounded up to whole restart cycles.
        Defaults to 200.
    restart : int, optional
        Number of GMRES iterations between restarts, at least 1. Ignored by BiCGSTAB. Defaults to 20.
    preconditioner : sparse matrix or LinearOperator, optional
        Approximation to the inverse Jacobian, applied as ``preconditioner @ v``. Takes precedence over the refreshed
        factorization. Defaults to ``None``.
    recompute_every : int or None, optional
        Number of direction computations between refreshes of the LU factorization used as the preconditioner, at
        least 1. ``None`` turns the factorization off, for systems too large to factorize. Ignored when
        ``preconditioner`` is given. Defaults to 5.
    """

    krylov_method: str = "gmres"
    eta_max: float = 0.9
    eta_min: float = 1e-6
    eisenstat_walker: bool = True
    max_krylov_iter: int = 200
    restart: int = 20
    preconditioner: sp.spmatrix | LinearOperator | None = None
    recompute_every: int | None = 5

    _eta: float = field(init=False, repr=False, default=0.0)
    _prev_res_norm: float = field(init=False, repr=False, default=0.0)
    _prev_pred_norm: float = field(init=False, repr=False, default=0.0)
    _factorization: _ScheduledFactorization = field(init=False, repr=False, default_factory=_ScheduledFactorization)

    def __post_init__(self):
        if self.krylov_method not in _KRYLOV_METHODS:
            raise ValueError(
                f"Unknown Krylov method {self.krylov_method!r}. Pass one of {sorted(_KRYLOV_METHODS)} as krylov_method."
            )
        if self.max_krylov_iter < 1:
            raise ValueError(f"max_krylov_iter must be at least 1, got {self.max_krylov_iter}.")
        if self.restart < 1:
            raise ValueError(f"restart must be at least 1, got {self.restart}.")
        if self.recompute_every is not None and self.recompute_every < 1:
            raise ValueError(
                f"recompute_every must be at least 1, got {self.recompute_every}. Pass None to turn it off."
            )
        self._eta = self.eta_max

    def reset(self) -> None:
        """Restore the forcing term to ``eta_max``, forget the previous residual, and discard the factorization."""
        self._eta = self.eta_max
        self._prev_res_norm = 0.0
        self._prev_pred_norm = 0.0
        self._factorization.reset()

    def compute(
        self,
        x: np.ndarray,  # noqa: ARG002
        res: np.ndarray,
        jac: sp.spmatrix,
    ) -> DirectionProposal:
        """Compute a search direction at the current iterate.

        Parameters
        ----------
        x : ndarray
            Current iterate. Unused, accepted for interface compatibility.
        res : ndarray
            Residuals at ``x``.
        jac : sparse matrix
            Jacobian at ``x``.

        Returns
        -------
        proposal : ~gEconpy.solvers.sparse_root.direction.DirectionProposal
            Direction, its slope against the current residuals, and a label describing how it was computed.
        """
        res_norm = float(np.linalg.norm(res))
        self._update_forcing_term(res_norm)

        krylov_solve = _KRYLOV_METHODS[self.krylov_method]
        iteration_limits = self._iteration_limits()
        preconditioner = self._current_preconditioner(jac)

        stopped_early = False

        def solve(A, b):
            nonlocal stopped_early
            # scipy stops at max(rtol * ||b||, atol), and its default rtol of 1e-5 would floor the forcing term.
            dx, info = krylov_solve(A, b, rtol=0.0, atol=self._eta * res_norm, M=preconditioner, **iteration_limits)
            if info < 0:
                raise ValueError(f"Krylov solve failed with info={info}")
            stopped_early = info > 0
            return dx

        dx = try_linear_solve(solve, jac, -res)
        kind = f"krylov_{self.krylov_method}"
        predicted_norm = None if dx is None else float(np.linalg.norm(res + jac @ dx))
        # An iterate that lowered ||r + J dx|| below ||r|| has r^T J dx < 0, so it descends the merit function even
        # when the inner solve stopped short of the forcing term.
        if stopped_early and predicted_norm is not None:
            if predicted_norm < res_norm:
                kind = f"{kind}_inexact"
            else:
                dx = None
        if dx is None:
            dx = -(jac.T @ res)
            kind = f"{kind}_gradient_fallback"
            predicted_norm = float(np.linalg.norm(res + jac @ dx))

        self._prev_res_norm = res_norm
        self._prev_pred_norm = predicted_norm

        return _descent_proposal(dx, res, jac, kind=kind)

    def _current_preconditioner(self, jac: sp.spmatrix) -> sp.spmatrix | LinearOperator | None:
        if self.preconditioner is not None or self.recompute_every is None:
            return self.preconditioner
        _, lu = self._factorization.update(jac, self.recompute_every)
        return None if lu is None else LinearOperator(jac.shape, matvec=lu.solve)

    def _iteration_limits(self) -> dict[str, int]:
        # scipy's gmres counts maxiter in restart cycles and bicgstab counts it in iterations, so one inner-iteration
        # cap means a different maxiter for each.
        if self.krylov_method == "gmres":
            restart = min(self.restart, self.max_krylov_iter)
            return {"restart": restart, "maxiter": math.ceil(self.max_krylov_iter / restart)}
        return {"maxiter": self.max_krylov_iter}

    def _update_forcing_term(self, res_norm: float) -> None:
        if not self.eisenstat_walker or self._prev_res_norm == 0:
            return
        eta_new = abs(res_norm - self._prev_pred_norm) / self._prev_res_norm
        # Squaring the previous forcing term bounds how fast eta may fall between iterations.
        eta_floor = self._eta**2
        self._eta = float(np.clip(max(eta_new, eta_floor), self.eta_min, self.eta_max))


def try_linear_solve(linear_solver: Callable, A: sp.spmatrix | SuperLU, b: np.ndarray) -> np.ndarray | None:
    """Solve ``A x = b`` and return ``None`` when the solver raises or produces non-finite values.

    Parameters
    ----------
    linear_solver : callable
        Solver called as ``linear_solver(A, b)``.
    A : sparse matrix or SuperLU
        System matrix, or a factorization of one.
    b : ndarray
        Right-hand side.

    Returns
    -------
    x : ndarray or None
        Solution, or ``None`` when the solve failed.
    """
    # A singular matrix is an expected outcome here, and the caller records it by falling back to the gradient, so
    # scipy's MatrixRankWarning is silenced rather than left to surface with no solver or iteration attached. The
    # solver is user-supplied and may fail in any other way, so the catch stays broad.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=MatrixRankWarning)
            x = linear_solver(A, b)
    except Exception:
        return None
    if not np.all(np.isfinite(x)):
        return None
    return x


def _try_factorize(jac: sp.spmatrix, permc_spec: str = "COLAMD") -> SuperLU | None:
    """Factorize ``jac`` and return ``None`` when it is singular."""
    # splu converts a non-CSC input itself and warns while doing it, which is fatal under warnings-as-errors.
    # tocsc returns the matrix unchanged when it is already CSC, so the common case costs nothing.
    try:
        return splu(jac.tocsc(), permc_spec=permc_spec)
    except Exception:
        return None


def _descent_proposal(dx: np.ndarray, res: np.ndarray, jac: sp.spmatrix, kind: str) -> DirectionProposal:
    """Flip ``dx`` when it is not a descent direction for the merit function at the current Jacobian."""
    slope = float(np.dot(res, jac @ dx))
    if slope == 0.0:
        # J^T r vanishing while r does not is a least-squares minimum rather than a root, so neither dx nor -dx
        # reduces the merit function. The comparison is exact because the case is an algebraic zero, and a tiny
        # nonzero slope is a bad step the line search already shrinks.
        return DirectionProposal(direction=dx, slope=0.0, kind=f"{kind}_stationary")
    if slope > 0:
        return DirectionProposal(direction=-dx, slope=-slope, kind=f"{kind}_flipped")
    return DirectionProposal(direction=dx, slope=slope, kind=kind)
