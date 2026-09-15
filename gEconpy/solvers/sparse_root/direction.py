from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import bicgstab, gmres, spsolve

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
        Directional derivative of the merit function along ``direction``. Always negative.
    kind : str
        Label describing how the direction was computed, such as ``"newton"`` or ``"gradient_fallback"``.
    """

    direction: np.ndarray
    slope: float
    kind: str


class DirectionStrategy(Protocol):
    """Strategy that proposes a search direction from the current residuals and Jacobian."""

    def compute(self, x: np.ndarray, res: np.ndarray, jac: sp.spmatrix) -> DirectionProposal: ...


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

    linear_solver: Callable = spsolve

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
class ChordDirection:
    """Direction strategy that solves with a cached Jacobian, refreshing it every ``recompute_every`` calls.

    The residual is always current, so only the linear solve reuses old data.

    Parameters
    ----------
    linear_solver : callable, optional
        Sparse linear solver called as ``linear_solver(A, b)``. Defaults to :func:`scipy.sparse.linalg.spsolve`.
    recompute_every : int, optional
        Number of direction computations between Jacobian refreshes. Defaults to 5.
    """

    linear_solver: Callable = spsolve
    recompute_every: int = 5
    _cached_jac: sp.spmatrix | None = field(init=False, repr=False, default=None)
    _call_count: int = field(init=False, repr=False, default=0)

    def reset(self) -> None:
        """Discard the cached Jacobian so the next call refreshes it."""
        self._cached_jac = None
        self._call_count = 0

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
            Jacobian at ``x``. Used for the slope even when the cached Jacobian is used for the solve.

        Returns
        -------
        proposal : ~gEconpy.solvers.sparse_root.direction.DirectionProposal
            Direction, its slope against the current residuals, and a label describing how it was computed.
        """
        if self._cached_jac is None or self._call_count % self.recompute_every == 0:
            self._cached_jac = jac
        self._call_count += 1

        cached_jac = self._cached_jac
        dx = try_linear_solve(self.linear_solver, cached_jac, -res)
        if dx is None:
            return _descent_proposal(-(cached_jac.T @ res), res, jac, kind="chord_gradient_fallback")
        return _descent_proposal(dx, res, jac, kind="chord")


@dataclass
class KrylovDirection:
    r"""Direction strategy that solves the Newton system inexactly with a Krylov method.

    Solves :math:`J \, dx = -r` approximately using GMRES or BiCGSTAB with absolute tolerance
    :math:`\eta_k \lVert r_k \rVert`. With Eisenstat-Walker forcing, :math:`\eta_k` tightens as the residual shrinks.

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
    """

    krylov_method: str = "gmres"
    eta_max: float = 0.9
    eta_min: float = 1e-6
    eisenstat_walker: bool = True

    _eta: float = field(init=False, repr=False, default=0.0)
    _prev_res_norm: float = field(init=False, repr=False, default=0.0)
    _prev_pred_norm: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        if self.krylov_method not in _KRYLOV_METHODS:
            raise ValueError(
                f"Unknown Krylov method {self.krylov_method!r}. Pass one of {sorted(_KRYLOV_METHODS)} as krylov_method."
            )
        self._eta = self.eta_max

    def reset(self) -> None:
        """Restore the forcing term to ``eta_max`` and forget the previous residual."""
        self._eta = self.eta_max
        self._prev_res_norm = 0.0
        self._prev_pred_norm = 0.0

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
        res_norm = np.linalg.norm(res)
        self._update_forcing_term(res_norm)

        krylov_solve = _KRYLOV_METHODS[self.krylov_method]

        def solve(A, b):
            dx, info = krylov_solve(A, b, atol=self._eta * res_norm)
            if info != 0:
                raise ValueError(f"Krylov solve stopped with info={info}")
            return dx

        dx = try_linear_solve(solve, jac, -res)
        kind = f"krylov_{self.krylov_method}"
        if dx is None:
            dx = -(jac.T @ res)
            kind = f"{kind}_gradient_fallback"

        self._prev_res_norm = res_norm
        self._prev_pred_norm = np.linalg.norm(res + jac @ dx)

        return _descent_proposal(dx, res, jac, kind=kind)

    def _update_forcing_term(self, res_norm: float) -> None:
        if not self.eisenstat_walker or self._prev_res_norm == 0:
            return
        eta_new = abs(res_norm - self._prev_pred_norm) / self._prev_res_norm
        # Squaring the previous forcing term bounds how fast eta may fall between iterations.
        eta_floor = self._eta**2
        self._eta = float(np.clip(max(eta_new, eta_floor), self.eta_min, self.eta_max))


def try_linear_solve(linear_solver: Callable, A: sp.spmatrix, b: np.ndarray) -> np.ndarray | None:
    """Solve ``A x = b`` and return ``None`` when the solver raises or produces non-finite values.

    Parameters
    ----------
    linear_solver : callable
        Solver called as ``linear_solver(A, b)``.
    A : sparse matrix
        System matrix.
    b : ndarray
        Right-hand side.

    Returns
    -------
    x : ndarray or None
        Solution, or ``None`` when the solve failed.
    """
    # The solver is user-supplied and may fail in any way. Under warnings-as-errors, scipy's MatrixRankWarning on a
    # singular matrix also arrives here as an exception, so the catch stays broad.
    try:
        x = linear_solver(A, b)
    except Exception:
        return None
    if not np.all(np.isfinite(x)):
        return None
    return x


def _descent_proposal(dx: np.ndarray, res: np.ndarray, jac: sp.spmatrix, kind: str) -> DirectionProposal:
    """Flip ``dx`` when it is not a descent direction for the merit function at the current Jacobian."""
    slope = float(np.dot(res, jac @ dx))
    if slope >= 0:
        return DirectionProposal(direction=-dx, slope=-slope, kind=f"{kind}_flipped")
    return DirectionProposal(direction=dx, slope=slope, kind=kind)
