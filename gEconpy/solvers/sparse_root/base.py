from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

DEFAULT_TOL = 1e-10
DEFAULT_MAXITER = 1000
DEFAULT_ARMIJO_C1 = 1e-4
DEFAULT_ARMIJO_BETA = 0.5
DEFAULT_ARMIJO_MAX_ITER = 50


class RootFunction(Protocol):
    """Callable that returns both residuals and a sparse Jacobian at a point."""

    def __call__(self, x: np.ndarray, *args: Any) -> tuple[np.ndarray, sp.spmatrix]: ...


class MeritFunction(Protocol):
    """Callable that returns only residuals, for cheap merit evaluation during a line search."""

    def __call__(self, x: np.ndarray, *args: Any) -> np.ndarray: ...


class RootSolver(Protocol):
    """Iteration strategy that builds an initial state and advances it one step at a time."""

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple[Any, ...]) -> "SolverState": ...

    def step(
        self, fun: RootFunction, state: "SolverState", args: tuple[Any, ...]
    ) -> tuple["SolverState", "StepInfo"]: ...


@dataclass(slots=True)
class IterationStats:
    """Running counts of the work done by a solver.

    Attributes
    ----------
    nit : int
        Number of accepted iterations.
    nfev : int
        Number of residual evaluations.
    njev : int
        Number of Jacobian evaluations.
    nsolve : int
        Number of linear solves.
    nreject : int
        Number of rejected trial steps.
    """

    nit: int = 0
    nfev: int = 0
    njev: int = 0
    nsolve: int = 0
    nreject: int = 0

    def update(self, nit: int = 0, nfev: int = 0, njev: int = 0, nsolve: int = 0, nreject: int = 0) -> "IterationStats":
        self.nit += nit
        self.nfev += nfev
        self.njev += njev
        self.nsolve += nsolve
        self.nreject += nreject
        return self


@dataclass(frozen=True, slots=True)
class SolverState:
    """Solver state at one iterate.

    Attributes
    ----------
    x : ndarray
        Current iterate.
    res : ndarray
        Residuals at ``x``.
    jac : ``sparse matrix``
        Jacobian at ``x``.
    phi : float
        Merit value at ``x``.
    stats : IterationStats
        Work counts accumulated so far.
    """

    x: np.ndarray
    res: np.ndarray
    jac: sp.spmatrix
    phi: float
    stats: IterationStats


@dataclass(frozen=True, slots=True)
class StepInfo:
    """Outcome of a single solver step.

    Attributes
    ----------
    accepted : bool
        Whether the step was accepted.
    step : ndarray
        The step taken, or zeros when no step was accepted.
    message : str
        Failure description when the step was rejected.
    """

    accepted: bool
    step: np.ndarray
    message: str = ""


def default_check_convergence(state: SolverState, f_tol: float, x_tol: float, last_step: np.ndarray | None) -> bool:
    """Report whether the residuals or the last step are small enough to stop iterating."""
    if np.max(np.abs(state.res)) < f_tol:
        return True
    return bool(last_step is not None and check_step_convergence(last_step, state.x, x_tol))


def default_failure_message(state: SolverState, maxiter: int) -> str:
    """Build the message reported when the solver exhausts its iteration budget."""
    return f"Did not converge after {maxiter} iterations (max |residual| = {np.max(np.abs(state.res)):.2e})"


def merit(res: np.ndarray) -> float:
    """Return the least-squares merit value ``0.5 * res^T res``."""
    return 0.5 * np.dot(res, res)


def check_step_convergence(dx: np.ndarray, x_new: np.ndarray, tol: float) -> bool:
    """Report whether the step ``dx`` is small relative to the size of ``x_new``."""
    return np.max(np.abs(dx)) < tol * (1.0 + np.max(np.abs(x_new)))


def validate_fused_fun(fun: RootFunction, x0: np.ndarray, args: tuple[Any, ...]) -> None:
    """Check that ``fun`` returns a dense residual ndarray and a sparse Jacobian, raising if it does not."""
    result = fun(x0, *args)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise ValueError("fun must return a tuple of (residuals, jacobian)")
    res, jac = result
    if not isinstance(res, np.ndarray):
        raise TypeError("fun must return residuals as ndarray")
    if not sp.issparse(jac):
        raise TypeError("fun must return a sparse jacobian")
