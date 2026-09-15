from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

DEFAULT_TOL = 1e-10
DEFAULT_MAXITER = 1000
DEFAULT_ARMIJO_C1 = 1e-4
DEFAULT_ARMIJO_BETA = 0.5
DEFAULT_ARMIJO_MAX_ITER = 50
TRUST_REGION_GROW_RHO = 0.75
TRUST_REGION_BOUNDARY_FRACTION = 0.9


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
        """Add to each count in place.

        Parameters
        ----------
        nit : int, optional
            Accepted iterations to add. Defaults to 0.
        nfev : int, optional
            Residual evaluations to add. Defaults to 0.
        njev : int, optional
            Jacobian evaluations to add. Defaults to 0.
        nsolve : int, optional
            Linear solves to add. Defaults to 0.
        nreject : int, optional
            Rejected trial steps to add. Defaults to 0.

        Returns
        -------
        stats : IterationStats
            This object, after the counts are added.
        """
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
    jac : sparse matrix
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


def initial_state(fun: RootFunction, x0: np.ndarray, args: tuple[Any, ...]) -> SolverState:
    """Evaluate ``fun`` at ``x0`` and build the state every solver starts from.

    Parameters
    ----------
    fun : callable
        Fused residual and Jacobian function, called as ``fun(x, *args)``.
    x0 : ndarray
        Initial guess for the root. Copied to a float64 array.
    args : tuple
        Extra positional arguments passed to ``fun``.

    Returns
    -------
    state : SolverState
        State holding the initial point, residuals, Jacobian, merit value and evaluation counts.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    res, jac = fun(x, *args)
    return SolverState(x=x, res=res, jac=jac, phi=merit(res), stats=IterationStats(nfev=1, njev=1))


def default_check_convergence(state: SolverState, f_tol: float, x_tol: float, last_step: np.ndarray | None) -> bool:
    """Report whether the residuals or the last step are small enough to stop iterating.

    Parameters
    ----------
    state : SolverState
        Current solver state.
    f_tol : float
        Tolerance on the maximum absolute residual.
    x_tol : float
        Tolerance on the step size relative to the iterate.
    last_step : ndarray or None
        Step that produced ``state``, or ``None`` before the first step.

    Returns
    -------
    converged : bool
        Whether either test passed.
    """
    if np.max(np.abs(state.res)) < f_tol:
        return True
    return bool(last_step is not None and check_step_convergence(last_step, state.x, x_tol))


def default_failure_message(state: SolverState, maxiter: int) -> str:
    """Build the message reported when the solver exhausts its iteration budget.

    Parameters
    ----------
    state : SolverState
        Final solver state.
    maxiter : int
        Iteration budget that was exhausted.

    Returns
    -------
    message : str
        Message naming the budget and the largest remaining residual.
    """
    return f"Did not converge after {maxiter} iterations (max |residual| = {np.max(np.abs(state.res)):.2e})"


def merit(res: np.ndarray) -> float:
    r"""Return the least-squares merit value :math:`\frac{1}{2} r^T r`.

    Parameters
    ----------
    res : ndarray
        Residual vector.

    Returns
    -------
    phi : float
        Half the squared Euclidean norm of ``res``.
    """
    return 0.5 * np.dot(res, res)


def check_step_convergence(dx: np.ndarray, x_new: np.ndarray, tol: float) -> bool:
    """Report whether the step ``dx`` is small relative to the size of ``x_new``.

    Parameters
    ----------
    dx : ndarray
        Step just taken.
    x_new : ndarray
        Iterate reached by the step.
    tol : float
        Relative tolerance.

    Returns
    -------
    converged : bool
        Whether ``max |dx| < tol * (1 + max |x_new|)``.
    """
    return np.max(np.abs(dx)) < tol * (1.0 + np.max(np.abs(x_new)))


def validate_fused_fun(fun: RootFunction, x0: np.ndarray, args: tuple[Any, ...]) -> None:
    """Call ``fun`` once and raise unless it returns a dense residual ndarray and a sparse Jacobian.

    Parameters
    ----------
    fun : callable
        Function to check, called as ``fun(x0, *args)``.
    x0 : ndarray
        Point at which to evaluate ``fun``.
    args : tuple
        Extra positional arguments passed to ``fun``.
    """
    result = fun(x0, *args)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise ValueError(
            f"fun must return a tuple (residuals, jacobian), but returned {type(result).__name__}. Return both the "
            "residual vector and its sparse Jacobian from a single call."
        )
    res, jac = result
    if not isinstance(res, np.ndarray):
        raise TypeError(
            f"fun must return residuals as a numpy ndarray, but returned {type(res).__name__}. Wrap the residuals in "
            "np.asarray before returning them."
        )
    if not sp.issparse(jac):
        raise TypeError(
            f"fun must return the Jacobian as a scipy sparse matrix, but returned {type(jac).__name__}. Convert it "
            "with scipy.sparse.csc_matrix before returning it."
        )
