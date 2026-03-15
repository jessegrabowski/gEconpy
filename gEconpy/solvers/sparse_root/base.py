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
    def __call__(self, x: np.ndarray, *args: Any) -> tuple[np.ndarray, sp.spmatrix]: ...


class RootSolver(Protocol):
    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple[Any, ...]) -> "SolverState": ...

    def step(
        self, fun: RootFunction, state: "SolverState", args: tuple[Any, ...]
    ) -> tuple["SolverState", "StepInfo"]: ...


@dataclass(frozen=True, slots=True)
class IterationStats:
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    nsolve: int = 0
    nreject: int = 0

    def update(self, nit: int = 0, nfev: int = 0, njev: int = 0, nsolve: int = 0, nreject: int = 0) -> "IterationStats":
        return IterationStats(
            self.nit + nit,
            self.nfev + nfev,
            self.njev + njev,
            self.nsolve + nsolve,
            self.nreject + nreject,
        )


@dataclass(frozen=True, slots=True)
class SolverState:
    x: np.ndarray
    res: np.ndarray
    jac: sp.spmatrix
    phi: float
    stats: IterationStats


@dataclass(frozen=True, slots=True)
class StepInfo:
    accepted: bool
    step: np.ndarray
    message: str = ""


def default_check_convergence(state: SolverState, f_tol: float, x_tol: float, last_step: np.ndarray | None) -> bool:
    if np.max(np.abs(state.res)) < f_tol:
        return True
    return bool(last_step is not None and check_step_convergence(last_step, state.x, x_tol))


def default_failure_message(state: SolverState, maxiter: int) -> str:
    return f"Did not converge after {maxiter} iterations (max |residual| = {np.max(np.abs(state.res)):.2e})"


def merit(res: np.ndarray) -> float:
    return 0.5 * np.dot(res, res)


def check_step_convergence(dx: np.ndarray, x_new: np.ndarray, tol: float) -> bool:
    return np.max(np.abs(dx)) < tol * (1.0 + np.max(np.abs(x_new)))


def validate_fused_fun(fun: RootFunction, x0: np.ndarray, args: tuple[Any, ...]) -> None:
    result = fun(x0, *args)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise ValueError("fun must return a tuple of (residuals, jacobian)")
    res, jac = result
    if not isinstance(res, np.ndarray):
        raise TypeError("fun must return residuals as ndarray")
    if not sp.issparse(jac):
        raise TypeError("fun must return a sparse jacobian")
