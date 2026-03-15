from functools import partial
from typing import Any

import numpy as np

from better_optimize.wrapper import ObjectiveWrapper, optimizer_early_stopping_wrapper
from scipy.optimize import OptimizeResult

from gEconpy.solvers.sparse_root.base import (
    DEFAULT_ARMIJO_MAX_ITER,
    DEFAULT_MAXITER,
    DEFAULT_TOL,
    IterationStats,
    RootFunction,
    RootSolver,
    SolverState,
    StepInfo,
    default_check_convergence,
    default_failure_message,
    validate_fused_fun,
)
from gEconpy.solvers.sparse_root.chord import Chord, chord
from gEconpy.solvers.sparse_root.direction import ChordDirection, DirectionProposal, NewtonDirection
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking, LineSearchResult
from gEconpy.solvers.sparse_root.newton import NewtonArmijo, newton_armijo

__all__ = [
    "ArmijoBacktracking",
    "Chord",
    "ChordDirection",
    "DirectionProposal",
    "IterationStats",
    "LineSearchResult",
    "NewtonArmijo",
    "NewtonDirection",
    "RootSolver",
    "SolverState",
    "StepInfo",
    "chord",
    "newton_armijo",
    "sparse_root",
]


def _make_result(state: SolverState, success: bool, message: str) -> OptimizeResult:
    return OptimizeResult(
        x=state.x,
        success=success,
        message=message,
        fun=state.res,
        jac=state.jac,
        nit=state.stats.nit,
        nfev=state.stats.nfev,
    )


def _root_find_core(fun, x0, solver, args, f_tol, x_tol, maxiter) -> OptimizeResult:
    state = solver.init(fun, x0, args)
    if np.max(np.abs(state.res)) < f_tol:
        return _make_result(state, True, "Converged")

    check_fn = getattr(solver, "check_convergence", default_check_convergence)
    fail_fn = getattr(solver, "failure_message", default_failure_message)
    last_step = None

    for _ in range(maxiter):
        state, info = solver.step(fun, state, args)
        if not info.accepted:
            return _make_result(state, False, info.message)
        last_step = info.step
        if check_fn(state, f_tol=f_tol, x_tol=x_tol, last_step=last_step):
            return _make_result(state, True, "Converged")

    return _make_result(state, False, fail_fn(state, maxiter))


def sparse_root(
    fun: RootFunction,
    x0: np.ndarray,
    *,
    solver: RootSolver | None = None,
    args: tuple[Any, ...] = (),
    tol: float = DEFAULT_TOL,
    f_tol: float | None = None,
    x_tol: float | None = None,
    maxiter: int = DEFAULT_MAXITER,
    progressbar: bool = True,
) -> OptimizeResult:
    """Find root of nonlinear system with sparse Jacobian."""
    if solver is None:
        solver = NewtonArmijo()
    if f_tol is None:
        f_tol = tol
    if x_tol is None:
        x_tol = tol

    validate_fused_fun(fun, x0, args)

    max_ls = getattr(getattr(solver, "globalization", None), "max_iter", DEFAULT_ARMIJO_MAX_ITER)
    objective = ObjectiveWrapper(
        maxeval=maxiter * (max_ls + 1),
        f=fun,
        jac=None,
        args=args,
        progressbar=progressbar,
        progressbar_update_interval=1,
        has_fused_f_and_grad=True,
        root=True,
    )

    f_optim = partial(
        _root_find_core,
        fun=objective,
        x0=x0,
        solver=solver,
        args=(),
        f_tol=f_tol,
        x_tol=x_tol,
        maxiter=maxiter,
    )
    return optimizer_early_stopping_wrapper(f_optim)
