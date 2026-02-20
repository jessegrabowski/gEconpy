from collections.abc import Callable
from functools import partial

import numpy as np

from better_optimize.wrapper import ObjectiveWrapper, optimizer_early_stopping_wrapper
from scipy import sparse
from scipy.optimize import OptimizeResult


def _validate_fused_fun(fun: Callable, x0: np.ndarray, args: tuple) -> None:
    """Validate that fun returns (residuals, sparse_jacobian)."""
    test_result = fun(x0, *args)
    n_expected_outputs = 2
    if not (isinstance(test_result, tuple) and len(test_result) == n_expected_outputs):
        raise ValueError("fun must return a tuple of (residuals, jacobian)")

    res, jac = test_result
    if not isinstance(res, np.ndarray):
        raise TypeError("fun must return residuals as ndarray")
    if not sparse.issparse(jac):
        raise TypeError("fun must return a sparse jacobian")


def _sparse_newton_core(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    tol: float = 1e-10,
    maxiter: int = 100,
    solver: Callable | None = None,
) -> OptimizeResult:
    """Core sparse Newton implementation."""
    if solver is None:
        solver = sparse.linalg.spsolve

    res, jac = fun(x0, *args)
    x = x0.copy()
    nfev = 1

    err = np.max(np.abs(res))
    if err < tol:
        return OptimizeResult(x=x, success=True, message="Converged", fun=res, jac=jac, nit=0, nfev=nfev)

    for iteration in range(1, 1 + maxiter):
        dx = solver(jac, -res)
        x = x + dx

        res, jac = fun(x, *args)
        nfev += 1

        err = np.max(np.abs(res))
        if err < tol:
            return OptimizeResult(
                x=x,
                success=True,
                message="Converged",
                fun=res,
                jac=jac,
                nit=iteration,
                nfev=nfev,
            )

    return OptimizeResult(
        x=x,
        success=False,
        message=f"Did not converge after {maxiter} iterations",
        fun=res,
        jac=jac,
        nit=maxiter,
        nfev=nfev,
    )


def sparse_newton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    tol: float = 1e-10,
    maxiter: int = 100,
    solver: Callable | None = None,
) -> OptimizeResult:
    """Solve a nonlinear system using Newton's method with sparse Jacobian.

    Parameters
    ----------
    fun : callable
        Function that returns (residuals, jacobian) given x and *args.
        Jacobian must be a scipy sparse matrix.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to fun.
    tol : float, default 1e-10
        Convergence tolerance on the infinity norm of residuals.
    maxiter : int, default 100
        Maximum number of iterations.
    solver : callable, optional
        Sparse linear solver with signature solver(A, b) -> x.
        Defaults to scipy.sparse.linalg.spsolve.

    Returns
    -------
    OptimizeResult
        Result object with solution and convergence info.

    Raises
    ------
    ValueError
        If fun does not return a (residuals, sparse_jacobian) tuple.
    """
    _validate_fused_fun(fun, x0, args)

    # TODO: progressbar=False until better_optimize supports sparse jacobian display
    objective = ObjectiveWrapper(
        maxeval=maxiter,
        f=fun,
        jac=None,
        args=args,
        progressbar=False,
        progressbar_update_interval=1,
        has_fused_f_and_grad=True,
        root=True,
    )

    f_optim = partial(
        _sparse_newton_core,
        fun=objective,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        solver=solver,
    )

    return optimizer_early_stopping_wrapper(f_optim)
