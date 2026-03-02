from collections.abc import Callable
from functools import partial

import numpy as np

from better_optimize.wrapper import ObjectiveWrapper, optimizer_early_stopping_wrapper
from scipy import sparse
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import spsolve


def _validate_fused_fun(fun: Callable, x0: np.ndarray, args: tuple) -> None:
    """Validate that fun returns (residuals, sparse_jacobian)."""
    test_result = fun(x0, *args)
    if not (isinstance(test_result, tuple) and len(test_result) == 2):
        raise ValueError("fun must return a tuple of (residuals, jacobian)")

    res, jac = test_result
    if not isinstance(res, np.ndarray):
        raise TypeError("fun must return residuals as ndarray")
    if not sparse.issparse(jac):
        raise TypeError("fun must return a sparse jacobian")


def _check_convergence(dx: np.ndarray, x_new: np.ndarray, tol: float) -> bool:
    return np.max(np.abs(dx)) < tol * (1.0 + np.max(np.abs(x_new)))


def _phi(res: np.ndarray) -> float:
    """Merit function for line search: 0.5 * ||res||^2."""
    return 0.5 * np.dot(res, res)


def _solve_newton_step(jac: sparse.spmatrix, res: np.ndarray, solver: Callable | None) -> tuple[np.ndarray, bool]:
    """Compute the Newton step dx = -J^{-1} F(x).

    Uses ``spsolve`` by default, or delegates to a caller-provided linear solver. Returns ``(dx, True)`` on success,
    or ``(steepest_descent_direction, False)`` on failure.
    """
    try:
        dx = spsolve(jac, -res) if solver is None else solver(jac, -res)

        if np.all(np.isfinite(dx)):
            return dx, True

    except Exception:
        pass

    return -(jac.T @ res), False


def _sparse_newton_core(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    f_tol: float = 1e-10,
    x_tol: float = 1e-10,
    max_ls: int = 50,
    beta: float = 0.5,
    c1: float = 1e-4,
    maxiter: int = 100,
    solver: Callable | None = None,
) -> OptimizeResult:
    """Core sparse Newton implementation with backtracking line search.

    Uses the Armijo condition to globalize convergence. If the Newton direction is not a descent direction for the
    merit function ``0.5 * ||F(x)||^2``, the direction is negated to guarantee descent. If the linear solve fails
    (singular Jacobian), the solver falls back to a steepest-descent step on the merit function.
    """
    x = x0.copy()
    res, jac = fun(x, *args)
    nfev = 1

    err = np.max(np.abs(res))
    if err < f_tol:
        return OptimizeResult(x=x, success=True, message="Converged", fun=res, jac=jac, nit=0, nfev=nfev)

    phi_current = _phi(res)
    phi_f_tol = 0.5 * f_tol * f_tol

    for iteration in range(1, maxiter + 1):
        dx, _solve_ok = _solve_newton_step(jac, res, solver)

        slope = np.dot(res, jac @ dx)

        if slope >= 0:
            dx = -dx
            slope = -slope

        # Backtracking line search with Armijo condition
        alpha = 1.0
        accepted = False

        for _ in range(max_ls):
            x_trial = x + alpha * dx
            res_trial, jac_trial = fun(x_trial, *args)
            nfev += 1

            phi_trial = _phi(res_trial)
            if phi_trial <= phi_current + c1 * alpha * slope:
                x, res, jac = x_trial, res_trial, jac_trial
                phi_current = phi_trial
                accepted = True
                break

            alpha *= beta

        if not accepted:
            return OptimizeResult(
                x=x,
                success=False,
                message=f"Line search failed after {max_ls} attempts (iteration {iteration})",
                fun=res,
                jac=jac,
                nit=iteration,
                nfev=nfev,
            )

        # phi = 0.5 * ||res||^2 >= 0.5 * max(|res|)^2, so if phi > 0.5 * f_tol^2
        # we know max(|res|) might still exceed f_tol but we need to check. When phi
        # is orders of magnitude above this bound, skip the more expensive checks.
        if phi_current <= phi_f_tol:
            err = np.max(np.abs(res))
            if err < f_tol:
                return OptimizeResult(
                    x=x,
                    success=True,
                    message="Converged",
                    fun=res,
                    jac=jac,
                    nit=iteration,
                    nfev=nfev,
                )

            if _check_convergence(alpha * dx, x, x_tol):
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
        message=f"Did not converge after {maxiter} iterations (max |residual| = {np.max(np.abs(res)):.2e})",
        fun=res,
        jac=jac,
        nit=maxiter,
        nfev=nfev,
    )


def sparse_newton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    tol: float | None = 1e-10,
    x_tol: float | None = None,
    f_tol: float | None = None,
    beta: float = 0.5,
    c1: float = 1e-4,
    max_ls: int = 50,
    maxiter: int = 1000,
    solver: Callable | None = None,
    progressbar: bool = True,
) -> OptimizeResult:
    """Solve a nonlinear system using Newton's method with sparse Jacobian.

    Finds ``x`` such that ``F(x) = 0`` where ``F`` returns both residuals and a sparse Jacobian. Convergence is
    globalized via backtracking line search on the merit function ``0.5 * ||F(x)||^2`` with the Armijo condition.

    The default linear solver is ``scipy.sparse.linalg.spsolve`` (sparse direct solve via LU). A custom ``solver``
    callable can be provided for iterative methods (e.g., GMRES) on very large systems. If the linear solve fails
    (singular Jacobian, non-finite result), the solver falls back to a steepest-descent step on the merit function.

    Parameters
    ----------
    fun : callable
        Function with signature ``fun(x, *args) -> (residuals, jacobian)``. The Jacobian must be a scipy sparse
        matrix.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to ``fun``.
    tol : float, default 1e-10
        Default convergence tolerance, used for both ``x_tol`` and ``f_tol`` when they are not provided.
    x_tol : float, optional
        Step-size convergence tolerance: the solver converges when
        ``max(|dx|) < x_tol * (1 + max(|x|))``. Defaults to ``tol``.
    f_tol : float, optional
        Residual convergence tolerance: the solver converges when ``max(|F(x)|) < f_tol``. Defaults to ``tol``.
    beta : float, default 0.5
        Line search step reduction factor.
    c1 : float, default 1e-4
        Armijo sufficient decrease constant.
    max_ls : int, default 50
        Maximum number of line search reductions per iteration.
    maxiter : int, default 1000
        Maximum number of Newton iterations.
    solver : callable, optional
        Sparse linear solver with signature ``solver(A, b) -> x``. When provided, this replaces the default
        ``spsolve``. Useful for iterative solvers like GMRES on very large systems where direct factorization is
        too expensive.
    progressbar : bool, default True
        Whether to display a progress bar during optimization.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``success``, ``message``, ``fun`` (final residuals), ``jac`` (final Jacobian),
        ``nit`` (iterations), and ``nfev`` (function evaluations).
    """
    _validate_fused_fun(fun, x0, args)

    if x_tol is None:
        x_tol = tol
    if f_tol is None:
        f_tol = tol

    # maxeval is set high enough that the iteration budget (not the eval budget) is the binding constraint.
    # Each iteration may consume multiple evals during line search.
    max_evals = maxiter * (max_ls + 1)

    objective = ObjectiveWrapper(
        maxeval=max_evals,
        f=fun,
        jac=None,
        args=args,
        progressbar=progressbar,
        progressbar_update_interval=1,
        has_fused_f_and_grad=True,
        root=True,
    )

    f_optim = partial(
        _sparse_newton_core,
        fun=objective,
        x0=x0,
        x_tol=x_tol,
        f_tol=f_tol,
        beta=beta,
        c1=c1,
        max_ls=max_ls,
        maxiter=maxiter,
        solver=solver,
    )

    return optimizer_early_stopping_wrapper(f_optim)
