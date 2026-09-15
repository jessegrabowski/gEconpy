from functools import partial
from typing import Any

import numpy as np

from better_optimize.wrapper import ObjectiveWrapper, optimizer_early_stopping_wrapper
from scipy.optimize import OptimizeResult

from gEconpy.solvers.sparse_root.base import (
    DEFAULT_ARMIJO_MAX_ITER,
    DEFAULT_MAXITER,
    DEFAULT_TOL,
    RootFunction,
    RootSolver,
    SolverState,
    default_check_convergence,
    default_failure_message,
    validate_fused_fun,
)
from gEconpy.solvers.sparse_root.line_search import NewtonArmijo


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
    """Find a root of a nonlinear system whose Jacobian is sparse.

    Parameters
    ----------
    fun : callable
        Fused residual and Jacobian function, called as ``fun(x, *args)`` and returning a tuple of a dense residual
        ndarray and a sparse Jacobian.
    x0 : array_like
        Initial guess for the root. Cast to a float64 array before ``fun`` is first called.
    solver : RootSolver, optional
        Iteration strategy, implementing ``init`` and ``step``. Defaults to
        :func:`~gEconpy.solvers.sparse_root.line_search.NewtonArmijo`.
    args : tuple, optional
        Extra positional arguments passed to ``fun``. Defaults to an empty tuple.
    tol : float, optional
        Tolerance used for both the residual and the step test when neither is given explicitly. Defaults to 1e-10.
    f_tol : float, optional
        Convergence tolerance on the maximum absolute residual. Defaults to ``tol``.
    x_tol : float, optional
        Convergence tolerance on the relative step size. Defaults to ``tol``.
    maxiter : int, optional
        Maximum number of solver iterations. Defaults to 1000.
    progressbar : bool, optional
        Whether to display a progress bar while iterating. Defaults to True.

    Returns
    -------
    result : OptimizeResult
        Result with fields ``x``, ``success``, ``message``, ``fun`` (final residuals), ``jac`` (final sparse
        Jacobian), ``nit``, and ``nfev``.

    Examples
    --------
    Solve a two-variable system whose Jacobian is diagonal, so the sparse form is a ``diags`` matrix:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        from gEconpy.solvers.sparse_root import sparse_root


        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")


        result = sparse_root(fun, np.array([2.0, 3.0]), progressbar=False)
        print(result.x)

    Swap in a trust region solver when the Jacobian may be singular along the way:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        from gEconpy.solvers.sparse_root import LevenbergMarquardt, sparse_root


        def fun(x):
            res = np.array([x[0] + x[1] - 3.0, x[0] * x[1] - 2.0])
            jac = sp.csc_matrix([[1.0, 1.0], [x[1], x[0]]])
            return res, jac


        result = sparse_root(fun, np.array([1.0, 1.0]), solver=LevenbergMarquardt(), progressbar=False)
        print(result.x)
    """
    if solver is None:
        solver = NewtonArmijo()
    if f_tol is None:
        f_tol = tol
    if x_tol is None:
        x_tol = tol

    x0 = np.asarray(x0, dtype=np.float64)
    validate_fused_fun(fun, x0, args)

    max_line_search_evals = getattr(getattr(solver, "globalization", None), "max_iter", DEFAULT_ARMIJO_MAX_ITER)
    objective = ObjectiveWrapper(
        maxeval=maxiter * (max_line_search_evals + 1),
        f=fun,
        jac=None,
        args=args,
        progressbar=progressbar,
        progressbar_update_interval=1,
        has_fused_f_and_grad=True,
        root=True,
    )

    run = partial(
        _iterate_to_root,
        fun=objective,
        x0=x0,
        solver=solver,
        args=(),
        f_tol=f_tol,
        x_tol=x_tol,
        maxiter=maxiter,
    )
    return optimizer_early_stopping_wrapper(run)


def _iterate_to_root(
    fun: RootFunction,
    x0: np.ndarray,
    solver: RootSolver,
    args: tuple[Any, ...],
    f_tol: float,
    x_tol: float,
    maxiter: int,
) -> OptimizeResult:
    state = solver.init(fun, x0, args)
    if np.max(np.abs(state.res)) < f_tol:
        return _make_result(state, success=True, message="Converged")

    check_convergence = getattr(solver, "check_convergence", default_check_convergence)
    failure_message = getattr(solver, "failure_message", default_failure_message)

    for _ in range(maxiter):
        state, info = solver.step(fun, state, args)
        if not info.accepted:
            return _make_result(state, success=False, message=info.message)
        if check_convergence(state, f_tol=f_tol, x_tol=x_tol, last_step=info.step):
            return _make_result(state, success=True, message="Converged")

    return _make_result(state, success=False, message=failure_message(state, maxiter))


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
