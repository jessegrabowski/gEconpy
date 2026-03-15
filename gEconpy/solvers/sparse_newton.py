from collections.abc import Callable

import numpy as np

from scipy.optimize import OptimizeResult

from gEconpy.solvers.sparse_root import sparse_root
from gEconpy.solvers.sparse_root.direction import NewtonDirection
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking
from gEconpy.solvers.sparse_root.newton import NewtonArmijo


def sparse_newton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    tol: float = 1e-10,
    x_tol: float | None = None,
    f_tol: float | None = None,
    beta: float = 0.5,
    c1: float = 1e-4,
    max_ls: int = 50,
    maxiter: int = 1000,
    solver: Callable | None = None,
    progressbar: bool = True,
) -> OptimizeResult:
    """Backward-compatible sparse Newton solver.

    New code should use :func:`~gEconpy.solvers.sparse_root.sparse_root` with a
    :class:`~gEconpy.solvers.sparse_root.newton.NewtonArmijo` solver instead.
    """
    newton_solver = NewtonArmijo(
        direction=NewtonDirection(linear_solver=solver),
        globalization=ArmijoBacktracking(c1=c1, beta=beta, max_iter=max_ls),
    )
    return sparse_root(
        fun=fun,
        x0=x0,
        solver=newton_solver,
        args=args,
        tol=tol,
        f_tol=f_tol,
        x_tol=x_tol,
        maxiter=maxiter,
        progressbar=progressbar,
    )
