from typing import Optional, Tuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike


@njit(cache=True)
def cycle_reduction(
    A0: ArrayLike,
    A1: ArrayLike,
    A2: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-7,
    verbose: bool = True,
) -> Tuple[Optional[ArrayLike], str, float]:

    """
    Solve quadratic matrix equation of the form $A0x^2 + A1x + A2 = 0$ via cycle reduction algorithm of [1].
    Useful in the DSGE context to solve for the implicit derivative of the policy function, g, with respect to
    state vector y.

    Adapted from the Dynare file cycle_reduction.m, found at
    https://github.com/DynareTeam/dynare/blob/master/matlab/cycle_reduction.m

    Parameters
    ----------
    A0: Arraylike
        Coefficient matrix associated with the constant term of the matrix quadratic equation. In DSGE models, this is
        dF_d_t-1, the derivative of the system with respect to variables that enter as lags
    A1: ArrayLike
        Coefficient matrix associated with the linear term of the matrix quadratic equation. In DSGE models, this is
        dF_d_t, the derivative of the system with respect to variables that enter at the current time
    A2: ArrayLike
        Coefficient matrix associated with the quadratic term of the matrix quadratic equation. In DSGE models, this is
        dF_d_t+1, the derivative of the system with respect to variables that enter in expectation
    max_iter: int, default: 1000
        Maximum number of iterations to perform before giving up.
    tol: float, default: 1e-7
        Floating point tolerance used to detect algorithmic convergence
    verbose: bool, default: True
        If true, prints the sum of squared residuals that result when the system is computed used the solution.

    Returns
    -------

    References
    -------
    ..[1] D.A. Bini, G. Latouche, B. Meini (2002), "Solving matrix polynomial equations
          arising in queueing problems", Linear Algebra and its Applications 340, pp. 222-244
    ..[2]

    """
    result = "Optimization successful"
    log_norm = 0
    X = None

    A0_initial = A0.copy()
    A1_hat = A1.copy()

    if verbose:
        A1_initial = A1.copy()
        A2_initial = A2.copy()

    n, _ = A0.shape
    idx_0 = np.arange(n)
    idx_1 = idx_0 + n

    # Pre-allocate this so it doesn't have to be repeatedly created
    EYE = np.eye(A1.shape[0])

    for i in range(max_iter):
        tmp = np.vstack((A0, A2)) @ np.linalg.solve(A1, EYE) @ np.hstack((A0, A2))

        A1 = A1 - tmp[idx_0, :][:, idx_1] - tmp[idx_1, :][:, idx_0]
        A0 = -tmp[idx_0, :][:, idx_0]
        A2 = -tmp[idx_1, :][:, idx_1]
        A1_hat = A1_hat - tmp[idx_1, :][:, idx_0]

        A0_L1_norm = np.linalg.norm(A0, ord=1)
        if A0_L1_norm < tol:
            # Algorithm is successful when the L1 norm of A2 is sufficiently small
            A2_L1_norm = np.linalg.norm(A2, ord=1)
            if A2_L1_norm < tol:
                break

        elif np.isnan(A0_L1_norm) or i == (max_iter - 1):
            # If we fail, figure out how far we got
            if A0_L1_norm < tol:
                result = "Iteration on matrix A0 and A1 converged towards a solution, but A2 did not."
                log_norm = np.log(np.linalg.norm(A2, 1))
            else:
                result = "Iteration on all matrices failed to converged"
                log_norm = np.log(np.linalg.norm(A1, 1))

            return X, result, log_norm

    X = -np.linalg.solve(A1_hat, A0_initial)

    if verbose:
        res = A0_initial + A1_initial @ X + A2_initial @ X @ X
        print("Solution found, sum of squared residuals: ", (res**2).sum())

    return X, result, log_norm


@njit(cache=True)
def solve_shock_matrix(B, C, D, G_1):
    """
    Given the partial solution to the linear approximate policy function G_1, solve for the remaining component of the
    policy function, R.

    Parameters
    ----------
    B: ArrayLike
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        are observed when decision making: those with t subscripts.
    C: Arraylike
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        enter in expectation when decision making: those with t+1 subscripts.
    D: ArrayLike
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to exogenous shocks.
    G_1: ArrayLike
        Transition matrix T in state space jargon. Gives the effect of variable values at time t on the
        values of the variables at time t+1.
    Returns
    -------
    impact: ArrayLike
        Selection matrix R in state space jargon. Gives the effect of exogenous shocks at time t on the values of
        system variables at time t+1.

    """

    return -np.linalg.solve(C @ G_1 + B, np.eye(C.shape[0])) @ D
