import pytensor.tensor as pt

from pytensor.tensor import TensorVariable


def stabilize(x, jitter=1e-16):
    """Add a small constant to the diagonal of a square matrix to keep linear solves well conditioned.

    Parameters
    ----------
    x : TensorVariable
        Square matrix to stabilize.
    jitter : float
        Amount added to each diagonal entry. Defaults to 1e-16.

    Returns
    -------
    x_stable : TensorVariable
        Copy of ``x`` with ``jitter`` added to its diagonal.
    """
    n = x.shape[0]
    idx = pt.arange(n)
    return x[idx, idx].inc(jitter)


def o1_policy_function_adjoints(
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,
    T: TensorVariable,
    T_bar: TensorVariable,
) -> list[TensorVariable, TensorVariable, TensorVariable]:
    """
    Compute the adjoint gradients to a matrix quadratic equation.

    The matrix quadratic equation is of the form:

    .. math::

        A + BT + CTT = 0

    It is associated with the first order approximation to a DSGE policy function.

    Parameters
    ----------
    A : TensorVariable
        Matrix of partial derivatives with respect to variables at t-1, evaluated at the steady-state
    B : TensorVariable
        Matrix of partial derivatives with respect to variables at t, evaluated at the steady-state
    C : TensorVariable
        Matrix of partial derivatives with respect to variables at t+1, evaluated at the steady-state
    T : TensorVariable
        Solved policy function matrix, such that :math:`X_t = T X_{t-1}`
    T_bar : TensorVariable
        Backward sensitivity of a scalar loss function with respect to the solved policy function T

    Returns
    -------
    adjoints : list of TensorVariable
        Adjoints of A, B and C, in that order.
    """
    vec_T_bar = T_bar.T.ravel()

    n = A.shape[0]

    # Compute matrix of lagrange multipliers S
    eye = pt.eye(n)
    M1 = pt.linalg.kron(T, C.T)
    M2 = pt.linalg.kron(eye, T.T @ C.T)
    M3 = pt.linalg.kron(eye, B.T)

    vec_S = pt.linalg.solve(stabilize(M1 + M2 + M3), -vec_T_bar, assume_a="gen", check_finite=False)
    S = vec_S.reshape((n, n)).T

    # With S, compute adjoints of the inputs
    A_bar = S
    B_bar = S @ T.T
    C_bar = S @ T.T @ T.T

    return [A_bar, B_bar, C_bar]


def pt_compute_selection_matrix(B, C, D, T):
    """Compute the shock selection matrix :math:`R = -(CT + B)^{-1} D`.

    Parameters
    ----------
    B : TensorVariable
        Matrix of partial derivatives with respect to variables at t, evaluated at the steady state.
    C : TensorVariable
        Matrix of partial derivatives with respect to variables at t+1, evaluated at the steady state.
    D : TensorVariable
        Matrix of partial derivatives with respect to exogenous shocks, evaluated at the steady state.
    T : TensorVariable
        Solved policy function matrix.

    Returns
    -------
    R : TensorVariable
        Selection matrix giving the effect of exogenous shocks on the variables.
    """
    return -pt.linalg.solve(C @ T + B, D, assume_a="gen", check_finite=False)
