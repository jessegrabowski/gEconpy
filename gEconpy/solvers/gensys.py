from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg


def qzdiv(
    stake: float, A: ArrayLike, B: ArrayLike, Q: ArrayLike, Z: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Christopher Sim's qzdiv http://sims.princeton.edu/yftp/gensys/mfiles/qzdiv.m
    :param stake: float, largest positive value for which an eigenvalue is considered stable
    :param A: Array, upper-triangular matrix
    :param B: Array, upper-triangular matrix
    :param Q: Array, matrix of left Schur vectors.
    :param Z: Array, matrix of right Schur vectors.
    :return: Tuple, A, B, Q, Z, sorted such that all unstable roots are placed in the lower-right corners of the
             matrices.

    Original docstring:
    Takes U.T. matrices A, B, orthonormal matrices Q,Z, rearranges them so that all cases of abs(B(i,i)/A(i,i))>stake
    are in lower right  corner, while preserving U.T. and orthonormal properties and Q'AZ' and Q'BZ'.
    The columns of v are sorted correspondingly.

    Additional notes:
    Matrices A, B, Q, and Z are the output of the generalized Schur decomposition (QZ decomposition) of the system
    matrices G0 and G1. A and B are upper triangular, with the properties Q @ A @ Z.T = G0 and Q @ B @ Z.T = G1.

    TODO: scipy offers a sorted qz routine, ordqz, which automatically sorts the matrices by size of eigenvalue. This
        seems to be what the functions qzdiv and qzswitch do, so it might be worthwhile to see if we can just use
        ordqz instead.

    TODO: Add shape information to the Typing (see PEP 646)
    """

    n, _ = A.shape

    root = np.hstack([np.diag(A)[:, None], np.diag(B)[:, None]])
    root = np.abs(root)
    root[:, 0] = root[:, 0] - (root[:, 0] < 1e-13) * (root[:, 0] + root[:, 1])
    root[:, 1] = root[:, 1] / root[:, 0]

    for i in range(n - 1, -1, -1):
        m = None
        for j in range(i, -1, -1):
            if (root[j, 1] > stake) or (root[j, 1] < -0.1):
                m = j
                break

        if m is None:
            return A, B, Q, Z

        for k in range(m, i):
            A, B, Q, Z = qzswitch(k, A, B, Q, Z)
            root[k, 1], root[k + 1, 1] = root[k + 1, 1], root[k, 1]

    return A, B, Q, Z


def qzswitch(
    i: int, A: ArrayLike, B: ArrayLike, Q: ArrayLike, Z: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Christopher Sim's qzswitch,
    :param i: int, index of matrix diagonal to switch
    :param A: Array, upper-triangular matrix
    :param B: Array, upper-triangular matrix
    :param Q: Array, matrix of left Schur vectors.
    :param Z: Array, matrix of right Schur vectors.
    :return: Tuple of A, B, Q, Z

    Original docstring:
    Takes U.T. matrices A, B, orthonormal matrices Q,Z, interchanges diagonal elements i and i+1 of both A and B,
    while maintaining Q'AZ' and Q'BZ' unchanged.  If diagonal elements of A and B are zero at matching positions,
    the returned A will have zeros at both positions on the diagonal.  This is natural behavior if this routine is used
    to drive all zeros on the diagonal of A to the lower right, but in this case the qz transformation is not unique
    and it is not possible simply to switch the positions of the diagonal elements of both A and B.

    TODO: Add shape information to the Typing (see PEP 646)
    """
    eps = np.spacing(1)

    a = A[i, i]
    b = A[i, i + 1]
    c = A[i + 1, i + 1]
    d = B[i, i]
    e = B[i, i + 1]
    f = B[i + 1, i + 1]

    if (abs(c) < eps) & (abs(f) < eps):
        if abs(a) < eps:
            return A, B, Q, Z

        else:
            wz = np.c_[b, -a].T
            wz = wz / np.sqrt(wz.conj().T @ wz)
            wz = np.hstack([wz, np.c_[wz[1].conj().T, -wz[0].conj().T].T])
            xy = np.eye(2)

    elif (abs(a) < eps) & (abs(d) < eps):
        if abs(c) < eps:
            return A, B, Q, Z
        else:
            wz = np.eye(2)
            xy = np.c_[c, -b].T
            xy = xy / np.sqrt(xy @ xy.conj().T)
            xy = np.hstack([np.c_[xy[1].conj().T, -xy[0].conj().T].T, xy])

    else:
        wz = np.c_[c * e - f * b, (c * d - f * a).conj()]
        xy = np.c_[(b * d - e * a).conj(), (c * d - f * a).conj()]

        n = np.sqrt(wz @ wz.conj().T)
        m = np.sqrt(xy @ xy.conj().T)

        if m < eps * 100:
            return A, B, Q, Z

        wz = wz / n
        xy = xy / m

        wz = np.vstack([wz, np.c_[-wz[:, 1].conj(), wz[:, 0].conj()]])
        xy = np.vstack([xy, np.c_[-xy[:, 1].conj(), xy[:, 0].conj()]])

    idx_slice = slice(i, i + 2)

    A[idx_slice, :] = xy @ A[idx_slice, :]
    B[idx_slice, :] = xy @ B[idx_slice, :]
    Q[idx_slice, :] = xy @ Q[idx_slice, :]

    A[:, idx_slice] = A[:, idx_slice] @ wz
    B[:, idx_slice] = B[:, idx_slice] @ wz
    Z[:, idx_slice] = Z[:, idx_slice] @ wz

    return A, B, Q, Z


def determine_n_unstable(
    A: ArrayLike, B: ArrayLike, div: float, realsmall: float
) -> Tuple[float, int, bool]:
    """
    :param A: array, upper-triangular matrix, output of QZ decomposition
    :param B: array, upper-triangular matrix, output of QZ decomposition
    :param div: float, largest positive value for which an eigenvalue is considered stable
    :param realsmall: an arbitrarily small number
    :return: Tuple

    Originally part of gensys, this helper function determines how many roots of the system described by A and B are
    unstable. Returns three values:
        div, a float representing which roots of the system can be considered stable,
        n_unstable, an int of how many unstable roots are in the system, and
        zxz, a bool that signals whether the system has a unique solution.
    """
    n, _ = A.shape
    n_unstable = 0
    zxz = False
    compute_div = div is None
    div = 1.01 if div is None else div

    for i in range(n):
        if compute_div:
            if abs(A[i, i]) > 0:
                divhat = abs(B[i, i] / A[i, i])
                if (1 + realsmall < divhat) and divhat <= div:
                    div = 0.5 * (1 + divhat)
        n_unstable += abs(B[i, i]) > div * abs(A[i, i])

        zxz = (abs(A[i, i]) < realsmall) & (abs(B[i, i]) < realsmall)

    return div, n_unstable, zxz


def split_matrix_on_eigen_stability(
    A: ArrayLike, n_unstable: int
) -> Tuple[ArrayLike, ArrayLike]:
    """
    :param A: Arrayline, array to split
    :param n_unstable: int, number of unstable roots in the
    :return: Tuple of (A1, A2), the A matrix split such that all stable roots are in the A1 matrix,
             and the unstable roots are in the A2 matrix.

    Originally in the gensys function, split out here for readability.
    """
    n, _ = A.shape
    stable_slice = slice(None, n - n_unstable)
    unstable_slice = slice(n - n_unstable, None)

    A1 = A[stable_slice]
    A2 = A[unstable_slice]

    return A1, A2


def build_u_v_d(eta: ArrayLike, realsmall: float):
    """

    :param eta:
    :param realsmall:
    :return: tuple, svd decomposition of eta plus an array of non-zero indices

    Piece of gensys adapted from Matlab code, split out as a helper function.
    """

    u_eta, d_eta, v_eta = linalg.svd(eta)
    d_eta = np.diag(d_eta)  # match matlab output of svd
    v_eta = v_eta.conj().T  # match matlab output of svd

    md = min(d_eta.shape)
    big_ev = np.where(np.diagonal(d_eta[:md, :md] > realsmall))[0]

    u_eta = u_eta[:, big_ev]
    v_eta = v_eta[:, big_ev]
    d_eta = d_eta[big_ev, big_ev]

    if d_eta.ndim == 1:
        d_eta = np.diag(d_eta)

    return u_eta, v_eta, d_eta, big_ev


def gensys(
    g0: ArrayLike,
    g1: ArrayLike,
    c: ArrayLike,
    psi: ArrayLike,
    pi: ArrayLike,
    div: Optional[float] = None,
    tol: Optional[float] = 1e-8,
) -> Tuple:
    """
    Christopher Sim's gensys, http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m

    Solves rational expectations equations as described in [1] by partitioning partitioning the system into stable
    and unstable roots, then eliminating the unstable roots via QZ decomposition.

    Original matlab docstring:
    System given as g0*y(t)=g1*y(t-1)+c+psi*z(t)+pi*eta(t), with z an exogenous variable process and eta being
    endogenously determined one-step-ahead expectational errors.  Returned system is
        y(t)=G1*y(t-1)+C+impact*z(t)+ywt*inv(I-fmat*inv(L))*fwt*z(t+1) .
    If z(t) is i.i.d., the last term drops out. If div is omitted from argument list, a div>1 is calculated.
    eu(1)=1 for existence,
    eu(2)=1 for uniqueness.
    eu(1)=-1 for existence only with not-s.c. z;
    eu=[-2,-2] for coincident zeros.

    Parameters
    ----------
    g0: ArrayLike
        Coefficient matrix of the dynamic system corresponding to the time-t variables
    g1: ArrayLike
        Coefficient matrix of the dynamic system corresponding to the time t-1 variables
    c: ArrayLike
        Vector of constant terms
    psi: ArrayLike
        Coefficient matrix of the dynamic system corresponding to the exogenous shock terms
    pi: ArrayLike
        Coefficient matrix of the dynamic system corresponding to the endogenously determined
        expectational errors.
    div: float
        # TODO: WRITE ME
    tol: float, default: 1e-8
        Level of floating point precision

    Returns
    -------
    G_1: ArrayLike
        Policy function relating the current timestep to the next, transition matrix T in state space jargon.
    C: ArrayLike
        Array of system means, intercept vector c in state space jargon.
    impact: ArrayLike
        Policy function component relating exogenous shocks observed at the t to variable values in t+1, selection
        matric R in state space jargon.
    f_mat: ArrayLike
        # TODO: WRITE ME
    f_wt: ArrayLike
        # TODO: WRITE ME
    y_wt: ArrayLike
        # TODO: WRITE ME
    gev: ArrayLike
        Generalized left and right eigenvalues generated by qz(g0, g1), sorted such that stable roots are in the
        top-left corner
    eu: tuple
        Tuple of two values indicting uniqueness and determinacy of the solution.
    loose: int
        Number of loose endogenous variables.

    References
    -------
    ..[1] Sims, Christopher A. "Solving linear rational expectations models." Computational economics 20.1-2 (2002): 1.

    TODO: Can this be written in Numba/Aesara?
    """
    eu = [0, 0, 0]

    n, _ = g1.shape
    A, B, Q, Z = linalg.qz(g0, g1, "complex")
    Q = Q.conj().T  # q is transposed relative to matlab, see scipy docs

    div, n_unstable, zxz = determine_n_unstable(A, B, div, tol)
    n_stable = n - n_unstable

    if zxz:
        eu = [-2, -2, 0]
        return None, None, None, None, None, None, None, eu, None

    A, B, Q, Z = qzdiv(div, A, B, Q, Z)
    gev = np.c_[np.diagonal(A), np.diagonal(B)]

    Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)

    eta_wt = Q2 @ pi
    _, n_eta = pi.shape

    # No stable roots
    if n_unstable == 0:
        big_ev = 0

        u_eta = np.zeros((0, 0))
        d_eta = np.zeros((0, 0))
        v_eta = np.zeros((n_eta, 0))

    else:
        u_eta, v_eta, d_eta, big_ev = build_u_v_d(eta_wt, tol)

    if len(big_ev) >= n_unstable:
        eu[0] = 1

    # All stable roots
    if n_unstable == n:
        eta_wt_1 = np.zeros((0, n_eta))
        u_eta_1 = np.zeros((0, 0))
        d_eta_1 = np.zeros((0, 0))
        v_eta_1 = np.zeros((n_eta, 0))

    else:
        eta_wt_1 = Q1 @ pi
        u_eta_1, v_eta_1, d_eta_1, big_ev = build_u_v_d(eta_wt_1, tol)

    if 0 in v_eta_1.shape:
        unique = True
    else:
        loose = v_eta_1 - v_eta @ v_eta.T @ v_eta_1
        ul, dl, vl = linalg.svd(loose)
        if dl.ndim == 1:
            dl = np.diag(dl)

        n_loose = (np.abs(np.diagonal(dl)) > (tol * n)).sum()
        eu[2] = n_loose
        unique = n_loose == 0

    if unique:
        eu[1] = 1

    inner_term = (
        u_eta
        @ linalg.solve(d_eta, v_eta.conj().T)
        @ v_eta_1
        @ d_eta_1
        @ u_eta_1.conj().T
    )

    T_mat = np.c_[np.eye(n_stable), -inner_term.conj().T]
    G_0 = np.r_[T_mat @ A, np.c_[np.zeros((n_unstable, n_stable)), np.eye(n_unstable)]]

    G_1 = np.r_[T_mat @ B, np.zeros((n_unstable, n))]

    G_0_inv = linalg.inv(G_0)
    G_1 = G_0_inv @ G_1

    idx = slice(n_stable, n)

    C = np.r_[T_mat @ Q @ c, linalg.solve(A[idx, idx] - B[idx, idx], Q2) @ c]

    impact = G_0_inv @ np.r_[T_mat @ Q @ psi, np.zeros((n_unstable, psi.shape[1]))]

    f_mat = linalg.solve(B[idx, idx], A[idx, idx])
    f_wt = -linalg.solve(B[idx, idx], Q2) @ psi
    y_wt = G_0_inv[:, idx]

    loose = (
        G_0_inv
        @ np.r_[
            eta_wt_1 @ (np.eye(n_eta) - v_eta @ v_eta.conj().T),
            np.zeros((n_unstable, n_eta)),
        ]
    )

    G_1 = (Z @ G_1 @ Z.conj().T).real
    C = (Z @ C).real
    impact = (Z @ impact).real
    loose = (Z @ loose).real
    y_wt = Z @ y_wt

    return G_1, C, impact, f_mat, f_wt, y_wt, gev, eu, loose


def interpret_gensys_output(eu):
    message = ""
    if eu[0] == -2 and eu[1] == -2:
        message = "Coincident zeros.  Indeterminacy and/or nonexistence."
    elif eu[0] == -1:
        message = (
            f"System is indeterminate. There are {eu[2]} loose endogenous variables."
        )
    elif eu[1] == -1:
        message = f"Solution exists, but it is not unique -- sunspots."

    return message
