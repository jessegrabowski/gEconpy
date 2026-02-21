import numba as nb
import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy import linalg

from gEconpy.solvers.shared import (
    o1_policy_function_adjoints,
    pt_compute_selection_matrix,
)

# A very small number
EPSILON = np.spacing(1)
floatX = pytensor.config.floatX


@nb.njit(cache=True)
def neg_conj_flip(x):
    x_conj = x.conj()
    x[:] = np.array((-x_conj[1], x_conj[0]))
    return x


@nb.njit(
    [
        "UniTuple(c16[::1, :], 4)(i8, c16[::1, :], c16[::1, :], c16[::1, :], c16[::1, :])",
        "UniTuple(f8[::1, :], 4)(i8, f8[::1, :], f8[::1, :] ,f8[::1, :], f8[::1, :])",
    ],
    cache=True,
)
def qzswitch(
    i: int, A: np.ndarray, B: np.ndarray, Q: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Christopher Sim's qzswitch.

    Takes upper-triangular matrices A, B, orthonormal matrices Q, Z, and interchanges diagonal elements i and i+1 of
    both A and B, while maintaining Q'AZ' and Q'BZ' unchanged. If diagonal elements of A and B are zero at matching
    positions, the returned A will have zeros at both positions on the diagonal. This is natural behavior if this
    routine is used to drive all zeros on the diagonal of A to the lower right, but in this case the qz transformation
    is not unique and it is not possible simply to switch the positions of the diagonal elements of both A and B.

    Parameters
    ----------
    i : int
        Index of matrix diagonal to switch.
    A : np.ndarray
        Upper-triangular matrix.
    B : np.ndarray
        Upper-triangular matrix.
    Q : np.ndarray
        Matrix of left Schur vectors.
    Z : np.ndarray
        Matrix of right Schur vectors.

    Returns
    -------
    tuple of np.ndarray
        Contains four elements:
            A : np.ndarray
                Upper-triangular matrix with switched diagonal elements.
            B : np.ndarray
                Upper-triangular matrix with switched diagonal elements.
            Q : np.ndarray
                Orthonormal matrix of left Schur vectors.
            Z : np.ndarray
                Orthonormal matrix of right Schur vectors.

    Notes
    -----
    Originally part of gensys. Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m
    """
    eps = np.spacing(1)

    a = A[i, i]
    b = A[i, i + 1]
    c = A[i + 1, i + 1]
    d = B[i, i]
    e = B[i, i + 1]
    f = B[i + 1, i + 1]

    wz = np.empty((2, 2), dtype=A.dtype)
    xy = np.empty((2, 2), dtype=A.dtype)

    if (abs(c) < eps) & (abs(f) < eps):
        if abs(a) < eps:
            return A, B, Q, Z

        wz_row = np.array((b, -a))
        wz_inner = (wz_row * wz_row.conj()).sum()
        wz_row = wz_row / np.sqrt(wz_inner)

        wz[:, 0] = wz_row
        wz[:, 1] = neg_conj_flip(wz_row)
        xy[:] = np.eye(2).astype(wz.dtype)

    elif (abs(a) < eps) & (abs(d) < eps):
        if abs(c) < eps:
            return A, B, Q, Z
        xy_row = np.array((c, -b))
        xy_inner = (xy_row * xy_row.conj()).sum()
        xy_row = xy_row / np.sqrt(xy_inner)

        xy[:, 0] = neg_conj_flip(xy_row)
        xy[:, 1] = xy_row
        wz[:] = np.eye(2).astype(xy.dtype)

    else:
        wz_row = np.array((c * e - f * b, (c * d - f * a).conjugate()))
        xy_row = np.array(((b * d - e * a).conjugate(), (c * d - f * a).conjugate()))

        wz_inner = (wz_row * wz_row.conj()).sum()
        xy_inner = (xy_row * xy_row.conj()).sum()

        n = np.sqrt(wz_inner)
        m = np.sqrt(xy_inner)

        if np.abs(m) < eps * 100:
            return A, B, Q, Z

        wz_row = wz_row / n
        xy_row = xy_row / m

        # xy = np.row_stack((xy, neg_conj_flip(xy)))
        xy[0, :] = xy_row
        xy[1, :] = neg_conj_flip(xy_row)

        # wz = np.row_stack((wz, neg_conj_flip(wz)))
        wz[0, :] = wz_row
        wz[1, :] = neg_conj_flip(wz_row)

    idx_slice = slice(i, i + 2)

    A[idx_slice, :] = xy @ np.asfortranarray(A[idx_slice, :])
    B[idx_slice, :] = xy @ np.asfortranarray(B[idx_slice, :])
    Q[idx_slice, :] = xy @ np.asfortranarray(Q[idx_slice, :])

    A[:, idx_slice] = np.asfortranarray(A[:, idx_slice]) @ wz
    B[:, idx_slice] = np.asfortranarray(B[:, idx_slice]) @ wz
    Z[:, idx_slice] = np.asfortranarray(Z[:, idx_slice]) @ wz

    return A, B, Q, Z


@nb.njit(
    [
        "UniTuple(c16[::1, :], 4)(f8, c16[::1, :], c16[::1, :] ,c16[::1, :], c16[::1, :])",
        "UniTuple(f8[::1, :], 4)(f8, f8[::1, :], f8[::1, :] ,f8[::1, :], f8[::1, :])",
    ],
    cache=True,
)
def qzdiv(
    stake: float, A: np.ndarray, B: np.ndarray, Q: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Christopher Sim's qzdiv.

    Takes upper-triangular matrices :math:`A`, :math:`B` and orthonormal matrices :math:`Q`, :math:`Z`, and rearranges
    them so that all cases of ``abs(B(i, i) / A(i, i)) > stake`` are in the lower-right corner, while preserving
    upper-triangular and orthonormal properties, and maintaining the relationships :math:`Q^TAZ'` and :math:`Q^TBZ'`.
    The columns of v are sorted correspondingly.

    Matrices :math:`A`, :math:`B`, :math:`Q`, and :math:`Z` are the output of the generalized Schur decomposition
    (QZ decomposition) of the system matrices :math:`G_0` and :math:`G_1`. A and B are upper triangular, with the
    properties :math:`QAZ^T = G_0` and :math:`QBZ^T = G_1`.

    Parameters
    ----------
     stake : float
         Largest positive value for which an eigenvalue is considered stable.
     A : np.ndarray
         Upper-triangular matrix.
     B : np.ndarray
         Upper-triangular matrix.
     Q : np.ndarray
         Matrix of left Schur vectors.
     Z : np.ndarray
         Matrix of right Schur vectors.

    Returns
    -------
     tuple of np.ndarray
         A, B, Q, Z matrices sorted such that all unstable roots are placed in the lower-right corners of the matrices.

    Notes
    -----
    Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/qzdiv.m
    """
    # TODO: scipy offers a sorted qz routine, ordqz, which automatically sorts the matrices by size of eigenvalue. This
    #     seems to be what the functions qzdiv and qzswitch do, so it might be worthwhile to see if we can just use
    #     ordqz instead.
    #
    # TODO: Add shape information to the Typing (see PEP 646)
    FLOAT_ZERO = 1e-13
    n, _ = A.shape

    root = np.hstack((np.diag(A)[:, None], np.diag(B)[:, None]))
    root = np.abs(root)
    root[:, 0] = root[:, 0] - (root[:, 0] < FLOAT_ZERO) * (root[:, 0] + root[:, 1])
    root[:, 1] = root[:, 1] / root[:, 0]

    for i in range(n - 1, -1, -1):
        m = None
        for j in range(i, -1, -1):
            # No idea why -0.1 appears here; it comes from the original MATLAB code.
            if (root[j, 1] > stake) or (root[j, 1] < -0.1):  # noqa: PLR2004
                m = j
                break

        if m is None:
            return A, B, Q, Z

        for k in range(m, i):
            A[:], B[:], Q[:], Z[:] = qzswitch(k, A, B, Q, Z)
            root[k, 1], root[k + 1, 1] = root[k + 1, 1], root[k, 1]

    return A, B, Q, Z


@nb.njit(
    [
        "Tuple((f8, i8, b1))(f8[::1, :], f8[::1, :], optional(f8), f8)",
        "Tuple((f8, i8, b1))(c16[::1, :], c16[::1, :], optional(f8), f8)",
    ],
    cache=True,
)
def determine_n_unstable(A: np.ndarray, B: np.ndarray, div: float | None, realsmall: float) -> tuple[float, int, bool]:
    """
    Determine how many roots of the system described by A and B are unstable.

    Parameters
    ----------
    A : array
        Upper-triangular matrix, output of QZ decomposition.
    B : array
        Upper-triangular matrix, output of QZ decomposition.
    div : float, Optional
        Largest positive value for which an eigenvalue is considered stable. If None, a suitable value is calculated
        based on the input matrices.
    realsmall : float
        An arbitrarily small number.

    Returns
    -------
    tuple
        Contains three elements:
            div : float
                Represents which roots of the system can be considered stable.
            n_unstable : int
                The number of unstable roots in the system.
            zxz : bool
                Signals whether the system has a unique solution.

    Notes
    -----
    Originally part of gensys. Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m
    """
    n, _ = A.shape
    n_unstable = 0
    zxz = False

    realsmall = np.spacing(1) if realsmall is None else realsmall
    compute_div = div is None

    if div is None:
        div = 1.01

    for i in range(n):
        if compute_div and abs(A[i, i]) > 0:
            divhat = abs(B[i, i] / A[i, i])
            if 1 + realsmall < divhat <= div:
                div = 0.5 * (1 + divhat)
        n_unstable += abs(B[i, i]) > div * abs(A[i, i])

        zxz = (abs(A[i, i]) < realsmall) & (abs(B[i, i]) < realsmall)

    return div, n_unstable, zxz


@nb.njit(
    [
        "UniTuple(f8[::1, :], 2)(f8[::1, :],  i8)",
        "UniTuple(c16[::1, :], 2)(c16[::1, :], i8)",
    ],
    cache=True,
)
def split_matrix_on_eigen_stability(A: np.ndarray, n_unstable: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a matrix into stable and unstable parts based on the number of unstable roots.

    Parameters
    ----------
    A : np.ndarray
        Array to split.
    n_unstable : int
        Number of unstable roots in the system.

    Returns
    -------
    tuple of np.ndarray
        Contains two elements:
            A1 : np.ndarray
                Matrix containing all stable roots.
            A2 : np.ndarray
                Matrix containing all unstable roots.

    Notes
    -----
    Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m
    """
    n, _ = A.shape
    stable_slice = slice(None, n - n_unstable)
    unstable_slice = slice(n - n_unstable, None)

    A1 = np.asfortranarray(A[stable_slice])
    A2 = np.asfortranarray(A[unstable_slice])

    return A1, A2


# @nb.njit(['Tuple((f8[:,::1], f8[:,::1], f8[:,::1], i8[::1]))(f8[:,::1], f8)',
#           'Tuple((c16[:,::1], c16[:,::1], c16[:,::1], i8[::1]))(c16[:,::1], f8)'],
#          cache=True)
def build_u_v_d(
    eta: np.ndarray, realsmall: float = EPSILON, invalid_system: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the singular value decomposition (SVD) of the input matrix `eta` and identifies non-zero indices.

    Alternatively, if the system is invalid, returns zero matrices.

    Parameters
    ----------
    eta : np.ndarray
        Input matrix for which to compute the SVD.
    realsmall : float
        A small threshold value to determine non-zero singular values.
    invalid_system : int
        If True, return a zero solution. If False, compute the SVD normally.

    Returns
    -------
    tuple
        Contains two elements:
            (U, V, D) : tuple of np.ndarray
                SVD decomposition of `eta` where `U` and `V` are orthogonal matrices and `D` is a diagonal matrix.
            non_zero_indices : np.ndarray
                Array of non-zero indices based on the threshold `realsmall`.

    Notes
    -----
    Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m
    """
    # No stable roots
    if invalid_system:
        big_ev = np.zeros(
            0,
        )

        u_eta = np.zeros((0, 0))
        d_eta = np.zeros((0, 0))
        v_eta = np.zeros((eta.shape[-1], 0))

        return u_eta, v_eta, d_eta, big_ev

    u_eta, d_eta, vh_eta = linalg.svd(eta, compute_uv=True, full_matrices=False)
    v_eta = vh_eta.conj().T

    big_ev = np.flatnonzero(d_eta > realsmall)

    u_eta = u_eta[:, big_ev]
    v_eta = v_eta[:, big_ev]
    d_eta = np.diag(d_eta[big_ev])

    return u_eta, v_eta, d_eta, big_ev


# @nb.njit(cache=True)
def gensys(
    g0: np.ndarray,
    g1: np.ndarray,
    c: np.ndarray,
    psi: np.ndarray,
    pi: np.ndarray,
    div: float | None = None,
    tol: float | None = 1e-8,
    return_all_matrices: bool = True,
) -> tuple:
    r"""
    Christopher Sim's gensys.

    Solves rational expectations equations by partitioning the system into stable and unstable roots,
    then eliminating the unstable roots via QZ decomposition, as described in [1]_.

    System given as:

    .. math::
       :nowrap:

       \begin{aligned}
       G_0 \cdot y_t &= G_1 \cdot y_{t-1} + c \\
                     &\quad + \psi \cdot z_t + \pi \cdot \eta_t
       \end{aligned}

    with :math:`z_t` an exogenous variable process and :math:`\eta_t` being endogenously determined one-step-ahead
    expectational errors.

    Returned system is:

    .. math::
       :nowrap:

       \begin{aligned}
       y_t &= G_1 \cdot y_{t-1} + C + \text{impact} \cdot z_t \\
           &\quad + ywt \cdot (I - fmat \cdot L^{-1})^{-1} \cdot fwt \cdot z_{t+1}
       \end{aligned}

    If :math:`z_t` is i.i.d., the last term drops out. If `div` is omitted from argument list, a :math:`div > 1` is
    calculated.

    Parameters
    ----------
    g0 : np.ndarray
        Coefficient matrix of the dynamic system corresponding to the time-t variables.
    g1 : np.ndarray
        Coefficient matrix of the dynamic system corresponding to the time t-1 variables.
    c : np.ndarray
        Vector of constant terms.
    psi : np.ndarray
        Coefficient matrix of the dynamic system corresponding to the exogenous shock terms.
    pi : np.ndarray
        Coefficient matrix of the dynamic system corresponding to the endogenously determined
        expectational errors.
    div : float
        Threshold value for determining stable and unstable roots.
    tol : float, default: 1e-8
        Level of floating point precision.
    return_all_matrices: bool, default True
        Whether to return all matrices or just the policy function.

    Returns
    -------
    G1 : np.ndarray
        Policy function relating the current timestep to the next, transition matrix T in state space jargon.
    C : np.ndarray
        Array of system means, intercept vector c in state space jargon.
    impact : np.ndarray
        Policy function component relating exogenous shocks observed at the t to variable values in t+1, selection
        matrix R in state space jargon.
    fmat : np.ndarray
        Matrix used in the transformation of the system to handle unstable roots.
    fwt : np.ndarray
        Weight matrix corresponding to fmat.
    ywt : np.ndarray
        Weight matrix corresponding to the stable part of the system.
    gev : np.ndarray
        Generalized left and right eigenvalues generated by qz(g0, g1), sorted such that stable roots are in the
        top-left corner.
    eu : tuple
        Tuple of two values indicating existence and uniqueness of the solution, with the following meanings:

        - eu[0] = 1 for existence,
        - eu[1] = 1 for uniqueness.
        - eu[0] = -1 for existence only with not-s.c. z;
        - eu = [-2, -2] for coincident zeros.

    loose : int
        Number of loose endogenous variables.

    References
    ----------
    .. [1] Sims, Christopher A. "Solving linear rational expectations models."
       *Computational Economics* 20.1-2 (2002): 1-20.

    Notes
    -----
    Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m
    """
    eu = [0, 0, 0]

    n, _ = g1.shape
    A, B, Q, Z = linalg.qz(g0, g1, "complex")
    Q = np.asfortranarray(Q.conj().T)  # q is transposed relative to matlab, see scipy docs

    div, n_unstable, zxz = determine_n_unstable(A, B, div, tol)
    n_stable = n - n_unstable

    if zxz:
        eu = [-2, -2, 0]
        return None, None, None, None, None, None, None, eu, None

    A[:], B[:], Q[:], Z[:] = qzdiv(div, A, B, Q, Z)
    gev = np.column_stack((np.diagonal(A), np.diagonal(B)))

    Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)

    eta_wt = Q2 @ pi
    _, n_eta = pi.shape
    u_eta, v_eta, d_eta, big_ev = build_u_v_d(eta_wt, tol, n_unstable == 0)

    if len(big_ev) >= n_unstable:
        eu[0] = 1

    # All stable roots
    eta_wt_1 = np.zeros((0, n_eta)) if n_unstable == n else Q1 @ pi

    u_eta_1, v_eta_1, d_eta_1, big_ev = build_u_v_d(eta_wt_1, tol, n_unstable == n)

    if 0 in v_eta_1.shape:
        unique = True
    else:
        loose = v_eta_1 - v_eta @ v_eta.T @ v_eta_1
        [_ul, dl, _vl] = linalg.svd(loose)
        if dl.ndim == 1:
            dl = np.diag(dl)

        n_loose = (np.abs(np.diagonal(dl)) > (tol * n)).sum()
        eu[2] = n_loose
        unique = n_loose == 0

    if unique:
        eu[1] = 1

    inner_term = u_eta @ linalg.solve(d_eta, v_eta.conj().T) @ v_eta_1 @ d_eta_1 @ u_eta_1.conj().T

    T_mat = np.column_stack((np.eye(n_stable), -inner_term.conj().T))
    G_0 = np.vstack(
        (
            T_mat @ A,
            np.column_stack((np.zeros((n_unstable, n_stable)), np.eye(n_unstable))),
        )
    )

    G_1 = np.vstack((T_mat @ B, np.zeros((n_unstable, n))))

    G_0_inv = linalg.inv(G_0)
    G_1 = G_0_inv @ G_1
    G_1 = (Z @ G_1 @ Z.conj().T).real

    if not return_all_matrices:
        return G_1, eu

    idx = slice(n_stable, n)

    C = np.vstack((T_mat @ Q @ c, linalg.solve(A[idx, idx] - B[idx, idx], Q2) @ c))

    impact = G_0_inv @ np.vstack((T_mat @ Q @ psi, np.zeros((n_unstable, psi.shape[1]))))

    f_mat = linalg.solve(B[idx, idx], A[idx, idx])
    f_wt = -linalg.solve(B[idx, idx], Q2) @ psi
    y_wt = G_0_inv[:, idx]

    loose = G_0_inv @ np.vstack(
        (
            eta_wt_1 @ (np.eye(n_eta) - v_eta @ v_eta.conj().T),
            np.zeros((n_unstable, n_eta)),
        )
    )

    C = (Z @ C).real
    impact = (Z @ impact).real
    loose = (Z @ loose).real
    y_wt = Z @ y_wt

    return G_1, C, impact, f_mat, f_wt, y_wt, gev, eu, loose


def interpret_gensys_output(eu):
    """
    Interprets the output of the gensys function.

    Gensys returns integer success codes like we're FORTRAN programmers in 1980. This function converts these codes
    to human-readable messages that describe the existence and uniqueness of the solution.

    Parameters
    ----------
    eu : tuple of int
        A tuple of two integers returned by the gensys function:
            eu[0] : int
                Indicates existence of the solution.
            eu[1] : int
                Indicates uniqueness of the solution.

    Returns
    -------
    message : str
        A message describing the existence and uniqueness of the solution based on the values in `eu`.
    """
    NON_EXISTENCE_CODE = 0
    COINCIDENT_ZEROS_CODE = -2
    INDETERMINATE_CODE = -1
    EXISTENCE_CODE = 1

    message = f"Gensys return codes: {' '.join(map(str, eu))}, with the following meaning:\n"
    if eu[0] == COINCIDENT_ZEROS_CODE and eu[1] == COINCIDENT_ZEROS_CODE:
        message += "Coincident zeros.  Indeterminacy and/or nonexistence. Check that your system is correctly defined."
    elif eu[0] == INDETERMINATE_CODE:
        message += f"System is indeterminate. There are {eu[2]} loose endogenous variables."
    elif eu[1] == INDETERMINATE_CODE:
        message += "Solution exists, but it is not unique -- sunspots."
    elif eu[0] == NON_EXISTENCE_CODE and eu[1] == NON_EXISTENCE_CODE:
        message += "Solution does not exist."
    elif eu[0] == EXISTENCE_CODE and eu[1] == NON_EXISTENCE_CODE:
        message += "Solution exists, but is not unique."
    elif eu[0] == EXISTENCE_CODE and eu[1] == EXISTENCE_CODE:
        message += "Gensys found a unique solution."
    else:
        message += "Unknown return code. Check the gensys documentation."
    return message.strip()


@nb.njit(cache=True)
def _get_variable_counts(A, D):
    n_eq, n_vars = A.shape
    _, n_shocks = D.shape

    return n_eq, n_vars, n_shocks


@nb.njit(cache=True)
def _find_lead_variables(C, tol=1e-8):
    return np.where(np.sum(np.abs(C), axis=0) > tol)[0]


@nb.njit(cache=True)
def _gensys_setup(A, B, C, D, tol=1e-8):
    n_eq, n_vars, n_shocks = _get_variable_counts(A, D)

    lead_var_idx = _find_lead_variables(C, tol)
    eqs_and_leads_idx = np.concatenate((np.arange(n_vars), lead_var_idx + n_vars), axis=0)
    n_leads = len(lead_var_idx)

    Gamma_0 = np.vstack((np.hstack((B, C)), np.hstack((-np.eye(n_eq), np.zeros((n_eq, n_eq))))))

    Gamma_1 = np.vstack(
        (
            np.hstack((A, np.zeros((n_eq, n_eq)))),
            np.hstack((np.zeros((n_eq, n_eq)), np.eye(n_eq))),
        )
    )

    Pi = np.vstack((np.zeros((n_eq, n_eq)), np.eye(n_eq)))

    Psi = np.vstack((D, np.zeros((n_eq, n_shocks))))

    Gamma_0 = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
    Gamma_1 = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
    Psi = Psi[eqs_and_leads_idx, :]
    Pi = Pi[eqs_and_leads_idx, :][:, lead_var_idx]

    G0 = -Gamma_0
    C = np.asfortranarray(np.zeros(shape=(n_vars + n_leads, 1)))

    return G0, Gamma_1, C, Psi, Pi


def solve_policy_function_with_gensys(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    tol: float = 1e-8,
    reutrn_all_matrices: bool = True,
) -> tuple:
    g0, g1, c, psi, pi = _gensys_setup(A, B, C, D, tol)
    G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = gensys(g0, g1, c, psi, pi)

    if reutrn_all_matrices:
        return G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose

    return G_1, eu


class GensysWrapper(Op):
    def __init__(self, tol=1e-8):
        self.tol = tol
        super().__init__()

    def make_node(self, A, B, C, D) -> Apply:
        inputs = list(map(pt.as_tensor, [A, B, C, D]))
        n_variables = inputs[0].type.shape[0]

        outputs = [
            pt.tensor("T", shape=(n_variables, n_variables)),
            pt.scalar("success", dtype="bool"),
        ]

        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        A, B, C, D = inputs
        G_1, eu = solve_policy_function_with_gensys(A, B, C, D, tol=self.tol, reutrn_all_matrices=False)

        n_vars = A.shape[0]
        T = G_1[:n_vars, :n_vars]
        success = all(x == 1 for x in eu[:2])

        outputs[0][0] = np.asarray(T)
        outputs[1][0] = np.asarray(success)

    def L_op(self, inputs, outputs, output_grads):
        A, B, C, D = inputs
        T, _success = outputs
        T_bar, _success_bar = output_grads

        A_bar, B_bar, C_bar = o1_policy_function_adjoints(A, B, C, T, T_bar)
        D_bar = pt.zeros_like(D).astype(floatX)

        return [A_bar, B_bar, C_bar, D_bar]


def gensys_pt(A, B, C, D, tol=1e-8):
    T, success = GensysWrapper(tol=tol)(A, B, C, D)
    R = pt_compute_selection_matrix(B, C, D, T)

    return T, R, success
