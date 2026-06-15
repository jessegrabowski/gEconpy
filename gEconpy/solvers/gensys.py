import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key

# `_lu_solve` can't be njit-dispatched (stub/impl arity mismatch); use `_getrs` directly.
from pytensor.link.numba.dispatch.linalg.decomposition.lu_factor import _lu_factor
from pytensor.link.numba.dispatch.linalg.decomposition.qz import _qz_complex_sort_eig
from pytensor.link.numba.dispatch.linalg.decomposition.svd import _svd_gesdd_full, _svd_gesdd_no_uv
from pytensor.link.numba.dispatch.linalg.solvers.lu_solve import _getrs
from pytensor.link.numba.dispatch.linalg.solvers.triangular import _solve_triangular

from gEconpy.solvers.shared import (
    o1_policy_function_adjoints,
    pt_compute_selection_matrix,
)

EPSILON = np.spacing(1)
floatX = pytensor.config.floatX


@numba_basic.numba_njit(final_function=True)
def _determine_n_unstable_core(
    alpha: np.ndarray,
    beta: np.ndarray,
    div: float,
    compute_div: bool,
    realsmall: float,
) -> tuple[float, int, bool]:
    """Numba-compiled heart of :func:`determine_n_unstable`. See its docstring."""
    n_unstable = 0
    zxz = False

    for i in range(alpha.size):
        abs_a = np.abs(alpha[i])
        abs_b = np.abs(beta[i])
        if compute_div and abs_a > 0:
            divhat = abs_b / abs_a
            if 1 + realsmall < divhat <= div:
                div = 0.5 * (1 + divhat)
        if abs_b > div * abs_a:
            n_unstable += 1
        zxz = (abs_a < realsmall) and (abs_b < realsmall)

    return div, n_unstable, zxz


def determine_n_unstable(
    alpha: np.ndarray,
    beta: np.ndarray,
    div: float | None,
    realsmall: float,
) -> tuple[float, int, bool]:
    """Classify generalized eigenvalues `beta / alpha` as stable or unstable.

    Parameters
    ----------
    alpha : ndarray of complex
        Left-side generalized eigenvalue components (diagonal of the triangular
        `A` factor from the QZ decomposition).
    beta : ndarray of complex
        Right-side generalized eigenvalue components (diagonal of the triangular
        `B` factor from the QZ decomposition).
    div : float, optional
        Cutoff above which an eigenvalue is considered unstable. When `None`,
        the cutoff is inferred from the spectrum using Sims's heuristic: shrink
        toward 1 whenever an eigenvalue lies just outside the unit circle, so
        borderline roots are grouped consistently.
    realsmall : float
        Tolerance for detecting coincident near-zero diagonal entries.

    Returns
    -------
    div : float
        The (possibly adjusted) stability cutoff.
    n_unstable : int
        Count of eigenvalues classified as unstable.
    zxz : bool
        True if the final pair of diagonal entries is coincident near-zero,
        signalling a non-unique / non-existent solution.

    Notes
    -----
    Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m.
    This is a thin Python shim over the numba-compiled
    :func:`_determine_n_unstable_core`, which can't accept `None` for `div`.
    """
    compute_div = div is None
    div_eff = 1.01 if compute_div else float(div)
    div_out, n_unstable, zxz = _determine_n_unstable_core(alpha, beta, div_eff, compute_div, realsmall)
    return float(div_out), int(n_unstable), bool(zxz)


@numba_basic.numba_njit(final_function=True)
def split_matrix_on_eigen_stability(A: np.ndarray, n_unstable: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a matrix row-wise into stable / unstable blocks.

    Parameters
    ----------
    A : ndarray
        Matrix whose rows are aligned with the sorted QZ eigenvalue ordering
        (stable first, unstable last).
    n_unstable : int
        Number of trailing rows corresponding to unstable eigenvalues.

    Returns
    -------
    A1 : ndarray
        Rows corresponding to stable eigenvalues.
    A2 : ndarray
        Rows corresponding to unstable eigenvalues.
    """
    n = A.shape[0]
    return A[: n - n_unstable], A[n - n_unstable :]


@numba_basic.numba_njit(final_function=True)
def _thin_svd_and_rank(eta: np.ndarray, realsmall: float):
    u, s, vh = _svd_gesdd_full(eta, full_matrices=False)
    keep = s > realsmall
    return u, s, vh, keep


def build_u_v_d(
    eta: np.ndarray, realsmall: float = EPSILON, invalid_system: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a thin SVD of `eta` and keep the non-negligible components.

    Parameters
    ----------
    eta : ndarray
        Input matrix to decompose.
    realsmall : float, optional
        Threshold below which singular values are treated as zero. Default
        :data:`EPSILON`.
    invalid_system : bool, optional
        If True, return zero-sized outputs so downstream code can short-circuit
        an ill-posed system. Default False.

    Returns
    -------
    u_eta : ndarray
        Left singular vectors for the retained components.
    v_eta : ndarray
        Right singular vectors for the retained components.
    d_eta : ndarray of float
        Retained singular values as a 1-D array.
    big_ev : ndarray of int
        Indices of the retained singular values in the full spectrum.

    Notes
    -----
    Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m
    """
    if invalid_system or eta.size == 0:
        dtype = eta.dtype
        u_eta = np.zeros((eta.shape[0], 0), dtype=dtype)
        d_eta = np.zeros(0, dtype=np.float64)
        v_eta = np.zeros((eta.shape[-1], 0), dtype=dtype)
        big_ev = np.zeros(0, dtype=np.int64)
        return u_eta, v_eta, d_eta, big_ev

    u, s, vh, keep = _thin_svd_and_rank(eta, realsmall)
    big_ev = np.flatnonzero(keep)
    u_eta = u[:, big_ev]
    v_eta = vh.conj().T[:, big_ev]
    d_eta = s[big_ev]
    return u_eta, v_eta, d_eta, big_ev


@numba_basic.numba_njit(final_function=True)
def _matrix_rank(A: np.ndarray, tol: float) -> int:
    """Numba-friendly rank-via-SVD. Assumes ``A`` has at least one element."""
    if A.shape[0] == 0 or A.shape[1] == 0:
        return 0
    # `_svd_gesdd_no_uv` returns NaN singular values on a LAPACK failure instead
    # of raising; NaN > tol is False, so a failed decomposition reads as rank 0.
    s = _svd_gesdd_no_uv(A)
    n_nonzero = 0
    for i in range(s.size):
        if s[i] > tol:
            n_nonzero += 1
    return n_nonzero


@numba_basic.numba_njit(final_function=True)
def _gensys_core(
    g0: np.ndarray,
    g1: np.ndarray,
    c: np.ndarray,
    psi: np.ndarray,
    pi: np.ndarray,
    tol: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Numba-compiled gensys core.

    Always returns the full 9-tuple. `eu` is a length-3 int64 ndarray. A `zxz`
    failure is signalled by `eu = [-2, -2, 0]`; the accompanying matrices are
    zero-filled so the output types stay consistent.

    Uses the scipy ``"ouc"`` sort convention (stable = strictly inside unit
    circle, threshold 1.0), dropping Sims's dynamic-div heuristic. The heuristic
    only matters for eigenvalues in (1.0, 1.01] — empirically absent from
    well-posed DSGE models.
    """
    n = g1.shape[0]
    n_eta = pi.shape[1]
    n_shocks = psi.shape[1]
    realsmall = tol if tol > 0 else np.spacing(1)

    # QZ works on complex; cast once at the boundary. These casts are fresh
    # buffers used only by the QZ, so we let gges overwrite them in place.
    g0_c = g0.astype(np.complex128)
    g1_c = g1.astype(np.complex128)

    # Sorted QZ: stable eigenvalues (|alpha/beta| > 1 in scipy convention =
    # |beta/alpha| < 1 in Sims convention) occupy the top-left.
    A, B, alpha, beta, Q_raw, Z = _qz_complex_sort_eig(g0_c, g1_c, "ouc", True, True)
    # Sims convention: g0 = Q^H A Z^H.
    Q = Q_raw.conj().T
    Z_H = Z.conj().T

    # Count n_unstable and detect coincident-zero diagonal pairs (zxz).
    n_unstable = 0
    zxz = False
    for i in range(n):
        abs_a = np.abs(alpha[i])
        abs_b = np.abs(beta[i])
        if abs_a < realsmall and abs_b < realsmall:
            zxz = True
        # "ouc" stable criterion: (alpha != 0 and beta == 0) or |alpha| > |beta|.
        is_stable = ((abs_b < realsmall) and (abs_a >= realsmall)) or ((abs_b >= realsmall) and (abs_a > abs_b))
        if not is_stable:
            n_unstable += 1

    n_stable = n - n_unstable

    eu = np.zeros(3, dtype=np.int64)
    gev = np.column_stack((alpha, beta))

    if zxz:
        eu[0] = -2
        eu[1] = -2
        G_1 = np.zeros((n, n), dtype=np.float64)
        C_out = np.zeros((n, c.shape[1]), dtype=np.float64)
        impact = np.zeros((n, n_shocks), dtype=np.float64)
        f_mat_out = np.zeros((n_unstable, n_unstable), dtype=np.complex128)
        f_wt_out = np.zeros((n_unstable, n_shocks), dtype=np.complex128)
        y_wt_out = np.zeros((n, n_unstable), dtype=np.complex128)
        loose_out = np.zeros((n, n_eta), dtype=np.float64)
        return G_1, C_out, impact, f_mat_out, f_wt_out, y_wt_out, gev, eu, loose_out

    Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)

    # eta partition + SVDs used to back out the expectational-error selection.
    eta_wt = Q2 @ pi.astype(np.complex128)
    if n_unstable == 0:
        u_eta = np.zeros((0, 0), dtype=np.complex128)
        d_eta = np.zeros(0, dtype=np.float64)
        v_eta = np.zeros((n_eta, 0), dtype=np.complex128)
    else:
        u_raw, s_raw, vh_raw, keep = _thin_svd_and_rank(eta_wt, realsmall)
        big_ev = np.flatnonzero(keep)
        u_eta = u_raw[:, big_ev]
        d_eta = s_raw[big_ev]
        v_eta = vh_raw.conj().T[:, big_ev]

    if d_eta.size >= n_unstable:
        eu[0] = 1

    if n_unstable == n:
        eta_wt_1 = np.zeros((0, n_eta), dtype=np.complex128)
        u_eta_1 = np.zeros((0, 0), dtype=np.complex128)
        d_eta_1 = np.zeros(0, dtype=np.float64)
        v_eta_1 = np.zeros((n_eta, 0), dtype=np.complex128)
    else:
        eta_wt_1 = Q1 @ pi.astype(np.complex128)
        u1_raw, s1_raw, vh1_raw, keep1 = _thin_svd_and_rank(eta_wt_1, realsmall)
        big_ev_1 = np.flatnonzero(keep1)
        u_eta_1 = u1_raw[:, big_ev_1]
        d_eta_1 = s1_raw[big_ev_1]
        v_eta_1 = vh1_raw.conj().T[:, big_ev_1]

    v_eta_H = v_eta.conj().T
    u_eta_1_H = u_eta_1.conj().T

    if v_eta_1.shape[0] == 0 or v_eta_1.shape[1] == 0:
        unique = True
    else:
        loose_for_rank = v_eta_1 - v_eta @ v_eta_H @ v_eta_1
        n_loose = _matrix_rank(loose_for_rank, realsmall * n)
        eu[2] = n_loose
        unique = n_loose == 0

    if unique:
        eu[1] = 1

    # inner_term = U_eta D_eta^{-1} V_eta^H V_eta_1 D_eta_1 U_eta_1^H, with the
    # two diagonal solves collapsed to element-wise scaling.
    d_eta_c = d_eta.astype(np.complex128)
    d_eta_1_c = d_eta_1.astype(np.complex128)
    scaled_vh = (v_eta_H / d_eta_c.reshape(-1, 1)) if d_eta.size else v_eta_H
    scaled_u1h = (d_eta_1_c.reshape(-1, 1) * u_eta_1_H) if d_eta_1.size else u_eta_1_H

    inner_term = u_eta @ scaled_vh @ v_eta_1 @ scaled_u1h
    inner_term_H = inner_term.conj().T

    T_mat = np.column_stack((np.eye(n_stable, dtype=np.complex128), -inner_term_H))
    top_block = T_mat @ A
    bottom_block = np.column_stack(
        (
            np.zeros((n_unstable, n_stable), dtype=np.complex128),
            np.eye(n_unstable, dtype=np.complex128),
        )
    )
    G_0 = np.vstack((top_block, bottom_block))

    # One LU of G_0, reused for every solve. _getrs wants 1-based pivots.
    lu, piv = _lu_factor(G_0.T, True)
    piv += np.int32(1)

    G_1_rhs = np.vstack(
        (
            T_mat @ B,
            np.zeros((n_unstable, n), dtype=np.complex128),
        )
    )
    G_1_c, _info = _getrs(lu, G_1_rhs, piv, 1, True)
    G_1 = (Z @ G_1_c @ Z_H).real

    A_idx = A[n_stable:n, n_stable:n]
    B_idx = B[n_stable:n, n_stable:n]
    Q2c = Q2  # already complex
    c_c = c.astype(np.complex128)
    psi_c = psi.astype(np.complex128)
    Tmat_Q = T_mat @ Q

    # A_idx and B_idx are upper-triangular blocks of the sorted QZ output.
    if n_unstable == 0:
        C_tail = np.zeros((0, c.shape[1]), dtype=np.complex128)
    else:
        C_tail = _solve_triangular(A_idx - B_idx, Q2c @ c_c, 0, False, False, True)
    C_complex = np.vstack((Tmat_Q @ c_c, C_tail))

    impact_rhs = np.vstack(
        (
            Tmat_Q @ psi_c,
            np.zeros((n_unstable, n_shocks), dtype=np.complex128),
        )
    )
    impact_complex, _info = _getrs(lu, impact_rhs, piv, 1, True)

    if n_unstable == 0:
        f_mat = np.zeros((0, 0), dtype=np.complex128)
        f_wt = np.zeros((0, n_shocks), dtype=np.complex128)
    else:
        # `A_idx` is a view into `A`; keep overwrite_b=False so the solve doesn't
        # clobber A's buffer. `Q2c @ psi_c` is a fresh temporary, so it can be.
        f_mat = _solve_triangular(B_idx, A_idx, 0, False, False, False)
        f_wt = -_solve_triangular(B_idx, Q2c @ psi_c, 0, False, False, True)

    # y_wt = G_0^{-1}[:, n_stable:n]  <=>  solve G_0 @ y_wt = I[:, n_stable:n].
    eye_cols = np.zeros((n, n_unstable), dtype=np.complex128)
    for j in range(n_unstable):
        eye_cols[n_stable + j, j] = 1.0 + 0.0j
    y_wt_complex, _info = _getrs(lu, eye_cols, piv, 1, True)
    y_wt = Z @ y_wt_complex

    loose_rhs = np.vstack(
        (
            eta_wt_1 @ (np.eye(n_eta, dtype=np.complex128) - v_eta @ v_eta_H),
            np.zeros((n_unstable, n_eta), dtype=np.complex128),
        )
    )
    loose_complex, _info = _getrs(lu, loose_rhs, piv, 1, True)

    C_out = (Z @ C_complex).real
    impact = (Z @ impact_complex).real
    loose_out = (Z @ loose_complex).real

    return G_1, C_out, impact, f_mat, f_wt, y_wt, gev, eu, loose_out


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
    div : float, optional
        Accepted for backward compatibility but ignored. The njit core uses the
        scipy `"ouc"` sort (strict unit circle, threshold 1.0). Sims's dynamic
        shrink-toward-1 heuristic is only needed for eigenvalues in (1.0, 1.01]
        and is empirically absent from well-posed DSGE models.
    tol : float, optional
        Level of floating point precision. Default 1e-8.
    return_all_matrices : bool, optional
        Whether to return all matrices or just the policy function. Default True.

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
    eu : list of int
        Three-element list indicating existence and uniqueness of the solution, with the following meanings:

        - eu[0] = 1 for existence,
        - eu[1] = 1 for uniqueness.
        - eu[0] = -1 for existence only with not-s.c. z;
        - eu = [-2, -2] for coincident zeros.

    loose : np.ndarray
        Matrix characterising the sunspot-indeterminacy directions.

    References
    ----------
    .. [1] Sims, Christopher A. "Solving linear rational expectations models."
       *Computational Economics* 20.1-2 (2002): 1-20.

    Notes
    -----
    Adapted from http://sims.princeton.edu/yftp/gensys/mfiles/gensys.m. The core
    is numba-njit compiled and uses pytensor's numba-compatible LAPACK wrappers.
    """
    del div  # accepted for backward compat, ignored; see docstring.

    tol_eff = tol if tol is not None and tol > 0 else np.spacing(1)
    g0_f = np.ascontiguousarray(g0, dtype=np.float64)
    g1_f = np.ascontiguousarray(g1, dtype=np.float64)
    c_f = np.ascontiguousarray(c, dtype=np.float64)
    psi_f = np.ascontiguousarray(psi, dtype=np.float64)
    pi_f = np.ascontiguousarray(pi, dtype=np.float64)

    G_1, C_out, impact, f_mat, f_wt, y_wt, gev, eu_arr, loose_out = _gensys_core(g0_f, g1_f, c_f, psi_f, pi_f, tol_eff)

    eu = [int(x) for x in eu_arr]

    if eu[0] == -2 and eu[1] == -2:
        return None, None, None, None, None, None, None, eu, None

    if not return_all_matrices:
        return G_1, eu

    return G_1, C_out, impact, f_mat, f_wt, y_wt, gev, eu, loose_out


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


@numba_basic.numba_njit(final_function=True)
def _gensys_setup(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the `(g0, g1, c, psi, pi)` quintuple expected by `gensys`."""
    n_eq = A.shape[0]
    n_shocks = D.shape[1]

    # Indices of "lead" columns — those C-columns with any non-negligible entry.
    col_abs_sum = np.zeros(C.shape[1], dtype=np.float64)
    for j in range(C.shape[1]):
        total = 0.0
        for i in range(C.shape[0]):
            total += abs(C[i, j])
        col_abs_sum[j] = total
    lead_var_idx = np.flatnonzero(col_abs_sum > tol)
    n_leads = lead_var_idx.size
    eqs_and_leads_idx = np.concatenate((np.arange(n_eq), lead_var_idx + n_eq))

    Gamma_0 = np.vstack(
        (
            np.hstack((B, C)),
            np.hstack((-np.eye(n_eq), np.zeros((n_eq, n_eq)))),
        )
    )
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
    const = np.zeros((n_eq + n_leads, 1))

    return G0, Gamma_1, const, Psi, Pi


def solve_policy_function_with_gensys(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    tol: float = 1e-8,
    return_all_matrices: bool = True,
) -> tuple:
    A_f = np.ascontiguousarray(A, dtype=np.float64)
    B_f = np.ascontiguousarray(B, dtype=np.float64)
    C_f = np.ascontiguousarray(C, dtype=np.float64)
    D_f = np.ascontiguousarray(D, dtype=np.float64)

    g0, g1, c, psi, pi = _gensys_setup(A_f, B_f, C_f, D_f, tol)
    return gensys(g0, g1, c, psi, pi, tol=tol, return_all_matrices=return_all_matrices)


class GensysWrapper(Op):
    __props__ = ("tol",)
    gufunc_signature = "(n,n),(n,n),(n,n),(n,k)->(n,n),()"

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

    def infer_shape(self, fgraph, node, input_shapes):
        n = input_shapes[0][0]
        return [(n, n), ()]

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        A, B, C, D = inputs
        G_1, eu = solve_policy_function_with_gensys(A, B, C, D, tol=self.tol, return_all_matrices=False)

        n_vars = A.shape[0]
        T = G_1[:n_vars, :n_vars]
        success = all(x == 1 for x in eu[:2])

        outputs[0][0] = np.asarray(T)
        outputs[1][0] = np.asarray(success)

    def pullback(self, inputs, outputs, cotangents):
        A, B, C, D = inputs
        T, _success = outputs
        T_bar, _success_bar = cotangents

        A_bar, B_bar, C_bar = o1_policy_function_adjoints(A, B, C, T, T_bar)
        D_bar = pt.zeros_like(D).astype(floatX)

        return [A_bar, B_bar, C_bar, D_bar]


def gensys_pt(A, B, C, D, tol=1e-8):
    T, success = GensysWrapper(tol=tol)(A, B, C, D)
    R = pt_compute_selection_matrix(B, C, D, T)

    return T, R, success


@register_funcify_default_op_cache_key(GensysWrapper)
def numba_funcify_GensysWrapper(op, node, **kwargs):  # noqa: ARG001
    """Route to the njit compute chain, avoiding an object-mode fallback on perform."""
    tol = op.tol
    # The kernel runs in float64; only upcast inputs that aren't already.
    if node.inputs[0].type.dtype != "float64":

        @numba_basic.numba_njit
        def _prep(X):
            return np.ascontiguousarray(X).astype(np.float64)
    else:

        @numba_basic.numba_njit
        def _prep(X):
            return np.ascontiguousarray(X)

    @numba_basic.numba_njit
    def gensys_wrapper(A, B, C, D):
        n_vars = A.shape[0]
        g0, g1, c, psi, pi = _gensys_setup(_prep(A), _prep(B), _prep(C), _prep(D), tol)
        G_1, _C_out, _impact, _f_mat, _f_wt, _y_wt, _gev, eu, _loose = _gensys_core(g0, g1, c, psi, pi, tol)

        T = np.ascontiguousarray(G_1[:n_vars, :n_vars])
        success = (eu[0] == 1) and (eu[1] == 1)
        return T, success

    cache_version = 2
    return gensys_wrapper, cache_version
