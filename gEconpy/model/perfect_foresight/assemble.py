"""Jacobian assembly for perfect foresight stacked system."""

import numpy as np

from scipy import sparse


def assemble_stacked_jacobian(
    period_jacobians: list[np.ndarray],
    n_vars: int,
    n_eq: int,
    T: int,
) -> sparse.csc_matrix:
    """Assemble block-tridiagonal Jacobian from period-wise Jacobians.

    Period Jacobian columns are ordered [y_{t-1}, y_t, y_{t+1}].

    Parameters
    ----------
    period_jacobians : list of ndarray
        List of T dense Jacobian matrices, each of shape (n_eq, 3*n_vars).
    n_vars : int
        Number of variables per period.
    n_eq : int
        Number of equations per period.
    T : int
        Number of time periods.

    Returns
    -------
    sparse.csc_matrix
        Block-tridiagonal sparse Jacobian of shape (T*n_eq, T*n_vars).
    """
    # Compute nonzero masks once from the first period (sparsity pattern is
    # identical across all periods for a given model).
    J0 = period_jacobians[0]
    r_tm1, c_tm1 = np.nonzero(J0[:, :n_vars])
    r_t, c_t = np.nonzero(J0[:, n_vars : 2 * n_vars])
    r_tp1, c_tp1 = np.nonzero(J0[:, 2 * n_vars : 3 * n_vars])

    nnz_tm1 = len(r_tm1)
    nnz_t = len(r_t)
    nnz_tp1 = len(r_tp1)

    # Interior periods contribute all three blocks; first/last drop one
    total_nnz = T * nnz_t + (T - 1) * nnz_tm1 + (T - 1) * nnz_tp1

    rows = np.empty(total_nnz, dtype=np.intp)
    cols = np.empty(total_nnz, dtype=np.intp)
    data = np.empty(total_nnz)
    pos = 0

    for t in range(T):
        J_period = period_jacobians[t]
        row_offset = t * n_eq

        if t > 0:
            col_offset = (t - 1) * n_vars
            rows[pos : pos + nnz_tm1] = row_offset + r_tm1
            cols[pos : pos + nnz_tm1] = col_offset + c_tm1
            data[pos : pos + nnz_tm1] = J_period[r_tm1, c_tm1]
            pos += nnz_tm1

        col_offset = t * n_vars
        rows[pos : pos + nnz_t] = row_offset + r_t
        cols[pos : pos + nnz_t] = col_offset + c_t
        data[pos : pos + nnz_t] = J_period[r_t, n_vars + c_t]
        pos += nnz_t

        if t < T - 1:
            col_offset = (t + 1) * n_vars
            rows[pos : pos + nnz_tp1] = row_offset + r_tp1
            cols[pos : pos + nnz_tp1] = col_offset + c_tp1
            data[pos : pos + nnz_tp1] = J_period[r_tp1, 2 * n_vars + c_tp1]
            pos += nnz_tp1

    return sparse.csc_matrix((data, (rows, cols)), shape=(T * n_eq, T * n_vars))
