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
    # Count total non-zeros across all blocks to pre-allocate
    total_nnz = 0
    for t in range(T):
        J = period_jacobians[t]
        total_nnz += np.count_nonzero(J[:, n_vars : 2 * n_vars])  # J_t (always present)
        if t > 0:
            total_nnz += np.count_nonzero(J[:, :n_vars])  # J_{t-1}
        if t < T - 1:
            total_nnz += np.count_nonzero(J[:, 2 * n_vars : 3 * n_vars])  # J_{t+1}

    rows = np.empty(total_nnz, dtype=np.intp)
    cols = np.empty(total_nnz, dtype=np.intp)
    data = np.empty(total_nnz)
    pos = 0

    for t in range(T):
        J_period = period_jacobians[t]
        row_offset = t * n_eq

        J_tm1 = J_period[:, :n_vars]
        J_t = J_period[:, n_vars : 2 * n_vars]
        J_tp1 = J_period[:, 2 * n_vars : 3 * n_vars]

        if t > 0:
            col_offset = (t - 1) * n_vars
            r, c = np.nonzero(J_tm1)
            n = len(r)
            rows[pos : pos + n] = row_offset + r
            cols[pos : pos + n] = col_offset + c
            data[pos : pos + n] = J_tm1[r, c]
            pos += n

        col_offset = t * n_vars
        r, c = np.nonzero(J_t)
        n = len(r)
        rows[pos : pos + n] = row_offset + r
        cols[pos : pos + n] = col_offset + c
        data[pos : pos + n] = J_t[r, c]
        pos += n

        if t < T - 1:
            col_offset = (t + 1) * n_vars
            r, c = np.nonzero(J_tp1)
            n = len(r)
            rows[pos : pos + n] = row_offset + r
            cols[pos : pos + n] = col_offset + c
            data[pos : pos + n] = J_tp1[r, c]
            pos += n

    return sparse.csc_matrix((data, (rows, cols)), shape=(T * n_eq, T * n_vars))
