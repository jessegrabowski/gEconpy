from collections.abc import Sequence

import numpy as np

from scipy import sparse


def assemble_stacked_jacobian(
    period_jacobians: Sequence[np.ndarray],
    sparsity_pattern: np.ndarray,
    n_vars: int,
    n_eq: int,
    T: int,
) -> sparse.csc_matrix:
    """
    Assemble the block-tridiagonal Jacobian of the stacked system from one dense Jacobian per period.

    Each period Jacobian has columns ordered ``[y_{t-1}, y_t, y_{t+1}]``. The ``y_{t-1}`` block of the first period
    and the ``y_{t+1}`` block of the last period multiply the fixed boundary conditions and are dropped.

    Parameters
    ----------
    period_jacobians : sequence of ndarray
        T dense matrices, each of shape ``(n_eq, 3 * n_vars)``.
    sparsity_pattern : ndarray of bool
        Structural nonzero mask of one period's Jacobian, of shape ``(n_eq, 3 * n_vars)``, shared by every
        period. A structurally nonzero entry that is numerically zero at this iterate keeps an explicit slot.
    n_vars : int
        Number of variables per period.
    n_eq : int
        Number of equations per period.
    T : int
        Number of periods.

    Returns
    -------
    jacobian : sparse.csc_matrix
        Block-tridiagonal matrix of shape ``(T * n_eq, T * n_vars)``.

    Raises
    ------
    ValueError
        If ``sparsity_pattern`` does not have shape ``(n_eq, 3 * n_vars)``.
    """
    # The caller supplies the pattern because reading it off ``period_jacobians`` cannot be made safe: an entry
    # can pass through zero in every period at once (a steady-state initial guess, a parameter path crossing
    # zero), and a pattern inferred from those values drops it, leaving Newton to step against a Jacobian that
    # is wrong rather than approximate.
    if sparsity_pattern.shape != (n_eq, 3 * n_vars):
        raise ValueError(f"sparsity_pattern has shape {sparsity_pattern.shape}, expected {(n_eq, 3 * n_vars)}.")

    rows_tm1, cols_tm1 = np.nonzero(sparsity_pattern[:, :n_vars])
    rows_t, cols_t = np.nonzero(sparsity_pattern[:, n_vars : 2 * n_vars])
    rows_tp1, cols_tp1 = np.nonzero(sparsity_pattern[:, 2 * n_vars : 3 * n_vars])

    nnz_tm1 = len(rows_tm1)
    nnz_t = len(rows_t)
    nnz_tp1 = len(rows_tp1)
    total_nnz = T * nnz_t + (T - 1) * nnz_tm1 + (T - 1) * nnz_tp1

    rows = np.empty(total_nnz, dtype=np.intp)
    cols = np.empty(total_nnz, dtype=np.intp)
    data = np.empty(total_nnz)
    pos = 0

    for t in range(T):
        period_jacobian = period_jacobians[t]
        row_offset = t * n_eq

        if t > 0:
            col_offset = (t - 1) * n_vars
            rows[pos : pos + nnz_tm1] = row_offset + rows_tm1
            cols[pos : pos + nnz_tm1] = col_offset + cols_tm1
            data[pos : pos + nnz_tm1] = period_jacobian[rows_tm1, cols_tm1]
            pos += nnz_tm1

        col_offset = t * n_vars
        rows[pos : pos + nnz_t] = row_offset + rows_t
        cols[pos : pos + nnz_t] = col_offset + cols_t
        data[pos : pos + nnz_t] = period_jacobian[rows_t, n_vars + cols_t]
        pos += nnz_t

        if t < T - 1:
            col_offset = (t + 1) * n_vars
            rows[pos : pos + nnz_tp1] = row_offset + rows_tp1
            cols[pos : pos + nnz_tp1] = col_offset + cols_tp1
            data[pos : pos + nnz_tp1] = period_jacobian[rows_tp1, 2 * n_vars + cols_tp1]
            pos += nnz_tp1

    return sparse.csc_matrix((data, (rows, cols)), shape=(T * n_eq, T * n_vars))
