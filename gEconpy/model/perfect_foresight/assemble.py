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
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for t in range(T):
        J_period = period_jacobians[t]
        row_offset = t * n_eq

        J_tm1 = J_period[:, :n_vars]
        J_t = J_period[:, n_vars : 2 * n_vars]
        J_tp1 = J_period[:, 2 * n_vars : 3 * n_vars]

        # J^{-1} block (skip at t=0, initial condition is fixed)
        if t > 0:
            col_offset = (t - 1) * n_vars
            r, c = np.nonzero(J_tm1)
            rows.extend(row_offset + r)
            cols.extend(col_offset + c)
            data.extend(J_tm1[r, c])

        # J^{0} block
        col_offset = t * n_vars
        r, c = np.nonzero(J_t)
        rows.extend(row_offset + r)
        cols.extend(col_offset + c)
        data.extend(J_t[r, c])

        # J^{+1} block (skip at t=T-1, terminal condition is fixed)
        if t < T - 1:
            col_offset = (t + 1) * n_vars
            r, c = np.nonzero(J_tp1)
            rows.extend(row_offset + r)
            cols.extend(col_offset + c)
            data.extend(J_tp1[r, c])

    return sparse.csc_matrix((data, (rows, cols)), shape=(T * n_eq, T * n_vars))
