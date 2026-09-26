from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scipy import sparse


@dataclass(frozen=True, eq=False)
class StackedJacobianLayout:
    """
    Where every entry of the stacked Jacobian comes from, fixed by the sparsity pattern and the horizon.

    Attributes
    ----------
    source_rows, source_cols : ndarray of int
        Row and column of the period Jacobian each entry reads, in fill order.
    period_bounds : ndarray of int
        Start of each period's entries in fill order, with the total in the last position.
    permutation : ndarray of int
        Reorders the filled entries into compressed-column order.
    indices, indptr : ndarray of int
        Row indices and column pointers of the compressed-column result.
    shape : tuple of int
        Shape of the stacked Jacobian.
    """

    source_rows: np.ndarray
    source_cols: np.ndarray
    period_bounds: np.ndarray
    permutation: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]


def build_stacked_jacobian_layout(
    sparsity_pattern: np.ndarray,
    n_vars: int,
    n_eq: int,
    T: int,
) -> StackedJacobianLayout:
    """
    Resolve the block-tridiagonal structure of the stacked system once, for reuse across Newton iterations.

    Each period Jacobian has columns ordered ``[y_{t-1}, y_t, y_{t+1}]``. The ``y_{t-1}`` block of the first period
    and the ``y_{t+1}`` block of the last period multiply the fixed boundary conditions and are dropped.

    Parameters
    ----------
    sparsity_pattern : ndarray of bool
        Structural nonzero mask of one period's Jacobian, of shape ``(n_eq, 3 * n_vars)``, shared by every
        period. A structurally nonzero entry that is numerically zero at a given iterate keeps an explicit slot.
    n_vars : int
        Number of variables per period.
    n_eq : int
        Number of equations per period.
    T : int
        Number of periods.

    Returns
    -------
    layout : StackedJacobianLayout
        Index arrays for :func:`assemble_stacked_jacobian`.

    Raises
    ------
    ValueError
        If ``sparsity_pattern`` does not have shape ``(n_eq, 3 * n_vars)``.
    """
    # The pattern is supplied rather than read off the period Jacobians because an entry can pass through zero in
    # every period at once (a steady-state initial guess, a parameter path crossing zero). A pattern inferred from
    # those values drops it, leaving Newton to step against a Jacobian that is wrong rather than approximate.
    if sparsity_pattern.shape != (n_eq, 3 * n_vars):
        raise ValueError(f"sparsity_pattern has shape {sparsity_pattern.shape}, expected {(n_eq, 3 * n_vars)}.")

    blocks = [np.nonzero(sparsity_pattern[:, start : start + n_vars]) for start in (0, n_vars, 2 * n_vars)]
    placements = ((-1, 0), (0, n_vars), (1, 2 * n_vars))

    source_rows, source_cols, rows, cols, bounds = [], [], [], [], [0]
    filled = 0
    for t in range(T):
        for (block_rows, block_cols), (shift, column_offset) in zip(blocks, placements, strict=True):
            if not 0 <= t + shift < T:
                continue
            source_rows.append(block_rows)
            source_cols.append(block_cols + column_offset)
            rows.append(t * n_eq + block_rows)
            cols.append((t + shift) * n_vars + block_cols)
            filled += block_rows.size
        bounds.append(filled)

    shape = (T * n_eq, T * n_vars)

    # Carrying the fill position as the data of a throwaway matrix turns the conversion into the permutation that
    # reorders filled entries into compressed-column order, so no later call has to convert from coordinate form.
    positions = sparse.coo_matrix(
        (np.arange(filled), (np.concatenate(rows), np.concatenate(cols))), shape=shape, dtype=np.int64
    ).tocsc()

    return StackedJacobianLayout(
        source_rows=np.concatenate(source_rows),
        source_cols=np.concatenate(source_cols),
        period_bounds=np.array(bounds, dtype=np.intp),
        permutation=positions.data,
        indices=positions.indices,
        indptr=positions.indptr,
        shape=shape,
    )


def assemble_stacked_jacobian(
    period_jacobians: Sequence[np.ndarray],
    layout: StackedJacobianLayout,
) -> sparse.csc_matrix:
    """
    Assemble the block-tridiagonal Jacobian of the stacked system from one dense Jacobian per period.

    Parameters
    ----------
    period_jacobians : sequence of ndarray
        T dense matrices, each of shape ``(n_eq, 3 * n_vars)``.
    layout : StackedJacobianLayout
        Index arrays from :func:`build_stacked_jacobian_layout`.

    Returns
    -------
    jacobian : sparse.csc_matrix
        Block-tridiagonal matrix of shape ``(T * n_eq, T * n_vars)``.
    """
    data = np.empty(layout.source_rows.size)
    for period_jacobian, start, stop in zip(
        period_jacobians, layout.period_bounds[:-1], layout.period_bounds[1:], strict=True
    ):
        entries = slice(start, stop)
        data[entries] = period_jacobian[layout.source_rows[entries], layout.source_cols[entries]]

    return sparse.csc_matrix((data[layout.permutation], layout.indices, layout.indptr), shape=layout.shape)
