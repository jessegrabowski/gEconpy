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
    gather : ndarray of int
        Flat indices into the ``(T, n_eq, 3 * n_vars)`` array of period Jacobians, in compressed-column order.
    indices, indptr : ndarray of int
        Row indices and column pointers of the compressed-column result.
    period_shape : tuple of int
        Shape ``(T, n_eq, 3 * n_vars)`` that ``gather`` indexes into.
    shape : tuple of int
        Shape of the stacked Jacobian.
    """

    gather: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    period_shape: tuple[int, int, int]
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
        If ``sparsity_pattern`` does not have shape ``(n_eq, 3 * n_vars)``, or if ``T`` is not positive.
    """
    if sparsity_pattern.shape != (n_eq, 3 * n_vars):
        raise ValueError(f"sparsity_pattern has shape {sparsity_pattern.shape}, expected {(n_eq, 3 * n_vars)}.")
    if T < 1:
        raise ValueError(f"T must be at least 1, got {T}.")

    # The pattern is supplied rather than read off the period Jacobians because an entry can pass through zero in
    # every period at once (a steady-state initial guess, a parameter path crossing zero). A pattern inferred from
    # those values drops it, leaving Newton to step against a Jacobian that is wrong rather than approximate.
    blocks = [np.nonzero(sparsity_pattern[:, start : start + n_vars]) for start in (0, n_vars, 2 * n_vars)]
    placements = ((-1, 0), (0, n_vars), (1, 2 * n_vars))
    period_shape = (T, n_eq, 3 * n_vars)

    source_blocks, row_blocks, column_blocks = [], [], []
    for t in range(T):
        for (block_rows, block_cols), (shift, column_offset) in zip(blocks, placements, strict=True):
            if not 0 <= t + shift < T:
                continue
            source_blocks.append(
                np.ravel_multi_index(
                    (np.full_like(block_rows, t), block_rows, block_cols + column_offset), period_shape
                )
            )
            row_blocks.append(t * n_eq + block_rows)
            column_blocks.append((t + shift) * n_vars + block_cols)

    sources = np.concatenate(source_blocks)
    rows = np.concatenate(row_blocks)
    columns = np.concatenate(column_blocks)
    shape = (T * n_eq, T * n_vars)

    # Carrying the source index as the data of a throwaway matrix turns the conversion into the gather that reads
    # the period Jacobians straight into compressed-column order, so no later call has to convert from coordinate
    # form. The blocks of a period land in distinct columns and the periods in distinct rows, so no two entries
    # share a position and nothing is summed away.
    positions = sparse.coo_matrix((sources, (rows, columns)), shape=shape, dtype=np.int64).tocsc()

    return StackedJacobianLayout(
        gather=positions.data,
        indices=positions.indices,
        indptr=positions.indptr,
        period_shape=period_shape,
        shape=shape,
    )


def assemble_stacked_jacobian(
    period_jacobians: Sequence[np.ndarray] | np.ndarray,
    layout: StackedJacobianLayout,
) -> sparse.csc_matrix:
    """
    Assemble the block-tridiagonal Jacobian of the stacked system from the Jacobian of every period.

    Parameters
    ----------
    period_jacobians : ndarray or sequence of ndarray
        Array of shape ``(T, n_eq, 3 * n_vars)``, or a sequence of T matrices of shape ``(n_eq, 3 * n_vars)``.
    layout : StackedJacobianLayout
        Index arrays from :func:`build_stacked_jacobian_layout`.

    Returns
    -------
    jacobian : sparse.csc_matrix
        Block-tridiagonal matrix of shape ``(T * n_eq, T * n_vars)``.

    Raises
    ------
    ValueError
        If ``period_jacobians`` does not have the shape the layout was built for.
    """
    jacobians = np.asarray(period_jacobians, dtype=float)
    if jacobians.shape != layout.period_shape:
        raise ValueError(f"period_jacobians has shape {jacobians.shape}, expected {layout.period_shape}.")

    data = jacobians.ravel()[layout.gather]

    return sparse.csc_matrix((data, layout.indices, layout.indptr), shape=layout.shape)
