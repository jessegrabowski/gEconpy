import numpy as np

from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.basic import (
    Join,
    MakeVector,
    Split,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.basic import (
    join as pt_join,
)
from pytensor.tensor.basic import (
    split as pt_split,
)
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.linalg import block_diag
from pytensor.tensor.rewriting.basic import register_canonicalize, register_stabilize

# Vendored and adapted from an open pytensor PR (Join/block rewrites: transpose pushdown, nested-join -> block_diag
# detection, and Split(Join) collapse). Drop this module once that work lands upstream.


def _const_int(var):
    """Return the static int value of a scalar variable, or ``None`` when it is not statically known."""
    try:
        return int(get_underlying_scalar_constant_value(var, raise_not_constant=True))
    except NotScalarConstantError:
        return None


def _const_int_vector(var):
    """Return the entries of a 1-D variable as ints, or ``None`` when they are not statically known.

    Handles a :class:`Constant` array and a :class:`MakeVector` of scalar constants.
    """
    if isinstance(var, Constant):
        arr = np.asarray(var.data)
        if arr.ndim != 1:
            return None
        return [int(x) for x in arr]
    if var.owner is not None and isinstance(var.owner.op, MakeVector):
        out = [_const_int(inp) for inp in var.owner.inputs]
        return None if any(v is None for v in out) else out
    return None


def _is_zero(var):
    """Return whether ``var`` is statically the all-zero tensor, seeing through ``Alloc`` and constants."""
    try:
        return get_underlying_scalar_constant_value(var, only_process_constants=False, raise_not_constant=True) == 0
    except NotScalarConstantError:
        return False


def _join_matmul_axis(var):
    """Return ``-1`` or ``-2`` when ``var`` is a :class:`Join` along the last or second-to-last axis, else ``None``."""
    owner = var.owner
    if owner is None or not isinstance(owner.op, Join):
        return None
    axis = owner.op.axis
    ndim = var.type.ndim
    if axis == ndim - 1:
        return -1
    if axis == ndim - 2:
        return -2
    return None


@node_rewriter([DimShuffle])
def local_transpose_of_join(_fgraph, node):
    """Push a matrix transpose inside a :class:`Join`.

    Rewrite ``Join(axis, *xs).mT`` to ``Join(swapped_axis, *[x.mT for x in xs])``, swapping ``axis`` between ``-1``
    and ``-2`` and leaving batch axes untouched. The transpose is a free ``DimShuffle``, so pushing it inward exposes
    each leaf's transpose to folding (``A.mT.mT -> A``, triangular-solve patterns).
    """
    if not node.op.is_matrix_transpose:
        return None

    [src] = node.inputs
    if src.owner is None or not isinstance(src.owner.op, Join):
        return None

    join_axis = src.owner.op.axis
    src_ndim = src.type.ndim

    if join_axis == src_ndim - 1:
        new_axis = src_ndim - 2
    elif join_axis == src_ndim - 2:
        new_axis = src_ndim - 1
    else:
        new_axis = join_axis  # batch axis, which mT does not touch

    new_out = pt_join(new_axis, *[inp.mT for inp in src.owner.inputs])
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@node_rewriter([Join])
def local_nested_join_to_block_diagonal(_fgraph, node):
    """Rewrite a square block grid with zero off-diagonals to :func:`block_diag`.

    Detect ``Join(-2, *Join(-1, ...))``, an outer row-concat whose every input is a column-concat, forming an
    ``n x n`` square grid whose off-diagonal blocks are statically zero. Replace it with ``BlockDiagonal`` so the
    ``BlockDiagonal`` rewrites (det, diag, trace, dot and solve pushdowns) can fire.
    """
    if _join_matmul_axis(node.outputs[0]) != -2:
        return None

    rows = list(node.inputs)
    n = len(rows)
    if n < 2:
        return None

    grid = []
    for row in rows:
        if _join_matmul_axis(row) != -1:
            return None
        cols = list(row.owner.inputs)
        if len(cols) != n:
            return None
        grid.append(cols)

    diag_blocks = []
    for i in range(n):
        for j in range(n):
            if i == j:
                diag_blocks.append(grid[i][j])
            elif not _is_zero(grid[i][j]):
                return None

    new_out = block_diag(*diag_blocks)
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


def _split_undoes_join(node, join_inputs, join_axis, splits_size_var):
    """Return the join's inputs when a same-axis split exactly reverses it, else ``None``."""
    if len(join_inputs) != len(node.outputs):
        return None
    join_sizes = [inp.type.shape[join_axis] for inp in join_inputs]
    split_sizes = _const_int_vector(splits_size_var)
    if None in join_sizes or split_sizes is None or join_sizes != split_sizes:
        return None
    for inp in join_inputs:
        copy_stack_trace(node.outputs[0], inp)
    return list(join_inputs)


def _split_distributes_through_join(node, join_inputs, join_axis, split_axis, splits_size_var):
    """Distribute a split through a join along an orthogonal axis (they commute)."""
    n_splits = len(node.outputs)
    per_input = [pt_split(inp, splits_size=splits_size_var, n_splits=n_splits, axis=split_axis) for inp in join_inputs]
    new_outputs = [pt_join(join_axis, *[part[k] for part in per_input]) for k in range(n_splits)]
    for new_out in new_outputs:
        copy_stack_trace(node.outputs[0], new_out)
    return new_outputs


@node_rewriter([Split])
def local_split_of_join(_fgraph, node):
    """Push :class:`Split` through :class:`Join`.

    Two cases are handled:

    - Same axis, matching sizes. ``Split(Join(a, *X), [|X_i|_a], axis=a)`` returns the join's inputs directly,
      because the split exactly undoes the concatenation.
    - Different axis. ``Split(Join(a, *X), s, axis=b)`` with ``a != b`` distributes the split through the join, so
      each cut becomes ``Join(a, *[Split(X_i, s, b)[k]])``. Slicing an orthogonal axis commutes with concatenation.

    Both collapse ``Split(Join(...))`` cascades back to their underlying blocks.
    """
    x, splits_size_var = node.inputs
    if x.owner is None or not isinstance(x.owner.op, Join):
        return None

    split_axis = node.op.axis
    join_axis = x.owner.op.axis
    join_inputs = list(x.owner.inputs)

    if split_axis == join_axis:
        return _split_undoes_join(node, join_inputs, join_axis, splits_size_var)
    return _split_distributes_through_join(node, join_inputs, join_axis, split_axis, splits_size_var)


# Registered with ``overwrite_existing`` so that re-importing the module (``importlib.reload``, or a harness that
# evicts gEconpy from ``sys.modules``) replaces the entries. Without it the duplicate name raises.
for _rewrite in (local_transpose_of_join, local_nested_join_to_block_diagonal, local_split_of_join):
    register_canonicalize(_rewrite, overwrite_existing=True)
    register_stabilize(_rewrite, overwrite_existing=True)
