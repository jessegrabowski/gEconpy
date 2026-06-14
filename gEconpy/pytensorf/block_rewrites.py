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

# Vendored and adapted from an open pytensor PR (Join/block rewrites: transpose pushdown,
# nested-join -> block_diag detection, and Split(Join) collapse). Drop this module once
# that work lands upstream.


def _const_int(var):
    """Return the static int value of a scalar variable, or ``None`` if not static.

    Parameters
    ----------
    var : TensorVariable
        Scalar variable to inspect.

    Returns
    -------
    value : int or None
        The constant value, or ``None`` when it is not statically known.
    """
    try:
        return int(get_underlying_scalar_constant_value(var, raise_not_constant=True))
    except NotScalarConstantError:
        return None


def _const_int_vector(var):
    """Return a ``list`` of ints from a 1-D variable whose entries are statically known.

    Handles a :class:`Constant` array and a :class:`MakeVector` of scalar constants.

    Parameters
    ----------
    var : TensorVariable
        Vector variable to inspect.

    Returns
    -------
    values : list of int or None
        The contents as Python ints, or ``None`` when they are not statically known.
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
    """Return ``True`` if ``var`` is statically the all-zero tensor.

    Sees through ``Alloc`` (so ``pytensor.tensor.zeros`` is recognised) and constants.

    Parameters
    ----------
    var : TensorVariable
        Variable to test.

    Returns
    -------
    bool
        Whether ``var`` is statically zero.
    """
    try:
        return get_underlying_scalar_constant_value(var, only_process_constants=False, raise_not_constant=True) == 0
    except NotScalarConstantError:
        return False


def _join_matmul_axis(var):
    """Return ``-1`` or ``-2`` if ``var`` is a :class:`Join` along a matmul axis, else ``None``.

    Parameters
    ----------
    var : TensorVariable
        Variable to inspect.

    Returns
    -------
    axis : int or None
        ``-1`` if the join concatenates the inner (last) matrix axis, ``-2`` for the
        outer axis, ``None`` if ``var`` is not such a join.
    """
    owner = var.owner
    if owner is None or not isinstance(owner.op, Join):
        return None
    axis = _const_int(owner.inputs[0])
    if axis is None:
        return None
    ndim = var.type.ndim
    if axis < 0:
        axis += ndim
    if axis == ndim - 1:
        return -1
    if axis == ndim - 2:
        return -2
    return None


@register_canonicalize
@register_stabilize
@node_rewriter([DimShuffle])
def local_transpose_of_join(_fgraph, node):
    r"""Push a matrix transpose inside a :class:`Join`.

    Rewrite ``Join(axis, *xs).mT`` to ``Join(swapped_axis, *[x.mT for x in xs])``,
    swapping ``axis`` between ``-1`` and ``-2`` and leaving batch axes untouched. The
    transpose is a free ``DimShuffle``; pushing it inward exposes each leaf's transpose
    to folding (``A.mT.mT -> A``, triangular-solve patterns).
    """
    if not node.op.is_matrix_transpose:
        return None

    [src] = node.inputs
    if src.owner is None or not isinstance(src.owner.op, Join):
        return None

    join_axis = _const_int(src.owner.inputs[0])
    if join_axis is None:
        return None

    src_ndim = src.type.ndim
    if join_axis < 0:
        join_axis += src_ndim

    if join_axis == src_ndim - 1:
        new_axis = src_ndim - 2
    elif join_axis == src_ndim - 2:
        new_axis = src_ndim - 1
    else:
        new_axis = join_axis  # batch axis -- mT does not touch it

    new_out = pt_join(new_axis, *[inp.mT for inp in src.owner.inputs[1:]])
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@register_canonicalize
@register_stabilize
@node_rewriter([Join])
def local_nested_join_to_block_diagonal(_fgraph, node):
    r"""Rewrite a square block grid with zero off-diagonals to :func:`block_diag`.

    Detect ``Join(-2, *Join(-1, ...))`` -- an outer row-concat whose every input is a
    column-concat -- forming an ``n x n`` square grid whose off-diagonal blocks are
    statically zero. Replace it with ``BlockDiagonal`` to unlock the targeted
    ``BlockDiagonal`` rewrites (det, diag, trace, dot and solve pushdowns).
    """
    if _join_matmul_axis(node.outputs[0]) != -2:
        return None

    rows = list(node.inputs[1:])
    n = len(rows)
    if n < 2:
        return None

    grid = []
    for row in rows:
        if _join_matmul_axis(row) != -1:
            return None
        cols = list(row.owner.inputs[1:])
        if len(cols) != n:  # not square
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


@register_canonicalize
@register_stabilize
@node_rewriter([Split])
def local_split_of_join(_fgraph, node):
    r"""Push :class:`Split` through :class:`Join`.

    Two cases are handled:

    - **Same axis, matching sizes.** ``Split(Join(a, *X), [|X_i|_a], axis=a)`` returns
      the join's inputs directly -- the split exactly undoes the concatenation.
    - **Different axis.** ``Split(Join(a, *X), s, axis=b)`` with ``a != b`` distributes
      the split through the join: each cut becomes ``Join(a, *[Split(X_i, s, b)[k]])``.
      Slicing an orthogonal axis commutes with concatenation.

    These collapse ``Split(Join(...))`` cascades back to their underlying blocks.
    """
    x, axis_var, splits_size_var = node.inputs
    if x.owner is None or not isinstance(x.owner.op, Join):
        return None

    split_axis = _const_int(axis_var)
    join_axis = _const_int(x.owner.inputs[0])
    if split_axis is None or join_axis is None:
        return None

    ndim = x.type.ndim
    split_axis %= ndim
    join_axis %= ndim
    join_inputs = list(x.owner.inputs[1:])

    if split_axis == join_axis:
        return _split_undoes_join(node, join_inputs, join_axis, splits_size_var)
    return _split_distributes_through_join(node, join_inputs, join_axis, split_axis, splits_size_var)
