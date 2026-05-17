import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.basic import (
    Eye,
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
from pytensor.tensor.basic import (
    stack as pt_stack,
)
from pytensor.tensor.basic import (
    zeros as pt_zeros,
)
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.extra_ops import concat_with_broadcast
from pytensor.tensor.linalg import block_diag
from pytensor.tensor.math import _dot, _matmul, add
from pytensor.tensor.rewriting.basic import register_canonicalize, register_stabilize

# Vendored and adapted from an open pytensor PR (Join/dot/block rewrites). Drop this
# module once that work lands upstream. The dot-of-join rewrite here is gated and
# drops zero blocks itself rather than relying on ``local_mul_zero`` -- gEconpy
# excludes that rewrite (see ``gEconpy/__init__.py``).


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


def _is_identity(var):
    """Return ``True`` if ``var`` is statically a square identity matrix.

    Recognizes an :class:`Eye` ``Op`` with equal row/column counts and zero offset, and
    a :class:`Constant` equal to :func:`numpy.eye`.

    Parameters
    ----------
    var : TensorVariable
        Variable to test.

    Returns
    -------
    bool
        Whether ``var`` is statically a square identity.
    """
    if var.type.ndim != 2:
        return False
    owner = var.owner
    if owner is not None and isinstance(owner.op, Eye):
        n, m, k = owner.inputs
        if _const_int(k) != 0:
            return False
        nv, mv = _const_int(n), _const_int(m)
        if nv is not None and mv is not None:
            return nv == mv
        return n is m
    if isinstance(var, Constant):
        arr = np.asarray(var.data)
        return (
            arr.ndim == 2
            and arr.shape[0] == arr.shape[1]
            and np.array_equal(arr, np.eye(arr.shape[0], dtype=arr.dtype))
        )
    return False


def _selection_map(var):
    """Return the row-to-column index map of a static row-selection matrix, else ``None``.

    A row-selection matrix is 2-D, ``0``/``1``-valued, with at most one ``1`` per row --
    identities, offset/shift eyes, and permutations all qualify. ``M @ X`` then equals
    ``X`` with its rows gathered by the map (an advanced index, not a GEMM); a row of
    ``M`` that is all zero maps to ``-1`` and becomes a zero row of the result.
    Recognizes an :class:`Eye` ``Op`` (any offset) and a ``0``/``1`` :class:`Constant`.

    Parameters
    ----------
    var : TensorVariable
        Variable to test.

    Returns
    -------
    idx : ndarray of int or None
        ``idx[i]`` is the column of the ``1`` in row ``i``, or ``-1`` for an all-zero
        row. ``None`` when ``var`` is not a statically-known row-selection matrix.
    """
    if var.type.ndim != 2:
        return None
    owner = var.owner
    if owner is not None and isinstance(owner.op, Eye):
        n, m, k = (_const_int(i) for i in owner.inputs)
        if n is None or m is None or k is None:
            return None
        rows = np.arange(n) + k
        return np.where((rows >= 0) & (rows < m), rows, -1)
    if isinstance(var, Constant):
        arr = np.asarray(var.data)
        if arr.ndim != 2 or not np.all((arr == 0) | (arr == 1)) or np.any(arr.sum(axis=1) > 1):
            return None
        idx = np.full(arr.shape[0], -1, dtype=int)
        nz_rows, nz_cols = np.nonzero(arr)
        idx[nz_rows] = nz_cols
        return idx
    return None


def _apply_selection(operand, idx):
    """Return ``M @ operand`` for a row-selection matrix ``M`` with row map ``idx``.

    Gathers rows of ``operand`` along its second-to-last axis; rows where ``idx == -1``
    (all-zero rows of ``M``) are zero-filled. Replaces the GEMM with an advanced index.

    Parameters
    ----------
    operand : TensorVariable
        The right operand of ``M @ operand``.
    idx : ndarray of int
        The row map from :func:`_selection_map`.

    Returns
    -------
    TensorVariable
        The gathered (and masked) operand.
    """
    empty = idx < 0
    gathered = pt.take(operand, np.where(empty, 0, idx), axis=-2)
    if not empty.any():
        return gathered
    keep = (~empty).astype(operand.type.dtype).reshape((-1, 1))
    return gathered * keep


def _block_kind(var):
    """Classify a matmul block, caching the result on ``var.tag``.

    Classifying a :class:`Constant` scans its data (``O(n^2)`` for the selection-matrix
    test). The rewrite revisits the same blocks many times -- across clients, recursive
    gate walks, and repeated compilations -- so the verdict is memoised on the
    variable's ``tag``; the structure a block is built from never changes, so the cache
    never goes stale.

    Parameters
    ----------
    var : TensorVariable
        The block to classify.

    Returns
    -------
    kind : str
        One of ``"zero"``, ``"identity"``, ``"selection"``, ``"dense"``.
    idx : ndarray of int or None
        The row-selection map when ``kind`` is ``"selection"``, else ``None``.
    """
    cached = getattr(var.tag, "block_rewrite_kind", None)
    if cached is not None:
        return cached
    if _is_zero(var):
        result = ("zero", None)
    elif _is_identity(var):
        result = ("identity", None)
    elif (sel := _selection_map(var)) is not None:
        result = ("selection", sel)
    else:
        result = ("dense", None)
    var.tag.block_rewrite_kind = result
    return result


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


def _decomposition_saves(var, join_is_left):
    """Return ``True`` if decomposing ``dot`` against ``var`` would eliminate a GEMM.

    Decomposition pays off only when a leaf -- reached transitively through nested
    matmul-axis :class:`Join` nodes, as produced by :func:`gEconpy.pytensorf.block.block`
    -- can be handled without a GEMM: a static zero (dropped), an identity (the product
    is the operand), or, when the block is *left*-multiplied, a row-selection matrix
    (the product is an advanced-index gather). An all-dense block matrix has no such
    leaf, so its single BLAS GEMM is left untouched.

    ``X @ M`` for a non-identity selection ``M`` is a column scatter rather than a clean
    gather, so selection blocks count only on the left.

    Parameters
    ----------
    var : TensorVariable
        Operand of the matmul (a possibly-nested ``Join``).
    join_is_left : bool
        ``True`` when the join is the left operand of the matmul.

    Returns
    -------
    bool
        Whether the nested-join tree holds a leaf the decomposition can cheapen.
    """
    kind, _ = _block_kind(var)
    if kind in ("zero", "identity"):
        return True
    if kind == "selection" and join_is_left:
        return True
    if _join_matmul_axis(var) is not None:
        return any(_decomposition_saves(leaf, join_is_left) for leaf in var.owner.inputs[1:])
    return False


def _maybe_join(items, axis):
    """Join ``items`` along ``axis``, returning the lone element unchanged when ``len == 1``."""
    return items[0] if len(items) == 1 else pt_join(axis, *items)


def _matmul_clients(fgraph, var):
    """Yield ``(matmul_node, input_index)`` for every direct ``Dot``/``matmul`` client of ``var``."""
    for client, input_idx in fgraph.clients[var]:
        if isinstance(client, str):
            continue
        if client.op in (_dot, _matmul):
            yield client, input_idx


def _split_operand(other, leaves, join_is_left):
    """Split ``other`` into one chunk per leaf along the contracted matmul axis."""
    size_axis, split_axis = (-1, -2) if join_is_left else (-2, -1)
    sizes = pt_stack([leaf.shape[size_axis] for leaf in leaves])
    return pt_split(other, splits_size=sizes, n_splits=len(leaves), axis=split_axis)


def _sided_dot(dot_op, left, right, join_is_left):
    """Apply ``dot_op`` with the join-side operand on the correct side."""
    return dot_op(left, right) if join_is_left else dot_op(right, left)


def _contracted_dense_terms(dense, n_leaves, dot_op, join_is_left):
    """Build the GEMM term(s) for the dense survivor blocks of a contracted decomposition.

    Re-joins the survivors into one GEMM when at least one block was handled without a
    GEMM (the survivors then form a strictly smaller ``Join``). When every leaf survived
    -- the gate fired on a *nested* block -- emits one product per leaf instead, since
    re-joining all of them would rebuild the parent ``Join`` and loop the rewrite.
    """
    if not dense:
        return []
    if len(dense) == n_leaves:
        return [_sided_dot(dot_op, leaf, chunk, join_is_left) for leaf, chunk in dense]
    leaves, chunks = (list(t) for t in zip(*dense, strict=True))
    return [_sided_dot(dot_op, _maybe_join(leaves, -1), _maybe_join(chunks, -2), join_is_left)]


def _decompose_contracted(leaves, other, dot_op, join_is_left):
    """Decompose ``dot`` when the join runs along the contracted axis.

    Splits ``other`` by the per-leaf sizes, then for each block: drops a zero, folds an
    identity to its operand chunk, gathers a left-multiplied row-selection block, and
    collects the rest as dense pairs handed to :func:`_contracted_dense_terms`.

    Parameters
    ----------
    leaves : list of TensorVariable
        The joined blocks along the contracted axis.
    other : TensorVariable
        The non-join matmul operand.
    dot_op : Op
        The matmul ``Op`` (``Dot`` or ``matmul``) to rebuild products with.
    join_is_left : bool
        ``True`` for ``Join @ other``, ``False`` for ``other @ Join``.

    Returns
    -------
    TensorVariable
        The decomposed result.
    """
    chunks = _split_operand(other, leaves, join_is_left)

    terms = []
    dense = []
    for leaf, chunk in zip(leaves, chunks, strict=True):
        match _block_kind(leaf):
            case ("zero", _):
                continue
            case ("identity", _):
                terms.append(chunk)
            case ("selection", idx) if join_is_left:
                terms.append(_apply_selection(chunk, idx))
            case _:
                dense.append((leaf, chunk))

    terms += _contracted_dense_terms(dense, len(leaves), dot_op, join_is_left)
    if not terms:
        # Degenerate: every block was zero. Keep one (zero-valued) product for the shape.
        return _sided_dot(dot_op, leaves[0], chunks[0], join_is_left)
    return terms[0] if len(terms) == 1 else add(*terms)


def _zero_block(leaf, other, join_is_left):
    """Build the all-zero result block for a zero leaf in an output-axis decomposition."""
    shape = (leaf.shape[-2], other.shape[-1]) if join_is_left else (other.shape[-2], leaf.shape[-1])
    return pt_zeros(shape, dtype=other.type.dtype)


def _decompose_output(leaves, other, dot_op, join_is_left, concat_axis):
    """Decompose ``dot`` when the join runs along an output (non-contracted) axis.

    Emits one block per leaf -- a zero block, ``other`` for an identity, an
    advanced-index gather of ``other`` for a left-multiplied row-selection block, a GEMM
    otherwise -- and concatenates them. Dense leaves are *not* re-joined here: doing so
    would reconstruct the parent ``Join``. This pass instead exposes nested joins for
    the contracted-axis decomposition to crunch.

    Parameters
    ----------
    leaves : list of TensorVariable
        The joined blocks along the output axis.
    other : TensorVariable
        The non-join matmul operand.
    dot_op : Op
        The matmul ``Op`` to rebuild products with.
    join_is_left : bool
        ``True`` for ``Join @ other``, ``False`` for ``other @ Join``.
    concat_axis : int
        Axis (``-1`` or ``-2``) to concatenate the result blocks along.

    Returns
    -------
    TensorVariable
        The decomposed result.
    """
    blocks = []
    for leaf in leaves:
        match _block_kind(leaf):
            case ("identity", _):
                blocks.append(other)
            case ("zero", _):
                blocks.append(_zero_block(leaf, other, join_is_left))
            case ("selection", idx) if join_is_left:
                blocks.append(_apply_selection(other, idx))
            case _:
                blocks.append(_sided_dot(dot_op, leaf, other, join_is_left))
    return concat_with_broadcast(blocks, axis=concat_axis)


@register_stabilize
@node_rewriter([Join])
def local_dot_of_join(fgraph, node):
    r"""Push ``dot`` inside a :class:`Join`, but only when it eliminates work.

    Decomposing ``dot(Join, Y)`` does not change the FLOP count; it trades one BLAS
    GEMM for several smaller ones plus a concat/add. That is a loss unless the
    decomposition lets work be dropped -- a statically-zero block (skip the product),
    an identity block (the product is the operand), or, on the left, a row-selection
    block (the product is an advanced-index gather). The rewrite fires per matmul
    client only when the nested-join tree transitively contains such a block (see
    :func:`_decomposition_saves`), so all-dense block matmuls keep their single GEMM.

    Along the contracted axis the surviving dense blocks are re-joined into one GEMM.
    Along an output axis blocks stay separate -- re-joining would rebuild the parent.
    """
    matmul_axis = _join_matmul_axis(node.outputs[0])
    if matmul_axis is None:
        return None

    leaves = list(node.inputs[1:])
    if len(leaves) < 2:
        return None

    join_out = node.outputs[0]

    replacements: dict = {}
    for client, client_idx in _matmul_clients(fgraph, join_out):
        old_out = client.outputs[0]
        if old_out in replacements:
            continue

        join_is_left = client_idx == 0
        if not _decomposition_saves(join_out, join_is_left):
            continue
        other = client.inputs[1 - client_idx]
        dot_op = client.op

        # Contracted axis: Join @ other along -1, or other @ Join along -2.
        contracted = (join_is_left and matmul_axis == -1) or (not join_is_left and matmul_axis == -2)
        if contracted:
            new_output = _decompose_contracted(leaves, other, dot_op, join_is_left)
        else:
            new_output = _decompose_output(leaves, other, dot_op, join_is_left, matmul_axis)

        copy_stack_trace(old_out, new_output)
        replacements[old_out] = new_output

    return replacements or None


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

    These collapse the ``Split(Join(...))`` cascades produced by :func:`local_dot_of_join`.
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
