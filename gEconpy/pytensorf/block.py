import builtins

from pytensor.tensor import as_tensor_variable, atleast_Nd, concatenate

# Vendored from an open pytensor PR. Drop this module and switch to ``pt.block``
# once that lands upstream.


def _block_check_depths_match(arrays, parent_index=()):
    """Walk a nested block-list and check every leaf sits at the same depth.

    Parameters
    ----------
    arrays : list or array_like
        Nested block-list to validate.
    parent_index : tuple of int, optional
        Indices accumulated from the root, used in error messages. Default ``()``.

    Returns
    -------
    structure : nested tuple of None
        Tree shape with ``None`` at each leaf position.
    leaf_depth : int
        Depth at which every leaf sits.
    max_leaf_ndim : int
        Largest ``ndim`` across all leaves.
    """
    if isinstance(arrays, list):
        if not arrays:
            raise ValueError("Block: empty list is not allowed")
        children = []
        first_leaf_depth = None
        max_ndim = 0
        for i, child in enumerate(arrays):
            child_struct, child_leaf_depth, child_ndim = _block_check_depths_match(child, (*parent_index, i))
            if first_leaf_depth is None:
                first_leaf_depth = child_leaf_depth
            elif first_leaf_depth != child_leaf_depth:
                raise ValueError(
                    "Block: all leaves must be at the same nesting depth "
                    f"(got depth {child_leaf_depth} at index {(*parent_index, i)}, "
                    f"expected {first_leaf_depth})"
                )
            max_ndim = max(max_ndim, child_ndim)
            children.append(child_struct)
        return tuple(children), first_leaf_depth, max_ndim
    if isinstance(arrays, tuple):
        raise TypeError("Block: tuples are not allowed as nested containers; use lists")
    leaf = as_tensor_variable(arrays)
    return None, len(parent_index), leaf.type.ndim


def block(arrays):
    """Assemble a tensor from nested lists of blocks, like ``numpy.block``.

    Parameters
    ----------
    arrays : nested list of array_like
        Tensors at the leaves, lists at the interior. Every leaf must sit at
        the same nesting depth ``d``; the concatenation spans the last ``d``
        axes.

    Returns
    -------
    result : TensorVariable
        Assembled block tensor. A bare tensor (no list wrapping) returns as
        ``atleast_1d(arrays)``.

    Examples
    --------
    .. testcode::

        import numpy as np
        import pytensor.tensor as pt

        from gEconpy.pytensorf.block import block

        A = pt.as_tensor_variable(np.array([[1, 2], [3, 4]]))
        B = pt.as_tensor_variable(np.array([[5], [6]]))
        C = pt.as_tensor_variable(np.array([[7, 8]]))
        D = pt.as_tensor_variable(np.array([[9]]))
        M = block([[A, B], [C, D]])
        print(M.eval())

    .. testoutput::

        [[1 2 5]
         [3 4 6]
         [7 8 9]]
    """
    structure, _, _ = _block_check_depths_match(arrays)

    if structure is None:
        return atleast_Nd(arrays, n=1)

    flat = []

    def _gather(node):
        if isinstance(node, list):
            for child in node:
                _gather(child)
        else:
            flat.append(as_tensor_variable(node))

    _gather(arrays)

    def _structure_depth(structure):
        if structure is None:
            return 0
        return 1 + _structure_depth(structure[0])

    list_ndim = _structure_depth(structure)
    result_ndim = builtins.max(list_ndim, *(inp.type.ndim for inp in flat))
    promoted = [atleast_Nd(inp, n=result_ndim) for inp in flat]

    def _unflatten_structure(flat, structure):
        """Rebuild a nested list from ``flat`` consumed in pre-order against ``structure``."""
        it = iter(flat)

        def _build(s):
            if s is None:
                return next(it)
            return [_build(child) for child in s]

        return _build(structure)

    nested = _unflatten_structure(promoted, structure)

    def _recurse(node, depth):
        if depth == list_ndim:
            return node
        children = [_recurse(child, depth + 1) for child in node]
        return concatenate(children, axis=-(list_ndim - depth))

    return _recurse(nested, 0)
