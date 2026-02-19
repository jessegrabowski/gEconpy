from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np
import pytensor.tensor as pt

from pytensor import sparse as pts
from pytensor.graph.traversal import ancestors, explicit_graph_inputs
from pytensor.sparse.variable import SparseVariable
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from pytensor.tensor.variable import TensorVariable
from scipy import sparse


def _get_used_indices(equation: TensorVariable, vector: TensorVariable) -> set[int]:
    """Find which indices of a vector are used in an equation.

    Traverses the computation graph to find Subtensor operations that
    access elements of the given vector.

    Parameters
    ----------
    equation : TensorVariable
        The equation to analyze.
    vector : TensorVariable
        The vector variable to find index accesses for.

    Returns
    -------
    indices : set of int
        The set of constant integer indices accessed from the vector.
    """
    indices: set[int] = set()

    for node in ancestors([equation]):
        if not hasattr(node, "owner") or node.owner is None:
            continue

        op = node.owner.op
        if not isinstance(op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1)):
            continue

        inputs = node.owner.inputs
        if inputs[0] is not vector:
            continue

        # Extract the index value if it's a constant
        if len(inputs) > 1:
            idx = inputs[1]
            if hasattr(idx, "data"):
                indices.add(int(idx.data))

    return indices


def _classify_variables(
    variables: Sequence[TensorVariable],
) -> tuple[dict[TensorVariable, dict[int, int]], dict[TensorVariable, int]]:
    """Classify variables into vector elements and scalars.

    Returns
    -------
    vector_to_cols : dict
        Map from root vector -> {index: column position}
    scalar_to_col : dict
        Map from scalar variable -> column position
    """
    vector_to_cols: dict[TensorVariable, dict[int, int]] = {}
    scalar_to_col: dict[TensorVariable, int] = {}

    for j, var in enumerate(variables):
        if var.owner is not None and isinstance(var.owner.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1)):
            inputs = var.owner.inputs
            vector = inputs[0]
            if len(inputs) > 1 and hasattr(inputs[1], "data"):
                idx = int(inputs[1].data)
                vector_to_cols.setdefault(vector, {})[idx] = j
        else:
            scalar_to_col[var] = j

    return vector_to_cols, scalar_to_col


def _get_jacobian_connectivity(
    variables: Sequence[TensorVariable],
    equations: Sequence[TensorVariable],
) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column indices for the non-zero Jacobian pattern.

    This inspects the PyTensor graphs and records which variables appear
    as explicit inputs in which equations. For vector variables accessed
    via indexing (x[i]), it tracks which specific indices are used.

    Parameters
    ----------
    variables : Sequence of TensorVariable
        The variables with respect to which derivatives are taken.
        Can be scalars or individual elements of a vector (x[i]).
    equations : Sequence of TensorVariable
        The equations (outputs) to differentiate.

    Returns
    -------
    rows : ndarray of int
        Row indices of non-zero entries in the Jacobian.
    cols : ndarray of int
        Column indices of non-zero entries in the Jacobian.
    """
    rows: list[int] = []
    cols: list[int] = []

    vector_to_cols, scalar_to_col = _classify_variables(variables)

    for i, eq in enumerate(equations):
        eq_inputs = set(explicit_graph_inputs(eq))

        # Check scalar variables
        for var, j in scalar_to_col.items():
            if var in eq_inputs:
                rows.append(i)
                cols.append(j)

        # Check vector element variables
        for vector, idx_to_col in vector_to_cols.items():
            if vector not in eq_inputs:
                continue
            used_indices = _get_used_indices(eq, vector)
            for idx, j in idx_to_col.items():
                if idx in used_indices:
                    rows.append(i)
                    cols.append(j)

    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _greedy_color(
    connectivity: sparse.spmatrix,
    strategy: str = "largest_first",
) -> tuple[np.ndarray, int]:
    """Color the output-connectivity graph.

    The coloring is used to pack independent equations into common projection
    vectors so that fewer Jacobian evaluations are required.

    Parameters
    ----------
    connectivity : sparse matrix
        Square symmetric connectivity matrix where entry (i, j) is non-zero
        if equations i and j share at least one variable.
    strategy : str, default "largest_first"
        Graph coloring strategy passed to networkx.

    Returns
    -------
    coloring : ndarray of int
        Color assignment for each equation.
    n_colors : int
        Total number of colors used.

    Raises
    ------
    ValueError
        If connectivity is not a square 2D matrix.
    """
    is_2d = connectivity.ndim == 2  # noqa: PLR2004
    is_square = connectivity.shape[0] == connectivity.shape[1]
    if not (is_2d and is_square):
        raise ValueError(f"Expected square 2D connectivity matrix, got shape {connectivity.shape}")

    G = nx.convert_matrix.from_scipy_sparse_array(connectivity)
    coloring_dict = nx.algorithms.coloring.greedy_color(G, strategy)

    indices, colors = zip(*coloring_dict.items(), strict=False)
    coloring = np.asarray(colors, dtype=int)[np.argsort(indices)]
    n_colors = int(np.unique(coloring).size)

    return coloring, n_colors


def _coo_to_csc(
    rows: np.ndarray | Iterable[int],
    cols: np.ndarray | Iterable[int],
    data: TensorVariable,
    shape: tuple[int, int],
) -> tuple[TensorVariable, TensorVariable, TensorVariable, tuple[int, int]]:
    """Build CSC triplets from COO-style indices and data.

    This operates symbolically on ``data`` while keeping the indices
    as constants.

    Parameters
    ----------
    rows : array-like of int
        Row indices in COO format.
    cols : array-like of int
        Column indices in COO format.
    data : TensorVariable
        Data values corresponding to the indices.
    shape : tuple of int
        Shape of the sparse matrix (n_rows, n_cols).

    Returns
    -------
    csc_data : TensorVariable
        Data array in CSC order.
    csc_indices : TensorVariable
        Row indices in CSC format.
    csc_indptr : TensorVariable
        Column pointer array for CSC format.
    shape : tuple of int
        Shape of the sparse matrix.
    """
    rows_pt = pt.as_tensor_variable(np.asarray(rows, dtype=int))
    cols_pt = pt.as_tensor_variable(np.asarray(cols, dtype=int))
    _n_rows, n_cols = shape

    order = pt.argsort(cols_pt)
    csc_data = data[order]
    csc_indices = rows_pt[order]
    sorted_cols = cols_pt[order]

    counts = pt.bincount(sorted_cols, minlength=n_cols)
    csc_indptr = pt.concatenate([pt.as_tensor([0]), pt.cumsum(counts).astype("int64")])

    return csc_data, csc_indices, csc_indptr, shape


def sparse_jacobian(
    equations: Sequence[TensorVariable],
    variables: Sequence[TensorVariable],
    use_vectorized_jacobian: bool = True,
    return_sparse: bool = False,
) -> TensorVariable | SparseVariable:
    """Compute a (potentially) sparse Jacobian matrix.

    The structure of the Jacobian is detected once and used together with a
    graph-coloring based compression scheme to reduce the number of required
    derivative evaluations.

    Parameters
    ----------
    equations : Sequence of TensorVariable
        The equations (outputs) to differentiate.
    variables : Sequence of TensorVariable
        The variables with respect to which derivatives are taken.
        Can be scalars or indexed elements of a vector (e.g., x[i]).
    use_vectorized_jacobian : bool
        If True, use PyTensor's vectorized Jacobian computation to compute derivatives for all equations. Otherwise,
        a scan is used.
    return_sparse : bool, default False
        If True, return a sparse CSC matrix. Otherwise return a dense matrix.

    Returns
    -------
    jacobian : TensorVariable or SparseVariable
        The Jacobian matrix. Shape is (len(equations), len(variables)).
    """
    rows, cols = _get_jacobian_connectivity(variables, equations)
    if rows.size == 0:
        N = len(equations)
        return pt.zeros((N, len(variables)))

    N = len(equations)
    M = len(variables)

    # Identify root vectors and build mapping from variable index to (root, idx)
    # For scalar variables, the root is the variable itself with idx=None
    root_vectors: dict[TensorVariable, list[tuple[int, int | None]]] = {}

    vector_to_cols, scalar_to_col = _classify_variables(variables)

    # Build root_vectors from the classification
    for vector, idx_to_col in vector_to_cols.items():
        root_vectors[vector] = [(col, idx) for idx, col in idx_to_col.items()]

    for scalar, col in scalar_to_col.items():
        root_vectors[scalar] = [(col, None)]

    sparsity = sparse.coo_array((np.ones_like(rows, dtype=bool), (rows, cols)), (N, M))
    output_connectivity = sparsity @ sparsity.T

    output_coloring, n_colors = _greedy_color(output_connectivity)

    if output_coloring.size != sparsity.shape[0]:
        raise ValueError(f"Coloring size {output_coloring.size} does not match number of equations {sparsity.shape[0]}")

    projection_matrix = np.equal.outer(np.arange(n_colors, dtype=int), output_coloring).astype(float)
    projected_eqs = projection_matrix @ pt.stack(equations)

    # Compute Jacobian wrt each root vector and extract relevant columns
    jac_columns: dict[int, TensorVariable] = {}

    for root, col_idx_pairs in root_vectors.items():
        # Differentiate projected equations wrt this root
        root_jac = pt.jacobian(
            projected_eqs, root, vectorize=use_vectorized_jacobian
        )  # shape: (n_colors, root_size) or (n_colors,) for scalar

        for var_col, vec_idx in col_idx_pairs:
            if vec_idx is None:
                # Scalar root - root_jac shape is (n_colors,)
                jac_columns[var_col] = root_jac
            else:
                # Vector root - extract the column for this index
                jac_columns[var_col] = root_jac[:, vec_idx]

    # Build the compressed jacobian by stacking columns in order
    compressed_jac = pt.stack([jac_columns[j] for j in range(M)], axis=-1)

    compressed_index = (output_coloring[rows], cols)
    data = compressed_jac[compressed_index]

    if return_sparse:
        data_csc, indices_csc, indptr_csc, shape = _coo_to_csc(rows, cols, data, (N, M))
        return pts.CSC(data_csc, indices_csc, indptr_csc, shape)

    dense = pt.zeros((N, M))
    return dense[rows, cols].set(data)
