from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np
import pytensor.tensor as pt

from pytensor import sparse as pts
from pytensor.gradient import Lop
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.sparse.variable import SparseVariable
from pytensor.tensor.variable import TensorVariable
from scipy import sparse


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

    scalar_to_col: dict[TensorVariable, int] = {var: j for j, var in enumerate(variables)}

    for i, eq in enumerate(equations):
        eq_inputs = set(explicit_graph_inputs(eq))

        for var, j in scalar_to_col.items():
            if var in eq_inputs:
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
    eval_points = [pt.dscalar(f"p_{i}") for i in range(N)]

    sparsity = sparse.coo_array((np.ones_like(rows, dtype=bool), (rows, cols)), (N, M))
    output_connectivity = sparsity @ sparsity.T

    output_coloring, n_colors = _greedy_color(output_connectivity)

    if output_coloring.size != sparsity.shape[0]:
        raise ValueError(f"Coloring size {output_coloring.size} does not match number of equations {sparsity.shape[0]}")

    projection_matrix = np.equal.outer(np.arange(n_colors, dtype=int), output_coloring).astype(float)

    jvp_stack = pt.stack(
        Lop(equations, variables, eval_points, disconnected_inputs="ignore", return_disconnected="zero")
    )
    P = pt.dvector("P", shape=(N,))
    jvp_stack = graph_replace(jvp_stack, {p: P[i] for i, p in enumerate(eval_points)})

    compressed_jac = vectorize_graph(jvp_stack, replace={P: pt.as_tensor_variable(projection_matrix)})

    compressed_index = (output_coloring[rows], cols)
    data = compressed_jac[compressed_index]

    if return_sparse:
        data_csc, indices_csc, indptr_csc, shape = _coo_to_csc(rows, cols, data, (N, M))
        return pts.CSC(data_csc, indices_csc, indptr_csc, shape)

    return pt.zeros((N, M))[rows, cols].set(data)
