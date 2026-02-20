import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import function
from pytensor.sparse.variable import SparseVariable

from gEconpy.model.perfect_foresight import _replace_scalars_with_vector
from gEconpy.pytensorf.sparse_jacobian import sparse_jacobian


def test_replace_scalars_with_vector_dense():
    v1 = pt.scalar("v1")
    v2 = pt.scalar("v2")
    x = pt.vector("x", shape=(2,))

    expr = v1 + 2 * v2
    replaced = _replace_scalars_with_vector(expr, [v1, v2], x)

    f = function([x], replaced)
    out = f(np.array([1.0, 3.0], dtype="float64"))
    assert out == pytest.approx(7.0)


def test_sparse_jacobian_dense_and_sparse_agree():
    variables = x1, x2, y1, y2 = pt.dscalars("x1", "x2", "y1", "y2")

    eq1 = x1**2 + 2 * y1
    eq2 = 3 * x2 - y2**2
    equations = [eq1, eq2]

    jac_dense = sparse_jacobian(equations, variables, return_sparse=False)
    jac_sparse = sparse_jacobian(equations, variables, return_sparse=True)

    f_dense = function(variables, jac_dense)
    J_dense_val = f_dense(*np.ones(4))

    assert J_dense_val.shape == (2, 4)

    assert isinstance(jac_sparse, SparseVariable)
