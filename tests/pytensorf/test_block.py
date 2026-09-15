import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose

from gEconpy.pytensorf.block import block


@pytest.mark.parametrize(
    ("arrays", "error", "match"),
    [
        ([], ValueError, r"list at index \(\) is empty"),
        ([[pt.zeros((2, 2))], []], ValueError, r"list at index \(1,\) is empty"),
        ([[pt.zeros((2, 2))], pt.zeros((2, 2))], ValueError, r"depth 1 at index \(1,\), expected 2"),
        ([(pt.zeros((2, 2)),)], TypeError, r"container at index \(0,\) is a tuple"),
    ],
    ids=["empty_root", "empty_nested", "ragged_depth", "tuple_container"],
)
def test_rejects_malformed_nesting(arrays, error, match):
    with pytest.raises(error, match=match):
        block(arrays)


def test_matches_numpy_block():
    """Leaves of different ndim at the same depth promote to the grid's ndim, as in ``numpy.block``."""
    A = pt.dmatrix("A")
    v = pt.dvector("v")
    outputs = [
        block([[A, v[:, None]], [v[None, :], pt.zeros((1, 1))]]),
        block([[A, A], [v, v]]),
        block([v, v]),
        block(A),
    ]
    f = pytensor.function([A, v], outputs)

    rng = np.random.default_rng(1234)
    for _ in range(3):
        A_val = rng.standard_normal((2, 2))
        v_val = rng.standard_normal(2)
        expected = [
            np.block([[A_val, v_val[:, None]], [v_val[None, :], np.zeros((1, 1))]]),
            np.block([[A_val, A_val], [v_val, v_val]]),
            np.block([v_val, v_val]),
            A_val,
        ]
        for actual, want in zip(f(A_val, v_val), expected, strict=True):
            assert_allclose(actual, want)
