import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import out2in
from pytensor.tensor.extra_ops import concat_with_broadcast
from pytensor.tensor.linalg import block_diag

import gEconpy

from gEconpy.pytensorf.block import block
from gEconpy.pytensorf.block_rewrites import (
    local_nested_join_to_block_diagonal,
    local_split_of_join,
    local_transpose_of_join,
)

N = 4
floatX = pytensor.config.floatX

# Apply only the block rewrites (not the full canonicalize/stabilize databases),
# so the rewritten graph is exactly what these rewrites emit and can be matched op-for-op.
_BLOCK_REWRITES = out2in(
    local_transpose_of_join,
    local_nested_join_to_block_diagonal,
    local_split_of_join,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _rewrite(*exprs):
    """Apply the block rewrites to fixpoint and return the rewritten output variables."""
    fg = FunctionGraph(outputs=list(exprs), clone=False)
    _BLOCK_REWRITES.rewrite(fg)
    return fg.outputs


def assert_equal_computations(rewritten, expected, *args, original=None, **kwargs):
    """Assert that ``rewritten`` computes the same graph as ``expected``.

    On failure, dump the original (if given), rewritten, and expected graphs.
    """
    __tracebackhide__ = True

    if equal_computations(rewritten, expected, *args, **kwargs):
        return True

    def _dprint(expr):
        return pytensor.dprint(expr, print_type=True, file="str")

    parts = []
    if original is not None:
        parts.append(f"\nOriginal:\n{_dprint(original)}")
    parts.append(f"\nRewritten:\n{_dprint(rewritten)}")
    parts.append(f"\nExpected:\n{_dprint(expected)}")
    raise AssertionError("equal_computations failed\n" + "".join(parts))


def _mat(name, shape):
    return pt.matrix(name, shape=shape)


def _vals(rng, *shapes):
    return [rng.normal(size=s).astype(floatX) for s in shapes]


class TestTransposeOfJoin:
    def test_transpose_pushes_to_leaves(self, rng):
        A, B = (_mat(s, (N, N)) for s in "AB")
        original = block([[A, B]]).mT
        (actual,) = _rewrite(original)
        assert_equal_computations([actual], [pt.join(0, A.mT, B.mT)], original=original)

        f = pytensor.function([A, B], block([[A, B]]).mT)
        Av, Bv = _vals(rng, (N, N), (N, N))
        assert_allclose(f(Av, Bv), np.block([[Av, Bv]]).T)


class TestNestedJoinToBlockDiagonal:
    def test_zero_off_diagonal_becomes_block_diag(self, rng):
        A, B = (_mat(s, (N, N)) for s in "AB")
        zero = pt.zeros((N, N))
        original = block([[A, zero], [zero, B]])
        (actual,) = _rewrite(original)
        assert_equal_computations([actual], [block_diag(A, B)], original=original)

        f = pytensor.function([A, B], block([[A, zero], [zero, B]]))
        Av, Bv = _vals(rng, (N, N), (N, N))
        z = np.zeros((N, N))
        assert_allclose(f(Av, Bv), np.block([[Av, z], [z, Bv]]))

    def test_dense_off_diagonal_unchanged(self):
        A, B, C = (_mat(s, (N, N)) for s in "ABC")
        original = block([[A, pt.zeros((N, N))], [C, B]])
        (actual,) = _rewrite(original)
        assert_equal_computations([actual], [original])

    def test_non_square_grid_unchanged(self):
        A, B, C = (_mat(s, (N, N)) for s in "ABC")
        z = pt.zeros((N, N))
        original = block([[A, z, z], [B, z, C]])
        (actual,) = _rewrite(original)
        assert_equal_computations([actual], [original])


class TestSplitOfJoin:
    def test_split_undoes_join_on_same_axis(self):
        A, B = (_mat(s, (N, N)) for s in "AB")
        a, b = _rewrite(*pt.split(pt.join(-1, A, B), splits_size=[N, N], n_splits=2, axis=-1))
        # The split exactly reverses the join: its inputs are returned verbatim.
        assert a is A
        assert b is B

    def test_split_distributes_through_join_on_different_axis(self, rng):
        A = _mat("A", (3, N))
        B = _mat("B", (4, N))
        joined = pt.join(-2, A, B)
        c0, c1 = _rewrite(*pt.split(joined, splits_size=[2, N - 2], n_splits=2, axis=-1))

        # The rewrite normalizes axes: split axis -1 -> 1, join axis -2 -> 0.
        a_parts = pt.split(A, splits_size=[2, N - 2], n_splits=2, axis=1)
        b_parts = pt.split(B, splits_size=[2, N - 2], n_splits=2, axis=1)
        assert_equal_computations([c0], [pt.join(0, a_parts[0], b_parts[0])])
        assert_equal_computations([c1], [pt.join(0, a_parts[1], b_parts[1])])

        f = pytensor.function([A, B], list(pt.split(joined, splits_size=[2, N - 2], n_splits=2, axis=-1)))
        Av, Bv = _vals(rng, (3, N), (4, N))
        joined_v = np.concatenate([Av, Bv], axis=-2)
        got0, got1 = f(Av, Bv)
        assert_allclose(got0, joined_v[:, :2])
        assert_allclose(got1, joined_v[:, 2:])
