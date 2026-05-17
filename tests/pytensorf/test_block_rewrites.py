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
    local_dot_of_join,
    local_nested_join_to_block_diagonal,
    local_split_of_join,
    local_transpose_of_join,
)

N = 4
floatX = pytensor.config.floatX

# Apply only the four block rewrites (not the full canonicalize/stabilize databases),
# so the rewritten graph is exactly what these rewrites emit and can be matched op-for-op.
_BLOCK_REWRITES = out2in(
    local_dot_of_join,
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


class TestDotOfJoin:
    def test_block_triangular_drops_zero_block(self, rng):
        T, F, C = (_mat(s, (N, N)) for s in "TFC")
        x = _mat("x", (2 * N, 3))
        zero = pt.zeros((N, N))
        original = block([[T, zero], [F, C]]) @ x
        (actual,) = _rewrite(original)

        x0 = pt.split(x, splits_size=pt.stack([T.shape[-1], zero.shape[-1]]), n_splits=2, axis=-2)[0]
        expected = concat_with_broadcast([T @ x0, pt.join(-1, F, C) @ x], axis=-2)
        assert_equal_computations([actual], [expected], original=original)

        f = pytensor.function([T, F, C, x], block([[T, zero], [F, C]]) @ x)
        Tv, Fv, Cv, xv = _vals(rng, (N, N), (N, N), (N, N), (2 * N, 3))
        assert_allclose(f(Tv, Fv, Cv, xv), np.block([[Tv, np.zeros((N, N))], [Fv, Cv]]) @ xv)

    def test_dense_block_matmul_unchanged(self):
        A, B, C, D = (_mat(s, (N, N)) for s in "ABCD")
        x = _mat("x", (2 * N, 3))
        original = block([[A, B], [C, D]]) @ x
        (actual,) = _rewrite(original)
        # No zero/identity/selection block -> the gate declines and the graph is untouched.
        assert_equal_computations([actual], [original])

    def test_contracted_zero_block_rejoined_into_one_gemm(self, rng):
        A, B = (_mat(s, (N, N)) for s in "AB")
        Y = _mat("Y", (3 * N, 3))
        zero = pt.zeros((N, N))
        original = block([[A, B, zero]]) @ Y
        (actual,) = _rewrite(original)

        sizes = pt.stack([A.shape[-1], B.shape[-1], zero.shape[-1]])
        chunks = pt.split(Y, splits_size=sizes, n_splits=3, axis=-2)
        expected = pt.join(-1, A, B) @ pt.join(-2, chunks[0], chunks[1])
        assert_equal_computations([actual], [expected], original=original)

        f = pytensor.function([A, B, Y], block([[A, B, zero]]) @ Y)
        Av, Bv, Yv = _vals(rng, (N, N), (N, N), (3 * N, 3))
        assert_allclose(f(Av, Bv, Yv), np.block([[Av, Bv, np.zeros((N, N))]]) @ Yv)

    def test_permutation_block_becomes_gather(self, rng):
        A = _mat("A", (N, N))
        Y = _mat("Y", (2 * N, 3))
        perm_idx = rng.permutation(N)
        perm = pt.as_tensor(np.eye(N)[perm_idx].astype(floatX))
        original = block([[A, perm]]) @ Y
        (actual,) = _rewrite(original)

        chunks = pt.split(Y, splits_size=pt.stack([A.shape[-1], perm.shape[-1]]), n_splits=2, axis=-2)
        # The permutation block is a row gather of its chunk; A keeps its GEMM.
        expected = pt.take(chunks[1], perm_idx, axis=-2) + A @ chunks[0]
        assert_equal_computations([actual], [expected], original=original)

        f = pytensor.function([A, Y], block([[A, perm]]) @ Y)
        Av, Yv = _vals(rng, (N, N), (2 * N, 3))
        assert_allclose(f(Av, Yv), np.block([[Av, np.eye(N)[perm_idx]]]) @ Yv)

    def test_offset_eye_block_becomes_masked_gather(self, rng):
        A = _mat("A", (N, N))
        Y = _mat("Y", (2 * N, 3))
        shift = pt.eye(N, N, 1)
        original = block([[A, shift]]) @ Y
        (actual,) = _rewrite(original)

        chunks = pt.split(Y, splits_size=pt.stack([A.shape[-1], shift.shape[-1]]), n_splits=2, axis=-2)
        # eye(N, N, 1) maps row i -> i + 1, with the last row out of range (zero-filled).
        gather_idx = np.array([1, 2, 3, 0])
        keep = np.array([1.0, 1.0, 1.0, 0.0], dtype=floatX).reshape((-1, 1))
        gathered = pt.take(chunks[1], gather_idx, axis=-2) * keep
        expected = gathered + A @ chunks[0]
        assert_equal_computations([actual], [expected], original=original)

        f = pytensor.function([A, Y], block([[A, shift]]) @ Y)
        Av, Yv = _vals(rng, (N, N), (2 * N, 3))
        assert_allclose(f(Av, Yv), np.block([[Av, np.eye(N, N, 1)]]) @ Yv)

    def test_right_multiply_block_triangular_is_correct(self, rng):
        # Right-multiply: a selection block on the right stays a dense GEMM (column
        # scatter, not a clean gather). Correctness is the contract checked here.
        A, B, C = (_mat(s, (N, N)) for s in "ABC")
        Y = _mat("Y", (3, 2 * N))
        f = pytensor.function([A, B, C, Y], Y @ block([[A, pt.zeros((N, N))], [B, C]]))
        Av, Bv, Cv, Yv = _vals(rng, (N, N), (N, N), (N, N), (3, 2 * N))
        assert_allclose(f(Av, Bv, Cv, Yv), Yv @ np.block([[Av, np.zeros((N, N))], [Bv, Cv]]))


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
