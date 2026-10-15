import subprocess
import sys

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.rewriting.basic import out2in
from pytensor.tensor.linalg import block_diag

from gEconpy.pytensorf.block import block
from gEconpy.pytensorf.block_rewrites import (
    local_nested_join_to_block_diagonal,
    local_split_of_join,
    local_transpose_of_join,
)
from tests._resources.rewrite_tester import RewriteTester

N = 4
floatX = pytensor.config.floatX

# Only the block rewrites run here, so the rewritten graph is exactly what they emit and can be matched op-for-op.
# The full canonicalize and stabilize databases stay out.
_BLOCK_REWRITES = out2in(
    local_transpose_of_join,
    local_nested_join_to_block_diagonal,
    local_split_of_join,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def block_rewrite_tester(inputs, outputs):
    return RewriteTester(inputs, outputs, include=None, custom_rewrite=_BLOCK_REWRITES)


def _mat(name, shape):
    return pt.matrix(name, shape=shape)


def _vals(rng, *shapes):
    return [rng.normal(size=s).astype(floatX) for s in shapes]


class TestTransposeOfJoin:
    def test_transpose_pushes_to_leaves(self, rng):
        A, B = (_mat(s, (N, N)) for s in "AB")
        tester = block_rewrite_tester([A, B], [block([[A, B]]).mT])

        tester.assert_graph(pt.join(0, A.mT, B.mT))
        tester.assert_eval(*_vals(rng, (N, N), (N, N)))


class TestNestedJoinToBlockDiagonal:
    def test_zero_off_diagonal_becomes_block_diag(self, rng):
        A, B = (_mat(s, (N, N)) for s in "AB")
        zero = pt.zeros((N, N))
        tester = block_rewrite_tester([A, B], [block([[A, zero], [zero, B]])])

        tester.assert_graph(block_diag(A, B))
        tester.assert_eval(*_vals(rng, (N, N), (N, N)))

    @pytest.mark.parametrize(
        "build_grid",
        [
            lambda A, B, C, z: block([[A, z], [C, B]]),
            lambda A, B, C, z: block([[A, z, z], [B, z, C]]),
        ],
        ids=["dense_off_diagonal", "non_square_grid"],
    )
    def test_grid_that_is_not_block_diagonal_is_unchanged(self, build_grid):
        A, B, C = (_mat(s, (N, N)) for s in "ABC")
        original = build_grid(A, B, C, pt.zeros((N, N)))
        block_rewrite_tester([A, B, C], [original]).assert_graph(original)


class TestSplitOfJoin:
    def test_split_undoes_join_on_same_axis(self):
        A, B = (_mat(s, (N, N)) for s in "AB")
        tester = block_rewrite_tester([A, B], pt.split(pt.join(-1, A, B), splits_size=[N, N], n_splits=2, axis=-1))

        # The split exactly reverses the join, so its outputs are the join's inputs.
        tester.assert_graph(A, B)

    def test_split_distributes_through_join_on_different_axis(self, rng):
        A = _mat("A", (3, N))
        B = _mat("B", (4, N))
        tester = block_rewrite_tester([A, B], pt.split(pt.join(-2, A, B), splits_size=[2, N - 2], n_splits=2, axis=-1))

        # The rewrite normalizes axes: split axis -1 -> 1, join axis -2 -> 0.
        a_parts = pt.split(A, splits_size=[2, N - 2], n_splits=2, axis=1)
        b_parts = pt.split(B, splits_size=[2, N - 2], n_splits=2, axis=1)
        tester.assert_graph(pt.join(0, a_parts[0], b_parts[0]), pt.join(0, a_parts[1], b_parts[1]))
        tester.assert_eval(*_vals(rng, (3, N), (4, N)))


def test_reimport_replaces_registered_rewrites():
    """Evicting gEconpy from ``sys.modules`` and importing again must not raise on the duplicate rewrite names."""
    script = (
        "import sys\n"
        "from pytensor import compile\n"
        "import gEconpy.pytensorf.block_rewrites\n"
        "db = compile.optdb['canonicalize']\n"
        "first = db['local_split_of_join']\n"
        "for name in [k for k in sys.modules if k.startswith('gEconpy')]:\n"
        "    del sys.modules[name]\n"
        "import gEconpy.pytensorf.block_rewrites\n"
        "assert db['local_split_of_join'] is not first\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
