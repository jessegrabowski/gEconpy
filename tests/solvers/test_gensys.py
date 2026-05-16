import unittest

import numpy as np

from gEconpy.solvers.gensys import (
    build_u_v_d,
    determine_n_unstable,
    split_matrix_on_eigen_stability,
)


class GensysComponentTests(unittest.TestCase):
    def setUp(self):
        # Upper-triangular A, B from MATLAB's qz on two seed=1337 normal matrices. Only
        # the diagonals are used by determine_n_unstable; the off-diagonals persist so
        # the split helper can still reuse the full 5x5 shape below.
        self.div = 1.01

        self.alpha = np.array(
            [
                -2.0123 - 0.5490j,
                -1.7594 + 0.4800j,
                0.9347 - 0.1598j,
                0.9237 + 0.1579j,
                1.0847 + 0.0000j,
            ]
        )
        self.beta = np.array(
            [
                2.2056 + 0.0000j,
                1.9284 + 0.0000j,
                2.4670 + 0.0000j,
                2.4382 + 0.0000j,
                1.3904 + 0.0000j,
            ]
        )

        self.Q = np.eye(5, dtype=complex)

    def test_determine_n_unstable(self):
        div, n_unstable, zxz = determine_n_unstable(self.alpha, self.beta, self.div, realsmall=1e-6)

        self.assertEqual(div, 1.01)
        self.assertEqual(n_unstable, 5)
        self.assertEqual(zxz, False)

    def test_determine_n_unstable_infers_div(self):
        # Initial div = 1.01. The first (alpha, beta) ratio sits in (1+eps, 1.01], so
        # the Sims heuristic halves toward 1: div := (1 + 1.005)/2 = 1.0025. The check
        # `|beta| > div * |alpha|` fires on the same iteration with the newly-shrunk
        # div, so both eigenvalues end up classified as unstable.
        alpha = np.array([1.0 + 0.0j, 1.0 + 0.0j])
        beta = np.array([1.005 + 0.0j, 2.0 + 0.0j])

        div, n_unstable, zxz = determine_n_unstable(alpha, beta, div=None, realsmall=1e-6)

        self.assertAlmostEqual(div, 0.5 * (1 + 1.005))
        self.assertEqual(n_unstable, 2)
        self.assertEqual(zxz, False)

    def test_split_matrix_on_eigen_stability(self):
        Q = np.arange(25, dtype=float).reshape(5, 5)
        n_unstable = 3

        Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)
        self.assertEqual(Q1.shape, (2, 5))
        self.assertEqual(Q2.shape, (3, 5))

        np.testing.assert_allclose(Q1, Q[:2, :])
        np.testing.assert_allclose(Q2, Q[2:, :])

    def test_build_u_v_d(self):
        # TODO: matching MATLAB's svd output conventions is fiddly; real correctness
        # coverage lives in tests/model/test_perturbation.py where gensys is
        # cross-checked against cycle_reduction.
        pass

    def test_gensys(self):
        # TODO: high-level correctness is asserted by
        # tests/model/test_perturbation.py::TestSolvePolicyFunction.
        pass


if __name__ == "__main__":
    unittest.main()
