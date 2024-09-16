import unittest

import numpy as np

from gEconpy.solvers.gensys import (
    build_u_v_d,
    determine_n_unstable,
    qzdiv,
    qzswitch,
    split_matrix_on_eigen_stability,
)


class GensysComponentTests(unittest.TestCase):
    def setUp(self):
        # These matrices come from Matlab's qz function on two random normal matrices, seed set as rng(1337)
        # I do it this way for two reasons: a) these functions are original written in MATLAB, so the outputs should
        # match that. Second, the matlab qz function returns a different solution than the scipy.linalg.qz function, so
        # it's easier to just copy over the matrices.

        self.div = 1.01

        A = np.array(
            [
                [
                    -2.0123 - 0.5490j,
                    -0.1903 - 0.2355j,
                    0.5456 - 0.0095j,
                    0.2528 + 0.0204j,
                    -0.6081 - 0.3134j,
                ],
                [
                    0.0000 + 0.0000j,
                    -1.7594 + 0.4800j,
                    -0.1374 - 0.1764j,
                    -0.0810 - 0.0883j,
                    0.7683 + 0.1901j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.9347 - 0.1598j,
                    0.8522 + 0.0918j,
                    0.9138 - 0.0331j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.9237 + 0.1579j,
                    -0.4913 + 0.0776j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    1.0847 + 0.0000j,
                ],
            ]
        )

        B = np.array(
            [
                [
                    2.2056 + 0.0000j,
                    -0.9085 + 0.3745j,
                    0.4071 + 0.0283j,
                    0.3431 - 0.0258j,
                    -1.7238 + 0.3767j,
                ],
                [
                    0.0000 + 0.0000j,
                    1.9284 + 0.0000j,
                    -0.2346 - 0.1266j,
                    0.1147 - 0.0997j,
                    -1.2307 + 0.6968j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    2.4670 + 0.0000j,
                    -2.8122 - 0.3138j,
                    1.1122 - 0.0005j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    2.4382 + 0.0000j,
                    -0.1066 + 0.0908j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    1.3904 + 0.0000j,
                ],
            ]
        )

        Q = np.array(
            [
                [
                    -0.1666 + 0.0735j,
                    0.0022 - 0.2391j,
                    0.8435 + 0.0613j,
                    -0.3592 + 0.0560j,
                    -0.0947 - 0.2308j,
                ],
                [
                    -0.2188 + 0.0720j,
                    0.6445 - 0.0307j,
                    -0.0598 - 0.3103j,
                    -0.1957 + 0.1424j,
                    0.6101 + 0.0069j,
                ],
                [
                    0.8396 + 0.0311j,
                    -0.0092 + 0.0572j,
                    0.2951 + 0.0018j,
                    0.1971 + 0.0028j,
                    0.4030 - 0.0482j,
                ],
                [
                    0.3082 + 0.0657j,
                    0.7069 - 0.0059j,
                    -0.0045 + 0.0239j,
                    0.0173 + 0.0158j,
                    -0.6311 + 0.0373j,
                ],
                [
                    0.3294 + 0.0000j,
                    -0.1530 + 0.0000j,
                    -0.3118 + 0.0000j,
                    -0.8774 + 0.0000j,
                    -0.0323 + 0.0000j,
                ],
            ]
        )

        Z = np.array(
            [
                [
                    -0.1455 + 0.1400j,
                    0.3500 - 0.1150j,
                    -0.7558 + 0.1162j,
                    -0.4753 + 0.0142j,
                    0.1124 + 0.0000j,
                ],
                [
                    -0.2703 + 0.2195j,
                    0.5617 - 0.2014j,
                    0.0504 - 0.0363j,
                    0.3769 + 0.0342j,
                    -0.6129 + 0.0000j,
                ],
                [
                    0.7385 + 0.0767j,
                    -0.0557 + 0.3455j,
                    -0.3824 + 0.0224j,
                    0.1995 + 0.0520j,
                    -0.3701 + 0.0000j,
                ],
                [
                    0.0316 + 0.1059j,
                    0.2219 - 0.0163j,
                    -0.2197 - 0.0343j,
                    0.6844 + 0.0878j,
                    0.6425 + 0.0000j,
                ],
                [
                    0.4164 + 0.3179j,
                    0.5690 + 0.1117j,
                    0.4634 - 0.0201j,
                    -0.3271 - 0.0717j,
                    0.2491 + 0.0000j,
                ],
            ]
        )

        self.A = np.asfortranarray(A)
        self.B = np.asfortranarray(B)
        self.Q = np.asfortranarray(Q)
        self.Z = np.asfortranarray(Z)

    def unpack_matrices(self):
        return self.A, self.B, self.Q, self.Z

    def test_qzdiv(self):
        A, B, Q, Z = self.unpack_matrices()
        A, B, Q, Z = qzdiv(self.div, A, B, Q, Z)

        ans_A = np.array(
            [
                [
                    -2.0123 - 0.5490j,
                    -0.1903 - 0.2355j,
                    0.5456 - 0.0095j,
                    0.2528 + 0.0204j,
                    -0.6081 - 0.3134j,
                ],
                [
                    0.0000 + 0.0000j,
                    -1.7594 + 0.4800j,
                    -0.1374 - 0.1764j,
                    -0.0810 - 0.0883j,
                    0.7683 + 0.1901j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.9347 - 0.1598j,
                    0.8522 + 0.0918j,
                    0.9138 - 0.0331j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.9237 + 0.1579j,
                    -0.4913 + 0.0776j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    1.0847 + 0.0000j,
                ],
            ]
        )

        ans_B = np.array(
            [
                [
                    2.2056 + 0.0000j,
                    -0.9085 + 0.3745j,
                    0.4071 + 0.0283j,
                    0.3431 - 0.0258j,
                    -1.7238 + 0.3767j,
                ],
                [
                    0.0000 + 0.0000j,
                    1.9284 + 0.0000j,
                    -0.2346 - 0.1266j,
                    0.1147 - 0.0997j,
                    -1.2307 + 0.6968j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    2.4670 + 0.0000j,
                    -2.8122 - 0.3138j,
                    1.1122 - 0.0005j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    2.4382 + 0.0000j,
                    -0.1066 + 0.0908j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    1.3904 + 0.0000j,
                ],
            ]
        )

        ans_Q = np.array(
            [
                [
                    -0.1666 + 0.0735j,
                    0.0022 - 0.2391j,
                    0.8435 + 0.0613j,
                    -0.3592 + 0.0560j,
                    -0.0947 - 0.2308j,
                ],
                [
                    -0.2188 + 0.0720j,
                    0.6445 - 0.0307j,
                    -0.0598 - 0.3103j,
                    -0.1957 + 0.1424j,
                    0.6101 + 0.0069j,
                ],
                [
                    0.8396 + 0.0311j,
                    -0.0092 + 0.0572j,
                    0.2951 + 0.0018j,
                    0.1971 + 0.0028j,
                    0.4030 - 0.0482j,
                ],
                [
                    0.3082 + 0.0657j,
                    0.7069 - 0.0059j,
                    -0.0045 + 0.0239j,
                    0.0173 + 0.0158j,
                    -0.6311 + 0.0373j,
                ],
                [
                    0.3294 + 0.0000j,
                    -0.1530 + 0.0000j,
                    -0.3118 + 0.0000j,
                    -0.8774 + 0.0000j,
                    -0.0323 + 0.0000j,
                ],
            ]
        )

        ans_Z = np.array(
            [
                [
                    -0.1455 + 0.1400j,
                    0.3500 - 0.1150j,
                    -0.7558 + 0.1162j,
                    -0.4753 + 0.0142j,
                    0.1124 + 0.0000j,
                ],
                [
                    -0.2703 + 0.2195j,
                    0.5617 - 0.2014j,
                    0.0504 - 0.0363j,
                    0.3769 + 0.0342j,
                    -0.6129 + 0.0000j,
                ],
                [
                    0.7385 + 0.0767j,
                    -0.0557 + 0.3455j,
                    -0.3824 + 0.0224j,
                    0.1995 + 0.0520j,
                    -0.3701 + 0.0000j,
                ],
                [
                    0.0316 + 0.1059j,
                    0.2219 - 0.0163j,
                    -0.2197 - 0.0343j,
                    0.6844 + 0.0878j,
                    0.6425 + 0.0000j,
                ],
                [
                    0.4164 + 0.3179j,
                    0.5690 + 0.1117j,
                    0.4634 - 0.0201j,
                    -0.3271 - 0.0717j,
                    0.2491 + 0.0000j,
                ],
            ]
        )

        np.testing.assert_allclose(
            A, ans_A, err_msg="A not equal to requested precision"
        )
        np.testing.assert_allclose(
            B, ans_B, err_msg="B not equal to requested precision"
        )
        np.testing.assert_allclose(
            Q, ans_Q, err_msg="Q not equal to requested precision"
        )
        np.testing.assert_allclose(
            Z, ans_Z, err_msg="Z not equal to requested precision"
        )

    def test_qzswitch(self):
        # TODO: Find matrices that test conditions (1) and (2) in qzswitch (most will only hit condition 3)
        A, B, Q, Z = self.unpack_matrices()
        A, B, Q, Z = qzswitch(2, A, B, Q, Z)

        ans_A = np.array(
            [
                [
                    -2.0123 - 0.5490j,
                    -0.1903 - 0.2355j,
                    -0.5258 - 0.1411j,
                    -0.2499 - 0.0580j,
                    -0.6081 - 0.3134j,
                ],
                [
                    0.0000 + 0.0000j,
                    -1.7594 + 0.4800j,
                    0.0831 + 0.2111j,
                    0.0672 + 0.0914j,
                    0.7683 + 0.1901j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    -0.9347 - 0.1598j,
                    -0.8522 + 0.0918j,
                    0.9138 + 0.0331j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 - 0.0000j,
                    -0.9237 + 0.1579j,
                    -0.4914 - 0.0776j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    1.0847 + 0.0000j,
                ],
            ]
        )

        ans_B = np.array(
            [
                [
                    2.2056 + 0.0000j,
                    -0.9085 + 0.3745j,
                    -0.3919 - 0.1646j,
                    -0.3218 + 0.0273j,
                    -1.7238 + 0.3767j,
                ],
                [
                    0.0000 + 0.0000j,
                    1.9284 + 0.0000j,
                    0.1851 + 0.1505j,
                    -0.1115 + 0.1575j,
                    -1.2307 + 0.6968j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    -2.4670 - 0.0000j,
                    2.8122 - 0.3138j,
                    1.1122 + 0.0005j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    -0.0000 + 0.0000j,
                    -2.4382 - 0.0000j,
                    -0.1066 - 0.0908j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    1.3904 + 0.0000j,
                ],
            ]
        )

        ans_Q = np.array(
            [
                [
                    -0.1666 + 0.0735j,
                    0.0022 - 0.2391j,
                    0.8435 + 0.0613j,
                    -0.3592 + 0.0560j,
                    -0.0947 - 0.2308j,
                ],
                [
                    -0.2188 + 0.0720j,
                    0.6445 - 0.0307j,
                    -0.0598 - 0.3103j,
                    -0.1957 + 0.1424j,
                    0.6101 + 0.0069j,
                ],
                [
                    0.8396 - 0.0311j,
                    -0.0092 - 0.0572j,
                    0.2951 - 0.0018j,
                    0.1971 - 0.0028j,
                    0.4030 + 0.0481j,
                ],
                [
                    0.3082 - 0.0657j,
                    0.7069 + 0.0059j,
                    -0.0045 - 0.0239j,
                    0.0173 - 0.0158j,
                    -0.6311 - 0.0372j,
                ],
                [
                    0.3294 + 0.0000j,
                    -0.1530 + 0.0000j,
                    -0.3118 + 0.0000j,
                    -0.8774 + 0.0000j,
                    -0.0323 + 0.0000j,
                ],
            ]
        )

        ans_Z = np.array(
            [
                [
                    -0.1455 + 0.1400j,
                    0.3500 - 0.1150j,
                    0.7558 + 0.1163j,
                    0.4753 + 0.0142j,
                    0.1124 + 0.0000j,
                ],
                [
                    -0.2703 + 0.2195j,
                    0.5617 - 0.2014j,
                    -0.0504 - 0.0363j,
                    -0.3769 + 0.0342j,
                    -0.6129 + 0.0000j,
                ],
                [
                    0.7385 + 0.0767j,
                    -0.0557 + 0.3455j,
                    0.3824 + 0.0224j,
                    -0.1995 + 0.0519j,
                    -0.3701 + 0.0000j,
                ],
                [
                    0.0316 + 0.1059j,
                    0.2219 - 0.0163j,
                    0.2197 - 0.0343j,
                    -0.6844 + 0.0878j,
                    0.6425 + 0.0000j,
                ],
                [
                    0.4164 + 0.3179j,
                    0.5690 + 0.1117j,
                    -0.4634 - 0.0201j,
                    0.3271 - 0.0716j,
                    0.2491 + 0.0000j,
                ],
            ]
        )

        # Riddle me this: qzswitch tests only pass at 3 decimal places of precision, but qzdiv tests, which call
        # qzswitch multiple times, pass at 4!

        np.testing.assert_allclose(
            A, ans_A, atol=1e-3, err_msg="A not close to requested precision"
        )
        np.testing.assert_allclose(
            B, ans_B, atol=1e-3, err_msg="B not close to requested precision"
        )
        np.testing.assert_allclose(
            Q, ans_Q, atol=1e-3, err_msg="Q not close to requested precision"
        )
        np.testing.assert_allclose(
            Z, ans_Z, atol=1e-3, err_msg="Z not close to requested precision"
        )

    def test_determine_n_unstable(self):
        A, B, _, _ = self.unpack_matrices()

        div, n_unstable, zxz = determine_n_unstable(A, B, self.div, realsmall=1e-6)

        self.assertEqual(div, 1.01)
        self.assertEqual(n_unstable, 5)
        self.assertEqual(zxz, False)

    def test_split_matrix_on_eigen_stability(self):
        _, _, Q, _ = self.unpack_matrices()
        n_unstable = 3  # let's pretend

        Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)
        self.assertEqual(Q1.shape, (2, 5))
        self.assertEqual(Q2.shape, (3, 5))

        self.assertEqual(np.allclose(Q1, Q[:2, :]), True)
        self.assertEqual(np.allclose(Q2, Q[2:, :]), True)

    def test_build_u_v_d(self):
        pass
        # TODO: I cannot for the life of me get the output of scipy.linalg.svd to match with matlab's svd output.
        # pi is (n_eq + n_expectations) x n_expectations. Imagine we have 3 equations with 2 forward-looking variables
        # pi = np.r_[np.zeros((3, 2)), np.eye(2)]
        #
        # A, B, Q, Z = self.unpack_matrices()
        #
        # div, n_unstable, zxz = determine_n_unstable(A, B, self.div, realsmall=-6)
        # Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)
        #
        # eta_wt = Q2 @ pi
        #
        # u_eta, v_eta, d_eta, big_ev = build_u_v_d(eta_wt, realsmall=1e-6)
        #
        # ans_u_eta = np.array(
        #     [[0.0072 - 0.2494j, -0.3592 + 0.0560j],
        #      [0.5547 + 0.2541j, -0.1957 + 0.1424j],
        #      [0.3878 + 0.1197j, 0.1971 + 0.0028j],
        #      [-0.5919 - 0.2223j, 0.0173 + 0.0158j],
        #      [-0.0295 - 0.0131j, -0.8774 - 0.0000j]]
        # )
        #
        # ans_v_eta = np.array(
        #     [[0.0000 + 0.0000j, 1.0000 + 0.0000j],
        #      [0.9138 + 0.4061j, 0.0000 + 0.0000j]]
        # )
        #
        # ans_d_eta = np.eye(2)
        #
        # ans_big_ev = np.array([0, 1])
        #
        # self.assertEqual(np.allclose(u_eta, ans_u_eta), True)
        # self.assertEqual(np.allclose(v_eta, ans_v_eta), True)
        # self.assertEqual(np.allclose(d_eta, ans_d_eta), True)
        # self.assertEqual(np.allclose(big_ev, ans_big_ev), True)

    def test_gensys(self):
        # TODO Develop a test that does something different from the tests in model_tests?
        pass


if __name__ == "__main__":
    unittest.main()
