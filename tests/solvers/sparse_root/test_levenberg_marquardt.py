import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import LevenbergMarquardt, sparse_root


class TestLevenbergMarquardtSuite(CommonSolverTests):
    solver = LevenbergMarquardt()


class TestLevenbergMarquardtSpecific:
    def test_handles_singular_jacobian(self, singular_system):
        fun, x0 = singular_system
        result = sparse_root(fun, x0, solver=LevenbergMarquardt(), progressbar=False)
        assert result.success
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-8)

    def test_lambda_increases_on_reject(self):
        n_calls = 0

        def bad_landscape(x):
            nonlocal n_calls
            n_calls += 1
            res = np.array([1e10, 1e10]) if 1 < n_calls <= 3 else x - np.array([1.0, 1.0])
            return res, sp.eye(2, format="csc")

        solver = LevenbergMarquardt(lam0=1e-6)
        state = solver.init(bad_landscape, np.array([5.0, 5.0]), ())
        solver.step(bad_landscape, state, ())
        assert solver._lam > solver.lam0

    def test_powell_badly_scaled(self, powell_badly_scaled):
        fun, x0 = powell_badly_scaled
        result = sparse_root(fun, x0, solver=LevenbergMarquardt(lam0=1.0), progressbar=False, maxiter=2000)
        assert result.success
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-4)
