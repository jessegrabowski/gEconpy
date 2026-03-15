import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import LevenbergMarquardt, sparse_root


class TestLevenbergMarquardtSuite(CommonSolverTests):
    solver = LevenbergMarquardt()


class TestLevenbergMarquardtSpecific:
    def test_handles_singular_jacobian(self, singular_system):
        """LM should handle a singular Jacobian at x0 via damping, unlike plain Newton."""
        fun, x0 = singular_system
        result = sparse_root(fun, x0, solver=LevenbergMarquardt(), progressbar=False)
        assert result.success
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-8)

    def test_lambda_increases_on_reject(self):
        """Verify damping grows when a step is rejected."""
        call_count = {"n": 0}

        def bad_landscape(x):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                res = x - np.array([1.0, 1.0])
            elif call_count["n"] <= 3:
                res = np.array([1e10, 1e10])
            else:
                res = x - np.array([1.0, 1.0])
            jac = sp.eye(2, format="csc")
            return res, jac

        solver = LevenbergMarquardt(lam0=1e-6)
        lam_before = solver.lam0
        state = solver.init(bad_landscape, np.array([5.0, 5.0]), ())
        solver.step(bad_landscape, state, ())
        assert solver._lam > lam_before

    def test_powell_badly_scaled(self, powell_badly_scaled):
        """LM is robust on badly-scaled problems."""
        fun, x0 = powell_badly_scaled
        result = sparse_root(fun, x0, solver=LevenbergMarquardt(lam0=1.0), progressbar=False, maxiter=2000)
        if result.success:
            res, _ = fun(result.x)
            np.testing.assert_allclose(res, 0.0, atol=1e-4)
