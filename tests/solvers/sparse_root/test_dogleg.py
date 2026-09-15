import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import SparseDogleg, sparse_root


class TestSparseDoglegSuite(CommonSolverTests):
    solver = SparseDogleg()


class TestSparseDoglegSpecific:
    def test_takes_newton_step_when_inside_region(self, quadratic_system):
        fun, x0, x_true = quadratic_system
        solver = SparseDogleg(delta0=1e6, delta_max=1e8)
        result = sparse_root(fun, x0, solver=solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-8)
        assert result.nit <= 5

    def test_falls_back_to_cauchy_on_singular(self, singular_system):
        fun, x0 = singular_system
        solver = SparseDogleg(delta0=0.1)
        result = sparse_root(fun, x0, solver=solver, progressbar=False, maxiter=500)
        assert result.success
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-6)


class TestDoglegStep:
    def test_newton_inside_region(self):
        solver = SparseDogleg(delta0=100.0)
        jac = sp.csc_matrix(np.eye(2))
        res = np.array([1.0, 2.0])
        p = solver._compute_dogleg_step(jac, res, jac.T @ res, delta=100.0)
        np.testing.assert_allclose(p, -res, atol=1e-10)

    def test_cauchy_outside_region(self):
        solver = SparseDogleg(delta0=0.1)
        jac = sp.csc_matrix(np.eye(2))
        res = np.array([10.0, 20.0])
        grad = jac.T @ res
        p = solver._compute_dogleg_step(jac, res, grad, delta=0.1)
        np.testing.assert_allclose(p, -0.1 * grad / np.linalg.norm(grad), atol=1e-10)

    def test_dogleg_interpolation(self):
        solver = SparseDogleg(delta0=1.5)
        jac = sp.csc_matrix(np.array([[2.0, 0.0], [0.0, 0.5]]))
        res = np.array([1.0, 1.0])
        grad = jac.T @ res
        p = solver._compute_dogleg_step(jac, res, grad, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-8)

        # The step sits on the segment from the Cauchy point to the Newton point, past the Cauchy point.
        cauchy = -(grad @ grad) / ((jac @ grad) @ (jac @ grad)) * grad
        newton = -res / np.array([2.0, 0.5])
        tau = (p - cauchy) / (newton - cauchy)
        np.testing.assert_allclose(tau[0], tau[1], atol=1e-8)
        assert 0.0 < tau[0] < 1.0
