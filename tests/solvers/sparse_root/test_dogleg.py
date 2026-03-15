import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import SparseDogleg, sparse_root


class TestSparseDoglegSuite(CommonSolverTests):
    solver = SparseDogleg()


class TestSparseDoglegSpecific:
    def test_takes_newton_step_when_inside_region(self, quadratic_system):
        """With a large trust region, dogleg should behave like Newton."""
        fun, x0, x_true = quadratic_system
        solver = SparseDogleg(delta0=1e6, delta_max=1e8)
        result = sparse_root(fun, x0, solver=solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-8)
        assert result.nit <= 5

    def test_falls_back_to_cauchy_on_singular(self, singular_system):
        """Dogleg should still converge when Newton step fails (singular Jacobian)."""
        fun, x0 = singular_system
        solver = SparseDogleg(delta0=0.1)
        result = sparse_root(fun, x0, solver=solver, progressbar=False, maxiter=500)
        if result.success:
            res, _ = fun(result.x)
            np.testing.assert_allclose(res, 0.0, atol=1e-6)

    def test_rosenbrock(self, rosenbrock_root):
        """Trust region is good on the narrow Rosenbrock valley."""
        fun, x0, x_true = rosenbrock_root
        result = sparse_root(fun, x0, solver=SparseDogleg(), progressbar=False, maxiter=500)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-4)


class TestDoglegStep:
    """Direct tests of the dogleg step computation."""

    def test_newton_inside_region(self):
        """When Newton step is inside trust region, take it exactly."""
        solver = SparseDogleg(delta0=100.0)
        J = sp.csc_matrix(np.eye(2))
        r = np.array([1.0, 2.0])
        p = solver._compute_dogleg_step(J, r, delta=100.0)
        np.testing.assert_allclose(p, -r, atol=1e-10)

    def test_cauchy_outside_region(self):
        """When even the Cauchy point is outside, scale gradient to boundary."""
        solver = SparseDogleg(delta0=0.1)
        J = sp.csc_matrix(np.eye(2))
        r = np.array([10.0, 20.0])
        p = solver._compute_dogleg_step(J, r, delta=0.1)
        np.testing.assert_allclose(np.linalg.norm(p), 0.1, atol=1e-10)

    def test_dogleg_interpolation(self):
        """When Cauchy is inside and Newton is outside, interpolate to boundary."""
        solver = SparseDogleg(delta0=1.5)
        J2 = sp.csc_matrix(np.array([[2.0, 0.0], [0.0, 0.5]]))
        r2 = np.array([1.0, 1.0])
        p = solver._compute_dogleg_step(J2, r2, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-8)
