import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import GaussNewtonTrustRegion, sparse_root
from gEconpy.solvers.sparse_root.gauss_newton import _steihaug_cg


class TestGaussNewtonTrustRegionSuite(CommonSolverTests):
    solver = GaussNewtonTrustRegion()


class TestGaussNewtonTrustRegionSpecific:
    def test_trust_region_shrinks_on_reject(self):
        """Verify solver converges even from far away with a small initial trust region."""

        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")

        solver = GaussNewtonTrustRegion(delta0=0.01)
        result = sparse_root(fun, np.array([100.0, 100.0]), solver=solver, progressbar=False)
        assert result.success

    def test_trust_region_radius_adapts(self, quadratic_system):
        """Trust region should grow when steps are very successful."""
        fun, x0, _ = quadratic_system
        solver = GaussNewtonTrustRegion(delta0=0.001)
        state = solver.init(fun, x0, ())
        initial_delta = solver._delta

        for _ in range(5):
            state, info = solver.step(fun, state, ())
            if not info.accepted:
                break

        assert solver._delta >= initial_delta

    def test_rosenbrock(self, rosenbrock_root):
        """Trust region is good on the narrow Rosenbrock valley."""
        fun, x0, x_true = rosenbrock_root
        result = sparse_root(fun, x0, solver=GaussNewtonTrustRegion(), progressbar=False, maxiter=500)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-4)


class TestStieahaugCG:
    """Direct tests of the Steihaug-CG subproblem solver.

    Adapted from scipy.optimize.tests.test_trustregion_krylov.
    """

    def test_interior_solution(self):
        """When the unconstrained minimum is inside the trust region, find it exactly."""
        H = sp.csc_matrix(np.eye(3))
        g = np.array([1.0, 2.0, 3.0])
        p = _steihaug_cg(H, g, delta=100.0)
        np.testing.assert_allclose(p, -g, atol=1e-8)
        assert np.linalg.norm(p) < 100.0

    def test_boundary_solution(self):
        """When the unconstrained minimum is outside, the step should hit the boundary."""
        H = sp.csc_matrix(np.eye(3))
        g = np.array([1.0, 2.0, 3.0])
        p = _steihaug_cg(H, g, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-8)

    def test_negative_curvature(self):
        """Negative curvature should cause step to trust region boundary."""
        H = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, -2.0]]))
        g = np.array([0.0, 1.0])
        p = _steihaug_cg(H, g, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-8)

    def test_zero_gradient(self):
        """Zero gradient should return zero step."""
        H = sp.csc_matrix(np.eye(3))
        g = np.zeros(3)
        p = _steihaug_cg(H, g, delta=1.0)
        np.testing.assert_allclose(p, 0.0, atol=1e-15)

    def test_scipy_krylov_easy_case(self):
        """Reproduce scipy test_trustregion_krylov easy case via our Steihaug-CG."""
        H = sp.csc_matrix(np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]]))
        g = np.array([5.0, 0.0, 4.0])
        p = _steihaug_cg(H, g, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-6)
        model_reduction = -(g @ p + 0.5 * p @ (H @ p))
        assert model_reduction > 0
