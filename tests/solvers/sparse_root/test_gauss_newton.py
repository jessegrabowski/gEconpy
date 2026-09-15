import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import GaussNewtonTrustRegion, sparse_root
from gEconpy.solvers.sparse_root.gauss_newton import _steihaug_cg


class TestGaussNewtonTrustRegionSuite(CommonSolverTests):
    solver = GaussNewtonTrustRegion()


class TestGaussNewtonTrustRegionSpecific:
    def test_converges_from_far_with_small_initial_region(self, quadratic_system):
        fun, _, x_true = quadratic_system
        solver = GaussNewtonTrustRegion(delta0=0.01)
        result = sparse_root(fun, np.array([100.0, 100.0]), solver=solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)

    def test_trust_region_radius_grows_on_good_steps(self, quadratic_system):
        fun, x0, _ = quadratic_system
        solver = GaussNewtonTrustRegion(delta0=0.001)
        state = solver.init(fun, x0, ())

        for _ in range(5):
            state, info = solver.step(fun, state, ())
            assert info.accepted

        assert solver._delta > solver.delta0


class TestSteihaugCG:
    def test_interior_solution(self):
        hessian = sp.csc_matrix(np.eye(3))
        grad = np.array([1.0, 2.0, 3.0])
        p = _steihaug_cg(hessian, grad, delta=100.0)
        np.testing.assert_allclose(p, -grad, atol=1e-8)

    def test_boundary_solution(self):
        hessian = sp.csc_matrix(np.eye(3))
        grad = np.array([1.0, 2.0, 3.0])
        p = _steihaug_cg(hessian, grad, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-8)

    def test_negative_curvature_reaches_boundary(self):
        hessian = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, -2.0]]))
        grad = np.array([0.0, 1.0])
        p = _steihaug_cg(hessian, grad, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-8)

    def test_zero_gradient(self):
        hessian = sp.csc_matrix(np.eye(3))
        p = _steihaug_cg(hessian, np.zeros(3), delta=1.0)
        np.testing.assert_allclose(p, 0.0, atol=1e-15)

    def test_scipy_krylov_easy_case(self):
        """Easy case from scipy's test_trustregion_krylov."""
        hessian = sp.csc_matrix(np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]]))
        grad = np.array([5.0, 0.0, 4.0])
        p = _steihaug_cg(hessian, grad, delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-6)
        model_reduction = -(grad @ p + 0.5 * p @ (hessian @ p))
        assert model_reduction > 0
