import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy.solvers.sparse_root import GaussNewtonTrustRegion, sparse_root
from gEconpy.solvers.sparse_root.gauss_newton import _steihaug_cg
from tests.solvers.sparse_root._solver_suite import CommonSolverTests


class TestGaussNewtonTrustRegionSuite(CommonSolverTests):
    @property
    def solver(self):
        return GaussNewtonTrustRegion()


class TestGaussNewtonTrustRegionSpecific:
    def test_converges_from_far_with_small_initial_region(self, quadratic_system):
        fun, _, x_true = quadratic_system
        solver = GaussNewtonTrustRegion(delta0=0.01)
        result = sparse_root(fun, np.array([100.0, 100.0]), solver=solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)

    def test_max_cg_iter_reaches_the_subproblem(self, helical_valley):
        fun, x0, _ = helical_valley

        uncapped = sparse_root(fun, x0, solver=GaussNewtonTrustRegion(), progressbar=False, maxiter=500)
        capped = sparse_root(fun, x0, solver=GaussNewtonTrustRegion(max_cg_iter=1), progressbar=False, maxiter=500)

        assert uncapped.success
        assert not capped.success
        assert capped.nit == 500

    def test_cg_rtol_reaches_the_subproblem(self, helical_valley):
        fun, x0, _ = helical_valley

        tight = sparse_root(fun, x0, solver=GaussNewtonTrustRegion(cg_rtol=1e-10), progressbar=False, maxiter=500)
        loose = sparse_root(fun, x0, solver=GaussNewtonTrustRegion(cg_rtol=1e-1), progressbar=False, maxiter=500)

        assert tight.success
        assert loose.success
        assert loose.nit > tight.nit

    @pytest.mark.parametrize("max_cg_iter", [0, -5], ids=["zero", "negative"])
    def test_max_cg_iter_below_one_is_rejected(self, max_cg_iter):
        with pytest.raises(ValueError, match="max_cg_iter must be at least 1"):
            GaussNewtonTrustRegion(max_cg_iter=max_cg_iter)

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

    def test_max_cg_iter_stops_the_iteration_early(self):
        rng = np.random.default_rng(0)
        basis = rng.normal(size=(20, 20))
        normal_matrix = basis.T @ basis + np.eye(20)
        hessian = sp.csc_matrix(normal_matrix)
        grad = rng.normal(size=20)
        exact = np.linalg.solve(normal_matrix, -grad)

        uncapped = _steihaug_cg(hessian, grad, delta=1e6)
        capped = _steihaug_cg(hessian, grad, delta=1e6, max_cg_iter=3)

        np.testing.assert_allclose(uncapped, exact, rtol=1e-6)
        assert np.linalg.norm(capped - exact) > 0.1 * np.linalg.norm(exact)

    def test_a_looser_rtol_leaves_a_larger_subproblem_residual(self):
        rng = np.random.default_rng(0)
        basis = rng.normal(size=(20, 20))
        hessian = sp.csc_matrix(basis.T @ basis + np.eye(20))
        grad = rng.normal(size=20)

        tight = _steihaug_cg(hessian, grad, delta=1e6, rtol=1e-12)
        loose = _steihaug_cg(hessian, grad, delta=1e6, rtol=1e-2)

        assert np.linalg.norm(hessian @ loose + grad) > np.linalg.norm(hessian @ tight + grad)

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
