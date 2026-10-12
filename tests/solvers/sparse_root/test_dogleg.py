import numpy as np
import pytest
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve

from gEconpy.solvers.sparse_root import SparseDogleg, sparse_root
from gEconpy.solvers.sparse_root.dogleg import _dogleg_geometry, _dogleg_step, _DoglegGeometry
from tests.solvers.sparse_root._solver_suite import CommonSolverTests


class TestSparseDoglegSuite(CommonSolverTests):
    @property
    def solver(self):
        return SparseDogleg()


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
    @staticmethod
    def _geometry(jac, res):
        return _dogleg_geometry(jac, res, jac.T @ res, spsolve)

    def test_newton_inside_region(self):
        jac = sp.csc_matrix(np.eye(2))
        res = np.array([1.0, 2.0])
        p = _dogleg_step(self._geometry(jac, res), delta=100.0)
        np.testing.assert_allclose(p, -res, atol=1e-10)

    def test_cauchy_outside_region(self):
        jac = sp.csc_matrix(np.eye(2))
        res = np.array([10.0, 20.0])
        grad = jac.T @ res
        p = _dogleg_step(self._geometry(jac, res), delta=0.1)
        np.testing.assert_allclose(p, -0.1 * grad / np.linalg.norm(grad), atol=1e-10)

    def test_dogleg_interpolation(self):
        jac = sp.csc_matrix(np.array([[2.0, 0.0], [0.0, 0.5]]))
        res = np.array([1.0, 1.0])
        grad = jac.T @ res
        p = _dogleg_step(self._geometry(jac, res), delta=1.0)
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-8)

        # The step sits on the segment from the Cauchy point to the Newton point, past the Cauchy point.
        cauchy = -(grad @ grad) / ((jac @ grad) @ (jac @ grad)) * grad
        newton = -res / np.array([2.0, 0.5])
        tau = (p - cauchy) / (newton - cauchy)
        np.testing.assert_allclose(tau[0], tau[1], atol=1e-8)
        assert 0.0 < tau[0] < 1.0

    @pytest.mark.parametrize(
        ("delta", "expected_norm"),
        [(10.0, 5.0), (5.0, 5.0), (3.0, 3.0), (0.5, 0.5)],
        ids=["newton_interior", "newton_on_boundary", "interpolated", "cauchy_truncated"],
    )
    def test_the_radius_selects_the_step(self, delta, expected_norm):
        geometry = _DoglegGeometry(
            gradient=np.array([1.0, 0.0]),
            gradient_norm=1.0,
            curvature=1.0,
            cauchy=np.array([-1.0, 0.0]),
            cauchy_norm=1.0,
            newton=np.array([-5.0, 0.0]),
            newton_norm=5.0,
        )

        p = _dogleg_step(geometry, delta=delta)

        np.testing.assert_allclose(np.linalg.norm(p), expected_norm, atol=1e-10)
