import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.direction import ChordDirection, KrylovDirection, NewtonDirection


class TestNewtonDirection:
    def test_fallback_on_singular_jacobian(self):
        direction = NewtonDirection()
        jac = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])
        proposal = direction.compute(np.zeros(2), np.array([1.0, 2.0]), jac)
        assert "fallback" in proposal.kind

    def test_fallback_on_nan_from_solver(self):
        direction = NewtonDirection(linear_solver=lambda _A, _b: np.array([np.nan, np.nan]))
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.eye(2, format="csc"))
        assert "fallback" in proposal.kind

    def test_ensures_descent_direction(self):
        direction = NewtonDirection()
        jac = sp.csc_matrix([[-1.0, 0.0], [0.0, -1.0]])
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), jac)
        assert proposal.slope < 0


class TestChordDirection:
    def test_cache_refreshes_every_recompute_every_calls(self):
        direction = ChordDirection(recompute_every=3)
        direction.reset()

        jac_first = sp.eye(2, format="csc")
        jac_second = 2.0 * sp.eye(2, format="csc")
        res = np.array([1.0, 1.0])
        x = np.zeros(2)

        direction.compute(x, res, jac_first)
        direction.compute(x, res, jac_second)
        direction.compute(x, res, jac_second)
        np.testing.assert_array_equal(direction._cached_jac.data, jac_first.data)

        direction.compute(x, res, jac_second)
        np.testing.assert_array_equal(direction._cached_jac.data, jac_second.data)


class TestKrylovDirection:
    def test_eisenstat_walker_tightens(self, broyden_system):
        fun, x0 = broyden_system
        direction = KrylovDirection(eta_max=0.9, eisenstat_walker=True)
        direction.reset()
        eta_initial = direction._eta

        x = x0.copy()
        res, jac = fun(x)
        for _ in range(10):
            proposal = direction.compute(x, res, jac)
            x = x + proposal.direction
            res, jac = fun(x)

        assert direction._eta < eta_initial

    def test_fallback_on_bad_krylov(self):
        direction = KrylovDirection(krylov_method="gmres")
        direction.reset()
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.csc_matrix((2, 2)))
        assert "fallback" in proposal.kind

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown Krylov method"):
            KrylovDirection(krylov_method="cg")
