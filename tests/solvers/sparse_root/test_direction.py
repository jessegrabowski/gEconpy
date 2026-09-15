import numpy as np
import pytest
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve

from gEconpy.solvers.sparse_root.direction import ChordDirection, KrylovDirection, NewtonDirection


class TestNewtonDirection:
    @pytest.mark.parametrize(
        ("direction", "kind"),
        [(NewtonDirection(), "gradient_fallback"), (ChordDirection(), "chord_gradient_fallback")],
        ids=["newton", "chord"],
    )
    def test_fallback_on_singular_jacobian(self, direction, kind):
        jac = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])
        res = np.array([1.0, 2.0])
        proposal = direction.compute(np.zeros(2), res, jac)

        assert proposal.kind == kind
        np.testing.assert_allclose(proposal.direction, -(jac.T @ res))
        assert proposal.slope < 0

    def test_fallback_on_nan_from_solver(self):
        direction = NewtonDirection(linear_solver=lambda _A, _b: np.array([np.nan, np.nan]))
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.eye(2, format="csc"))
        assert "fallback" in proposal.kind

    def test_ascent_direction_from_solver_is_flipped(self):
        """An exact Newton step is always a descent direction, so only a wrong-signed solver reaches the flip."""
        direction = NewtonDirection(linear_solver=lambda A, b: -spsolve(A, b))
        jac = sp.csc_matrix([[2.0, 0.0], [0.0, 1.0]])
        res = np.array([1.0, 1.0])
        proposal = direction.compute(np.zeros(2), res, jac)

        assert proposal.kind == "newton_flipped"
        np.testing.assert_allclose(proposal.direction, spsolve(jac, -res))
        np.testing.assert_allclose(proposal.slope, -float(res @ res))


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

    def test_fixed_forcing_term_without_eisenstat_walker(self, broyden_system):
        fun, x0 = broyden_system
        direction = KrylovDirection(eta_max=0.3, eisenstat_walker=False)
        direction.reset()

        x = x0.copy()
        res, jac = fun(x)
        for _ in range(5):
            proposal = direction.compute(x, res, jac)
            x = x + proposal.direction
            res, jac = fun(x)

        assert direction._eta == 0.3

    def test_fallback_on_bad_krylov(self):
        direction = KrylovDirection(krylov_method="gmres")
        direction.reset()
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.csc_matrix((2, 2)))
        assert "fallback" in proposal.kind

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown Krylov method"):
            KrylovDirection(krylov_method="cg")
