import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve

import gEconpy.solvers.sparse_root.direction as direction_module

from gEconpy.solvers.sparse_root.direction import (
    ChordDirection,
    KrylovDirection,
    NewtonDirection,
    _descent_proposal,
)


class TestNewtonDirection:
    @pytest.mark.parametrize(
        ("direction", "kind"),
        [(NewtonDirection(), "gradient_fallback"), (ChordDirection(), "chord_gradient_fallback")],
        ids=["newton", "chord"],
    )
    def test_fallback_on_singular_jacobian(self, direction, kind):
        jac = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])
        res = np.array([1.0, 2.0])
        # The suite turns warnings into errors, which the solve seam would catch either way, so recording them is
        # the only way to see whether one escapes.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proposal = direction.compute(np.zeros(2), res, jac)

        assert caught == []
        assert proposal.kind == kind
        np.testing.assert_allclose(proposal.direction, -(jac.T @ res))
        assert proposal.slope < 0

    def test_fallback_on_nan_from_solver(self):
        direction = NewtonDirection(linear_solver=lambda _A, _b: np.array([np.nan, np.nan]))
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.eye(2, format="csc"))
        assert "fallback" in proposal.kind

    def test_a_vanished_gradient_is_refused_rather_than_flipped(self):
        """A square nonsingular Jacobian cannot be stationary at a non-root, so this reaches the branch directly.

        ``J^T r = 0`` with ``r != 0`` is a least-squares minimum, where neither the step nor its negation reduces
        the merit function. Flipping a zero slope would hand the line search a zero step it then accepts.
        """
        jac = sp.csc_matrix([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        res = np.array([0.0, 0.0, 1.0])

        proposal = _descent_proposal(-(jac.T @ res), res, jac, kind="gradient_fallback")

        assert proposal.kind == "gradient_fallback_stationary"
        assert proposal.slope == 0.0

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

    def test_factorizes_once_per_refresh_window(self, monkeypatch):
        calls = {"n": 0}
        real_splu = direction_module.splu

        def counting_splu(A):
            calls["n"] += 1
            return real_splu(A)

        monkeypatch.setattr(direction_module, "splu", counting_splu)
        direction = ChordDirection(recompute_every=3)
        direction.reset()
        jac = sp.eye(2, format="csc")

        for _ in range(7):
            direction.compute(np.zeros(2), np.array([1.0, 1.0]), jac)

        assert calls["n"] == 3

    def test_a_non_csc_jacobian_is_converted_before_factorizing(self):
        direction = ChordDirection()
        direction.reset()
        jac = sp.eye(2, format="csr")

        proposal = direction.compute(np.zeros(2), np.array([1.0, 2.0]), jac)

        assert proposal.kind == "chord"
        np.testing.assert_allclose(proposal.direction, [-1.0, -2.0])

    def test_a_singular_jacobian_is_not_refactorized_until_the_window_ends(self, monkeypatch):
        calls = {"n": 0}
        real_splu = direction_module.splu

        def counting_splu(A):
            calls["n"] += 1
            return real_splu(A)

        monkeypatch.setattr(direction_module, "splu", counting_splu)
        direction = ChordDirection(recompute_every=3)
        direction.reset()
        singular = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])

        for _ in range(3):
            proposal = direction.compute(np.zeros(2), np.array([1.0, 2.0]), singular)

        assert proposal.kind == "chord_gradient_fallback"
        assert calls["n"] == 1


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

    def test_forcing_term_starts_at_eta_max(self):
        assert KrylovDirection(eta_max=0.5)._eta == 0.5

    def test_fallback_on_bad_krylov(self):
        direction = KrylovDirection(krylov_method="gmres")
        direction.reset()
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.csc_matrix((2, 2)))
        assert "fallback" in proposal.kind

    @pytest.mark.parametrize("eta", [1e-6, 1e-8], ids=["eta_min_default", "tighter"])
    def test_the_forcing_term_is_the_stopping_rule(self, eta):
        n = 300
        jac = (4.0 * sp.eye(n) + sp.random(n, n, density=0.02, random_state=1)).tocsc()
        res = np.random.default_rng(0).normal(size=n)
        direction = KrylovDirection(eta_max=eta, eta_min=eta, eisenstat_walker=False)
        direction.reset()

        proposal = direction.compute(np.zeros(n), res, jac)

        assert proposal.kind == "krylov_gmres"
        assert np.linalg.norm(res + jac @ proposal.direction) <= eta * np.linalg.norm(res)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown Krylov method"):
            KrylovDirection(krylov_method="cg")
