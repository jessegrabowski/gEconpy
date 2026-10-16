import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from scipy.sparse.linalg import LinearOperator, splu, spsolve

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
        np.testing.assert_array_equal(direction._factorization.matrix.data, jac_first.data)

        direction.compute(x, res, jac_second)
        np.testing.assert_array_equal(direction._factorization.matrix.data, jac_second.data)

    def test_factorizes_once_per_refresh_window(self, monkeypatch):
        calls = {"n": 0}
        real_splu = direction_module.splu

        def counting_splu(A, **kwargs):
            calls["n"] += 1
            return real_splu(A, **kwargs)

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

        def counting_splu(A, **kwargs):
            calls["n"] += 1
            return real_splu(A, **kwargs)

        monkeypatch.setattr(direction_module, "splu", counting_splu)
        direction = ChordDirection(recompute_every=3)
        direction.reset()
        singular = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])

        for _ in range(3):
            proposal = direction.compute(np.zeros(2), np.array([1.0, 2.0]), singular)

        assert proposal.kind == "chord_gradient_fallback"
        assert calls["n"] == 1

    @pytest.mark.parametrize(
        ("second_residual", "refreshed"), [(0.05, False), (0.5, True)], ids=["fast_progress", "slow_progress"]
    )
    def test_a_residual_that_falls_too_slowly_refreshes_the_factorization(self, second_residual, refreshed):
        direction = ChordDirection(recompute_every=None, refresh_ratio=0.1)
        direction.reset()
        jac_first = sp.eye(2, format="csc")
        jac_second = 2.0 * sp.eye(2, format="csc")

        direction.compute(np.zeros(2), np.array([1.0, 1.0]), jac_first)
        direction.compute(np.zeros(2), np.array([second_residual, second_residual]), jac_second)

        expected = jac_second if refreshed else jac_first
        np.testing.assert_array_equal(direction._factorization.matrix.data, expected.data)

    @pytest.mark.parametrize(("refresh_step_tol", "refreshed"), [(1e-10, True), (None, False)], ids=["on", "off"])
    def test_a_short_step_from_an_old_factorization_refreshes_it(self, refresh_step_tol, refreshed):
        direction = ChordDirection(recompute_every=None, refresh_step_tol=refresh_step_tol)
        direction.reset()
        jac_first = sp.eye(2, format="csc")
        jac_second = 2.0 * sp.eye(2, format="csc")

        direction.compute(np.zeros(2), np.array([1.0, 1.0]), jac_first)
        proposal = direction.compute(np.zeros(2), np.array([1e-12, 1e-12]), jac_second)

        expected_step = -1e-12 / (2.0 if refreshed else 1.0)
        np.testing.assert_allclose(proposal.direction, [expected_step, expected_step])

    def test_a_short_step_from_a_fresh_factorization_is_not_refactorized(self, monkeypatch):
        calls = {"n": 0}
        real_splu = direction_module.splu

        def counting_splu(A, **kwargs):
            calls["n"] += 1
            return real_splu(A, **kwargs)

        monkeypatch.setattr(direction_module, "splu", counting_splu)
        direction = ChordDirection(recompute_every=None, refresh_step_tol=1e-10)
        direction.reset()

        direction.compute(np.zeros(2), np.array([1e-12, 1e-12]), sp.eye(2, format="csc"))

        assert calls["n"] == 1

    @pytest.mark.parametrize("keep_across_solves", [True, False])
    def test_reset_keeps_the_factorization_only_when_asked(self, keep_across_solves):
        direction = ChordDirection(keep_across_solves=keep_across_solves)
        direction.reset()
        direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.eye(2, format="csc"))

        direction.reset()

        assert (direction._factorization.lu is not None) == keep_across_solves

    def test_a_kept_factorization_of_another_shape_is_refreshed(self):
        direction = ChordDirection(keep_across_solves=True)
        direction.reset()
        direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.eye(2, format="csc"))
        direction.reset()

        proposal = direction.compute(np.zeros(3), np.array([1.0, 2.0, 3.0]), 2.0 * sp.eye(3, format="csc"))

        assert proposal.kind == "chord"
        np.testing.assert_allclose(proposal.direction, [-0.5, -1.0, -1.5])

    def test_without_a_schedule_a_failed_factorization_is_retried_on_the_next_call(self):
        direction = ChordDirection(recompute_every=None)
        direction.reset()
        singular = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])

        first = direction.compute(np.zeros(2), np.array([1.0, 2.0]), singular)
        second = direction.compute(np.zeros(2), np.array([1.0, 2.0]), sp.eye(2, format="csc"))

        assert first.kind == "chord_gradient_fallback"
        assert second.kind == "chord"
        np.testing.assert_allclose(second.direction, [-1.0, -2.0])

    def test_a_recompute_every_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="recompute_every must be at least 1"):
            ChordDirection(recompute_every=0)


class TestKrylovDirection:
    def test_eisenstat_walker_tightens(self, broyden_system):
        fun, x0 = broyden_system
        direction = KrylovDirection(eta_max=0.9, eisenstat_walker=True)
        direction.reset()
        eta_initial = direction._eta

        x = x0.copy()
        res, jac = fun(x)
        # Past the root the forcing-term ratio is rounding noise, so stepping stops once the residual has converged.
        for _ in range(10):
            if np.linalg.norm(res) < 1e-8:
                break
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
        direction = KrylovDirection(eta_max=eta, eta_min=eta, eisenstat_walker=False, recompute_every=None)
        direction.reset()

        proposal = direction.compute(np.zeros(n), res, jac)

        assert proposal.kind == "krylov_gmres"
        assert np.linalg.norm(res + jac @ proposal.direction) <= eta * np.linalg.norm(res)

    @pytest.mark.parametrize(
        ("restart", "max_krylov_iter", "kind"),
        [
            (1, 200, "krylov_gmres_gradient_fallback"),
            (20, 2, "krylov_gmres_inexact"),
            (20, 200, "krylov_gmres"),
        ],
        ids=["no_progress_falls_back", "capped_short_keeps_the_iterate", "defaults_converge"],
    )
    def test_the_iteration_limits_reach_the_inner_solve(self, restart, max_krylov_iter, kind):
        # Skew-symmetric, so GMRES(1) makes no progress at all, while 10 unrestarted iterations solve it exactly.
        blocks = [sp.csc_matrix([[0.0, w], [-w, 0.0]]) for w in np.linspace(1.0, 2.0, 5)]
        jac = sp.block_diag(blocks, format="csc")
        res = np.random.default_rng(0).normal(size=10)
        direction = KrylovDirection(
            eta_max=1e-8,
            eta_min=1e-8,
            eisenstat_walker=False,
            restart=restart,
            max_krylov_iter=max_krylov_iter,
            recompute_every=None,
        )
        direction.reset()

        proposal = direction.compute(np.zeros(10), res, jac)

        assert proposal.kind == kind
        assert proposal.slope < 0

    @pytest.mark.parametrize(
        ("max_krylov_iter", "kind"),
        [(1, "krylov_bicgstab_inexact"), (200, "krylov_bicgstab")],
        ids=["capped_keeps_the_iterate", "uncapped_converges"],
    )
    def test_bicgstab_takes_the_cap_as_its_iteration_count(self, max_krylov_iter, kind):
        n = 300
        jac = (4.0 * sp.eye(n) + sp.random(n, n, density=0.02, random_state=1)).tocsc()
        res = np.random.default_rng(0).normal(size=n)
        direction = KrylovDirection(
            krylov_method="bicgstab",
            eta_max=1e-8,
            eta_min=1e-8,
            eisenstat_walker=False,
            max_krylov_iter=max_krylov_iter,
            recompute_every=None,
        )
        direction.reset()

        proposal = direction.compute(np.zeros(n), res, jac)

        assert proposal.kind == kind
        assert proposal.slope < 0

    def test_a_krylov_breakdown_falls_back_to_the_gradient(self):
        # BiCGSTAB breaks down on a skew-symmetric system, which scipy reports as a negative info.
        blocks = [sp.csc_matrix([[0.0, w], [-w, 0.0]]) for w in np.linspace(1.0, 2.0, 5)]
        jac = sp.block_diag(blocks, format="csc")
        res = np.random.default_rng(0).normal(size=10)
        direction = KrylovDirection(
            krylov_method="bicgstab", eta_max=1e-8, eta_min=1e-8, eisenstat_walker=False, recompute_every=None
        )
        direction.reset()

        proposal = direction.compute(np.zeros(10), res, jac)

        assert proposal.kind == "krylov_bicgstab_gradient_fallback"
        np.testing.assert_allclose(proposal.direction, -(jac.T @ res))

    def test_a_refreshed_lu_preconditions_by_default(self, stacked_system):
        res, jac = stacked_system.fun(stacked_system.x0)
        direction = KrylovDirection(eta_max=1e-8, eta_min=1e-8, eisenstat_walker=False, max_krylov_iter=3)
        direction.reset()

        proposal = direction.compute(stacked_system.x0, res, jac)

        assert proposal.kind == "krylov_gmres"
        np.testing.assert_allclose(proposal.direction, spsolve(jac, -res), rtol=1e-6)

    @pytest.mark.parametrize(
        ("options", "factorizations"),
        [({}, 3), ({"recompute_every": None}, 0), ({"preconditioner": sp.eye(2, format="csc")}, 0)],
        ids=["refreshed_every_5", "turned_off", "user_preconditioner_wins"],
    )
    def test_the_preconditioner_is_refactorized_on_schedule(self, monkeypatch, options, factorizations):
        calls = {"n": 0}
        real_splu = direction_module.splu

        def counting_splu(A, **kwargs):
            calls["n"] += 1
            return real_splu(A, **kwargs)

        monkeypatch.setattr(direction_module, "splu", counting_splu)
        direction = KrylovDirection(**options)
        direction.reset()
        jac = sp.csc_matrix([[2.0, 1.0], [0.0, 3.0]])

        for _ in range(12):
            direction.compute(np.zeros(2), np.array([1.0, 1.0]), jac)

        assert calls["n"] == factorizations

    def test_a_singular_jacobian_solves_unpreconditioned(self):
        jac = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])
        res = np.array([1.0, 2.0])
        direction = KrylovDirection()
        direction.reset()

        proposal = direction.compute(np.zeros(2), res, jac)

        assert proposal.kind == "krylov_gmres"
        assert proposal.slope < 0

    def test_the_preconditioner_reaches_the_inner_solve(self, stacked_system):
        res, jac = stacked_system.fun(stacked_system.x0)
        lu = splu(jac.tocsc())
        exact_inverse = LinearOperator(jac.shape, matvec=lu.solve)
        direction = KrylovDirection(
            eta_max=1e-8, eta_min=1e-8, eisenstat_walker=False, max_krylov_iter=3, preconditioner=exact_inverse
        )
        direction.reset()

        proposal = direction.compute(stacked_system.x0, res, jac)

        assert proposal.kind == "krylov_gmres"
        np.testing.assert_allclose(proposal.direction, spsolve(jac, -res), rtol=1e-6)

    @pytest.mark.parametrize("field_name", ["max_krylov_iter", "restart", "recompute_every"])
    def test_an_iteration_limit_below_one_is_rejected(self, field_name):
        with pytest.raises(ValueError, match=f"{field_name} must be at least 1"):
            KrylovDirection(**{field_name: 0})

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown Krylov method"):
            KrylovDirection(krylov_method="cg")
