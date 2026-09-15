import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy.solvers.sparse_root import (
    GaussNewtonTrustRegion,
    LevenbergMarquardt,
    NewtonArmijo,
    SparseDogleg,
    sparse_root,
)
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking


def scaled_quadratic(x, scale):
    return scale * (x**2 - np.array([1.0, 4.0])), sp.diags(2.0 * scale * x, format="csc")


def uphill_everywhere(x):
    """Residuals that are small only at the starting point [5, 5], so every trial step is rejected."""
    res = x - 1.0 if np.all(x == 5.0) else np.array([1e3, 1e3])
    return res, sp.eye(2, format="csc")


class TestSparseRootInputs:
    @pytest.mark.parametrize(
        ("bad_fun", "error", "match"),
        [
            (lambda x: x, ValueError, "must return a tuple"),
            (lambda x: (list(x), sp.eye(2, format="csc")), TypeError, "residuals as a numpy ndarray"),
            (lambda x: (x, np.eye(2)), TypeError, "Jacobian as a scipy sparse matrix"),
        ],
        ids=["not_a_tuple", "list_residuals", "dense_jacobian"],
    )
    def test_rejects_malformed_fun(self, bad_fun, error, match):
        with pytest.raises(error, match=match):
            sparse_root(bad_fun, np.zeros(2), progressbar=False)

    def test_args_are_forwarded_to_fun(self):
        result = sparse_root(scaled_quadratic, np.array([2.0, 3.0]), args=(3.0,), progressbar=False)

        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-8)
        np.testing.assert_allclose(result.fun, 0.0, atol=1e-8)

    def test_list_x0_reaches_fun_as_float_array(self):
        seen = []

        def recording_quadratic(x):
            seen.append(x)
            return scaled_quadratic(x, 1.0)

        result = sparse_root(recording_quadratic, [2, 3], progressbar=False)

        assert result.success
        assert all(isinstance(x, np.ndarray) and x.dtype == np.float64 for x in seen)

    def test_integer_x0_is_solved_in_float(self):
        result = sparse_root(scaled_quadratic, np.array([2, 3]), args=(1.0,), progressbar=False)

        assert result.success
        assert result.x.dtype == np.float64
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-8)

    def test_step_tolerance_stops_before_residual_tolerance(self):
        result = sparse_root(
            scaled_quadratic, np.array([2.0, 3.0]), args=(1.0,), f_tol=1e-300, x_tol=1e-6, progressbar=False
        )

        assert result.success
        assert result.message == "Converged"
        assert np.max(np.abs(result.fun)) > 1e-300


class TestSparseRootFailureMessages:
    def test_line_search_failure_is_reported(self):
        solver = NewtonArmijo(globalization=ArmijoBacktracking(max_iter=1, c1=0.99))
        result = sparse_root(scaled_quadratic, np.array([100.0, 100.0]), args=(1.0,), solver=solver, progressbar=False)

        assert not result.success
        assert result.message.startswith("Line search failed after 1 reductions")
        assert result.nit == 0

    @pytest.mark.parametrize(
        ("solver", "suffix"),
        [
            (SparseDogleg(max_reject=3), "dogleg rejected 3 consecutive steps"),
            (GaussNewtonTrustRegion(max_reject=3), "trust region rejected 3 consecutive steps"),
            (LevenbergMarquardt(max_reject=3), "damping reached 1.0e+00"),
            (LevenbergMarquardt(lam0=0.5, max_lam=1.0), "damping reached 1.0e+00"),
        ],
        ids=["dogleg", "gauss_newton", "levenberg_marquardt_max_reject", "levenberg_marquardt_max_lam"],
    )
    def test_trust_region_rejection_is_reported(self, solver, suffix):
        result = sparse_root(uphill_everywhere, np.array([5.0, 5.0]), solver=solver, progressbar=False)

        assert not result.success
        assert "fatal" in result.message
        assert result.message.endswith(suffix)
        np.testing.assert_array_equal(result.x, [5.0, 5.0])


class TestTrustRegionAccounting:
    @staticmethod
    def _counting(fun):
        calls = {"n": 0}

        def counting_fun(x):
            calls["n"] += 1
            return fun(x)

        return counting_fun, calls

    @pytest.mark.parametrize(
        "solver",
        [SparseDogleg(max_reject=3), GaussNewtonTrustRegion(max_reject=3), LevenbergMarquardt(lam0=0.5, max_lam=1.0)],
        ids=["dogleg", "gauss_newton", "levenberg_marquardt"],
    )
    def test_fatal_rejection_counts_trial_evaluations(self, solver):
        counting_fun, calls = self._counting(uphill_everywhere)
        state = solver.init(counting_fun, np.array([5.0, 5.0]), ())
        state, info = solver.step(counting_fun, state, ())

        assert not info.accepted
        assert calls["n"] > 1
        assert state.stats.nfev == calls["n"]
        assert state.stats.njev == calls["n"]
        assert state.stats.nreject == calls["n"] - 1
        assert state.stats.nsolve == calls["n"] - 1

    @pytest.mark.parametrize(
        "solver",
        [SparseDogleg(delta0=0.05), GaussNewtonTrustRegion(delta0=0.05), LevenbergMarquardt(lam0=1e-6)],
        ids=["dogleg", "gauss_newton", "levenberg_marquardt"],
    )
    def test_accepted_step_counts_one_solve_per_trial(self, solver, rosenbrock_root):
        fun, x0, _ = rosenbrock_root
        state = solver.init(fun, x0, ())
        for _ in range(200):
            state, info = solver.step(fun, state, ())
            if not info.accepted or np.max(np.abs(state.res)) < 1e-8:
                break

        assert state.stats.nreject > 0
        assert state.stats.nsolve == state.stats.nit + state.stats.nreject
