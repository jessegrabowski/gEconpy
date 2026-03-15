import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.direction import DirectionProposal
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking, NonmonotoneBacktracking


class TestArmijoBacktracking:
    def test_backtracks_when_needed(self):
        ls = ArmijoBacktracking(c1=0.9, beta=0.5, max_iter=10)

        def fun(x):
            return x, sp.eye(1, format="csc")

        proposal = DirectionProposal(np.array([-10.0]), slope=-10.0, kind="newton")
        result = ls.search(fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())
        assert result.alpha < 1.0

    def test_raises_after_max_iter(self):
        ls = ArmijoBacktracking(max_iter=3)

        def fun(_x):
            return np.array([100.0]), sp.eye(1, format="csc")

        proposal = DirectionProposal(np.array([1.0]), slope=-1.0, kind="newton")
        with pytest.raises(RuntimeError):
            ls.search(fun, np.array([0.0]), phi_current=0.5, proposal=proposal, args=())

    def test_merit_fun_avoids_jacobian_at_rejected_points(self):
        """When merit_fun is provided, rejected trial points should not trigger full fun evaluations."""
        jac_eval_count = {"n": 0}
        merit_eval_count = {"n": 0}

        def fun(x):
            jac_eval_count["n"] += 1
            return x, sp.eye(len(x), format="csc")

        def merit_fun(x):
            merit_eval_count["n"] += 1
            return x

        # Use strict c1 so that backtracking occurs
        ls = ArmijoBacktracking(c1=0.9, beta=0.5, max_iter=20, merit_fun=merit_fun)
        proposal = DirectionProposal(np.array([-10.0]), slope=-10.0, kind="newton")
        result = ls.search(fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())

        assert result.alpha < 1.0
        # Full fun should be called exactly once (at the accepted point)
        assert jac_eval_count["n"] == 1
        # Merit fun should have been called at least once (for the accepted trial + possibly rejected ones)
        assert merit_eval_count["n"] >= 1

    def test_merit_fun_same_acceptance_as_without(self):
        """With merit_fun returning the same residuals, the accepted alpha should match the no-merit_fun case."""

        def fun(x):
            return x, sp.eye(len(x), format="csc")

        def merit_fun(x):
            return x

        x = np.array([5.0])
        proposal = DirectionProposal(np.array([-10.0]), slope=-50.0, kind="newton")

        ls_plain = ArmijoBacktracking(c1=0.5, beta=0.5, max_iter=20)
        ls_merit = ArmijoBacktracking(c1=0.5, beta=0.5, max_iter=20, merit_fun=merit_fun)

        result_plain = ls_plain.search(fun, x, phi_current=12.5, proposal=proposal, args=())
        result_merit = ls_merit.search(fun, x, phi_current=12.5, proposal=proposal, args=())

        np.testing.assert_allclose(result_plain.alpha, result_merit.alpha)
        np.testing.assert_allclose(result_plain.phi_new, result_merit.phi_new)

    def test_merit_fun_n_evals_counts_final_fun_call(self):
        """n_evals should include the final full fun call when merit_fun is used."""

        def fun(x):
            return x, sp.eye(len(x), format="csc")

        # merit_fun that accepts on first try
        ls = ArmijoBacktracking(c1=1e-4, merit_fun=lambda x: x)
        proposal = DirectionProposal(np.array([-1.0]), slope=-1.0, kind="newton")
        result = ls.search(fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())

        # 1 merit eval + 1 fun eval = 2
        assert result.n_evals == 2


class TestNonmonotoneBacktracking:
    def test_memory_one_equals_armijo(self):
        """With memory=1, nonmonotone backtracking should behave like standard Armijo."""

        def fun(x):
            return x.copy(), sp.eye(len(x), format="csc")

        x = np.array([2.0])
        proposal = DirectionProposal(direction=np.array([-1.0]), slope=-2.0, kind="newton")

        armijo = ArmijoBacktracking(c1=1e-4, beta=0.5, max_iter=50)
        nm = NonmonotoneBacktracking(c1=1e-4, beta=0.5, max_iter=50, memory=1)

        result_a = armijo.search(fun, x, phi_current=2.0, proposal=proposal, args=())
        result_nm = nm.search(fun, x, phi_current=2.0, proposal=proposal, args=())

        np.testing.assert_allclose(result_a.alpha, result_nm.alpha)

    def test_memory_growth(self):
        """History should not exceed memory size."""
        nm = NonmonotoneBacktracking(memory=3)
        for i in range(10):
            nm._phi_history.append(float(i))
        assert len(nm._phi_history) <= 3

    def test_merit_fun_avoids_jacobian_at_rejected_points(self):
        """When merit_fun is provided, rejected trial points should not trigger full fun evaluations."""
        jac_eval_count = {"n": 0}

        def fun(x):
            jac_eval_count["n"] += 1
            return x.copy(), sp.eye(len(x), format="csc")

        def merit_fun(x):
            return x.copy()

        nm = NonmonotoneBacktracking(c1=0.9, beta=0.5, max_iter=20, memory=1, merit_fun=merit_fun)
        proposal = DirectionProposal(np.array([-10.0]), slope=-10.0, kind="newton")
        result = nm.search(fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())

        assert result.alpha < 1.0
        assert jac_eval_count["n"] == 1

    def test_merit_fun_same_acceptance_as_without(self):
        """With merit_fun returning the same residuals, the accepted alpha should match the no-merit_fun case."""

        def fun(x):
            return x.copy(), sp.eye(len(x), format="csc")

        x = np.array([5.0])
        proposal = DirectionProposal(np.array([-10.0]), slope=-50.0, kind="newton")

        nm_plain = NonmonotoneBacktracking(c1=0.5, beta=0.5, max_iter=20, memory=1)
        nm_merit = NonmonotoneBacktracking(c1=0.5, beta=0.5, max_iter=20, memory=1, merit_fun=lambda x: x.copy())

        result_plain = nm_plain.search(fun, x, phi_current=12.5, proposal=proposal, args=())
        result_merit = nm_merit.search(fun, x, phi_current=12.5, proposal=proposal, args=())

        np.testing.assert_allclose(result_plain.alpha, result_merit.alpha)
        np.testing.assert_allclose(result_plain.phi_new, result_merit.phi_new)
