import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.direction import DirectionProposal
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking, NonmonotoneBacktracking


def identity_fun(x):
    return x.copy(), sp.eye(len(x), format="csc")


def identity_merit(x):
    return x.copy()


class TestArmijoBacktracking:
    def test_backtracks_when_needed(self):
        line_search = ArmijoBacktracking(c1=0.9, beta=0.5, max_iter=10)
        proposal = DirectionProposal(np.array([-10.0]), slope=-10.0, kind="newton")
        result = line_search.search(identity_fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())
        assert result.alpha < 1.0

    def test_raises_after_max_iter(self):
        line_search = ArmijoBacktracking(max_iter=3)

        def never_decreasing(_x):
            return np.array([100.0]), sp.eye(1, format="csc")

        proposal = DirectionProposal(np.array([1.0]), slope=-1.0, kind="newton")
        with pytest.raises(RuntimeError, match="Line search failed after 3 reductions"):
            line_search.search(never_decreasing, np.array([0.0]), phi_current=0.5, proposal=proposal, args=())

    def test_merit_fun_calls_fun_once_at_accepted_point(self):
        n_fun_calls = 0

        def counting_fun(x):
            nonlocal n_fun_calls
            n_fun_calls += 1
            return identity_fun(x)

        line_search = ArmijoBacktracking(c1=0.9, beta=0.5, max_iter=20, merit_fun=identity_merit)
        proposal = DirectionProposal(np.array([-10.0]), slope=-10.0, kind="newton")
        result = line_search.search(counting_fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())

        assert result.alpha < 1.0
        assert n_fun_calls == 1

    def test_merit_fun_same_acceptance_as_without(self):
        x = np.array([5.0])
        proposal = DirectionProposal(np.array([-10.0]), slope=-50.0, kind="newton")

        plain = ArmijoBacktracking(c1=0.5, beta=0.5, max_iter=20)
        with_merit = ArmijoBacktracking(c1=0.5, beta=0.5, max_iter=20, merit_fun=identity_merit)

        result_plain = plain.search(identity_fun, x, phi_current=12.5, proposal=proposal, args=())
        result_merit = with_merit.search(identity_fun, x, phi_current=12.5, proposal=proposal, args=())

        np.testing.assert_allclose(result_plain.alpha, result_merit.alpha)
        np.testing.assert_allclose(result_plain.phi_new, result_merit.phi_new)

    def test_merit_fun_n_evals_counts_final_fun_call(self):
        line_search = ArmijoBacktracking(c1=1e-4, merit_fun=identity_merit)
        proposal = DirectionProposal(np.array([-1.0]), slope=-1.0, kind="newton")
        result = line_search.search(identity_fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())
        assert result.n_evals == 2


class TestNonmonotoneBacktracking:
    def test_memory_one_equals_armijo(self):
        x = np.array([2.0])
        proposal = DirectionProposal(direction=np.array([-1.0]), slope=-2.0, kind="newton")

        armijo = ArmijoBacktracking(c1=1e-4, beta=0.5, max_iter=50)
        nonmonotone = NonmonotoneBacktracking(c1=1e-4, beta=0.5, max_iter=50, memory=1)

        result_armijo = armijo.search(identity_fun, x, phi_current=2.0, proposal=proposal, args=())
        result_nonmonotone = nonmonotone.search(identity_fun, x, phi_current=2.0, proposal=proposal, args=())

        np.testing.assert_allclose(result_armijo.alpha, result_nonmonotone.alpha)

    def test_history_bounded_by_memory(self):
        nonmonotone = NonmonotoneBacktracking(memory=3)
        for i in range(10):
            nonmonotone._phi_history.append(float(i))
        assert len(nonmonotone._phi_history) == 3

    def test_accepts_step_armijo_rejects(self):
        x = np.array([0.1])
        proposal = DirectionProposal(direction=np.array([0.9]), slope=-0.01, kind="test")

        nonmonotone = NonmonotoneBacktracking(c1=1e-4, memory=5)
        nonmonotone._phi_history.append(10.0)
        result = nonmonotone.search(identity_fun, x, phi_current=0.005, proposal=proposal, args=())
        assert result.alpha == 1.0

        armijo = ArmijoBacktracking(c1=1e-4, max_iter=5)
        with pytest.raises(RuntimeError):
            armijo.search(identity_fun, x, phi_current=0.005, proposal=proposal, args=())

    def test_merit_fun_calls_fun_once_at_accepted_point(self):
        n_fun_calls = 0

        def counting_fun(x):
            nonlocal n_fun_calls
            n_fun_calls += 1
            return identity_fun(x)

        nonmonotone = NonmonotoneBacktracking(c1=0.9, beta=0.5, max_iter=20, memory=1, merit_fun=identity_merit)
        proposal = DirectionProposal(np.array([-10.0]), slope=-10.0, kind="newton")
        result = nonmonotone.search(counting_fun, np.array([1.0]), phi_current=0.5, proposal=proposal, args=())

        assert result.alpha < 1.0
        assert n_fun_calls == 1

    def test_merit_fun_same_acceptance_as_without(self):
        x = np.array([5.0])
        proposal = DirectionProposal(np.array([-10.0]), slope=-50.0, kind="newton")

        plain = NonmonotoneBacktracking(c1=0.5, beta=0.5, max_iter=20, memory=1)
        with_merit = NonmonotoneBacktracking(c1=0.5, beta=0.5, max_iter=20, memory=1, merit_fun=identity_merit)

        result_plain = plain.search(identity_fun, x, phi_current=12.5, proposal=proposal, args=())
        result_merit = with_merit.search(identity_fun, x, phi_current=12.5, proposal=proposal, args=())

        np.testing.assert_allclose(result_plain.alpha, result_merit.alpha)
        np.testing.assert_allclose(result_plain.phi_new, result_merit.phi_new)
