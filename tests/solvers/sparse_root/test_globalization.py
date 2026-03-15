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
