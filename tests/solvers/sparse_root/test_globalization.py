import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.direction import DirectionProposal
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking


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
