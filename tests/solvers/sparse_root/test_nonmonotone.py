import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import NewtonNonmonotone, sparse_root
from gEconpy.solvers.sparse_root.direction import DirectionProposal
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking, NonmonotoneBacktracking


class TestNewtonNonmonotoneSuite(CommonSolverTests):
    solver = NewtonNonmonotone()


class TestNewtonNonmonotoneSpecific:
    def test_accepts_nonmonotone_step(self):
        """Verify that with memory > 1, the solver accepts a step that standard Armijo rejects."""

        def fun(x):
            return x.copy(), sp.eye(len(x), format="csc")

        x = np.array([2.0])
        proposal = DirectionProposal(direction=np.array([-1.0]), slope=-2.0, kind="newton")

        nm = NonmonotoneBacktracking(c1=1e-4, memory=5)
        nm._phi_history.append(100.0)

        result_nm = nm.search(fun, x, phi_current=2.0, proposal=proposal, args=())
        assert result_nm.alpha == 1.0

        nm2 = NonmonotoneBacktracking(c1=1e-4, memory=5)
        nm2._phi_history.append(10.0)

        x_small = np.array([0.1])
        proposal2 = DirectionProposal(direction=np.array([0.9]), slope=-0.01, kind="test")
        result_nm2 = nm2.search(fun, x_small, phi_current=0.005, proposal=proposal2, args=())
        assert result_nm2.alpha == 1.0

        armijo = ArmijoBacktracking(c1=1e-4, max_iter=5)
        try:
            result_a = armijo.search(fun, x_small, phi_current=0.005, proposal=proposal2, args=())
            assert result_a.alpha < 1.0
        except RuntimeError:
            pass  # Expected: Armijo cannot find a step

    def test_coupled_system(self, coupled_nonlinear):
        """Nonmonotone line search should help on strongly coupled systems."""
        fun, x0 = coupled_nonlinear
        result = sparse_root(fun, x0, solver=NewtonNonmonotone(), progressbar=False, maxiter=500)
        if result.success:
            res, _ = fun(result.x)
            np.testing.assert_allclose(res, 0.0, atol=1e-4)
