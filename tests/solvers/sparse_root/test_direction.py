import numpy as np
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.direction import NewtonDirection


class TestNewtonDirectionFallback:
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
