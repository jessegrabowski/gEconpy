import numpy as np
import pytest
import scipy.sparse as sp

from gEconpy.solvers.sparse_root import sparse_root


@pytest.fixture
def quadratic_system():
    def fun(x):
        return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")

    return fun, np.array([2.0, 3.0]), np.array([1.0, 2.0])


@pytest.fixture
def trig_system():
    """a*x0*cos(x1) - 4 = 0, x1*x0 - x1 - 5 = 0. Solution approx [6.504, 0.908]."""

    def fun(x):
        res = np.array([x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5])
        jac = sp.csc_matrix(
            [[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]],
        )
        return res, jac

    return fun, np.array([1.0, 1.0]), np.array([6.50409711, 0.90841421])


@pytest.fixture
def broyden_system():
    n = 100

    def fun(x):
        res = (3.0 - 2.0 * x) * x + 1.0
        res[:-1] -= 2.0 * x[1:]
        res[1:] -= x[:-1]
        jac = sp.diags(
            [-np.ones(n - 1), 3.0 - 4.0 * x, -2 * np.ones(n - 1)],
            [-1, 0, 1],
            format="csc",
        )
        return res, jac

    return fun, -np.ones(n)


class CommonSolverTests:
    """Mixin base class providing standard convergence and robustness tests.

    Subclass in each ``test_*.py`` and set ``solver`` to run the full suite
    against a particular solver.
    """

    solver = None

    def test_quadratic(self, quadratic_system):
        fun, x0, x_true = quadratic_system
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)

    def test_trig(self, trig_system):
        fun, x0, x_true = trig_system
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-4)

    def test_broyden(self, broyden_system):
        fun, x0 = broyden_system
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False)
        assert result.success

    def test_already_at_root(self, quadratic_system):
        """Starting at the solution should return immediately."""
        fun, _, x_true = quadratic_system
        result = sparse_root(fun, x_true, solver=self.solver, progressbar=False)
        assert result.success
        assert result.nit == 0

    def test_far_start(self):
        """Starting very far from root. Solver should not crash."""

        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")

        result = sparse_root(fun, np.array([1e4, 1e4]), solver=self.solver, progressbar=False)
        if result.success:
            np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-4)

    def test_maxiter_respected(self, quadratic_system):
        """Solver must not exceed maxiter."""
        fun, _, _ = quadratic_system
        result = sparse_root(fun, np.array([1e6, 1e6]), solver=self.solver, maxiter=3, progressbar=False)
        assert not result.success
        assert result.nit <= 3

    def test_single_variable(self):
        """1D root-finding."""

        def fun(x):
            return np.array([x[0] ** 3 - 1.0]), sp.csc_matrix([[3.0 * x[0] ** 2]])

        result = sparse_root(fun, np.array([0.5]), solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, [1.0], rtol=1e-6)
