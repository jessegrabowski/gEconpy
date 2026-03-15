import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests
from scipy.sparse.linalg import gmres

from gEconpy.solvers.sparse_root import NewtonArmijo, sparse_root
from gEconpy.solvers.sparse_root.direction import NewtonDirection
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking


class TestNewtonArmijoSuite(CommonSolverTests):
    solver = NewtonArmijo()


class TestNewtonArmijoSpecific:
    def test_strict_vs_loose_c1(self, quadratic_system):
        fun, _, _ = quadratic_system
        strict = NewtonArmijo(globalization=ArmijoBacktracking(c1=0.99))
        loose = NewtonArmijo(globalization=ArmijoBacktracking(c1=1e-6))
        r_strict = sparse_root(fun, np.array([100.0, 100.0]), solver=strict, progressbar=False)
        r_loose = sparse_root(fun, np.array([100.0, 100.0]), solver=loose, progressbar=False)
        assert r_loose.nfev <= r_strict.nfev

    def test_custom_linear_solver(self, trig_system):
        def gmres_solver(_A, _b):
            x, _ = gmres(_A, _b, atol=1e-12)
            return x

        fun, x0, x_true = trig_system
        solver = NewtonArmijo(direction=NewtonDirection(linear_solver=gmres_solver))
        result = sparse_root(fun, x0, solver=solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)

    def test_line_search_backtracks_from_far_start(self):
        """Regression: line search must backtrack, not just take a full step."""
        nfev_counter = {"n": 0}

        def tracking_fun(x):
            nfev_counter["n"] += 1
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")

        result = sparse_root(tracking_fun, np.array([100.0, 100.0]), tol=1e-10, progressbar=False)
        assert result.success
        assert nfev_counter["n"] > result.nit + 1
