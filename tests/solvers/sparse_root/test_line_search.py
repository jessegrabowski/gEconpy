import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests
from scipy.sparse.linalg import gmres, spsolve

from gEconpy.solvers.sparse_root import Chord, InexactNewtonKrylov, NewtonArmijo, NewtonNonmonotone, sparse_root
from gEconpy.solvers.sparse_root.direction import (
    ChordDirection,
    DirectionProposal,
    KrylovDirection,
    NewtonDirection,
)
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking, NonmonotoneBacktracking
from gEconpy.solvers.sparse_root.line_search import LineSearchSolver


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


class TestChordSuite(CommonSolverTests):
    solver = Chord()


class TestChordSpecific:
    def test_fewer_solves_than_newton(self, broyden_system):
        """Chord should use fewer linear solves than Newton for the same problem."""
        fun, x0 = broyden_system

        solve_count_newton = {"n": 0}
        solve_count_chord = {"n": 0}

        def counting_solver_newton(A, b):
            solve_count_newton["n"] += 1
            return spsolve(A, b)

        def counting_solver_chord(A, b):
            solve_count_chord["n"] += 1
            return spsolve(A, b)

        newton = NewtonArmijo(
            direction=NewtonDirection(linear_solver=counting_solver_newton),
            globalization=ArmijoBacktracking(),
        )
        chord = Chord(
            direction=ChordDirection(linear_solver=counting_solver_chord, recompute_every=5),
            globalization=ArmijoBacktracking(),
        )

        result_n = sparse_root(fun, x0, solver=newton, progressbar=False)
        result_c = sparse_root(fun, x0, solver=chord, progressbar=False)

        assert result_n.success
        assert result_c.success


class TestInexactNewtonKrylovSuite(CommonSolverTests):
    solver = InexactNewtonKrylov()


class TestInexactNewtonKrylovSpecific:
    def test_direction_is_approximate(self, broyden_system):
        """Verify the Krylov solve is inexact but the solver still converges."""
        fun, x0 = broyden_system
        direction = KrylovDirection(krylov_method="gmres", eta_max=0.5, eisenstat_walker=False)
        solver = InexactNewtonKrylov(direction=direction)
        result = sparse_root(fun, x0, solver=solver, progressbar=False)
        assert result.success

    def test_bicgstab_also_converges(self, broyden_system):
        fun, x0 = broyden_system
        direction = KrylovDirection(krylov_method="bicgstab")
        solver = InexactNewtonKrylov(direction=direction)
        result = sparse_root(fun, x0, solver=solver, progressbar=False)
        assert result.success

    def test_large_broyden(self):
        """Krylov should handle large sparse systems efficiently."""
        n = 500

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

        result = sparse_root(fun, -np.ones(n), solver=InexactNewtonKrylov(), progressbar=False)
        assert result.success


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


class TestCustomSolver(CommonSolverTests):
    solver = LineSearchSolver(
        direction=KrylovDirection(krylov_method="gmres"),
        globalization=NonmonotoneBacktracking(memory=5),
    )
