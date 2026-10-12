import numpy as np
import pytest
import scipy.sparse as sp

from scipy.sparse.linalg import gmres

from gEconpy.solvers.sparse_root import chord, inexact_newton_krylov, newton_armijo, newton_nonmonotone, sparse_root
from gEconpy.solvers.sparse_root.direction import ChordDirection, KrylovDirection, NewtonDirection
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking, NonmonotoneBacktracking
from gEconpy.solvers.sparse_root.line_search import LineSearchSolver
from tests.solvers.sparse_root._solver_suite import CommonSolverTests


@pytest.mark.parametrize(
    ("factory", "direction_type", "globalization_type"),
    [
        (newton_armijo, NewtonDirection, ArmijoBacktracking),
        (chord, ChordDirection, ArmijoBacktracking),
        (inexact_newton_krylov, KrylovDirection, ArmijoBacktracking),
        (newton_nonmonotone, NewtonDirection, NonmonotoneBacktracking),
    ],
    ids=["newton_armijo", "chord", "inexact_newton_krylov", "newton_nonmonotone"],
)
def test_each_factory_builds_the_strategies_it_names(factory, direction_type, globalization_type):
    solver = factory()

    assert isinstance(solver.direction, direction_type)
    assert isinstance(solver.globalization, globalization_type)


@pytest.mark.parametrize(
    ("factory", "direction_type"),
    [
        (newton_armijo, NewtonDirection),
        (chord, ChordDirection),
        (inexact_newton_krylov, KrylovDirection),
        (newton_nonmonotone, NewtonDirection),
    ],
    ids=["newton_armijo", "chord", "inexact_newton_krylov", "newton_nonmonotone"],
)
def test_each_factory_keeps_the_strategy_it_is_handed(factory, direction_type):
    direction = direction_type()

    solver = factory(direction=direction)

    assert solver.direction is direction


class TestNewtonArmijoSuite(CommonSolverTests):
    @property
    def solver(self):
        return newton_armijo()


class TestNewtonArmijoSpecific:
    def test_strict_vs_loose_c1(self, quadratic_system):
        fun, _, _ = quadratic_system
        strict = newton_armijo(globalization=ArmijoBacktracking(c1=0.99))
        loose = newton_armijo(globalization=ArmijoBacktracking(c1=1e-6))
        result_strict = sparse_root(fun, np.array([100.0, 100.0]), solver=strict, progressbar=False)
        result_loose = sparse_root(fun, np.array([100.0, 100.0]), solver=loose, progressbar=False)
        assert result_loose.nfev <= result_strict.nfev

    def test_custom_linear_solver(self, trig_system):
        def gmres_solver(A, b):
            x, _ = gmres(A, b, atol=1e-12)
            return x

        fun, x0, x_true = trig_system
        solver = newton_armijo(direction=NewtonDirection(linear_solver=gmres_solver))
        result = sparse_root(fun, x0, solver=solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)

    def test_line_search_backtracks_from_far_start(self, quadratic_system):
        fun, _, _ = quadratic_system
        n_calls = 0

        def counting_fun(x):
            nonlocal n_calls
            n_calls += 1
            return fun(x)

        result = sparse_root(counting_fun, np.array([100.0, 100.0]), tol=1e-10, progressbar=False)
        assert result.success
        assert n_calls > result.nit + 1

    def test_merit_fun_reduces_jacobian_evaluations(self, quadratic_system):
        fun, _, _ = quadratic_system
        n_calls = {"plain": 0, "merit": 0}

        def make_counting_fun(label):
            def counting_fun(x):
                n_calls[label] += 1
                return fun(x)

            return counting_fun

        def merit_fun(x):
            return fun(x)[0]

        x0 = np.array([100.0, 100.0])
        solver_plain = newton_armijo(globalization=ArmijoBacktracking(c1=0.5))
        solver_merit = newton_armijo(globalization=ArmijoBacktracking(c1=0.5, merit_fun=merit_fun))
        result_plain = sparse_root(make_counting_fun("plain"), x0, solver=solver_plain, tol=1e-10, progressbar=False)
        result_merit = sparse_root(make_counting_fun("merit"), x0, solver=solver_merit, tol=1e-10, progressbar=False)

        assert result_plain.success
        assert result_merit.success
        np.testing.assert_allclose(result_plain.x, result_merit.x, rtol=1e-8)
        assert n_calls["merit"] < n_calls["plain"]

    def test_merit_fun_counts_one_jacobian_per_step(self, quadratic_system):
        fun, _, _ = quadratic_system

        def merit_fun(x):
            return fun(x)[0]

        solver = newton_armijo(globalization=ArmijoBacktracking(c1=0.5, merit_fun=merit_fun))
        state = solver.init(fun, np.array([100.0, 100.0]), ())
        state, info = solver.step(fun, state, ())

        assert info.accepted
        assert state.stats.nfev > 2
        assert state.stats.njev == 2

    def test_merit_fun_converges_correctly(self, trig_system):
        fun, x0, x_true = trig_system

        def merit_fun(x):
            return fun(x)[0]

        solver = newton_armijo(globalization=ArmijoBacktracking(merit_fun=merit_fun))
        result = sparse_root(fun, x0, solver=solver, tol=1e-10, progressbar=False)

        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)


class TestChordSuite(CommonSolverTests):
    @property
    def solver(self):
        return chord()


class TestInexactNewtonKrylovSuite(CommonSolverTests):
    @property
    def solver(self):
        return inexact_newton_krylov()


class TestInexactNewtonKrylovSpecific:
    def test_direction_is_approximate(self, broyden_system):
        fun, x0 = broyden_system
        direction = KrylovDirection(krylov_method="gmres", eta_max=0.5, eisenstat_walker=False)
        # A fixed loose forcing term converges linearly, so the step tolerance is reached just short of the
        # default residual tolerance. `f_tol` asks for what this configuration can actually deliver.
        result = sparse_root(fun, x0, solver=inexact_newton_krylov(direction=direction), progressbar=False, f_tol=1e-9)
        assert result.success
        np.testing.assert_allclose(fun(result.x)[0], 0.0, atol=1e-8)

    def test_bicgstab_also_converges(self, broyden_system):
        fun, x0 = broyden_system
        direction = KrylovDirection(krylov_method="bicgstab")
        result = sparse_root(fun, x0, solver=inexact_newton_krylov(direction=direction), progressbar=False)
        assert result.success
        np.testing.assert_allclose(fun(result.x)[0], 0.0, atol=1e-8)

    def test_large_broyden(self):
        n = 500

        def fun(x):
            res = (3.0 - 2.0 * x) * x + 1.0
            res[:-1] -= 2.0 * x[1:]
            res[1:] -= x[:-1]
            jac = sp.diags([-np.ones(n - 1), 3.0 - 4.0 * x, -2 * np.ones(n - 1)], [-1, 0, 1], format="csc")
            return res, jac

        result = sparse_root(fun, -np.ones(n), solver=inexact_newton_krylov(), progressbar=False)
        assert result.success
        np.testing.assert_allclose(fun(result.x)[0], 0.0, atol=1e-8)


class TestNewtonNonmonotoneSuite(CommonSolverTests):
    @property
    def solver(self):
        return newton_nonmonotone()


class TestNewtonNonmonotoneSpecific:
    def test_coupled_system(self, coupled_nonlinear):
        fun, x0 = coupled_nonlinear
        result = sparse_root(fun, x0, solver=newton_nonmonotone(), progressbar=False, maxiter=500)
        assert result.success
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-4)

    def test_reused_solver_forgets_previous_merit_history(self, quadratic_system):
        fun, x0, _ = quadratic_system
        solver = newton_nonmonotone()

        sparse_root(fun, x0 * 100.0, solver=solver, progressbar=False)
        solver.init(fun, x0, ())
        assert len(solver.globalization._phi_history) == 0


class TestCustomSolver(CommonSolverTests):
    @property
    def solver(self):
        return LineSearchSolver(
            direction=KrylovDirection(krylov_method="gmres"),
            globalization=NonmonotoneBacktracking(memory=5),
        )
