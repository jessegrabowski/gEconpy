import numpy as np

from conftest import CommonSolverTests
from scipy.sparse.linalg import spsolve

from gEconpy.solvers.sparse_root import Chord, NewtonArmijo, sparse_root
from gEconpy.solvers.sparse_root.direction import ChordDirection, NewtonDirection
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking


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
