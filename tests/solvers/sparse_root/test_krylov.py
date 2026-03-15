import numpy as np
import scipy.sparse as sp

from conftest import CommonSolverTests

from gEconpy.solvers.sparse_root import InexactNewtonKrylov, sparse_root
from gEconpy.solvers.sparse_root.direction import KrylovDirection


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
