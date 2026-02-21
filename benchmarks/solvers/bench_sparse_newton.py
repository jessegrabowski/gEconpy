import numpy as np

from scipy import sparse

SOLVER_TOL = 1e-10
SOLVER_MAXITER = 100


class SparseNewton:
    params = [[100, 500, 1000]]
    param_names = ["system_size"]

    def setup(self, system_size):
        from gEconpy.solvers.sparse_newton import sparse_newton

        rng = np.random.default_rng()

        self.sparse_newton = sparse_newton
        self.x0 = rng.standard_normal(system_size)

        # Diagonal dominance ensures the system is well-conditioned
        diag = 2.0 + rng.random(system_size)
        A = sparse.diags(diag, format="csc")
        b = rng.standard_normal(system_size)

        def fun(x):
            return A @ x - b, A

        self.fun = fun

    def time_solve(self, system_size):
        self.sparse_newton(self.fun, self.x0, tol=SOLVER_TOL, maxiter=SOLVER_MAXITER)

    def peakmem_solve(self, system_size):
        self.sparse_newton(self.fun, self.x0, tol=SOLVER_TOL, maxiter=SOLVER_MAXITER)
