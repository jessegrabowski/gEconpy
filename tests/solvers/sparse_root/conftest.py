import numpy as np
import pytest
import scipy.sparse as sp

from scipy.special import lambertw

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


@pytest.fixture
def singular_system():
    """System with singular Jacobian at x0=[1,1]. Solution at [1,2] or [2,1]."""

    def fun(x):
        res = np.array([x[0] + x[1] - 3.0, x[0] * x[1] - 2.0])
        jac = sp.csc_matrix([[1.0, 1.0], [x[1], x[0]]])
        return res, jac

    return fun, np.array([1.0, 1.0])


@pytest.fixture
def powell_badly_scaled():
    """Powell's badly scaled system (scipy F4_powell).

    A = 1e4
    f0 = A*x0*x1 - 1
    f1 = exp(-x0) + exp(-x1) - (1 + 1/A)

    This is extremely ill-conditioned due to the 1e4 scaling.
    """

    def fun(x):
        A = 1e4
        res = np.array([A * x[0] * x[1] - 1.0, np.exp(-x[0]) + np.exp(-x[1]) - (1.0 + 1.0 / A)])
        jac = sp.csc_matrix(
            [[A * x[1], A * x[0]], [-np.exp(-x[0]), -np.exp(-x[1])]],
        )
        return res, jac

    return fun, np.array([0.0, 1.0])


@pytest.fixture
def pressure_network():
    """Pressure network from scipy.optimize.tests.test_minpack.

    Models pressures and flows in a system of n parallel pipes.
    P = k * Q^2, with constraint sum(Q) = Qtot.
    """
    Qtot = 4.0
    k = np.array([0.5, 0.5, 0.5, 0.5])
    n = len(k)

    def fun(x):
        P = k * x**2
        res = np.hstack((P[1:] - P[0], x.sum() - Qtot))
        # Jacobian
        rows, cols, vals = [], [], []
        # dP[i] - dP[0] for i=1..n-1
        for i in range(n - 1):
            rows.append(i)
            cols.append(0)
            vals.append(-2 * k[0] * x[0])
            rows.append(i)
            cols.append(i + 1)
            vals.append(2 * k[i + 1] * x[i + 1])
        # sum(Q) - Qtot row
        for j in range(n):
            rows.append(n - 1)
            cols.append(j)
            vals.append(1.0)
        jac = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        return res, jac

    return fun, np.array([2.0, 0.0, 2.0, 0.0])


@pytest.fixture
def rosenbrock_root():
    """2D Rosenbrock system as a root-finding problem.

    f0 = 1 - x0
    f1 = 10*(x1 - x0^2)

    Solution at [1, 1]. From scipy.optimize rosen / optimistix rosenbrock.
    """

    def fun(x):
        res = np.array([1.0 - x[0], 10.0 * (x[1] - x[0] ** 2)])
        jac = sp.csc_matrix([[-1.0, 0.0], [-20.0 * x[0], 10.0]])
        return res, jac

    return fun, np.array([-1.2, 1.0]), np.array([1.0, 1.0])


@pytest.fixture
def linear_system():
    """Random sparse linear system A x = b as a root-finding problem.

    Derived from scipy.optimize.tests.test_nonlin.TestLinear.
    """
    rng = np.random.default_rng(42)
    n = 20
    A_dense = rng.standard_normal((n, n))
    A_dense = A_dense + A_dense.T + 3 * np.eye(n)  # Make well-conditioned
    b = rng.standard_normal(n)
    x_true = np.linalg.solve(A_dense, b)

    A_sp = sp.csc_matrix(A_dense)

    def fun(x):
        return A_sp @ x - b, A_sp

    return fun, np.zeros(n), x_true


@pytest.fixture
def exponential_fixed_point():
    """exp(-x) = x as a root-finding problem.

    f(x) = exp(-x) - x. Scalar-valued, solution near 0.5671.
    From optimistix _exponential fixed point.
    """

    def fun(x):
        res = np.array([np.exp(-x[0]) - x[0]])
        jac = sp.csc_matrix([[-np.exp(-x[0]) - 1.0]])
        return res, jac

    x_true = np.array([np.real(lambertw(1.0))])
    return fun, np.array([0.5]), x_true


@pytest.fixture
def trigonometric_system():
    """Trigonometric function from More, Garbow, Hillstrom (TUOS eqn 26).

    f_i = n - sum(cos(x_j)) + i*(1 - cos(x_i)) - sin(x_i)

    Adapted from optimistix helpers. Hard for many solvers due to many
    near-solutions.
    """
    n = 20

    def fun(x):
        sumcos = np.sum(np.cos(x))
        idx = np.arange(1, n + 1, dtype=np.float64)
        res = float(n) - sumcos + idx * (1.0 - np.cos(x)) - np.sin(x)

        # Jacobian: df_i/dx_j
        # Off-diagonal: sin(x_j)
        # Diagonal: sin(x_i) + i*sin(x_i) - cos(x_i)
        # = (1 + i)*sin(x_i) - cos(x_i)
        diag_vals = (1.0 + idx) * np.sin(x) - np.cos(x)
        off_diag = np.sin(x)

        # Dense column of off-diagonal contributions: each row i gets sin(x_j) for all j
        # J_ij = sin(x_j) for i != j, J_ii = diag_vals[i]
        # J = ones_matrix * sin(x) row + diag_correction
        J_dense = np.tile(off_diag, (n, 1))
        np.fill_diagonal(J_dense, diag_vals)
        jac = sp.csc_matrix(J_dense)
        return res, jac

    return fun, np.ones(n) / n


@pytest.fixture
def helical_valley():
    """Helical valley function (More, Garbow, Hillstrom #7).

    A 3D nonlinear system with a helical geometry.
    Solution at [1, 0, 0].
    """

    def theta(x):
        if x[0] > 0:
            return np.arctan(x[1] / x[0]) / (2 * np.pi)
        return np.arctan(x[1] / x[0]) / (2 * np.pi) + 0.5

    def fun(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        t = theta(x)
        res = np.array([10.0 * (x[2] - 10.0 * t), 10.0 * (r - 1.0), x[2]])

        # Numerical Jacobian via finite differences (analytical is complex)
        eps = 1e-8
        n = len(x)
        jac_dense = np.zeros((n, n))
        r0 = res.copy()
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            r_pert = np.array(
                [
                    10.0 * (x_pert[2] - 10.0 * theta(x_pert)),
                    10.0 * (np.sqrt(x_pert[0] ** 2 + x_pert[1] ** 2) - 1.0),
                    x_pert[2],
                ]
            )
            jac_dense[:, j] = (r_pert - r0) / eps
        jac = sp.csc_matrix(jac_dense)
        return res, jac

    return fun, np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])


@pytest.fixture
def coupled_nonlinear():
    """Strongly coupled nonlinear system from scipy F6.

    Uses a preconditioner matrix. Tests solver robustness with coupling.
    """
    J0 = np.array([[-4.256, 14.7], [0.8394989, 0.59964207]])

    def fun(x):
        v = np.array([(x[0] + 3) * (x[1] ** 5 - 7) + 3 * 6, np.sin(x[1] * np.exp(x[0]) - 1)])
        res = -np.linalg.solve(J0, v)

        eps = 1e-8
        n = len(x)
        jac_dense = np.zeros((n, n))
        for j in range(n):
            xp = x.copy()
            xp[j] += eps
            vp = np.array([(xp[0] + 3) * (xp[1] ** 5 - 7) + 3 * 6, np.sin(xp[1] * np.exp(xp[0]) - 1)])
            res_p = -np.linalg.solve(J0, vp)
            jac_dense[:, j] = (res_p - res) / eps
        jac = sp.csc_matrix(jac_dense)
        return res, jac

    return fun, np.array([-0.5, 1.4])


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

    def test_linear(self, linear_system):
        fun, x0, x_true = linear_system
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)

    def test_exponential_fixed_point(self, exponential_fixed_point):
        fun, x0, x_true = exponential_fixed_point
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-6)

    def test_rosenbrock(self, rosenbrock_root):
        fun, x0, x_true = rosenbrock_root
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False, maxiter=500)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-4)

    def test_broyden(self, broyden_system):
        fun, x0 = broyden_system
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False)
        assert result.success

    def test_pressure_network(self, pressure_network):
        fun, x0 = pressure_network
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False, maxiter=500)
        if result.success:
            res, _ = fun(result.x)
            np.testing.assert_allclose(res, 0.0, atol=1e-6)

    def test_helical_valley(self, helical_valley):
        fun, x0, _ = helical_valley
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False, maxiter=500)
        if result.success:
            res, _ = fun(result.x)
            np.testing.assert_allclose(res, 0.0, atol=1e-4)

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

    def test_zero_residual_at_x0(self):
        """If x0 is already a root, return immediately."""

        def fun(x):
            return x - np.array([1.0, 2.0]), sp.eye(2, format="csc")

        result = sparse_root(fun, np.array([1.0, 2.0]), solver=self.solver, progressbar=False)
        assert result.success
        assert result.nit == 0

    def test_single_variable(self):
        """1D root-finding."""

        def fun(x):
            return np.array([x[0] ** 3 - 1.0]), sp.csc_matrix([[3.0 * x[0] ** 2]])

        result = sparse_root(fun, np.array([0.5]), solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, [1.0], rtol=1e-6)
