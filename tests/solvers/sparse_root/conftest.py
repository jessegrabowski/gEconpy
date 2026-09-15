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
    """x0*cos(x1) - 4 = 0, x1*x0 - x1 - 5 = 0. Solution approx [6.504, 0.908]."""

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
    """System with singular Jacobian at x0 = [1, 1]. Solution at [1, 2] or [2, 1]."""

    def fun(x):
        res = np.array([x[0] + x[1] - 3.0, x[0] * x[1] - 2.0])
        jac = sp.csc_matrix([[1.0, 1.0], [x[1], x[0]]])
        return res, jac

    return fun, np.array([1.0, 1.0])


@pytest.fixture
def powell_badly_scaled():
    """Powell's badly scaled system (scipy F4_powell), ill-conditioned because of the 1e4 scaling.

    f0 = A * x0 * x1 - 1
    f1 = exp(-x0) + exp(-x1) - (1 + 1 / A)
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

    Models pressures and flows in a system of n parallel pipes with P = k * Q^2 and the constraint sum(Q) = Qtot.
    """
    Qtot = 4.0
    k = np.array([0.5, 0.5, 0.5, 0.5])
    n = len(k)

    def fun(x):
        P = k * x**2
        res = np.hstack((P[1:] - P[0], x.sum() - Qtot))

        jac = np.zeros((n, n))
        jac[: n - 1, 0] = -2 * k[0] * x[0]
        jac[np.arange(n - 1), np.arange(1, n)] = 2 * k[1:] * x[1:]
        jac[n - 1, :] = 1.0
        return res, sp.csc_matrix(jac)

    return fun, np.array([2.0, 0.0, 2.0, 0.0])


@pytest.fixture
def rosenbrock_root():
    """2D Rosenbrock system as a root-finding problem, with solution at [1, 1].

    f0 = 1 - x0
    f1 = 10 * (x1 - x0^2)
    """

    def fun(x):
        res = np.array([1.0 - x[0], 10.0 * (x[1] - x[0] ** 2)])
        jac = sp.csc_matrix([[-1.0, 0.0], [-20.0 * x[0], 10.0]])
        return res, jac

    return fun, np.array([-1.2, 1.0]), np.array([1.0, 1.0])


@pytest.fixture
def linear_system():
    """Random well-conditioned linear system A x = b, after scipy.optimize.tests.test_nonlin.TestLinear."""
    rng = np.random.default_rng(42)
    n = 20
    A_dense = rng.standard_normal((n, n))
    A_dense = A_dense + A_dense.T + 3 * np.eye(n)
    b = rng.standard_normal(n)
    x_true = np.linalg.solve(A_dense, b)

    A_sparse = sp.csc_matrix(A_dense)

    def fun(x):
        return A_sparse @ x - b, A_sparse

    return fun, np.zeros(n), x_true


@pytest.fixture
def exponential_fixed_point():
    """Scalar fixed point exp(-x) = x, with solution W(1) near 0.5671."""

    def fun(x):
        res = np.array([np.exp(-x[0]) - x[0]])
        jac = sp.csc_matrix([[-np.exp(-x[0]) - 1.0]])
        return res, jac

    x_true = np.array([np.real(lambertw(1.0))])
    return fun, np.array([0.5]), x_true


@pytest.fixture
def helical_valley():
    """Helical valley function (More, Garbow and Hillstrom #7), a 3D system with solution at [1, 0, 0]."""

    def theta(x):
        if x[0] > 0:
            return np.arctan(x[1] / x[0]) / (2 * np.pi)
        return np.arctan(x[1] / x[0]) / (2 * np.pi) + 0.5

    def residual(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return np.array([10.0 * (x[2] - 10.0 * theta(x)), 10.0 * (r - 1.0), x[2]])

    def fun(x):
        res = residual(x)
        return res, _finite_difference_jacobian(residual, x, res)

    return fun, np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])


@pytest.fixture
def coupled_nonlinear():
    """Strongly coupled nonlinear system from scipy F6, preconditioned by a fixed matrix."""
    J0 = np.array([[-4.256, 14.7], [0.8394989, 0.59964207]])

    def residual(x):
        v = np.array([(x[0] + 3) * (x[1] ** 5 - 7) + 3 * 6, np.sin(x[1] * np.exp(x[0]) - 1)])
        return -np.linalg.solve(J0, v)

    def fun(x):
        res = residual(x)
        return res, _finite_difference_jacobian(residual, x, res)

    return fun, np.array([-0.5, 1.4])


def _finite_difference_jacobian(residual, x, res, eps=1e-8):
    jac = np.zeros((len(x), len(x)))
    for j in range(len(x)):
        x_perturbed = x.copy()
        x_perturbed[j] += eps
        jac[:, j] = (residual(x_perturbed) - res) / eps
    return sp.csc_matrix(jac)


class CommonSolverTests:
    """Convergence and robustness tests shared by every solver.

    Subclass in each ``test_*.py`` and set ``solver`` to run the full suite against a particular solver.
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
        assert result.success
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-6)

    def test_helical_valley(self, helical_valley):
        fun, x0, _ = helical_valley
        result = sparse_root(fun, x0, solver=self.solver, progressbar=False, maxiter=500)
        assert result.success
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-4)

    def test_already_at_root(self, quadratic_system):
        fun, _, x_true = quadratic_system
        result = sparse_root(fun, x_true, solver=self.solver, progressbar=False)
        assert result.success
        assert result.nit == 0

    def test_far_start(self, quadratic_system):
        fun, _, x_true = quadratic_system
        result = sparse_root(fun, np.array([1e4, 1e4]), solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, x_true, rtol=1e-4)

    def test_maxiter_respected(self, quadratic_system):
        fun, _, _ = quadratic_system
        result = sparse_root(fun, np.array([1e6, 1e6]), solver=self.solver, maxiter=3, progressbar=False)
        assert not result.success
        assert result.nit <= 3

    def test_single_variable(self):
        def fun(x):
            return np.array([x[0] ** 3 - 1.0]), sp.csc_matrix([[3.0 * x[0] ** 2]])

        result = sparse_root(fun, np.array([0.5]), solver=self.solver, progressbar=False)
        assert result.success
        np.testing.assert_allclose(result.x, [1.0], rtol=1e-6)
