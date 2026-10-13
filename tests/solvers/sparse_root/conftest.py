from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
import scipy.sparse as sp

from scipy.special import lambertw

from gEconpy.model.perfect_foresight.assemble import (
    StackedJacobianLayout,
    assemble_stacked_jacobian,
    build_stacked_jacobian_layout,
)
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
def structurally_singular_system():
    """System where x[1] never enters the residuals. Solution at x[0] = 2, with x[1] free.

    Column 1 of the Jacobian is structurally empty, so ``J.T @ J`` stores no diagonal entry for it.
    """

    def fun(x):
        res = np.array([x[0] ** 2 - 4.0, x[0] - 2.0])
        jac = sp.csc_matrix((np.array([2.0 * x[0], 1.0]), (np.array([0, 1]), np.array([0, 0]))), shape=(2, 2))
        return res, jac

    return fun, np.array([3.0, 1.0])


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


class StackedSystem(NamedTuple):
    """A nonlinear system with the block-tridiagonal structure a perfect foresight solve produces.

    Attributes
    ----------
    fun : callable
        Fused residual and Jacobian function over the stacked unknowns.
    x0 : ndarray
        Starting guess, perturbed off the root.
    x_true : ndarray
        The root, exact by construction.
    layout : StackedJacobianLayout
        Layout the Jacobian is assembled through.
    n_vars : int
        Unknowns per period.
    horizon : int
        Number of periods.
    """

    fun: Callable
    x0: np.ndarray
    x_true: np.ndarray
    layout: StackedJacobianLayout
    n_vars: int
    horizon: int


def _period_sparsity(n_vars):
    """Mask of ``d r_t / d [y_{t-1}, y_t, y_{t+1}]``, four entries per equation."""
    pattern = np.zeros((n_vars, 3 * n_vars), dtype=bool)
    equations = np.arange(n_vars)
    neighbor = (equations + 1) % n_vars
    pattern[equations, equations] = True
    pattern[equations, n_vars + equations] = True
    pattern[equations, n_vars + neighbor] = True
    pattern[equations, 2 * n_vars + equations] = True
    return pattern


@pytest.fixture
def stacked_system():
    """Discretized nonlinear boundary value problem, stacked over a horizon the way perfect foresight stacks one.

    Each period's equations couple ``y_{t-1}``, ``y_t`` and ``y_{t+1}``, so the stacked Jacobian is genuinely
    block-tridiagonal and is assembled through the same layout the perfect foresight solver uses. Per-period
    constants are chosen so ``x_true`` is a root to the bit rather than to a tolerance.
    """
    n_vars, horizon = 24, 200
    nonlinear_gain, coupling_gain = 0.4, 0.3
    rng = np.random.default_rng(0)
    equations = np.arange(n_vars)
    neighbor = (equations + 1) % n_vars

    y_true = 0.3 + 0.1 * rng.normal(size=(horizon, n_vars))
    y_initial = 0.3 + 0.1 * rng.normal(size=n_vars)
    y_terminal = 0.3 + 0.1 * rng.normal(size=n_vars)

    def period_residual(previous, current, following):
        return (
            following
            - 2.0 * current
            + previous
            + nonlinear_gain * (current - current**3)
            + coupling_gain * (current[..., neighbor] - current)
        )

    true_stack = np.vstack([y_initial, y_true, y_terminal])
    offsets = period_residual(true_stack[:-2], true_stack[1:-1], true_stack[2:])

    layout = build_stacked_jacobian_layout(_period_sparsity(n_vars), n_vars=n_vars, n_eq=n_vars, T=horizon)

    def fun(x):
        stack = np.vstack([y_initial, x.reshape(horizon, n_vars), y_terminal])
        current = stack[1:-1]
        residuals = period_residual(stack[:-2], current, stack[2:]) - offsets

        jacobians = np.zeros((horizon, n_vars, 3 * n_vars))
        jacobians[:, equations, equations] = 1.0
        jacobians[:, equations, n_vars + equations] = -2.0 + nonlinear_gain * (1.0 - 3.0 * current**2) - coupling_gain
        jacobians[:, equations, n_vars + neighbor] = coupling_gain
        jacobians[:, equations, 2 * n_vars + equations] = 1.0
        return residuals.ravel(), assemble_stacked_jacobian(jacobians, layout)

    x_true = y_true.ravel()
    return StackedSystem(
        fun=fun,
        x0=x_true + 0.05 * rng.normal(size=x_true.size),
        x_true=x_true,
        layout=layout,
        n_vars=n_vars,
        horizon=horizon,
    )
