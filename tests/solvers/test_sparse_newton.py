import numpy as np
import pytest

from scipy import sparse
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import gmres

from gEconpy.solvers.sparse_newton import sparse_newton


def make_linear_system(A_dense, b):
    """Create a test function that solves Ax = b."""
    A_sparse = sparse.csc_matrix(A_dense)

    def fun(x):
        return A_sparse @ x - b, A_sparse

    return fun


def make_nonlinear_system():
    """Create a simple nonlinear system: x^2 - [1, 4] = 0."""

    def fun(x):
        res = x**2 - np.array([1.0, 4.0])
        jac = sparse.diags(2 * x, format="csc")
        return res, jac

    return fun


def make_trig_system(a=1.0, b=1.0):
    """Scipy's func2 test problem for root-finding.

    System:
        a * x[0] * cos(x[1]) - 4 = 0
        x[1] * x[0] - b * x[1] - 5 = 0

    Solution (for a=1, b=1): approximately [6.50409711, 0.90841421]
    """

    def fun(x):
        res = np.array(
            [
                a * x[0] * np.cos(x[1]) - 4,
                x[1] * x[0] - b * x[1] - 5,
            ]
        )
        jac = sparse.csc_matrix(
            [
                [a * np.cos(x[1]), -a * x[0] * np.sin(x[1])],
                [x[1], x[0] - b],
            ]
        )
        return res, jac

    return fun


class TestSparseNewtonValidation:
    def test_rejects_non_tuple_return(self):
        def bad_fun(x):
            return x

        with pytest.raises(ValueError, match="must return a tuple"):
            sparse_newton(bad_fun, np.array([1.0]))

    def test_rejects_wrong_tuple_length(self):
        def bad_fun(x):
            return (x, x, x)

        with pytest.raises(ValueError, match="must return a tuple"):
            sparse_newton(bad_fun, np.array([1.0]))

    def test_rejects_non_array_residuals(self):
        def bad_fun(_x):
            return [1.0, 2.0], sparse.eye(2)

        with pytest.raises(TypeError, match="residuals as ndarray"):
            sparse_newton(bad_fun, np.array([1.0, 2.0]))

    def test_rejects_dense_jacobian(self):
        def bad_fun(_x):
            return np.array([1.0]), np.array([[1.0]])

        with pytest.raises(TypeError, match="sparse jacobian"):
            sparse_newton(bad_fun, np.array([1.0]))


class TestSparseNewtonConvergence:
    def test_linear_system(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x_true = np.linalg.solve(A, b)

        result = sparse_newton(make_linear_system(A, b), np.zeros(2))

        assert result.success
        assert result.nit == 1
        np.testing.assert_allclose(result.x, x_true)

    def test_nonlinear_system(self):
        result = sparse_newton(make_nonlinear_system(), np.array([2.0, 3.0]))

        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-8)

    def test_trig_system(self):
        result = sparse_newton(make_trig_system(a=1.0, b=1.0), np.array([1.0, 1.0]))

        assert result.success
        np.testing.assert_allclose(result.x, [6.50409711, 0.90841421], rtol=1e-6)

    def test_converged_initial_guess(self):
        result = sparse_newton(make_nonlinear_system(), np.array([1.0, 2.0]))

        assert result.success
        assert result.nit == 0
        assert result.nfev == 1

    def test_maxiter_reached(self):
        result = sparse_newton(make_nonlinear_system(), np.array([100.0, 100.0]), maxiter=2)

        assert not result.success
        assert result.nit == 2
        assert "Did not converge" in result.message


class TestSparseNewtonOptions:
    def test_gmres_solver(self):
        def gmres_solver(A, b):
            x, _ = gmres(A, b, atol=1e-12)
            return x

        result = sparse_newton(make_trig_system(), np.array([1.0, 1.0]), solver=gmres_solver)

        assert result.success
        np.testing.assert_allclose(result.x, [6.50409711, 0.90841421], rtol=1e-6)

    def test_args_passed_to_fun(self):
        result = sparse_newton(
            make_trig_system(a=2.0, b=0.5),
            np.array([1.0, 1.0]),
        )

        assert result.success
        # Verify solution satisfies the system
        x = result.x
        np.testing.assert_allclose(2.0 * x[0] * np.cos(x[1]) - 4, 0, atol=1e-8)
        np.testing.assert_allclose(x[1] * x[0] - 0.5 * x[1] - 5, 0, atol=1e-8)
