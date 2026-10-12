import numpy as np
import scipy.sparse as sp

from gEconpy.solvers.sparse_root import sparse_root


class CommonSolverTests:
    """Convergence and robustness tests shared by every solver.

    Subclass in each ``test_*.py`` and set ``solver`` to run the full suite against a particular solver.
    """

    solver = None

    def test_converged_implies_the_residual_tolerance(self, quadratic_system):
        """A ``Converged`` report has to be true, which is the assertion whose absence let a stall claim success.

        A tolerance this tight is met only by landing on the root exactly, which some solvers do and some do not.
        That split is what makes the regime useful: a solver that stops short must not call it convergence, and
        asserting the implication on an ordinary solve cannot tell the difference.
        """
        fun, x0, _ = quadratic_system
        f_tol = 1e-300

        result = sparse_root(fun, x0, solver=self.solver, progressbar=False, f_tol=f_tol, x_tol=1e-6)
        residual = np.max(np.abs(result.fun))

        assert not (result.message == "Converged" and residual >= f_tol), (
            f"reported Converged with max |residual| = {residual:.2e}, above f_tol={f_tol:.0e}"
        )

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
        res, _ = fun(result.x)
        np.testing.assert_allclose(res, 0.0, atol=1e-8)

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
