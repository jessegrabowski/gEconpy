import numpy as np
import scipy.sparse as sp

from gEconpy.solvers.sparse_root.direction import ChordDirection, KrylovDirection, NewtonDirection


class TestNewtonDirectionFallback:
    def test_fallback_on_singular_jacobian(self):
        direction = NewtonDirection()
        jac = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])
        proposal = direction.compute(np.zeros(2), np.array([1.0, 2.0]), jac)
        assert "fallback" in proposal.kind

    def test_fallback_on_nan_from_solver(self):
        direction = NewtonDirection(linear_solver=lambda _A, _b: np.array([np.nan, np.nan]))
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), sp.eye(2, format="csc"))
        assert "fallback" in proposal.kind

    def test_ensures_descent_direction(self):
        direction = NewtonDirection()
        jac = sp.csc_matrix([[-1.0, 0.0], [0.0, -1.0]])
        proposal = direction.compute(np.zeros(2), np.array([1.0, 1.0]), jac)
        assert proposal.slope < 0


class TestChordDirectionSpecific:
    def test_cache_persists(self):
        """Cached Jacobian should be reused for recompute_every steps."""
        cd = ChordDirection(recompute_every=3)
        cd.reset()

        jac1 = sp.eye(2, format="csc")
        jac2 = 2.0 * sp.eye(2, format="csc")
        res = np.array([1.0, 1.0])
        x = np.zeros(2)

        # Call 0: _call_count=0, 0%3==0 -> cache jac1, increment to 1
        cd.compute(x, res, jac1)
        assert cd._cached_jac is not None
        cached_data = cd._cached_jac.data.copy()

        # Call 1: _call_count=1, 1%3!=0 -> reuse jac1, increment to 2
        cd.compute(x, res, jac2)
        np.testing.assert_array_equal(cd._cached_jac.data, cached_data)

        # Call 2: _call_count=2, 2%3!=0 -> still reuse jac1, increment to 3
        cd.compute(x, res, jac2)
        np.testing.assert_array_equal(cd._cached_jac.data, cached_data)

        # Call 3: _call_count=3, 3%3==0 -> refresh to jac2, increment to 4
        cd.compute(x, res, jac2)
        np.testing.assert_array_equal(cd._cached_jac.data, jac2.data)


class TestKrylovDirectionSpecific:
    def test_eisenstat_walker_tightens(self, broyden_system):
        """Forcing term eta should decrease as residual decreases."""
        fun, x0 = broyden_system
        kd = KrylovDirection(eta_max=0.9, eisenstat_walker=True)
        kd.reset()

        res, jac = fun(x0)
        etas = [kd._eta]

        x = x0.copy()
        for _ in range(10):
            proposal = kd.compute(x, res, jac)
            x = x + proposal.direction
            res, jac = fun(x)
            etas.append(kd._eta)

        assert etas[-1] < etas[0]

    def test_fallback_on_bad_krylov(self):
        """If Krylov solve fails, direction should fall back to gradient."""
        kd = KrylovDirection(krylov_method="gmres")
        kd.reset()
        jac = sp.csc_matrix((2, 2))
        res = np.array([1.0, 1.0])
        proposal = kd.compute(np.zeros(2), res, jac)
        assert "fallback" in proposal.kind
