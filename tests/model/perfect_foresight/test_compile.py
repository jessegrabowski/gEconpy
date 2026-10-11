import numpy as np
import pytest
import sympy as sp

from numpy.testing import assert_allclose

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.perfect_foresight import compile_perfect_foresight_problem
from gEconpy.model.perfect_foresight.compile import _check_system_is_solvable
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.fixture
def rbc_model():
    return load_and_cache_model("one_block_1.gcn")


class TestCompile:
    def test_problem_has_correct_dimensions(self, rbc_model):
        T = 25
        problem = compile_perfect_foresight_problem(rbc_model, T)

        assert problem.T == T
        assert problem.n_vars == len(rbc_model.variables)
        assert problem.n_shocks == len(rbc_model.shocks)
        assert problem.n_eq == len(rbc_model.equations)

    def test_residuals_vanish_at_steady_state_and_jacobian_matches_finite_differences(self, rbc_model):
        problem = compile_perfect_foresight_problem(rbc_model, 5)
        ss = rbc_model.steady_state(verbose=False)
        params = rbc_model.parameters()

        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        x_zero = np.zeros(problem.n_shocks)
        param_values = [params[name] for name in problem.param_names]

        residuals, jacobian = problem.f_resid_and_jac(y_ss, y_ss, y_ss, x_zero, *param_values)
        assert_allclose(residuals, 0.0, atol=1e-8)
        assert jacobian.shape == (problem.n_eq, 3 * problem.n_vars)

        rng = np.random.default_rng(0)
        point = np.concatenate([y_ss * (1 + 0.05 * rng.normal(size=problem.n_vars)) for _ in range(3)])

        def residual_at(stacked):
            y_tm1, y_t, y_tp1 = np.split(stacked, 3)
            return problem.f_resid_and_jac(y_tm1, y_t, y_tp1, x_zero, *param_values)[0]

        _, jacobian = problem.f_resid_and_jac(*np.split(point, 3), x_zero, *param_values)
        step = 1e-6
        numerical = np.stack(
            [
                (residual_at(point + step * basis) - residual_at(point - step * basis)) / (2 * step)
                for basis in np.eye(point.size)
            ],
            axis=1,
        )
        assert_allclose(jacobian, numerical, rtol=1e-5, atol=1e-6)

        (residuals_only,) = problem.f_resid_only(*np.split(point, 3), x_zero, *param_values)
        assert_allclose(residuals_only, residual_at(point))

    def test_jacobian_sparsity_covers_every_numerically_nonzero_entry(self, rbc_model):
        problem = compile_perfect_foresight_problem(rbc_model, 5)
        ss = rbc_model.steady_state(verbose=False)
        params = rbc_model.parameters()

        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        param_values = [params[name] for name in problem.param_names]

        assert problem.jacobian_sparsity.shape == (problem.n_eq, 3 * problem.n_vars)

        rng = np.random.default_rng(0)
        observed = np.zeros_like(problem.jacobian_sparsity)
        for _ in range(10):
            point = [y_ss * (1 + 0.1 * rng.normal(size=problem.n_vars)) for _ in range(3)]
            _, jacobian = problem.f_resid_and_jac(*point, rng.normal(size=problem.n_shocks), *param_values)
            observed |= jacobian != 0

        missing_from_mask = observed & ~problem.jacobian_sparsity
        assert not missing_from_mask.any()

    @pytest.mark.include_nk
    def test_sparsity_keeps_entries_that_vanish_at_the_steady_state(self):
        """
        The steady state is a degenerate point, so the mask cannot be read off a Jacobian evaluated there.

        Investment adjustment costs carry a factor of :math:`I_t / I_{t-1} - 1`, which is zero at the steady
        state. Since the steady state is also the default starting path, a pattern taken from the numbers there
        drops those entries from every period at once and the first Newton step solves against a wrong Jacobian.
        """
        model = load_and_cache_model("full_nk.gcn")
        problem = compile_perfect_foresight_problem(model, 5)
        ss = model.steady_state(verbose=False)
        params = model.parameters()

        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        param_values = [params[name] for name in problem.param_names]
        _, jacobian = problem.f_resid_and_jac(y_ss, y_ss, y_ss, np.zeros(problem.n_shocks), *param_values)

        vanishing = problem.jacobian_sparsity & (jacobian == 0)
        assert vanishing.any(), "expected entries that are structurally nonzero but zero at the steady state"

        missing_from_mask = (jacobian != 0) & ~problem.jacobian_sparsity
        assert not missing_from_mask.any()

    def test_steady_state_reference_without_analytic_solution_raises(self):
        model = load_and_cache_model("full_nk_no_ss.gcn")
        with pytest.raises(ValueError, match="following do not: pi_ss, r_G_ss"):
            compile_perfect_foresight_problem(model, 5)

    def test_calibrated_parameter_raises(self):
        """A calibrated parameter is solved for with the steady state, so it has no value at compile time."""
        model = load_and_cache_model("one_block_2.gcn")
        with pytest.raises(ValueError, match=r"does not support calibrated parameters.*alpha"):
            compile_perfect_foresight_problem(model, 5)


class TestSystemChecks:
    """``_check_system_is_solvable`` guards the two shapes that otherwise fail deep inside pytensor or scipy."""

    def test_variables_absent_at_a_lag_or_lead_are_allowed(self):
        """``x`` never appears at ``t+1`` and ``y`` never at ``t-1``; only absence at ``t`` is a problem."""
        _check_system_is_solvable(
            equations=[sp.Symbol("x"), sp.Symbol("y")],
            vars_tm1=[TimeAwareSymbol("x", -1)],
            vars_t=[TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)],
            vars_tp1=[TimeAwareSymbol("y", 1)],
        )

    def test_variable_never_at_time_t_raises(self):
        with pytest.raises(ValueError, match=r"appear only at t-1 or t\+1: z"):
            _check_system_is_solvable(
                equations=[sp.Symbol("x"), sp.Symbol("y")],
                vars_tm1=[TimeAwareSymbol("z", -1)],
                vars_t=[TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)],
                vars_tp1=[],
            )

    def test_non_square_system_raises(self):
        with pytest.raises(ValueError, match="2 equations and 3 variables"):
            _check_system_is_solvable(
                equations=[sp.Symbol("x"), sp.Symbol("y")],
                vars_tm1=[],
                vars_t=[TimeAwareSymbol(name, 0) for name in ("x", "y", "z")],
                vars_tp1=[],
            )
