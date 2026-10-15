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


def _exogenous_inputs(problem, model):
    """Zero shocks and the model's own parameters, held flat across the horizon."""
    params = model.parameters()

    return (
        np.zeros((problem.T, problem.n_shocks)),
        np.tile([params[name] for name in problem.param_names], (problem.T, 1)),
    )


class TestCompile:
    def test_problem_has_correct_dimensions(self, rbc_model):
        T = 25
        problem = compile_perfect_foresight_problem(rbc_model, T)

        assert problem.T == T
        assert problem.n_vars == len(rbc_model.variables)
        assert problem.n_shocks == len(rbc_model.shocks)
        assert problem.n_eq == len(rbc_model.equations)
        assert problem.deterministic_param_names == ()

    def test_residuals_vanish_at_the_steady_state(self, rbc_model):
        T = 5
        problem = compile_perfect_foresight_problem(rbc_model, T)
        ss = rbc_model.steady_state(verbose=False)

        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        Y_ss = np.tile(y_ss, (T, 1))
        X, P = _exogenous_inputs(problem, rbc_model)

        residuals, jacobian = problem.f_resid_and_jac(Y_ss, Y_ss, Y_ss, X, P)

        assert_allclose(residuals, 0.0, atol=1e-8)
        assert residuals.shape == (T, problem.n_eq)
        assert jacobian.shape == (T, problem.n_eq, 3 * problem.n_vars)

    def test_jacobian_matches_finite_differences(self, rbc_model):
        T = 5
        problem = compile_perfect_foresight_problem(rbc_model, T)
        ss = rbc_model.steady_state(verbose=False)
        X, P = _exogenous_inputs(problem, rbc_model)

        rng = np.random.default_rng(0)
        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        point = np.concatenate([y_ss * (1 + 0.05 * rng.normal(size=problem.n_vars)) for _ in range(3)])

        def residuals_at(stacked):
            Y_tm1, Y_t, Y_tp1 = (np.tile(block, (T, 1)) for block in np.split(stacked, 3))
            return problem.f_resid_and_jac(Y_tm1, Y_t, Y_tp1, X, P)[0][0]

        step = 1e-6
        numerical = np.stack(
            [
                (residuals_at(point + step * basis) - residuals_at(point - step * basis)) / (2 * step)
                for basis in np.eye(point.size)
            ],
            axis=1,
        )

        Y_tm1, Y_t, Y_tp1 = (np.tile(block, (T, 1)) for block in np.split(point, 3))
        _, jacobian = problem.f_resid_and_jac(Y_tm1, Y_t, Y_tp1, X, P)

        assert_allclose(jacobian[0], numerical, rtol=1e-5, atol=1e-6)

    def test_both_compiled_functions_agree_on_the_residuals(self, rbc_model):
        """
        The line search trusts the residual-only function to stand in for the fused one.

        They are two compilations of one graph, so pytensor may order the arithmetic differently and the two
        agree to a few ULP rather than exactly.
        """
        T = 5
        problem = compile_perfect_foresight_problem(rbc_model, T)
        ss = rbc_model.steady_state(verbose=False)
        X, P = _exogenous_inputs(problem, rbc_model)

        rng = np.random.default_rng(0)
        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        blocks = [y_ss * (1 + 0.05 * rng.normal(size=(T, problem.n_vars))) for _ in range(3)]

        fused, _ = problem.f_resid_and_jac(*blocks, X, P)
        (residuals_only,) = problem.f_resid_only(*blocks, X, P)

        assert_allclose(residuals_only, fused, rtol=1e-13, atol=0)

    def test_every_period_is_evaluated_at_its_own_inputs(self, rbc_model):
        """One call covers the horizon, so a row of the output must depend only on the matching row of the input."""
        T = 4
        problem = compile_perfect_foresight_problem(rbc_model, T)
        ss = rbc_model.steady_state(verbose=False)

        rng = np.random.default_rng(1)
        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        Y = y_ss * (1 + 0.05 * rng.normal(size=(T, problem.n_vars)))
        X, P = _exogenous_inputs(problem, rbc_model)

        batched_residuals, batched_jacobian = problem.f_resid_and_jac(Y, Y, Y, X, P)
        for t in range(T):
            one_row = np.tile(Y[t], (T, 1))
            residuals, jacobian = problem.f_resid_and_jac(one_row, one_row, one_row, X, P)

            assert_allclose(batched_residuals[t], residuals[0], rtol=1e-12)
            assert_allclose(batched_jacobian[t], jacobian[0], rtol=1e-12)

    def test_jacobian_sparsity_covers_every_numerically_nonzero_entry(self, rbc_model):
        T = 10
        problem = compile_perfect_foresight_problem(rbc_model, T)
        ss = rbc_model.steady_state(verbose=False)

        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        _, P = _exogenous_inputs(problem, rbc_model)

        assert problem.jacobian_sparsity.shape == (problem.n_eq, 3 * problem.n_vars)

        rng = np.random.default_rng(0)
        blocks = [y_ss * (1 + 0.1 * rng.normal(size=(T, problem.n_vars))) for _ in range(3)]
        _, jacobian = problem.f_resid_and_jac(*blocks, rng.normal(size=(T, problem.n_shocks)), P)

        observed = (jacobian != 0).any(axis=0)
        assert not (observed & ~problem.jacobian_sparsity).any()

        # A mask of all True would satisfy the superset check above while carrying no information.
        assert not problem.jacobian_sparsity.all()
        assert problem.jacobian_sparsity.sum() >= observed.sum()

    @pytest.mark.include_nk
    def test_sparsity_keeps_entries_that_vanish_at_the_steady_state(self):
        """
        The steady state is a degenerate point, so the mask cannot be read off a Jacobian evaluated there.

        Investment adjustment costs carry a factor of :math:`I_t / I_{t-1} - 1`, which is zero at the steady
        state. Since the steady state is also the default starting path, a pattern taken from the numbers there
        drops those entries from every period at once and the first Newton step solves against a wrong Jacobian.
        """
        T = 5
        model = load_and_cache_model("full_nk.gcn")
        # The sparsity pattern is the same in every mode, and FAST_COMPILE skips minutes of numba compilation on CI.
        problem = compile_perfect_foresight_problem(model, T, mode="FAST_COMPILE")
        ss = model.steady_state(verbose=False)

        y_ss = np.array([ss[f"{name}_ss"] for name in problem.var_names])
        Y_ss = np.tile(y_ss, (T, 1))
        X, P = _exogenous_inputs(problem, model)
        _, jacobian = problem.f_resid_and_jac(Y_ss, Y_ss, Y_ss, X, P)

        vanishing = problem.jacobian_sparsity & (jacobian[0] == 0)
        assert vanishing.any(), "expected entries that are structurally nonzero but zero at the steady state"

        assert not ((jacobian[0] != 0) & ~problem.jacobian_sparsity).any()

    def test_deterministic_parameters_are_reported(self):
        problem = compile_perfect_foresight_problem(load_and_cache_model("open_rbc.gcn"), 5)

        assert "rstar" in problem.deterministic_param_names
        assert set(problem.deterministic_param_names) <= set(problem.param_names)

    def test_compiled_functions_are_reused_across_horizons_and_calls(self, rbc_model):
        """The compiled graph does not depend on T, and compiling costs more than a whole solve."""
        first = compile_perfect_foresight_problem(rbc_model, 10)
        again = compile_perfect_foresight_problem(rbc_model, 10)
        longer = compile_perfect_foresight_problem(rbc_model, 40)

        assert again.f_resid_and_jac is first.f_resid_and_jac
        assert longer.f_resid_and_jac is first.f_resid_and_jac
        assert longer.f_resid_only is first.f_resid_only

        assert longer.T == 40
        assert longer.jacobian_layout.shape == (40 * first.n_eq, 40 * first.n_vars)
        assert first.jacobian_layout.shape == (10 * first.n_eq, 10 * first.n_vars)

    def test_different_models_do_not_share_compiled_functions(self, rbc_model):
        """A cache keyed on anything but the model would hand one model's equations to another."""
        other_model = load_and_cache_model("open_rbc.gcn")

        problem = compile_perfect_foresight_problem(rbc_model, 10)
        other = compile_perfect_foresight_problem(other_model, 10)

        assert other.f_resid_and_jac is not problem.f_resid_and_jac
        assert other.var_names != problem.var_names
        assert other.jacobian_sparsity.shape != problem.jacobian_sparsity.shape

    def test_differing_compile_kwargs_get_their_own_functions(self, rbc_model):
        default = compile_perfect_foresight_problem(rbc_model, 10)
        excepting = compile_perfect_foresight_problem(rbc_model, 10, on_unused_input="raise")

        assert excepting.f_resid_and_jac is not default.f_resid_and_jac

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
