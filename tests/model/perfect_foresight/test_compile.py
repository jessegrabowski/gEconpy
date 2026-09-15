import numpy as np
import pytest

from numpy.testing import assert_allclose

from gEconpy.model.perfect_foresight import compile_perfect_foresight_problem
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
        assert problem.n_eq == problem.n_vars

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

    def test_steady_state_reference_without_analytic_solution_raises(self):
        model = load_and_cache_model("full_nk_no_ss.gcn")
        with pytest.raises(ValueError, match="following do not: pi_ss, r_G_ss"):
            compile_perfect_foresight_problem(model, 5)
