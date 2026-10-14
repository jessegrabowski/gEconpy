from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_allclose

from gEconpy import impulse_response_function
from gEconpy.model.perfect_foresight import compile_perfect_foresight_problem, solve_perfect_foresight
from gEconpy.model.perfect_foresight.solve import (
    _build_param_matrix,
    _compute_stacked_residuals,
    _compute_stacked_residuals_and_jacobian,
    _normalize_condition_keys,
    make_piecewise_x0,
)
from gEconpy.solvers.sparse_root import LevenbergMarquardt, newton_armijo
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking
from tests._resources.cache_compiled_models import load_and_cache_example, load_and_cache_model


@pytest.fixture
def rbc_model():
    return load_and_cache_model("one_block_1.gcn")


@pytest.fixture
def backward_var_model():
    return load_and_cache_model("backward_var.gcn")


@pytest.fixture
def forward_model():
    return load_and_cache_model("3_eq_linear_nk.gcn")


@pytest.fixture
def open_rbc_model():
    return load_and_cache_model("open_rbc.gcn")


class TestSolve:
    def test_steady_state_is_fixed_point(self, rbc_model):
        trajectory, result = solve_perfect_foresight(rbc_model, simulation_length=20)

        assert result.success
        ss = rbc_model.steady_state(verbose=False)
        for var in trajectory.columns:
            assert_allclose(trajectory[var].values, ss[f"{var}_ss"], rtol=1e-6)


class TestBackwardOnlyModel:
    def test_steady_state_is_fixed_point(self, backward_var_model):
        trajectory, result = solve_perfect_foresight(backward_var_model, simulation_length=20)

        assert result.success
        ss = backward_var_model.steady_state(verbose=False)
        for var in trajectory.columns:
            assert_allclose(trajectory[var].values, ss[f"{var}_ss"], atol=1e-10)

    def test_matches_analytical_var1_solution(self, backward_var_model):
        simulation_length = 30
        y0 = np.array([1.0, 0.5])

        trajectory, result = solve_perfect_foresight(
            backward_var_model,
            simulation_length=simulation_length,
            initial_conditions={"x": y0[0], "y": y0[1]},
        )
        assert result.success

        params = backward_var_model.parameters()
        A = np.array([[params["rho_xx"], params["rho_xy"]], [params["rho_yx"], params["rho_yy"]]])
        expected = np.array([np.linalg.matrix_power(A, t + 1) @ y0 for t in range(simulation_length)])

        assert_allclose(trajectory[["x", "y"]].values, expected, rtol=1e-10)

    @pytest.mark.parametrize("path_type", [np.asarray, list], ids=["array", "list"])
    def test_shock_response_matches_irf(self, backward_var_model, path_type):
        simulation_length = 30
        shock_path = np.zeros(simulation_length)
        shock_path[0] = 0.1

        trajectory, result = solve_perfect_foresight(
            backward_var_model,
            simulation_length=simulation_length,
            shocks={"epsilon_x": path_type(shock_path)},
        )
        assert result.success

        irf = (
            impulse_response_function(
                backward_var_model,
                simulation_length=simulation_length,
                shock_size={"epsilon_x": 0.1},
            )
            .isel(shock=0)
            .to_pandas()
        )
        assert_allclose(trajectory["x"].values, irf["x"].values)


class TestForwardOnlyModel:
    def test_steady_state_is_fixed_point(self, forward_model):
        trajectory, result = solve_perfect_foresight(forward_model, simulation_length=20)

        assert result.success
        ss = forward_model.steady_state(verbose=False)
        for var in trajectory.columns:
            assert_allclose(trajectory[var].values, ss[f"{var}_ss"], atol=1e-10)

    def test_natural_rate_shock_response(self, forward_model):
        simulation_length = 30
        shock_rn = np.zeros(simulation_length)
        shock_rn[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            forward_model,
            simulation_length=simulation_length,
            shocks={"epsilon_rn": shock_rn},
        )
        assert result.success

        assert trajectory["x"].iloc[0] > 0
        assert trajectory["pi"].iloc[0] > 0
        assert trajectory["i"].iloc[0] > 0

        rho = forward_model.parameters()["rho"]
        assert_allclose(trajectory["rn"].values, 0.01 * rho ** np.arange(simulation_length), rtol=1e-10)

        assert abs(trajectory["x"].iloc[-1]) < abs(trajectory["x"].iloc[0])

    def test_taylor_rule_holds(self, forward_model):
        simulation_length = 20
        shock_rn = np.zeros(simulation_length)
        shock_rn[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            forward_model,
            simulation_length=simulation_length,
            shocks={"epsilon_rn": shock_rn},
        )
        assert result.success

        params = forward_model.parameters()
        expected_i = params["phi_pi"] * trajectory["pi"] + params["phi_x"] * trajectory["x"]
        assert_allclose(trajectory["i"].values, expected_i.values, rtol=1e-10)

    def test_shock_response_matches_irf(self, forward_model):
        simulation_length = 100
        shock_rn = np.zeros(simulation_length)
        shock_rn[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            forward_model,
            simulation_length=simulation_length,
            shocks={"epsilon_rn": shock_rn},
        )
        assert result.success

        irf = (
            impulse_response_function(
                forward_model,
                simulation_length=simulation_length,
                shock_size={"epsilon_rn": 0.01},
            )
            .isel(shock=0)
            .to_pandas()
        )
        assert_allclose(
            trajectory[["x", "pi", "i", "rn"]].values,
            irf[["x", "pi", "i", "rn"]].values,
            atol=1e-8,
            rtol=1e-8,
        )


class TestX0Dispatch:
    def test_x0_dict_skips_steady_state_call(self, rbc_model):
        ss_dict = rbc_model.steady_state(verbose=False)

        with patch.object(type(rbc_model), "steady_state") as mock_ss:
            _, result = solve_perfect_foresight(rbc_model, simulation_length=10, x0=dict(ss_dict))

        assert result.success
        mock_ss.assert_not_called()

    def test_x0_dict_missing_variables_raises(self, rbc_model):
        with pytest.raises(ValueError, match=r"missing values for: .*C"):
            solve_perfect_foresight(rbc_model, simulation_length=10, x0={"K_ss": 3.0})

    def test_x0_array_with_full_conditions_skips_steady_state(self, rbc_model):
        ss_dict = rbc_model.steady_state(verbose=False)
        var_names = [v.base_name for v in rbc_model.variables]
        simulation_length = 10

        x0_mat = np.tile([ss_dict[f"{name}_ss"] for name in var_names], (simulation_length, 1))
        full_conditions = {name: ss_dict[f"{name}_ss"] for name in var_names}

        with patch.object(type(rbc_model), "steady_state") as mock_ss:
            _, result = solve_perfect_foresight(
                rbc_model,
                simulation_length=simulation_length,
                x0=x0_mat,
                initial_conditions=full_conditions,
                terminal_conditions=full_conditions,
            )

        assert result.success
        mock_ss.assert_not_called()

    @pytest.mark.parametrize("x0_type", ["ndarray", "dataframe"])
    def test_x0_with_full_initial_but_no_terminal_computes_terminal_ss(self, rbc_model, x0_type):
        ss_dict = rbc_model.steady_state(verbose=False)
        var_names = [v.base_name for v in rbc_model.variables]
        simulation_length = 10

        x0_mat = np.tile([ss_dict[f"{name}_ss"] for name in var_names], (simulation_length, 1))
        full_initial = {name: ss_dict[f"{name}_ss"] for name in var_names}

        x0 = pd.DataFrame(x0_mat, columns=var_names) if x0_type == "dataframe" else x0_mat

        _, result = solve_perfect_foresight(
            rbc_model, simulation_length=simulation_length, x0=x0, initial_conditions=full_initial
        )
        assert result.success

    def test_x0_dataframe_reindexes_columns(self, rbc_model):
        """A guess with columns in reverse order is already the steady state, so Newton converges in one step."""
        ss_dict = rbc_model.steady_state(verbose=False)
        var_names = [v.base_name for v in rbc_model.variables]
        simulation_length = 10

        reversed_names = list(reversed(var_names))
        x0_df = pd.DataFrame(
            np.tile([ss_dict[f"{name}_ss"] for name in reversed_names], (simulation_length, 1)),
            columns=reversed_names,
        )

        trajectory, result = solve_perfect_foresight(rbc_model, simulation_length=simulation_length, x0=x0_df)
        assert result.success
        assert result.nit <= 1
        assert list(trajectory.columns) == var_names
        for name in var_names:
            assert_allclose(trajectory[name].values, ss_dict[f"{name}_ss"], rtol=1e-8)

    @pytest.mark.parametrize("bad_shape", [(5, 3), (10,), (20, 1)])
    def test_x0_array_wrong_shape_raises(self, rbc_model, bad_shape):
        with pytest.raises(ValueError, match="x0 array must have shape"):
            solve_perfect_foresight(rbc_model, simulation_length=20, x0=np.zeros(bad_shape))

    def test_x0_dataframe_wrong_length_raises(self, rbc_model):
        with pytest.raises(ValueError, match="rows"):
            solve_perfect_foresight(rbc_model, simulation_length=20, x0=pd.DataFrame(np.zeros((5, 1))))


class TestSolverArgument:
    @pytest.mark.parametrize(
        "solver",
        [newton_armijo(), LevenbergMarquardt()],
        ids=["newton-armijo-with-globalization", "levenberg-marquardt-without-globalization"],
    )
    def test_user_solver_reaches_the_same_path_when_reused(self, rbc_model, solver):
        """A caller-owned solver instance works on the first call and again on a second, with no state carried over."""
        ss_dict = rbc_model.steady_state(verbose=False)
        conditions = {"initial_conditions": {"K": 0.9 * ss_dict["K_ss"]}}

        reference, _ = solve_perfect_foresight(rbc_model, simulation_length=20, **conditions)
        first, first_result = solve_perfect_foresight(rbc_model, simulation_length=20, solver=solver, **conditions)
        second, second_result = solve_perfect_foresight(rbc_model, simulation_length=20, solver=solver, **conditions)

        assert first_result.success
        assert second_result.success
        assert first_result.nit == second_result.nit
        assert_allclose(first.values, reference.values, rtol=1e-8)
        assert_allclose(second.values, reference.values, rtol=1e-8)


class TestSteadyStateKwargs:
    def test_kwargs_reach_the_steady_state_with_verbose_defaulting_off(self, rbc_model):
        with patch.object(type(rbc_model), "steady_state", wraps=rbc_model.steady_state) as mock_ss:
            solve_perfect_foresight(rbc_model, simulation_length=10, steady_state_kwargs={"progressbar": False})

        assert mock_ss.call_args.kwargs["verbose"] is False
        assert mock_ss.call_args.kwargs["progressbar"] is False

    def test_explicit_verbose_not_overridden(self, rbc_model):
        with patch.object(type(rbc_model), "steady_state", wraps=rbc_model.steady_state) as mock_ss:
            solve_perfect_foresight(rbc_model, simulation_length=10, steady_state_kwargs={"verbose": True})

        assert mock_ss.call_args.kwargs["verbose"] is True


class TestDeterministicParameters:
    def test_steady_state_is_fixed_point(self, open_rbc_model):
        trajectory, result = solve_perfect_foresight(open_rbc_model, simulation_length=20)

        assert result.success
        ss = open_rbc_model.steady_state(verbose=False)
        for var in trajectory.columns:
            # IIP, CA, TB and TBtoY are zero at the steady state, where a relative tolerance says nothing.
            assert_allclose(trajectory[var].values, ss[f"{var}_ss"], rtol=1e-6, atol=1e-10)

    def test_shock_response_follows_the_technology_process_exactly(self, open_rbc_model):
        """``log(A)`` is an AR(1) in the shock alone, so its whole path is known in closed form."""
        T = 50
        shock_path = np.zeros(T)
        shock_path[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            open_rbc_model, simulation_length=T, shocks={"epsilon_A": shock_path}
        )
        assert result.success

        rho_A = open_rbc_model.parameters()["rho_A"]
        assert_allclose(trajectory["A"].values, np.exp(0.01 * rho_A ** np.arange(T)), rtol=1e-10)

    def test_shock_response_mean_reverts_in_the_flow_variables(self, open_rbc_model):
        """
        Consumption, labor and output return toward the steady state after a transitory technology shock.

        The stock variables do not: with ``psi2 = 0.000742`` the debt-elastic premium is weak enough that net
        foreign assets are near a unit root, so ``IIP`` is still drifting away from its steady state at the end
        of any horizon this test could afford. Reading the whole trajectory on one scale hides that, because the
        absolute deviations are dominated by ``K``.
        """
        T = 50
        shock_path = np.zeros(T)
        shock_path[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            open_rbc_model, simulation_length=T, shocks={"epsilon_A": shock_path}
        )
        assert result.success

        ss = open_rbc_model.steady_state(verbose=False)
        for var in ("C", "N", "Y", "KtoN"):
            gap = np.abs(trajectory[var].values - ss[f"{var}_ss"])
            assert gap[0] > 1e-4 * abs(ss[f"{var}_ss"]), f"the shock should move {var} on impact"
            assert gap[-1] < gap[0] / 10, f"{var} should be most of the way back by period {T}"

    def test_path_on_a_free_parameter_rederives_its_deterministic_children(self, open_rbc_model):
        """A path for beta must carry rstar with it.

        ``rstar = 1 / beta - 1``, so leaving rstar at its default would solve one model in the periods
        and another at the boundaries, and report success anyway.
        """
        beta = 0.95
        ss_override = open_rbc_model.steady_state(verbose=False, beta=beta)

        trajectory, result = solve_perfect_foresight(open_rbc_model, simulation_length=20, param_paths={"beta": beta})

        assert result.success
        for var in trajectory.columns:
            assert_allclose(trajectory[var].values, ss_override[f"{var}_ss"], rtol=1e-8, atol=1e-10)

    def test_time_varying_free_parameter_moves_its_deterministic_children(self, open_rbc_model):
        """``r`` is ``rstar`` plus a premium in IIP, so a step in beta must move r by the step in rstar."""
        T = 20
        beta_path = np.full(T, 0.99)
        beta_path[T // 2 :] = 0.95

        trajectory, result = solve_perfect_foresight(
            open_rbc_model, simulation_length=T, param_paths={"beta": beta_path}
        )
        assert result.success

        parameters = open_rbc_model.parameters
        expected_jump = parameters(beta=0.95)["rstar"] - parameters(beta=0.99)["rstar"]
        assert_allclose(trajectory["r"].iloc[-1] - trajectory["r"].iloc[0], expected_jump, rtol=1e-3)

    def test_path_on_a_deterministic_parameter_is_rejected(self, open_rbc_model):
        with pytest.raises(ValueError, match="Cannot set a path for deterministic parameters: rstar"):
            solve_perfect_foresight(open_rbc_model, simulation_length=10, param_paths={"rstar": 0.05})


class TestConditionKeyNormalization:
    @pytest.mark.parametrize(
        "input_dict, expected",
        [
            ({"K_ss": 1.0, "C_ss": 2.0}, {"K": 1.0, "C": 2.0}),
            ({"K": 1.0, "C": 2.0}, {"K": 1.0, "C": 2.0}),
            ({"K_ss": 1.0, "C": 2.0}, {"K": 1.0, "C": 2.0}),
        ],
    )
    def test_normalize_strips_ss_suffix(self, input_dict, expected):
        assert _normalize_condition_keys(input_dict) == expected

    def test_ss_dict_as_conditions_matches_default(self, rbc_model):
        ss_dict = rbc_model.steady_state(verbose=False)

        traj_default, _ = solve_perfect_foresight(
            rbc_model,
            simulation_length=50,
            initial_conditions={"K": 0.9 * ss_dict["K_ss"]},
        )
        traj_explicit, _ = solve_perfect_foresight(
            rbc_model,
            simulation_length=50,
            initial_conditions={"K_ss": 0.9 * ss_dict["K_ss"]},
            terminal_conditions=dict(ss_dict),
        )

        assert_allclose(traj_default.values, traj_explicit.values, rtol=1e-10)


class TestMakePiecewiseX0:
    INIT_SS = {"K_ss": 1.0, "C_ss": 0.5, "Y_ss": 0.8}
    TERM_SS = {"K_ss": 2.0, "C_ss": 1.0, "Y_ss": 1.6}

    def test_returns_dataframe_with_correct_structure(self):
        x0 = make_piecewise_x0(self.INIT_SS, self.TERM_SS, simulation_length=100)

        assert len(x0) == 100
        assert sorted(x0.columns) == ["C", "K", "Y"]

    def test_step_transition_at_midpoint(self):
        x0 = make_piecewise_x0(self.INIT_SS, self.TERM_SS, simulation_length=100)

        assert_allclose(x0.loc[49, "K"], 1.0)
        assert_allclose(x0.loc[50, "K"], 2.0)

    def test_smooth_transition_interpolates_linearly(self):
        x0 = make_piecewise_x0(
            self.INIT_SS,
            self.TERM_SS,
            simulation_length=100,
            transition_start=40,
            transition_periods=21,
        )

        assert_allclose(x0.loc[39, "K"], 1.0)
        assert_allclose(x0.loc[61, "K"], 2.0)

        assert_allclose(x0.loc[40, "K"], 1.0)
        assert_allclose(x0.loc[60, "K"], 2.0)

        assert_allclose(x0.loc[50, "K"], 1.5)
        assert_allclose(x0.loc[50, "C"], 0.75)

    def test_infers_var_names_and_accepts_unsuffixed_keys(self):
        x0_suffixed = make_piecewise_x0(self.INIT_SS, self.TERM_SS, simulation_length=10)
        x0_bare = make_piecewise_x0(
            {"K": 1.0, "C": 0.5, "Y": 0.8}, {"K": 2.0, "C": 1.0, "Y": 1.6}, simulation_length=10
        )

        assert_allclose(x0_suffixed.values, x0_bare.values)
        assert list(x0_suffixed.columns) == list(x0_bare.columns)

    def test_explicit_var_names_controls_column_order(self):
        var_names = ["Y", "K", "C"]
        x0 = make_piecewise_x0(self.INIT_SS, self.TERM_SS, simulation_length=10, var_names=var_names)

        assert list(x0.columns) == var_names
        assert_allclose(x0.iloc[0].values, [0.8, 1.0, 0.5])

    def test_var_name_missing_from_a_steady_state_raises(self):
        with pytest.raises(ValueError, match="missing values for: L"):
            make_piecewise_x0(self.INIT_SS, self.TERM_SS, simulation_length=10, var_names=["K", "L"])

    def test_transition_running_past_the_horizon_is_clamped(self):
        """A window that overruns the horizon interpolates over the remaining periods and ends at the terminal value."""
        x0 = make_piecewise_x0(
            self.INIT_SS, self.TERM_SS, simulation_length=10, transition_start=7, transition_periods=5
        )

        assert_allclose(x0["K"].values, [1.0] * 7 + [1.0, 1.5, 2.0])


class TestParamPaths:
    def test_scalar_path_is_a_fixed_point_at_the_overridden_steady_state(self, rbc_model):
        delta = 1.5 * rbc_model.parameters()["delta"]
        ss_override = rbc_model.steady_state(verbose=False, delta=delta)

        trajectory, result = solve_perfect_foresight(rbc_model, simulation_length=20, param_paths={"delta": delta})

        assert result.success
        for var in trajectory.columns:
            assert_allclose(trajectory[var].values, ss_override[f"{var}_ss"], rtol=1e-6)

    def test_none_param_paths_matches_default(self, rbc_model):
        traj_default, _ = solve_perfect_foresight(rbc_model, simulation_length=20)
        traj_explicit, _ = solve_perfect_foresight(rbc_model, simulation_length=20, param_paths=None)

        assert_allclose(traj_default.values, traj_explicit.values)

    def test_time_varying_path_enters_each_period_with_its_own_value(self, rbc_model):
        """The capital accumulation constraint holds period by period with the delta of that period."""
        params = rbc_model.parameters()
        T = 20
        delta_path = np.full(T, params["delta"])
        delta_path[T // 2 :] *= 1.5

        trajectory, result = solve_perfect_foresight(rbc_model, simulation_length=T, param_paths={"delta": delta_path})
        assert result.success

        ss_initial = rbc_model.steady_state(verbose=False)
        K_lag = np.r_[ss_initial["K_ss"], trajectory["K"].values[:-1]]
        resources = trajectory["A"].values * K_lag ** params["alpha"] + (1 - delta_path) * K_lag
        assert_allclose(trajectory["C"].values + trajectory["K"].values, resources, rtol=1e-8)
        assert not np.allclose(trajectory["K"].values[T // 2 :], ss_initial["K_ss"])

    def test_list_path_matches_array_path(self, rbc_model):
        T = 20
        delta_path = np.full(T, rbc_model.parameters()["delta"])
        delta_path[T // 2 :] *= 1.5

        traj_array, _ = solve_perfect_foresight(rbc_model, simulation_length=T, param_paths={"delta": delta_path})
        traj_list, _ = solve_perfect_foresight(
            rbc_model, simulation_length=T, param_paths={"delta": delta_path.tolist()}
        )

        assert_allclose(traj_list.values, traj_array.values)

    @pytest.mark.parametrize("rho_0", [0.0, 1e-8], ids=["exact-zero", "near-zero"])
    def test_zero_entry_at_first_period_keeps_jacobian_structure(self, rbc_model, rho_0):
        """A Jacobian entry that is zero at t=0 but nonzero later must stay in the stacked sparsity pattern."""
        T = 30
        shock = np.zeros(T)
        shock[0] = 0.05
        rho_path = np.full(T, rbc_model.parameters()["rho"])
        rho_path[0] = rho_0

        _, result = solve_perfect_foresight(
            rbc_model,
            simulation_length=T,
            shocks={"epsilon": shock},
            param_paths={"rho": rho_path},
        )

        assert result.success
        assert result.nit <= 5

    def test_boundary_ss_uses_param_paths(self, rbc_model):
        params = rbc_model.parameters()
        T = 50
        delta_init = params["delta"]
        delta_term = delta_init * 1.5
        delta_path = np.where(np.arange(T) < T // 2, delta_init, delta_term)

        ss_init = rbc_model.steady_state(verbose=False, delta=delta_init)
        ss_term = rbc_model.steady_state(verbose=False, delta=delta_term)

        x0 = make_piecewise_x0(ss_init, ss_term, simulation_length=T, transition_periods=10)
        traj_auto, res_auto = solve_perfect_foresight(
            rbc_model,
            simulation_length=T,
            x0=x0,
            param_paths={"delta": delta_path},
        )

        traj_explicit, res_explicit = solve_perfect_foresight(
            rbc_model,
            simulation_length=T,
            x0=x0,
            initial_conditions=ss_init,
            terminal_conditions=ss_term,
            param_paths={"delta": delta_path},
        )

        assert res_auto.success
        assert res_explicit.success
        assert_allclose(traj_auto.values, traj_explicit.values, rtol=1e-8)


class TestBoundaryConditionsAreReached:
    def test_a_longer_horizon_lands_closer_to_the_terminal_steady_state(self, rbc_model):
        """The terminal condition is an approximation whose error a longer horizon pushes further away."""
        ss = rbc_model.steady_state(verbose=False)
        conditions = {"initial_conditions": {"K": 0.5 * ss["K_ss"]}}

        def terminal_deviation(simulation_length):
            trajectory, result = solve_perfect_foresight(rbc_model, simulation_length=simulation_length, **conditions)
            assert result.success
            ss_values = np.array([ss[f"{var}_ss"] for var in trajectory.columns])
            return np.max(np.abs(trajectory.values[-1] - ss_values) / np.abs(ss_values))

        assert terminal_deviation(400) < terminal_deviation(200) / 10

    def test_terminal_condition_on_a_lead_variable_binds(self, rbc_model):
        """Only the lead block multiplies the terminal vector, so lambda is where the condition can bite."""
        ss = rbc_model.steady_state(verbose=False)
        T = 50

        reference, _ = solve_perfect_foresight(rbc_model, simulation_length=T)
        bent, result = solve_perfect_foresight(
            rbc_model, simulation_length=T, terminal_conditions={"lambda": 1.1 * ss["lambda_ss"]}
        )
        assert result.success

        assert_allclose(reference["lambda"].values, ss["lambda_ss"], rtol=1e-8)
        assert bent["lambda"].iloc[-1] > reference["lambda"].iloc[-1]
        assert bent["lambda"].iloc[-1] < 1.1 * ss["lambda_ss"]
        assert np.all(np.diff(bent["lambda"].values) > 0), "the path should climb toward the imposed value"

    def test_initial_period_respects_the_initial_condition(self, rbc_model):
        """K[0] is pinned by the accumulation constraint read at the supplied K[-1]."""
        ss = rbc_model.steady_state(verbose=False)
        params = rbc_model.parameters()
        K_init = 0.9 * ss["K_ss"]

        trajectory, result = solve_perfect_foresight(rbc_model, simulation_length=50, initial_conditions={"K": K_init})
        assert result.success

        resources = trajectory["A"].iloc[0] * K_init ** params["alpha"] + (1 - params["delta"]) * K_init
        assert_allclose(trajectory["C"].iloc[0] + trajectory["K"].iloc[0], resources, rtol=1e-8)


class TestInputGuards:
    def test_x0_dataframe_missing_a_column_raises(self, rbc_model):
        """Reindexing would fill it with NaN, which reaches Newton as a singular Jacobian."""
        ss = rbc_model.steady_state(verbose=False)
        x0 = make_piecewise_x0(ss, ss, simulation_length=10)

        with pytest.raises(ValueError, match="x0 DataFrame is missing columns for: K"):
            solve_perfect_foresight(rbc_model, simulation_length=10, x0=x0.drop(columns=["K"]))

    def test_x0_dataframe_with_extra_columns_is_accepted(self, rbc_model):
        """A steady-state dictionary may carry entries that are not model variables."""
        ss = rbc_model.steady_state(verbose=False)
        x0 = make_piecewise_x0(ss, ss, simulation_length=10)
        x0["not_a_variable"] = 0.0

        _, result = solve_perfect_foresight(rbc_model, simulation_length=10, x0=x0)
        assert result.success

    @pytest.mark.parametrize("simulation_length", [0, -1], ids=["zero", "negative"])
    def test_non_positive_simulation_length_raises(self, rbc_model, simulation_length):
        with pytest.raises(ValueError, match="simulation_length must be at least 1"):
            solve_perfect_foresight(rbc_model, simulation_length=simulation_length)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf], ids=["nan", "inf"])
    @pytest.mark.parametrize(
        "named_input",
        ["x0", "initial_conditions", "terminal_conditions", "shocks", "param_paths"],
    )
    def test_non_finite_input_raises_naming_the_offender(self, rbc_model, named_input, bad_value):
        T = 10
        ss = rbc_model.steady_state(verbose=False)
        var_names = [v.base_name for v in rbc_model.variables]

        kwargs = {
            "x0": {"x0": np.tile([ss[f"{name}_ss"] for name in var_names], (T, 1))},
            "initial_conditions": {"initial_conditions": {"K": 0.0}},
            "terminal_conditions": {"terminal_conditions": {"K": 0.0}},
            "shocks": {"shocks": {"epsilon": np.zeros(T)}},
            "param_paths": {"param_paths": {"delta": np.full(T, 0.02)}},
        }[named_input]
        if named_input == "x0":
            kwargs["x0"][0, 0] = bad_value
        elif named_input == "shocks":
            kwargs["shocks"]["epsilon"][0] = bad_value
        elif named_input == "param_paths":
            kwargs["param_paths"]["delta"][0] = bad_value
        else:
            kwargs[named_input]["K"] = bad_value

        with pytest.raises(ValueError, match=f"{named_input} contains non-finite values"):
            solve_perfect_foresight(rbc_model, simulation_length=T, **kwargs)

    def test_unknown_parameter_is_reported_before_the_steady_state_is_computed(self, rbc_model):
        """The model's own error for a bad name does not say which names a parameter path may use."""
        with (
            patch.object(type(rbc_model), "steady_state") as mock_ss,
            pytest.raises(ValueError, match="Unknown parameters in param_paths"),
        ):
            solve_perfect_foresight(rbc_model, simulation_length=10, param_paths={"not_a_param": 1.0})

        mock_ss.assert_not_called()


class TestResidualScaling:
    @pytest.fixture
    def wage_setting_nk_model(self):
        # Its wage-setting auxiliary is of order 1e5, with a large exponent on inflation in the same equation.
        return load_and_cache_example("sims_2024/nk_complete_more_shocks")

    def test_an_equation_with_large_terms_does_not_stall_the_line_search(self, wage_setting_nk_model):
        shock_path = np.zeros(20)
        shock_path[0] = 1e-4

        _, result = solve_perfect_foresight(
            wage_setting_nk_model, simulation_length=20, shocks={"epsilon_beta": shock_path}
        )

        assert result.success
        assert result.nit <= 3

    def test_result_reports_the_unscaled_residuals_and_jacobian(self, wage_setting_nk_model):
        simulation_length = 20
        shock_path = np.zeros(simulation_length)
        shock_path[0] = 1e-4

        _trajectory, result = solve_perfect_foresight(
            wage_setting_nk_model, simulation_length=simulation_length, shocks={"epsilon_beta": shock_path}
        )

        problem = compile_perfect_foresight_problem(wage_setting_nk_model, simulation_length)
        steady_state = wage_setting_nk_model.steady_state(verbose=False)
        boundary = np.array([steady_state[f"{name}_ss"] for name in problem.var_names])
        shocks = np.zeros((simulation_length, len(problem.shock_names)))
        shocks[0, problem.shock_names.index("epsilon_beta")] = 1e-4
        params = _build_param_matrix(None, problem.param_names, wage_setting_nk_model, simulation_length)

        residuals, jacobian = _compute_stacked_residuals_and_jacobian(
            result.x, boundary, boundary, shocks, params, problem
        )
        assert_allclose(result.fun, residuals, rtol=1e-12)
        assert_allclose(result.jac.toarray(), jacobian.toarray(), rtol=1e-12)

    def test_a_caller_supplied_raw_merit_function_is_scaled_to_match(self, wage_setting_nk_model):
        simulation_length = 20
        shock_path = np.zeros(simulation_length)
        shock_path[0] = 1e-4

        problem = compile_perfect_foresight_problem(wage_setting_nk_model, simulation_length)
        steady_state = wage_setting_nk_model.steady_state(verbose=False)
        boundary = np.array([steady_state[f"{name}_ss"] for name in problem.var_names])
        shocks = np.zeros((simulation_length, len(problem.shock_names)))
        shocks[0, problem.shock_names.index("epsilon_beta")] = 1e-4
        params = _build_param_matrix(None, problem.param_names, wage_setting_nk_model, simulation_length)

        def raw_merit_fun(x, *_args):
            return _compute_stacked_residuals(x, boundary, boundary, shocks, params, problem)

        _, result = solve_perfect_foresight(
            wage_setting_nk_model,
            simulation_length=simulation_length,
            shocks={"epsilon_beta": shock_path},
            solver=newton_armijo(globalization=ArmijoBacktracking(merit_fun=raw_merit_fun)),
        )

        assert result.success
        assert result.nit <= 3


class TestMeritFunction:
    def test_a_caller_owned_solver_is_not_mutated(self, rbc_model):
        """The merit function is installed on a copy, so a solver the caller keeps is unchanged afterwards."""
        solver = newton_armijo(globalization=ArmijoBacktracking())
        _, result = solve_perfect_foresight(rbc_model, simulation_length=10, solver=solver)

        assert result.success
        assert solver.globalization.merit_fun is None

    def test_a_caller_supplied_merit_function_is_kept(self, rbc_model):
        """A merit function that reports every trial point as a root is observable in the solver's behavior."""
        calls = []

        def flat_merit_fun(x, *_args):
            calls.append(x)
            return np.zeros_like(x)

        solver = newton_armijo(globalization=ArmijoBacktracking(merit_fun=flat_merit_fun))
        ss = rbc_model.steady_state(verbose=False)

        _, result = solve_perfect_foresight(
            rbc_model,
            simulation_length=10,
            solver=solver,
            initial_conditions={"K": 0.5 * ss["K_ss"]},
        )

        assert calls, "the caller's merit function was replaced"
        assert solver.globalization.merit_fun is flat_merit_fun
        assert not result.success, "a merit function that lies should keep the solve from converging"
