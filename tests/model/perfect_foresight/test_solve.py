from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_allclose

from gEconpy import impulse_response_function
from gEconpy.model.perfect_foresight import solve_perfect_foresight
from gEconpy.model.perfect_foresight.solve import _normalize_condition_keys, make_piecewise_x0
from tests._resources.cache_compiled_models import load_and_cache_model


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

    def test_shock_response_matches_irf(self, backward_var_model):
        simulation_length = 30
        shock_path = np.zeros(simulation_length)
        shock_path[0] = 0.1

        trajectory, result = solve_perfect_foresight(
            backward_var_model,
            simulation_length=simulation_length,
            shocks={"epsilon_x": shock_path},
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

        # Expansionary shock raises all variables on impact
        assert trajectory["x"].iloc[0] > 0
        assert trajectory["pi"].iloc[0] > 0
        assert trajectory["i"].iloc[0] > 0

        # rn follows AR(1) exactly
        rho = forward_model.parameters()["rho"]
        assert_allclose(trajectory["rn"].values, 0.01 * rho ** np.arange(simulation_length), rtol=1e-10)

        # All variables decay toward steady state
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
        ss_dict = rbc_model.steady_state(verbose=False)
        var_names = [v.base_name for v in rbc_model.variables]
        simulation_length = 10

        reversed_names = list(reversed(var_names))
        x0_df = pd.DataFrame(
            np.tile([ss_dict[f"{name}_ss"] for name in reversed_names], (simulation_length, 1)),
            columns=reversed_names,
        )

        _, result = solve_perfect_foresight(rbc_model, simulation_length=simulation_length, x0=x0_df)
        assert result.success

    @pytest.mark.parametrize("bad_shape", [(5, 3), (10,), (20, 1)])
    def test_x0_array_wrong_shape_raises(self, rbc_model, bad_shape):
        with pytest.raises(ValueError, match="x0 array must have shape"):
            solve_perfect_foresight(rbc_model, simulation_length=20, x0=np.zeros(bad_shape))

    def test_x0_dataframe_wrong_length_raises(self, rbc_model):
        with pytest.raises(ValueError, match="rows"):
            solve_perfect_foresight(rbc_model, simulation_length=20, x0=pd.DataFrame(np.zeros((5, 1))))


class TestSteadyStateKwargs:
    def test_setdefault_verbose_false_and_forwarding(self, rbc_model):
        with patch.object(type(rbc_model), "steady_state", wraps=rbc_model.steady_state) as mock_ss:
            solve_perfect_foresight(rbc_model, simulation_length=10, steady_state_kwargs={"progressbar": False})
            kwargs = mock_ss.call_args[1]
            assert kwargs["verbose"] is False
            assert kwargs["progressbar"] is False

    def test_explicit_verbose_not_overridden(self, rbc_model):
        with patch.object(type(rbc_model), "steady_state", wraps=rbc_model.steady_state) as mock_ss:
            solve_perfect_foresight(rbc_model, simulation_length=10, steady_state_kwargs={"verbose": True})
            assert mock_ss.call_args[1]["verbose"] is True


class TestDeterministicParameters:
    def test_steady_state_is_fixed_point(self, open_rbc_model):
        """Models with deterministic parameters (e.g. rstar = 1/beta - 1) should solve correctly."""
        trajectory, result = solve_perfect_foresight(open_rbc_model, simulation_length=20)

        assert result.success
        ss = open_rbc_model.steady_state(verbose=False)
        for var in trajectory.columns:
            assert_allclose(trajectory[var].values, ss[f"{var}_ss"], rtol=1e-6)

    def test_shock_response_converges(self, open_rbc_model):
        """Perfect foresight with a shock should converge and trend toward steady state."""
        simulation_length = 50
        shock_path = np.zeros(simulation_length)
        shock_path[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            open_rbc_model,
            simulation_length=simulation_length,
            shocks={"epsilon_A": shock_path},
        )
        assert result.success

        # The trajectory should not be constant (the shock had an effect)
        ss = open_rbc_model.steady_state(verbose=False)
        ss_vals = np.array([ss[f"{var}_ss"] for var in trajectory.columns])
        deviations = np.abs(trajectory.values - ss_vals)

        # Deviations should be smaller at the end than right after the shock
        assert np.sum(deviations[-1]) < np.sum(deviations[1])


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

        assert isinstance(x0, pd.DataFrame)
        assert len(x0) == 100
        assert sorted(x0.columns) == ["C", "K", "Y"]

    def test_step_transition_at_midpoint(self):
        x0 = make_piecewise_x0(self.INIT_SS, self.TERM_SS, simulation_length=100)

        # Default: step at simulation_length // 2
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

        # Before and after transition region
        assert_allclose(x0.loc[39, "K"], 1.0)
        assert_allclose(x0.loc[61, "K"], 2.0)

        # Endpoints of the transition region (weight=0 and weight=1)
        assert_allclose(x0.loc[40, "K"], 1.0)
        assert_allclose(x0.loc[60, "K"], 2.0)

        # Exact midpoint (index 10 of 21, weight = 10/20 = 0.5)
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


class TestParamPaths:
    def test_scalar_path_changes_steady_state(self, rbc_model):
        ss_default = rbc_model.steady_state(verbose=False)
        traj_default, res_default = solve_perfect_foresight(rbc_model, simulation_length=20)

        # A different depreciation rate should yield a different fixed point
        params = rbc_model.parameters()
        delta = params["delta"]
        traj_override, res_override = solve_perfect_foresight(
            rbc_model,
            simulation_length=20,
            param_paths={"delta": delta * 1.5},
        )

        assert res_default.success and res_override.success
        assert not np.allclose(traj_default["K"].values, traj_override["K"].values)

    def test_none_param_paths_matches_default(self, rbc_model):
        traj_default, _ = solve_perfect_foresight(rbc_model, simulation_length=20)
        traj_explicit, _ = solve_perfect_foresight(rbc_model, simulation_length=20, param_paths=None)

        assert_allclose(traj_default.values, traj_explicit.values)

    def test_time_varying_path_accepted(self, rbc_model):
        params = rbc_model.parameters()
        T = 20
        delta_path = np.full(T, params["delta"])
        delta_path[T // 2 :] *= 1.5

        _, result = solve_perfect_foresight(rbc_model, simulation_length=T, param_paths={"delta": delta_path})
        assert result.success

    def test_boundary_ss_uses_param_paths(self, rbc_model):
        params = rbc_model.parameters()
        T = 50
        delta_init = params["delta"]
        delta_term = delta_init * 1.5
        delta_path = np.where(np.arange(T) < T // 2, delta_init, delta_term)

        ss_init = rbc_model.steady_state(verbose=False, delta=delta_init)
        ss_term = rbc_model.steady_state(verbose=False, delta=delta_term)

        # Without explicit conditions, solver should auto-compute distinct boundary SS values
        x0 = make_piecewise_x0(ss_init, ss_term, simulation_length=T, transition_periods=10)
        traj_auto, res_auto = solve_perfect_foresight(
            rbc_model,
            simulation_length=T,
            x0=x0,
            param_paths={"delta": delta_path},
        )

        # With explicit conditions matching the same SS values
        traj_explicit, res_explicit = solve_perfect_foresight(
            rbc_model,
            simulation_length=T,
            x0=x0,
            initial_conditions=ss_init,
            terminal_conditions=ss_term,
            param_paths={"delta": delta_path},
        )

        assert res_auto.success and res_explicit.success
        assert_allclose(traj_auto.values, traj_explicit.values, rtol=1e-8)
