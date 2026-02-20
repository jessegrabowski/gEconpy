import numpy as np
import pytest

from numpy.testing import assert_allclose

from gEconpy import impulse_response_function
from gEconpy.model.perfect_foresight import solve_perfect_foresight
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.fixture
def rbc_model():
    return load_and_cache_model("one_block_1.gcn", backend="numpy")


@pytest.fixture
def backward_var_model():
    return load_and_cache_model("backward_var.gcn", backend="numpy")


@pytest.fixture
def forward_model():
    return load_and_cache_model("3_eq_linear_nk.gcn", backend="numpy")


class TestSolve:
    def test_steady_state_is_fixed_point(self, rbc_model):
        trajectory, result = solve_perfect_foresight(rbc_model, simulation_length=20)

        assert result.success
        ss = rbc_model.steady_state(verbose=False)

        for var in trajectory.columns:
            expected = ss[f"{var}_ss"]
            assert_allclose(trajectory[var].values, expected, rtol=1e-6)

    def test_converges_toward_terminal_from_perturbed_initial(self, rbc_model):
        ss = rbc_model.steady_state(verbose=False)
        K_ss = ss["K_ss"]

        trajectory, result = solve_perfect_foresight(
            rbc_model,
            simulation_length=50,
            initial_conditions={"K": 0.9 * K_ss},
        )

        assert result.success
        # Should move toward steady state (closer at end than start)
        initial_gap = abs(trajectory["K"].iloc[0] - K_ss)
        final_gap = abs(trajectory["K"].iloc[-1] - K_ss)
        assert final_gap < initial_gap

    def test_shock_moves_trajectory_away_from_steady_state(self, rbc_model):
        ss = rbc_model.steady_state(verbose=False)
        simulation_length = 50
        shock_path = np.zeros(simulation_length)
        shock_path[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            rbc_model,
            simulation_length=simulation_length,
            shocks={"epsilon": shock_path},
        )

        assert result.success
        # First period should differ from steady state due to shock
        assert not np.isclose(trajectory["K"].iloc[0], ss["K_ss"])


class TestBackwardOnlyModel:
    def test_steady_state_is_fixed_point(self, backward_var_model):
        trajectory, result = solve_perfect_foresight(backward_var_model, simulation_length=20)

        assert result.success
        ss = backward_var_model.steady_state(verbose=False)

        for var in trajectory.columns:
            expected = ss[f"{var}_ss"]
            assert_allclose(trajectory[var].values, expected, atol=1e-10)

    def test_converges_from_perturbed_initial(self, backward_var_model):
        simulation_length = 30
        y0 = np.array([1.0, 0.5])

        trajectory, result = solve_perfect_foresight(
            backward_var_model,
            simulation_length=simulation_length,
            initial_conditions={"x": y0[0], "y": y0[1]},
        )
        assert result.success

        # VAR(1) coefficient matrix from model parameters
        params = backward_var_model.parameters()
        A = np.array(
            [
                [params["rho_xx"], params["rho_xy"]],
                [params["rho_yx"], params["rho_yy"]],
            ]
        )

        # Analytical solution: y_t = A^t @ y_0
        expected = np.zeros((simulation_length, 2))
        for t in range(simulation_length):
            expected[t] = np.linalg.matrix_power(A, t + 1) @ y0

        assert_allclose(trajectory[["x", "y"]].values, expected, rtol=1e-10)

    def test_shock_response(self, backward_var_model):
        simulation_length = 30
        shock_path = np.zeros(simulation_length)
        shock_path[0] = 0.1

        trajectory, result = solve_perfect_foresight(
            backward_var_model,
            simulation_length=simulation_length,
            shocks={"epsilon_x": shock_path},
        )

        irf = (
            impulse_response_function(
                backward_var_model,
                simulation_length=1 + simulation_length,
                shock_size={"epsilon_x": 0.1},
            )
            .isel(shock=0)
            .to_pandas()
        )

        # For a linear model, the IRF should match the perfect foresight response to a shock at t=0
        assert result.success
        np.testing.assert_allclose(trajectory["x"].values, irf["x"].values[1:])


class TestForwardOnlyModel:
    def test_steady_state_is_fixed_point(self, forward_model):
        trajectory, result = solve_perfect_foresight(forward_model, simulation_length=20)

        assert result.success
        ss = forward_model.steady_state(verbose=False)

        for var in trajectory.columns:
            expected = ss[f"{var}_ss"]
            assert_allclose(trajectory[var].values, expected, atol=1e-10)

    def test_natural_rate_shock_response(self, forward_model):
        simulation_length = 30
        shock_rn = np.zeros(simulation_length)
        shock_rn[0] = 0.01  # 1% shock

        trajectory, result = solve_perfect_foresight(
            forward_model,
            simulation_length=simulation_length,
            shocks={"epsilon_rn": shock_rn},
        )

        assert result.success

        # Natural rate shock is expansionary: raises x, pi, and i
        assert trajectory["x"].iloc[0] > 0
        assert trajectory["pi"].iloc[0] > 0
        assert trajectory["i"].iloc[0] > 0

        # rn follows AR(1), so rn[t] = shock * rho^t
        params = forward_model.parameters()
        rho = params["rho"]
        expected_rn = 0.01 * (rho ** np.arange(simulation_length))
        assert_allclose(trajectory["rn"].values, expected_rn, rtol=1e-10)

        # All variables should decay toward steady state
        assert abs(trajectory["x"].iloc[-1]) < abs(trajectory["x"].iloc[0])
        assert abs(trajectory["pi"].iloc[-1]) < abs(trajectory["pi"].iloc[0])

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

        # Taylor rule: i = phi_pi * pi + phi_x * x
        params = forward_model.parameters()
        phi_pi = params["phi_pi"]
        phi_x = params["phi_x"]

        expected_i = phi_pi * trajectory["pi"] + phi_x * trajectory["x"]
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

        irf = (
            impulse_response_function(
                forward_model,
                simulation_length=1 + simulation_length,
                shock_size={"epsilon_rn": 0.01},
            )
            .isel(shock=0)
            .to_pandas()
        )

        assert result.success
        # IRF has shock at t=0 with response starting at t=1, so compare with offset
        var_names = ["x", "pi", "i", "rn"]
        assert_allclose(trajectory[var_names].values, irf[var_names].iloc[1:].values, atol=1e-8, rtol=1e-8)
