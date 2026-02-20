import numpy as np
import pytest

from numpy.testing import assert_allclose

from gEconpy.model.perfect_foresight import solve_perfect_foresight
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.fixture
def rbc_model():
    return load_and_cache_model("one_block_1.gcn", backend="numpy")


class TestSolve:
    def test_steady_state_is_fixed_point(self, rbc_model):
        trajectory, result = solve_perfect_foresight(rbc_model, T=20)

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
            T=50,
            initial_conditions={"K": 0.9 * K_ss},
        )

        assert result.success
        # Should move toward steady state (closer at end than start)
        initial_gap = abs(trajectory["K"].iloc[0] - K_ss)
        final_gap = abs(trajectory["K"].iloc[-1] - K_ss)
        assert final_gap < initial_gap

    def test_shock_moves_trajectory_away_from_steady_state(self, rbc_model):
        ss = rbc_model.steady_state(verbose=False)
        T = 50
        shock_path = np.zeros(T)
        shock_path[0] = 0.01

        trajectory, result = solve_perfect_foresight(
            rbc_model,
            T=T,
            shocks={"epsilon": shock_path},
        )

        assert result.success
        # First period should differ from steady state due to shock
        assert not np.isclose(trajectory["K"].iloc[0], ss["K_ss"])
