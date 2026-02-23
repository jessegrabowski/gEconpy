"""Tests for backward-looking model workflows using the SARIMA(2,12) model."""

import numpy as np
import pytest

from numpy.testing import assert_allclose

from gEconpy.model.model import (
    impulse_response_function,
    simulate,
    stationary_covariance_matrix,
)
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.fixture(scope="module")
def sarima_model():
    return load_and_cache_model("sarima2_12.gcn", backend="numpy")


@pytest.fixture(scope="module")
def sarima_solved(sarima_model):
    T, R = sarima_model.solve_model(verbose=False)
    return sarima_model, T, R


class TestBackwardLookingDetection:
    def test_sarima_is_backward_looking(self, sarima_model):
        assert sarima_model._backward_looking

    def test_forward_looking_model_is_not_backward_looking(self):
        model = load_and_cache_model("one_block_1.gcn", backend="numpy")
        assert not model._backward_looking

    def test_backward_direct_rejects_forward_looking_model(self):
        model = load_and_cache_model("one_block_1.gcn", backend="numpy")
        with pytest.raises(ValueError, match="backward_direct"):
            model.solve_model(solver="backward_direct", verbose=False)


class TestBackwardLookingSolve:
    def test_policy_satisfies_linear_system(self, sarima_model):
        """A + B @ T = 0 and B @ R + D = 0 when C = 0."""
        A, B, C, D = sarima_model.linearize_model()
        T, R = sarima_model.solve_model(verbose=False)

        assert_allclose(C, 0.0)
        assert_allclose(A + B @ T, 0.0, atol=1e-10)
        assert_allclose(B @ R + D, 0.0, atol=1e-10)

    def test_T_encodes_ar_coefficients(self, sarima_solved):
        model, T, _ = sarima_solved
        params = model.parameters()

        rho_1, rho_2, rho_12 = params["rho_1"], params["rho_2"], params["rho_12"]

        var_names = [v.base_name for v in model.variables]
        x = var_names.index("x")

        # After time-index expansion: x[-1] -> x, x[-2] -> x__lag1,
        # x[-12] -> x__lag11, x[-13] -> x__lag12, x[-14] -> x__lag13
        assert_allclose(T[x, x], rho_1, atol=1e-12)
        assert_allclose(T[x, var_names.index("x__lag1")], rho_2, atol=1e-12)
        assert_allclose(T[x, var_names.index("x__lag11")], rho_12, atol=1e-12)
        assert_allclose(T[x, var_names.index("x__lag12")], rho_1 * rho_12, atol=1e-12)
        assert_allclose(T[x, var_names.index("x__lag13")], rho_2 * rho_12, atol=1e-12)


class TestBackwardLookingIRF:
    def test_irf_matches_manual_simulation(self, sarima_solved):
        model, T, R = sarima_solved
        n_vars = len(model.variables)
        n_steps = 50

        irf = impulse_response_function(model, T, R, simulation_length=n_steps, shock_size=1.0)

        # Convention: t=0 is pre-shock, impact lands at t=1
        expected = np.zeros((n_steps, n_vars))
        expected[1] = R.squeeze()
        for t in range(2, n_steps):
            expected[t] = T @ expected[t - 1]

        assert_allclose(irf.values.squeeze(), expected, atol=1e-10)

    def test_irf_decays_to_zero(self, sarima_solved):
        model, T, R = sarima_solved
        irf = impulse_response_function(model, T, R, simulation_length=500, shock_size=1.0)
        assert np.all(np.abs(irf.isel(time=-1).values) < 1e-3)


class TestBackwardLookingSimulation:
    def test_simulate_variance_scales_with_shock_std(self, sarima_solved):
        model, T, R = sarima_solved
        kwargs = {"simulation_length": 500, "n_simulations": 500, "random_seed": 42}

        data_1 = simulate(model, T, R, shock_std=1.0, **kwargs)
        data_2 = simulate(model, T, R, shock_std=2.0, **kwargs)

        var_1 = data_1.sel(variable="x").values[:, -1].var()
        var_2 = data_2.sel(variable="x").values[:, -1].var()

        assert_allclose(var_2 / var_1, 4.0, rtol=0.2)


class TestBackwardLookingMoments:
    def test_stationary_variance_matches_psd(self, sarima_solved):
        """Verify Sigma[x,x] matches the variance from integrating the power spectral density."""
        model, T, R = sarima_solved
        params = model.parameters()
        rho_1, rho_2, rho_12 = params["rho_1"], params["rho_2"], params["rho_12"]
        sigma = 1.0  # shock std; not yet exposed via model.parameters()

        # AR polynomial: (1 - rho_1*L - rho_2*L^2)(1 - rho_12*L^12) x_t = eps_t
        phi = np.zeros(14)
        phi[0] = rho_1
        phi[1] = rho_2
        phi[11] = rho_12
        phi[12] = rho_1 * rho_12
        phi[13] = rho_2 * rho_12

        # Var(x) = (sigma^2 / 2pi) * integral |1 - sum phi_k exp(-ikw)|^{-2} dw
        omega = np.linspace(0, 2 * np.pi, 100_000, endpoint=False)
        k = np.arange(1, len(phi) + 1)
        ar_poly = 1.0 - phi @ np.exp(-1j * np.outer(k, omega))
        psd_variance = sigma**2 * np.mean(1.0 / np.abs(ar_poly) ** 2)

        Sigma = stationary_covariance_matrix(model, T, R, shock_std=sigma, return_df=False)
        x_idx = [v.base_name for v in model.variables].index("x")

        assert_allclose(Sigma[x_idx, x_idx], psd_variance, rtol=1e-6)
