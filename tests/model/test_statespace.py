import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

from gEconpy import statespace_from_gcn
from gEconpy.model.statespace import data_from_prior, prepare_mixed_frequency_data
from tests._resources.cache_compiled_models import (
    load_and_cache_model,
    load_and_cache_statespace,
)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
        "rbc_linearized.gcn",
        "sarima2_12.gcn",
    ],
)
def test_statespace_matrices_agree_with_model(gcn_file):
    ss_mod = load_and_cache_statespace(gcn_file)
    model = load_and_cache_model(gcn_file)

    inputs = pm.inputvars(ss_mod.linearized_system)
    input_names = [x.name for x in inputs]
    f = pytensor.function(inputs, ss_mod.linearized_system, on_unused_input="ignore")
    mod_matrices = model.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})

    param_dict = model.parameters()
    ss_matrices = f(**{k: param_dict[k] for k in input_names})

    for mod_matrix, ss_matrix in zip(mod_matrices, ss_matrices, strict=False):
        np.testing.assert_allclose(mod_matrix, ss_matrix, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        "full_nk.gcn",
        "rbc_linearized.gcn",
        "sarima2_12.gcn",
    ],
)
def test_model_to_pymc(gcn_file):
    ss_mod = load_and_cache_statespace(gcn_file)
    with pm.Model() as m:
        ss_mod.to_pymc()
    rv_names = [rv.name for rv in m.free_RVs]

    assert all(name in rv_names for name in ss_mod.param_priors)

    hyper_prior_names = [
        name for dist in ss_mod.shock_priors.values() for name in dist.param_name_to_hyper_name.values()
    ]

    assert all(name in rv_names for name in hyper_prior_names)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
        "rbc_linearized.gcn",
        "sarima2_12.gcn",
    ],
)
def test_model_config(gcn_file):
    ss_mod = load_and_cache_statespace(gcn_file)


def test_backward_direct_statespace_logp():
    ss_mod = load_and_cache_statespace("sarima2_12.gcn")
    ss_mod.configure(
        observed_states=["x"],
        solver="backward_direct",
        verbose=False,
    )

    rng = np.random.default_rng()
    n_obs = 100
    data = pd.DataFrame(
        rng.normal(size=(n_obs,)),
        columns=["x"],
        index=pd.date_range("2000-01-01", periods=n_obs, freq="MS"),
    )

    with pm.Model() as m:
        ss_mod.to_pymc()
        ss_mod.build_statespace_graph(data)

        point = m.initial_point()
        logp = m.compile_logp()(point)
        assert np.isfinite(logp)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values and will be automatically imputed")
def test_data_from_prior_with_constant_params():
    ss_mod = statespace_from_gcn("tests/_resources/test_gcns/rbc_linearized.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y", "C", "L"],
        measurement_error=["Y", "C", "L"],
        constant_params=["beta", "delta"],
        solver="scan_cycle_reduction",
        mode="JAX",
        verbose=False,
    )

    with pm.Model(coords=ss_mod.coords) as pm_mod:
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        for var_name in ss_mod.observed_states:
            pm.Gamma(f"error_sigma_{var_name}", alpha=2, beta=100)

    true_params, data, _prior_idata = data_from_prior(
        ss_mod,
        pm_mod,
        n_samples=5,
        random_seed=42,
    )

    assert data.shape[1] == 3
    assert "beta" not in true_params.data_vars
    assert "delta" not in true_params.data_vars


def test_constant_params_auto():
    ss_mod = statespace_from_gcn("tests/_resources/test_gcns/open_rbc.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y"],
        constant_params="auto",
        solver="scan_cycle_reduction",
        verbose=False,
    )

    params_with_priors = set(ss_mod.param_priors.keys())
    input_param_names = {x.name for x in ss_mod.input_parameters}
    expected_constant = input_param_names - params_with_priors

    assert len(expected_constant) > 0, "Test requires a model with some prior-less parameters"
    assert set(ss_mod.constant_parameters) == expected_constant

    # Constant params should not appear in param_names
    for name in expected_constant:
        assert name not in ss_mod.param_names

    # Params with priors should still be free
    for name in params_with_priors:
        assert name in ss_mod.param_names


def _eval_augmented_matrices(ss_mod):
    """Compile and evaluate the augmented T, R, Z matrices at default parameter values."""
    inputs = pm.inputvars(ss_mod.linearized_system)
    input_names = [x.name for x in inputs]
    f = pytensor.function(inputs, [ss_mod.ssm["transition"], ss_mod.ssm["selection"]], on_unused_input="ignore")
    param_dict = load_and_cache_model("rbc_linearized.gcn").parameters()
    T, R = f(**{k: param_dict[k] for k in input_names})
    Z = ss_mod._make_design_matrix()
    return T, R, Z


def test_cumulator_augmentation_structure():
    """T, R, Z matrices have correct cumulator block for flow variable temporal aggregation."""
    ss_mod = statespace_from_gcn("tests/_resources/test_gcns/rbc_linearized.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y"],
        flow_variables=["Y"],
        solver="gensys",
        verbose=False,
    )
    T, R, Z = _eval_augmented_matrices(ss_mod)

    k_orig = ss_mod._k_orig_states
    y_idx = ss_mod._orig_state_names.index("Y")

    # Transition: first cumulator copies Y row; remaining shift by one
    np.testing.assert_allclose(T[k_orig, :k_orig], T[y_idx, :k_orig], atol=1e-10)
    assert T[k_orig + 1, k_orig] == 1.0
    assert T[k_orig + 2, k_orig + 1] == 1.0

    # Selection: first cumulator inherits Y's shock response; rest zero
    np.testing.assert_allclose(R[k_orig, :], R[y_idx, :], atol=1e-10)
    np.testing.assert_allclose(R[k_orig + 1 :, :], 0.0, atol=1e-10)

    # Design: sums current Y + 3 cumulator lags
    expected_Z = np.zeros_like(Z)
    expected_Z[0, y_idx] = 1.0
    expected_Z[0, k_orig : k_orig + 3] = 1.0
    np.testing.assert_allclose(Z, expected_Z)


def test_mixed_flow_and_stock_design():
    """With Y as flow and K as stock, Z has cumulator sum for Y and a plain selector for K."""
    ss_mod = statespace_from_gcn("tests/_resources/test_gcns/rbc_linearized.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y", "K"],
        measurement_error=["Y", "K"],
        flow_variables=["Y"],
        solver="gensys",
        verbose=False,
    )
    _T, _R, Z = _eval_augmented_matrices(ss_mod)

    k_orig = ss_mod._k_orig_states
    orig_names = ss_mod._orig_state_names
    y_idx = orig_names.index("Y")
    k_idx = orig_names.index("K")

    # Y row: sums current + 3 cumulator lags (flow)
    expected_y_row = np.zeros(ss_mod.k_states)
    expected_y_row[y_idx] = 1.0
    expected_y_row[k_orig : k_orig + 3] = 1.0
    np.testing.assert_allclose(Z[0], expected_y_row)

    # K row: plain selector into original state block (stock)
    expected_k_row = np.zeros(ss_mod.k_states)
    expected_k_row[k_idx] = 1.0
    np.testing.assert_allclose(Z[1], expected_k_row)


@pytest.mark.parametrize("aggregation,weight", [("sum", 1.0), ("average", 0.25)])
def test_flow_design_matrix_weighting(aggregation, weight):
    """Design matrix entries scale correctly for sum vs average temporal aggregation."""
    ss_mod = statespace_from_gcn("tests/_resources/test_gcns/rbc_linearized.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y"],
        flow_variables=["Y"],
        flow_aggregation=aggregation,
        solver="gensys",
        verbose=False,
    )
    Z = ss_mod._make_design_matrix()
    np.testing.assert_allclose(Z[0, Z[0] != 0], weight)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
def test_mixed_frequency_logp():
    """Kalman filter produces finite logp with NaN-padded annual data on flow + stock obs."""
    ss_mod = statespace_from_gcn("tests/_resources/test_gcns/rbc_linearized.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y", "K"],
        measurement_error=["Y", "K"],
        flow_variables=["Y"],
        solver="gensys",
        verbose=False,
    )

    rng = np.random.default_rng(42)
    n_quarters = 40
    hf_index = pd.date_range("2000-01-01", periods=n_quarters, freq="QS")
    data = pd.DataFrame(np.nan, index=hf_index, columns=["Y", "K"])
    data.iloc[3::4] = rng.normal(0, 0.1, size=(n_quarters // 4, 2))

    with pm.Model():
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        pm.Gamma("error_sigma_Y", alpha=2, beta=100)
        pm.Gamma("error_sigma_K", alpha=2, beta=100)
        ss_mod.build_statespace_graph(data, add_norm_check=False)
        logp = pm.modelcontext(None).compile_logp()(pm.modelcontext(None).initial_point())
        assert np.isfinite(logp)


def test_prepare_mixed_frequency_data():
    """Annual data expands to quarterly with values at correct positions and NaN elsewhere."""
    annual = pd.DataFrame(
        {"GDP": [100.0, 110.0], "R": [0.05, 0.04]},
        index=pd.to_datetime(["2020-01-01", "2021-01-01"]),
    )

    # Default: all columns at Q4 (last period)
    quarterly = prepare_mixed_frequency_data(annual, high_freq="QS")
    assert quarterly.shape == (8, 2)
    np.testing.assert_array_equal(quarterly.iloc[3].values, [100.0, 0.05])
    np.testing.assert_array_equal(quarterly.iloc[7].values, [110.0, 0.04])
    assert quarterly.isna().sum().sum() == 12

    # First period: all columns at Q1
    quarterly = prepare_mixed_frequency_data(annual, high_freq="QS", observation_position="first")
    assert quarterly.loc["2020-01-01", "GDP"] == 100.0
    assert quarterly.loc["2020-01-01", "R"] == 0.05
    assert np.isnan(quarterly.loc["2020-10-01", "GDP"])
