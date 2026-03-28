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


@pytest.fixture
def rbc_statespace():
    return statespace_from_gcn("tests/_resources/test_gcns/rbc_linearized.gcn", verbose=False)


def _eval_augmented_matrices(ss_mod):
    inputs = pm.inputvars(ss_mod.linearized_system)
    input_names = [x.name for x in inputs]
    f = pytensor.function(inputs, [ss_mod.ssm["transition"], ss_mod.ssm["selection"]], on_unused_input="ignore")
    param_dict = load_and_cache_model("rbc_linearized.gcn").parameters()
    T, R = f(**{k: param_dict[k] for k in input_names})
    Z = ss_mod._make_design_matrix()
    return T, R, Z


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
def test_to_pymc_creates_rvs_for_priors(gcn_file):
    ss_mod = load_and_cache_statespace(gcn_file)
    with pm.Model() as m:
        ss_mod.to_pymc()
    rv_names = {rv.name for rv in m.free_RVs}

    assert set(ss_mod.param_priors.keys()) <= rv_names

    hyper_prior_names = {
        name for dist in ss_mod.shock_priors.values() for name in dist.param_name_to_hyper_name.values()
    }
    assert hyper_prior_names <= rv_names


def test_backward_direct_solver_produces_finite_logp():
    ss_mod = load_and_cache_statespace("sarima2_12.gcn")
    ss_mod.configure(observed_states=["x"], solver="backward_direct", verbose=False)

    rng = np.random.default_rng()
    data = pd.DataFrame(
        rng.normal(size=(100,)),
        columns=["x"],
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )

    with pm.Model() as m:
        ss_mod.to_pymc()
        ss_mod.build_statespace_graph(data)
        logp = m.compile_logp()(m.initial_point())
        assert np.isfinite(logp)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values and will be automatically imputed")
def test_constant_params_excluded_from_prior_samples():
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

    true_params, data, _ = data_from_prior(ss_mod, pm_mod, n_samples=5, random_seed=17031)

    assert data.shape[1] == 3
    assert "beta" not in true_params.data_vars
    assert "delta" not in true_params.data_vars


def test_constant_params_auto_excludes_priorless_params():
    ss_mod = statespace_from_gcn("tests/_resources/test_gcns/open_rbc.gcn", verbose=False)
    ss_mod.configure(observed_states=["Y"], constant_params="auto", solver="scan_cycle_reduction", verbose=False)

    params_with_priors = set(ss_mod.param_priors.keys())
    input_param_names = {x.name for x in ss_mod.input_parameters}
    expected_constant = input_param_names - params_with_priors

    assert expected_constant, "Test requires a model with some prior-less parameters"
    assert set(ss_mod.constant_parameters) == expected_constant
    assert not (expected_constant & set(ss_mod.param_names))
    assert params_with_priors <= set(ss_mod.param_names)


def test_temporal_aggregation_sum_accumulates_over_window(rbc_statespace):
    rbc_statespace.configure(
        observed_states=["Y"],
        temporal_aggregation={"Y": "sum"},
        solver="gensys",
        verbose=False,
    )
    T, _R, Z = _eval_augmented_matrices(rbc_statespace)

    k_orig = rbc_statespace._k_orig_states
    y_idx = rbc_statespace._orig_state_names.index("Y")

    rng = np.random.default_rng()
    x = np.zeros(T.shape[0])
    x[:k_orig] = rng.normal(0, 0.1, size=k_orig)

    quarterly_Y = [x[y_idx]]
    for _ in range(3):
        x = T @ x
        quarterly_Y.append(x[y_idx])

    observation = (Z @ x)[0]
    np.testing.assert_allclose(observation, sum(quarterly_Y), rtol=1e-10)


@pytest.mark.parametrize("aggregation,weight", [("sum", 1.0), ("mean", 0.25)])
def test_temporal_aggregation_design_matrix_weighting(rbc_statespace, aggregation, weight):
    rbc_statespace.configure(
        observed_states=["Y"],
        temporal_aggregation={"Y": aggregation},
        solver="gensys",
        verbose=False,
    )
    Z = rbc_statespace._make_design_matrix()
    np.testing.assert_allclose(Z[0, Z[0] != 0], weight)


def test_aggregated_and_direct_variables_have_correct_design_rows(rbc_statespace):
    rbc_statespace.configure(
        observed_states=["Y", "K"],
        measurement_error=["Y", "K"],
        temporal_aggregation={"Y": "sum"},
        solver="gensys",
        verbose=False,
    )
    _T, _R, Z = _eval_augmented_matrices(rbc_statespace)

    k_orig = rbc_statespace._k_orig_states
    y_idx = rbc_statespace._orig_state_names.index("Y")
    k_idx = rbc_statespace._orig_state_names.index("K")

    expected_y_row = np.zeros(rbc_statespace.k_states)
    expected_y_row[y_idx] = 1.0
    expected_y_row[k_orig : k_orig + 3] = 1.0
    np.testing.assert_allclose(Z[0], expected_y_row)

    expected_k_row = np.zeros(rbc_statespace.k_states)
    expected_k_row[k_idx] = 1.0
    np.testing.assert_allclose(Z[1], expected_k_row)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
def test_mixed_frequency_data_produces_finite_logp(rbc_statespace):
    rbc_statespace.configure(
        observed_states=["Y", "K"],
        measurement_error=["Y", "K"],
        temporal_aggregation={"Y": "sum"},
        solver="gensys",
        verbose=False,
    )

    rng = np.random.default_rng()
    n_quarters = 40
    hf_index = pd.date_range("2000-01-01", periods=n_quarters, freq="QS")

    data = pd.DataFrame(index=hf_index, columns=["Y", "K"])
    data["Y"] = np.nan
    data.loc[data.index[3::4], "Y"] = rng.normal(0, 0.1, size=n_quarters // 4)
    data["K"] = rng.normal(0, 0.1, size=n_quarters)

    with pm.Model():
        rbc_statespace.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        pm.Gamma("error_sigma_Y", alpha=2, beta=100)
        pm.Gamma("error_sigma_K", alpha=2, beta=100)
        rbc_statespace.build_statespace_graph(data, add_norm_check=False)
        logp = pm.modelcontext(None).compile_logp()(pm.modelcontext(None).initial_point())
        assert np.isfinite(logp)


@pytest.mark.parametrize(
    "aggregation_period,expected_shape",
    [(4, (3, 3)), (3, (2, 2)), (2, (1, 1))],
    ids=["quarterly-to-annual", "monthly-to-quarterly", "period-2"],
)
def test_cumulator_shift_matrix_shape_matches_aggregation_period(rbc_statespace, aggregation_period, expected_shape):
    rbc_statespace.configure(
        observed_states=["Y"],
        temporal_aggregation={"Y": "sum"},
        aggregation_period=aggregation_period,
        solver="gensys",
        verbose=False,
    )
    T, _R, _Z = _eval_augmented_matrices(rbc_statespace)
    k_orig = rbc_statespace._k_orig_states
    C = T[k_orig:, k_orig:]
    assert C.shape == expected_shape


def test_multiple_aggregated_variables_produce_block_diagonal_structure(rbc_statespace):
    rbc_statespace.configure(
        observed_states=["Y", "C"],
        measurement_error=["Y", "C"],
        temporal_aggregation={"Y": "sum", "C": "sum"},
        aggregation_period=4,
        solver="gensys",
        verbose=False,
    )
    T, _R, _Z = _eval_augmented_matrices(rbc_statespace)

    k_orig = rbc_statespace._k_orig_states
    C = T[k_orig:, k_orig:]

    shift = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    expected_C = np.kron(np.eye(2, dtype=np.float32), shift)
    np.testing.assert_allclose(C, expected_C)


@pytest.mark.parametrize("observation_position", ["first", "last"])
def test_prepare_mixed_frequency_data_positions_values_correctly(observation_position):
    annual = pd.DataFrame(
        {"GDP": [100.0, 110.0], "R": [0.05, 0.04]},
        index=pd.to_datetime(["2020-01-01", "2021-01-01"]),
    )

    quarterly = prepare_mixed_frequency_data(annual, high_freq="QS", observation_position=observation_position)

    if observation_position == "last":
        assert quarterly.shape == (8, 2)
        np.testing.assert_array_equal(quarterly.iloc[3].values, [100.0, 0.05])
        np.testing.assert_array_equal(quarterly.iloc[7].values, [110.0, 0.04])
    else:
        assert quarterly.loc["2020-01-01", "GDP"] == 100.0
        assert quarterly.loc["2021-01-01", "GDP"] == 110.0


@pytest.mark.parametrize(
    "high_freq,aggregation_period,expected_len,value_position",
    [("MS", 12, 12, "2020-12-01"), ("MS", 3, 6, "2020-03-01")],
    ids=["monthly-to-annual", "monthly-to-quarterly"],
)
def test_prepare_mixed_frequency_data_various_periods(high_freq, aggregation_period, expected_len, value_position):
    low_freq = pd.DataFrame({"GDP": [100.0]}, index=pd.to_datetime(["2020-01-01"]))
    if aggregation_period == 3:
        low_freq = pd.DataFrame({"GDP": [100.0, 110.0]}, index=pd.to_datetime(["2020-01-01", "2020-04-01"]))

    result = prepare_mixed_frequency_data(
        low_freq, high_freq=high_freq, aggregation_period=aggregation_period, observation_position="last"
    )

    assert len(result) == expected_len
    assert result.loc[value_position, "GDP"] == 100.0


def test_mixed_sum_and_mean_aggregation_weights(rbc_statespace):
    rbc_statespace.configure(
        observed_states=["Y", "C"],
        measurement_error=["Y", "C"],
        temporal_aggregation={"Y": "sum", "C": "mean"},
        aggregation_period=4,
        solver="gensys",
        verbose=False,
    )
    _T, _R, Z = _eval_augmented_matrices(rbc_statespace)

    k_orig = rbc_statespace._k_orig_states
    y_idx = rbc_statespace._orig_state_names.index("Y")
    c_idx = rbc_statespace._orig_state_names.index("C")

    np.testing.assert_allclose(Z[0, y_idx], 1.0)
    np.testing.assert_allclose(Z[0, k_orig : k_orig + 3], 1.0)

    np.testing.assert_allclose(Z[1, c_idx], 0.25)
    np.testing.assert_allclose(Z[1, k_orig + 3 : k_orig + 6], 0.25)


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_and_last_aggregation_use_direct_selector(rbc_statespace, method):
    rbc_statespace.configure(
        observed_states=["Y"],
        temporal_aggregation={"Y": method},
        solver="gensys",
        verbose=False,
    )

    assert rbc_statespace._n_cumulator_states == 0

    _T, _R, Z = _eval_augmented_matrices(rbc_statespace)
    y_idx = rbc_statespace._orig_state_names.index("Y")

    expected_row = np.zeros(rbc_statespace.k_states)
    expected_row[y_idx] = 1.0
    np.testing.assert_allclose(Z[0], expected_row)
