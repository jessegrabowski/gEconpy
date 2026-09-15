import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
import xarray as xr

from pytensor.graph.traversal import ancestors
from pytensor.tensor.variable import TensorConstant

from gEconpy import statespace_from_gcn
from gEconpy.model.statespace import data_from_prior, prepare_mixed_frequency_data
from gEconpy.model.statistics import autocorrelation_matrix
from tests._resources.cache_compiled_models import (
    load_and_cache_model,
    load_and_cache_statespace,
)
from tests.conftest import TEST_GCNS


@pytest.fixture
def rbc_statespace():
    return statespace_from_gcn(TEST_GCNS / "rbc_linearized.gcn", verbose=False)


def _evaluate_at(ss_mod, outputs, param_values):
    """Compile ``outputs`` as a function of the statespace input parameters and evaluate it at ``param_values``."""
    inputs = [v for v in ss_mod.input_parameters if v.name in param_values]
    fn = pytensor.function(inputs, outputs, on_unused_input="ignore")
    return fn(*[param_values[v.name] for v in inputs])


def _eval_augmented_matrices(ss_mod):
    param_values = load_and_cache_model("rbc_linearized.gcn").parameters()
    T, R = _evaluate_at(ss_mod, [ss_mod.ssm["transition"], ss_mod.ssm["selection"]], param_values)
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

    inputs = pm.pytensorf.inputvars(ss_mod.linearized_system)
    input_names = [x.name for x in inputs]
    f = pytensor.function(inputs, ss_mod.linearized_system, on_unused_input="ignore")
    mod_matrices = model.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})

    param_dict = model.parameters()
    ss_matrices = f(**{k: param_dict[k] for k in input_names})

    # ``ss_mod.linearized_system`` keeps the rows in ``model.eq_order`` and the columns of A, B, and C in
    # ``ss_mod.var_order``, whereas ``model.linearize_model`` undoes both permutations before returning.
    inv_eq = np.argsort(model.eq_order)
    inv_var = ss_mod.inv_var_order
    A_ss, B_ss, C_ss, D_ss = ss_matrices
    A_ss = A_ss[inv_eq][:, inv_var]
    B_ss = B_ss[inv_eq][:, inv_var]
    C_ss = C_ss[inv_eq][:, inv_var]
    D_ss = D_ss[inv_eq]
    ss_matrices = [A_ss, B_ss, C_ss, D_ss]

    for mod_matrix, ss_matrix in zip(mod_matrices, ss_matrices, strict=True):
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

    rng = np.random.default_rng(0)
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
def test_norm_check_is_zero_at_solution(rbc_statespace):
    """The residual norms must be computed in one variable ordering, so they vanish at the policy solution."""
    assert not np.array_equal(rbc_statespace.var_order, np.arange(len(rbc_statespace.var_order)))
    rbc_statespace.configure(observed_states=["Y"], solver="gensys", verbose=False)

    with pm.Model() as m:
        rbc_statespace.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        rbc_statespace.build_statespace_graph(np.full((40, 1), np.nan), add_norm_check=True)
        norm_fn = m.compile_fn(
            [m["deterministic_norm"], m["stochastic_norm"]], inputs=m.value_vars, on_unused_input="ignore"
        )
        norms = norm_fn(m.initial_point())

    np.testing.assert_allclose(norms, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"observed_states": ["NotAState"]}, "unknown to the model and cannot be set as observed: NotAState"),
        ({"observed_states": ["Y"], "measurement_error": ["K"]}, "not observed, and cannot have measurement error: K"),
        (
            {"observed_states": ["Y"], "temporal_aggregation": {"K": "sum"}},
            "temporal_aggregation variables are not in observed_states: K",
        ),
        (
            {"observed_states": ["Y"], "temporal_aggregation": {"Y": "median"}},
            "Invalid aggregation methods: Y='median'",
        ),
        (
            {"observed_states": ["Y"], "temporal_aggregation": {"Y": "sum"}, "aggregation_period": 1},
            "aggregation_period must be >= 2",
        ),
        (
            {"observed_states": ["Y"], "ss_obs_intercept": ["K"]},
            "ss_obs_intercept entries are not in observed_states: K",
        ),
        ({"observed_states": ["Y"], "constant_params": ["not_a_param"]}, "cannot be set as constant: not_a_param"),
        ({"observed_states": ["Y"], "solver": "not_a_solver"}, "Unknown solver 'not_a_solver'"),
        (
            {"observed_states": ["Y", "K", "C"]},
            "Stochastic singularity! You requested 3 observed timeseries, but there is only 1 source",
        ),
        ({"observed_states": ["Y", "K", "C"], "measurement_error": ["Y"]}, "there are only 2 sources"),
    ],
    ids=[
        "unknown-observed-state",
        "measurement-error-on-unobserved-state",
        "aggregation-of-unobserved-state",
        "unknown-aggregation-method",
        "aggregation-period-below-two",
        "ss-obs-intercept-on-unobserved-state",
        "unknown-constant-param",
        "unknown-solver",
        "stochastic-singularity-one-source",
        "stochastic-singularity-two-sources",
    ],
)
def test_configure_rejects_inconsistent_arguments(rbc_statespace, kwargs, match):
    with pytest.raises(ValueError, match=match):
        rbc_statespace.configure(verbose=False, **kwargs)


def test_to_pymc_skips_excluded_and_constant_priors(rbc_statespace):
    rbc_statespace.configure(observed_states=["Y"], constant_params=["beta"], solver="gensys", verbose=False)
    assert {"alpha", "beta", "rho_A"} <= set(rbc_statespace.param_priors)

    with pm.Model() as m:
        rbc_statespace.to_pymc(exclude_priors=["alpha"])
    rv_names = {rv.name for rv in m.free_RVs}

    assert "rho_A" in rv_names
    assert "alpha" not in rv_names
    assert "beta" not in rv_names


def test_full_shock_covariance_takes_one_matrix_parameter(rbc_statespace):
    rbc_statespace.configure(observed_states=["Y"], full_shock_covariance=True, solver="gensys", verbose=False)

    assert "state_cov" in rbc_statespace.param_names
    assert "sigma_epsilon_A" not in rbc_statespace.param_names
    assert rbc_statespace.param_dims["state_cov"] == ("shock", "shock_aux")

    Q = rbc_statespace.ssm["state_cov"]
    (state_cov_input,) = pm.pytensorf.inputvars([Q])
    assert state_cov_input.name == "state_cov"
    np.testing.assert_allclose(pytensor.function([state_cov_input], Q)(np.array([[0.04]])), [[0.04]])


@pytest.mark.filterwarnings("ignore:Provided data contains missing values and will be automatically imputed")
def test_data_from_prior_drops_constant_params_and_blanks_requested_fraction():
    ss_mod = statespace_from_gcn(TEST_GCNS / "rbc_linearized.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y", "C", "L"],
        measurement_error=["Y", "C", "L"],
        constant_params=["beta", "delta"],
        solver="scan_cycle_reduction",
        verbose=False,
    )

    with pm.Model(coords=ss_mod.coords) as pm_mod:
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        for var_name in ss_mod.observed_states:
            pm.Gamma(f"error_sigma_{var_name}", alpha=2, beta=100)

    index = pd.date_range("1990-01-01", periods=40, freq="QS")
    true_params, data, prior_idata = data_from_prior(
        statespace_mod=ss_mod, pymc_model=pm_mod, index=index, n_samples=5, pct_missing=0.1, random_seed=17031
    )

    assert isinstance(true_params, xr.Dataset)
    assert "beta" not in true_params.data_vars
    assert "delta" not in true_params.data_vars
    assert int(true_params["param_idx"]) in range(5)

    assert list(data.columns) == ["Y", "C", "L"]
    assert data.index.equals(index)
    assert (data.isna().sum() == 4).all()
    assert "unconditional_prior" in prior_idata.children


def test_constant_params_auto_excludes_priorless_params():
    ss_mod = statespace_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)
    ss_mod.configure(observed_states=["Y"], constant_params="auto", solver="scan_cycle_reduction", verbose=False)

    params_with_priors = set(ss_mod.param_priors.keys())
    input_param_names = {x.name for x in ss_mod.input_parameters}
    expected_constant = input_param_names - params_with_priors

    assert expected_constant, "Test requires a model with some prior-less parameters"
    assert set(ss_mod.constant_parameters) == expected_constant
    assert not (expected_constant & set(ss_mod.param_names))
    assert params_with_priors <= set(ss_mod.param_names)


def test_statespace_from_gcn_forwards_filter_settings_and_mode():
    ss_mod = statespace_from_gcn(
        TEST_GCNS / "rbc_linearized.gcn", verbose=False, cov_jitter=1e-6, missing_fill_value=-1.0
    )
    assert ss_mod.cov_jitter == 1e-6
    assert ss_mod.missing_fill_value == -1.0
    assert ss_mod.mode is None

    ss_mod.configure(observed_states=["Y"], solver="gensys", mode="NUMBA", verbose=False)
    assert ss_mod.mode == "NUMBA"
    assert ss_mod.cov_jitter == 1e-6
    assert ss_mod.missing_fill_value == -1.0


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

    rng = np.random.default_rng(0)
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

    rng = np.random.default_rng(0)
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


@pytest.mark.parametrize(
    "observation_position, expected_rows",
    [("first", [0, 4]), ("last", [3, 7])],
    ids=["first", "last"],
)
def test_prepare_mixed_frequency_data_positions_values_correctly(observation_position, expected_rows):
    annual = pd.DataFrame(
        {"GDP": [100.0, 110.0], "R": [0.05, 0.04]},
        index=pd.to_datetime(["2020-01-01", "2021-01-01"]),
    )

    quarterly = prepare_mixed_frequency_data(annual, high_freq="QS", observation_position=observation_position)

    assert quarterly.shape == (expected_rows[-1] + 1, 2)
    np.testing.assert_array_equal(quarterly.iloc[expected_rows].values, annual.values)
    assert quarterly.drop(quarterly.index[expected_rows]).isna().all().all()


@pytest.mark.parametrize(
    "low_freq_dates, aggregation_period, expected_len, value_position",
    [
        (["2020-01-01"], 12, 12, "2020-12-01"),
        (["2020-01-01", "2020-04-01"], 3, 6, "2020-03-01"),
    ],
    ids=["monthly-to-annual", "monthly-to-quarterly"],
)
def test_prepare_mixed_frequency_data_various_periods(low_freq_dates, aggregation_period, expected_len, value_position):
    low_freq = pd.DataFrame(
        {"GDP": 100.0 + 10.0 * np.arange(len(low_freq_dates))}, index=pd.to_datetime(low_freq_dates)
    )

    result = prepare_mixed_frequency_data(
        low_freq, high_freq="MS", aggregation_period=aggregation_period, observation_position="last"
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


# RBC with a non-trivial analytic steady state (Y_ss is about 3).
NL_GCN = "rbc_2_block_ss.gcn"

# The same RBC plus an OBSERVATION block carrying ``Y_obs = log(Y)`` and ``dY_obs = log(Y) - log(Y[-1])`` as model
# identities, the Dynare way of adding observed series. It is the reference for observation-equation equivalence.
OBS_EQ_GCN = "rbc_2_block_obs_eq.gcn"


@pytest.fixture
def rbc_nonlinear_ss():
    return statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)


def _eval_obs_pieces(ss_mod, param_values):
    return _evaluate_at(ss_mod, [ss_mod.ssm["obs_intercept"], ss_mod.ssm["design"]], param_values)


def _model_ss_values():
    model = load_and_cache_model(NL_GCN)
    ss_vals = model.steady_state(verbose=False, progressbar=False)
    return model, ss_vals


def test_ss_obs_intercept_evaluates_to_log_v_ss(rbc_nonlinear_ss):
    model, ss_vals = _model_ss_values()
    rbc_nonlinear_ss.configure(
        observed_states=["Y", "C", "L"],
        measurement_error=["Y", "C", "L"],
        constant_params="auto",
        ss_obs_intercept=["Y", "C", "L"],
        verbose=False,
    )
    d, Z = _eval_obs_pieces(rbc_nonlinear_ss, model.parameters())

    expected = np.array([np.log(float(ss_vals[f"{v}_ss"])) for v in ("Y", "C", "L")])
    np.testing.assert_allclose(d, expected, rtol=1e-10)

    for i, name in enumerate(("Y", "C", "L")):
        idx = rbc_nonlinear_ss._orig_state_names.index(name)
        np.testing.assert_allclose(Z[i, idx], 1.0)
        np.testing.assert_allclose(Z[i].sum(), 1.0)


def test_ss_obs_intercept_zero_for_unmentioned_states(rbc_nonlinear_ss):
    model, ss_vals = _model_ss_values()
    rbc_nonlinear_ss.configure(
        observed_states=["Y", "C"],
        measurement_error=["Y", "C"],
        constant_params="auto",
        ss_obs_intercept=["Y"],
        verbose=False,
    )
    d, _ = _eval_obs_pieces(rbc_nonlinear_ss, model.parameters())
    assert d[0] == pytest.approx(np.log(float(ss_vals["Y_ss"])), rel=1e-10)
    assert d[1] == 0.0


def test_ss_obs_intercept_sum_aggregation_scales_intercept(rbc_nonlinear_ss):
    model, ss_vals = _model_ss_values()
    rbc_nonlinear_ss.configure(
        observed_states=["Y"],
        measurement_error=["Y"],
        constant_params="auto",
        ss_obs_intercept=["Y"],
        temporal_aggregation={"Y": "sum"},
        aggregation_period=4,
        verbose=False,
    )
    d, _ = _eval_obs_pieces(rbc_nonlinear_ss, model.parameters())
    assert d[0] == pytest.approx(4 * np.log(float(ss_vals["Y_ss"])), rel=1e-10)


def test_ss_obs_intercept_default_is_zero(rbc_nonlinear_ss):
    """Omitting ``ss_obs_intercept`` leaves the framework's default zero obs_intercept in place."""
    rbc_nonlinear_ss.configure(
        observed_states=["Y"],
        measurement_error=["Y"],
        constant_params="auto",
        verbose=False,
    )
    d = rbc_nonlinear_ss.ssm["obs_intercept"]
    fn = pytensor.function([], d, on_unused_input="ignore")
    np.testing.assert_allclose(fn(), np.zeros(rbc_nonlinear_ss.k_endog))


@pytest.mark.parametrize(
    "measurement_error, expected_diagonal",
    [(["Y", "K"], [0.25, 0.25]), (["K"], [0.0, 0.25]), (["Y"], [0.25, 0.0])],
    ids=["all", "second_only", "first_only"],
)
def test_measurement_error_lands_on_its_observed_state(rbc_statespace, measurement_error, expected_diagonal):
    rbc_statespace.configure(
        observed_states=["Y", "K"],
        measurement_error=measurement_error,
        solver="gensys",
        verbose=False,
    )
    H = rbc_statespace.ssm["obs_cov"]
    sigmas = pm.pytensorf.inputvars([H])
    fn = pytensor.function(sigmas, H)
    np.testing.assert_allclose(fn(*[0.5] * len(sigmas)), np.diag(expected_diagonal))


def test_ss_obs_intercept_unknown_state_raises(rbc_nonlinear_ss):
    with pytest.raises(ValueError, match="not in observed_states"):
        rbc_nonlinear_ss.configure(
            observed_states=["Y"],
            measurement_error=["Y"],
            constant_params="auto",
            ss_obs_intercept=["NotObserved"],
            verbose=False,
        )


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        pytest.param(
            {"observation_equations": {"Y": "log(Y[])"}, "ss_obs_intercept": ["Y"]},
            ValueError,
            "appear in both observation_equations and ss_obs_intercept",
            id="overlap_with_ss_obs_intercept",
        ),
        pytest.param(
            {"observation_equations": {"NotObserved": "log(Y[])"}},
            ValueError,
            "not in observed_states",
            id="key_not_in_observed_states",
        ),
        pytest.param(
            {"observation_equations": {"Y": "log(NotAVariable[])"}},
            ValueError,
            "unknown model variable",
            id="unknown_variable",
        ),
        pytest.param(
            {"observation_equations": {"Y": "log(Y[]) + not_a_param"}},
            ValueError,
            "unknown symbol",
            id="unknown_symbol",
        ),
        pytest.param(
            {"observation_equations": {"Y": "log(Y[1])"}},
            ValueError,
            "lead reference",
            id="lead_reference",
        ),
    ],
)
def test_observation_equations_validation_errors(rbc_nonlinear_ss, kwargs, exception, match):
    with pytest.raises(exception, match=match):
        rbc_nonlinear_ss.configure(
            observed_states=["Y"],
            measurement_error=["Y"],
            constant_params="auto",
            verbose=False,
            **kwargs,
        )


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
def test_observation_equations_aggregation_produces_finite_logp():
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["dlog_Y_annual"],
        measurement_error=["dlog_Y_annual"],
        constant_params="auto",
        observation_equations={"dlog_Y_annual": "log(Y[]) - log(Y[-1])"},
        temporal_aggregation={"dlog_Y_annual": "sum"},
        aggregation_period=4,
        verbose=False,
    )
    rng = np.random.default_rng(11)
    n = 32
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    data = pd.DataFrame(np.nan, index=idx, columns=["dlog_Y_annual"])
    # Annual observations land at the last quarter of each year.
    data.iloc[3::4, 0] = rng.normal(scale=0.02, size=n // 4)
    with pm.Model() as m:
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        pm.Gamma("error_sigma_dlog_Y_annual", alpha=2, beta=100)
        ss_mod.build_statespace_graph(data, add_norm_check=False)
        logp = m.compile_logp()(m.initial_point())
    assert np.isfinite(logp)


def test_observation_equations_produce_finite_logp(rbc_nonlinear_ss):
    """Two simultaneous observation equations flow through to a finite Kalman likelihood."""
    rbc_nonlinear_ss.configure(
        observed_states=["Y", "C"],
        measurement_error=["Y", "C"],
        constant_params="auto",
        observation_equations={"Y": "log(C[] + I[])", "C": "log(C[])"},
        verbose=False,
    )
    rng = np.random.default_rng(42)
    n = 30
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    data = pd.DataFrame(rng.normal(scale=0.1, size=(n, 2)), index=idx, columns=["Y", "C"])
    with pm.Model() as m:
        rbc_nonlinear_ss.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        pm.Gamma("error_sigma_Y", alpha=2, beta=100)
        pm.Gamma("error_sigma_C", alpha=2, beta=100)
        rbc_nonlinear_ss.build_statespace_graph(data, add_norm_check=False)
        logp = m.compile_logp()(m.initial_point())
    assert np.isfinite(logp)


def _obs_eq_logp(ss_mod, obs_name, data, point=None):
    """
    Build the pymc statespace graph for ``ss_mod`` and return ``(logp, point)``.

    The shock and measurement-error standard deviations get identical priors in both representations, so reusing
    ``point`` across models evaluates the log-likelihood at exactly the same parameter values.
    """
    with pm.Model() as m:
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        pm.Gamma(f"error_sigma_{obs_name}", alpha=2, beta=100)
        ss_mod.build_statespace_graph(data, add_norm_check=False)
        ip = m.initial_point() if point is None else point
        logp = m.compile_logp()(ip)
    return logp, ip


@pytest.mark.parametrize(
    "obs_name, obs_eq, ss_intercept",
    [
        pytest.param("Y_obs", "log(Y[])", True, id="contemporaneous"),
        pytest.param("dY_obs", "log(Y[]) - log(Y[-1])", False, id="first_difference"),
    ],
)
def test_observation_equation_matches_model_variable_equivalent(obs_name, obs_eq, ss_intercept):
    """
    An observation equation gives the same Kalman likelihood as the Dynare way.

    The observation equation linearizes ``obs_eq`` into ``Z`` and appends an observation-lag block to ``T`` for any
    lagged terms. The reference gcn ``rbc_2_block_obs_eq.gcn`` carries the series as a level-linearized model
    identity observed through a plain selector. The two representations have different state dimensions but
    imply the same distribution over the observable, so the log-likelihoods agree.
    """
    rng = np.random.default_rng(0)
    n = 50
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    center = np.log(3.0) if ss_intercept else 0.0
    data = pd.DataFrame(center + rng.normal(scale=0.05, size=(n, 1)), index=idx, columns=[obs_name])

    dm = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    dm.configure(
        observed_states=[obs_name],
        measurement_error=[obs_name],
        constant_params="auto",
        observation_equations={obs_name: obs_eq},
        verbose=False,
    )
    logp_dm, point = _obs_eq_logp(dm, obs_name, data)

    dyn = statespace_from_gcn(TEST_GCNS / OBS_EQ_GCN, not_loglin_variables=["Y_obs", "dY_obs"], verbose=False)
    dyn.configure(
        observed_states=[obs_name],
        measurement_error=[obs_name],
        constant_params="auto",
        ss_obs_intercept=[obs_name] if ss_intercept else None,
        verbose=False,
    )
    logp_dyn, _ = _obs_eq_logp(dyn, obs_name, data, point=point)

    # The obs-equation representation has fewer states.
    assert dm.k_states < dyn.k_states
    np.testing.assert_allclose(logp_dm, logp_dyn, rtol=1e-7, atol=1e-7)


def _build_expected_Z(ss_mod, per_period_coeffs, agg_method, agg_period):
    """Broadcast per-period coefficients across the aggregation window into a ``(k_endog, k_states)`` array."""
    Z = np.zeros((ss_mod.k_endog, ss_mod.k_states))
    if agg_method == "sum":
        n_periods, weight = agg_period, 1.0
    elif agg_method == "mean":
        n_periods, weight = agg_period, 1.0 / agg_period
    else:
        n_periods, weight = 1, 1.0

    for (vname, lag), coeff in per_period_coeffs.items():
        for d in range(n_periods):
            effective_lag = lag - d
            col = (
                ss_mod._orig_state_names.index(vname)
                if effective_lag == 0
                else ss_mod._obs_lag_column(vname, effective_lag)
            )
            Z[0, col] += weight * coeff
    return Z


def _ss_floats():
    ss = load_and_cache_model(NL_GCN).steady_state(verbose=False, progressbar=False)
    return {key.removesuffix("_ss"): float(val) for key, val in ss.items()}


def _params_at_calib():
    return load_and_cache_model(NL_GCN).parameters()


# Each entry is (eq_string, coeffs_fn, intercept_fn, agg, period, expected_depths). ``coeffs_fn(ss, params)`` returns
# the per-period linearized coefficients before any aggregation broadcasting and ``intercept_fn(ss, params)`` the
# per-period intercept. The test does the broadcasting and the scaling.
_BATTERY = [
    pytest.param(
        "log(Y[])",
        lambda ss, p: {("Y", 0): 1.0},
        lambda ss, p: np.log(ss["Y"]),
        None,
        1,
        {},
        id="trivial_log_Y",
    ),
    pytest.param(
        "2 * log(Y[])",
        lambda ss, p: {("Y", 0): 2.0},
        lambda ss, p: 2.0 * np.log(ss["Y"]),
        None,
        1,
        {},
        id="scaled_log_Y",
    ),
    pytest.param(
        "log(Y[]) - log(C[])",
        lambda ss, p: {("Y", 0): 1.0, ("C", 0): -1.0},
        lambda ss, p: np.log(ss["Y"]) - np.log(ss["C"]),
        None,
        1,
        {},
        id="log_ratio_Y_over_C",
    ),
    pytest.param(
        "log(Y[] + C[])",
        lambda ss, p: {
            ("Y", 0): ss["Y"] / (ss["Y"] + ss["C"]),
            ("C", 0): ss["C"] / (ss["Y"] + ss["C"]),
        },
        lambda ss, p: np.log(ss["Y"] + ss["C"]),
        None,
        1,
        {},
        id="log_of_sum",
    ),
    pytest.param(
        "log(Y[]) + alpha",
        lambda ss, p: {("Y", 0): 1.0},
        lambda ss, p: np.log(ss["Y"]) + p["alpha"],
        None,
        1,
        {},
        id="intercept_with_parameter",
    ),
    pytest.param(
        "log(Y[]) - log(Y[ss])",
        lambda ss, p: {("Y", 0): 1.0},
        lambda ss, p: 0.0,
        None,
        1,
        {},
        id="steady_state_reference",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-1])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -1): -1.0},
        lambda ss, p: 0.0,
        None,
        1,
        {"Y": 1},
        id="first_difference_Y",
    ),
    pytest.param(
        "log(Y[-1])",
        lambda ss, p: {("Y", -1): 1.0},
        lambda ss, p: np.log(ss["Y"]),
        None,
        1,
        {"Y": 1},
        id="lag_only_no_current",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-2])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -2): -1.0},
        lambda ss, p: 0.0,
        None,
        1,
        {"Y": 2},
        id="depth_2_difference",
    ),
    pytest.param(
        "log(Y[]) - 2 * log(Y[-1]) + log(Y[-2])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -1): -2.0, ("Y", -2): 1.0},
        lambda ss, p: 0.0,
        None,
        1,
        {"Y": 2},
        id="second_difference",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-1]) + log(A[])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -1): -1.0, ("A", 0): 1.0},
        lambda ss, p: np.log(ss["A"]),
        None,
        1,
        {"Y": 1},
        id="bgp_growth_rate_with_trend",
    ),
    pytest.param(
        "log(Y[])",
        lambda ss, p: {("Y", 0): 1.0},
        lambda ss, p: np.log(ss["Y"]),
        "sum",
        4,
        {"Y": 3},
        id="sum_agg_contemporaneous",
    ),
    pytest.param(
        "log(Y[])",
        lambda ss, p: {("Y", 0): 1.0},
        lambda ss, p: np.log(ss["Y"]),
        "mean",
        4,
        {"Y": 3},
        id="mean_agg_contemporaneous",
    ),
    pytest.param(
        "log(Y[] + C[])",
        lambda ss, p: {
            ("Y", 0): ss["Y"] / (ss["Y"] + ss["C"]),
            ("C", 0): ss["C"] / (ss["Y"] + ss["C"]),
        },
        lambda ss, p: np.log(ss["Y"] + ss["C"]),
        "sum",
        2,
        {"Y": 1, "C": 1},
        id="sum_agg_log_of_sum",
    ),
    pytest.param(
        "log(C[] + I[])",
        lambda ss, p: {
            ("C", 0): ss["C"] / (ss["C"] + ss["I"]),
            ("I", 0): ss["I"] / (ss["C"] + ss["I"]),
        },
        lambda ss, p: np.log(ss["C"] + ss["I"]),
        "sum",
        4,
        {"C": 3, "I": 3},
        id="sum_agg_two_variable_broadcast",
    ),
    pytest.param(
        "log(C[] + I[])",
        lambda ss, p: {
            ("C", 0): ss["C"] / (ss["C"] + ss["I"]),
            ("I", 0): ss["I"] / (ss["C"] + ss["I"]),
        },
        lambda ss, p: np.log(ss["C"] + ss["I"]),
        "mean",
        4,
        {"C": 3, "I": 3},
        id="mean_agg_two_variable_broadcast",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-1])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -1): -1.0},
        lambda ss, p: 0.0,
        "sum",
        4,
        {"Y": 4},
        id="sum_agg_first_difference_telescopes_4",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-1])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -1): -1.0},
        lambda ss, p: 0.0,
        "sum",
        2,
        {"Y": 2},
        id="sum_agg_first_difference_telescopes_2",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-1])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -1): -1.0},
        lambda ss, p: 0.0,
        "mean",
        4,
        {"Y": 4},
        id="mean_agg_first_difference_telescopes",
    ),
    pytest.param(
        "log(Y[-1])",
        lambda ss, p: {("Y", -1): 1.0},
        lambda ss, p: np.log(ss["Y"]),
        "sum",
        3,
        {"Y": 3},
        id="sum_agg_lag_only",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-2])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -2): -1.0},
        lambda ss, p: 0.0,
        "sum",
        2,
        {"Y": 3},
        id="sum_agg_two_step_difference",
    ),
    pytest.param(
        "log(Y[]) - log(Y[-1])",
        lambda ss, p: {("Y", 0): 1.0, ("Y", -1): -1.0},
        lambda ss, p: 0.0,
        "last",
        4,
        {"Y": 1},
        id="last_agg_first_difference",
    ),
    pytest.param(
        "log(Y[])",
        lambda ss, p: {("Y", 0): 1.0},
        lambda ss, p: np.log(ss["Y"]),
        "first",
        4,
        {},
        id="first_agg_contemporaneous",
    ),
]


@pytest.mark.parametrize("obs_eq, coeffs_fn, intercept_fn, agg, period, depths", _BATTERY)
def test_observation_equation_Z_matches_analytical_expectation(
    obs_eq,
    coeffs_fn,
    intercept_fn,
    agg,
    period,
    depths,
):
    """Z and the intercept match a hand-computed reference for a battery of observation equations."""
    name = "y_obs"
    cfg = {
        "observed_states": [name],
        "measurement_error": [name],
        "constant_params": "auto",
        "observation_equations": {name: obs_eq},
        "verbose": False,
    }
    if agg is not None:
        cfg["temporal_aggregation"] = {name: agg}
        cfg["aggregation_period"] = period

    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(**cfg)

    assert ss_mod._obs_lag_depths == depths

    ss = _ss_floats()
    params = _params_at_calib()
    per_period_coeffs = coeffs_fn(ss, params)
    per_period_intercept = float(intercept_fn(ss, params))

    expected_Z = _build_expected_Z(ss_mod, per_period_coeffs, agg, period)
    expected_d = period * per_period_intercept if agg == "sum" else per_period_intercept

    d, Z = _eval_obs_pieces(ss_mod, params)
    np.testing.assert_allclose(Z, expected_Z, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d[0], expected_d, rtol=1e-10, atol=1e-12)


def test_observation_equation_level_linearized_variable_coefficient():
    r"""
    A level-linearized variable linearizes with the plain derivative and no ``v_ss`` chain-rule factor.

    Every other observation-equation test uses log-linearized variables, where ``log(Y)`` has unit coefficient. Here
    ``Y`` is level-linearized, so the state holds ``Y - Y_ss`` and the coefficient of ``log(Y)`` is
    :math:`\\partial \\log Y / \\partial Y = 1 / Y_{ss}`.
    """
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, not_loglin_variables=["Y"], verbose=False)
    ss_mod.configure(
        observed_states=["y_obs"],
        measurement_error=["y_obs"],
        constant_params="auto",
        observation_equations={"y_obs": "log(Y[])"},
        verbose=False,
    )
    ss = _ss_floats()
    d, Z = _eval_obs_pieces(ss_mod, _params_at_calib())
    y_idx = ss_mod._orig_state_names.index("Y")
    np.testing.assert_allclose(Z[0, y_idx], 1.0 / ss["Y"], rtol=1e-10)
    np.testing.assert_allclose(d[0], np.log(ss["Y"]), rtol=1e-10)


def test_multiple_obs_equations_share_lag_block_correctly():
    """Two obs equations with overlapping lag depths share one lag chain, sized to the deeper one."""
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["fast", "slow"],
        measurement_error=["fast", "slow"],
        constant_params="auto",
        observation_equations={
            "fast": "log(Y[]) - log(Y[-1])",
            "slow": "log(Y[]) - log(Y[-3])",
        },
        verbose=False,
    )
    assert ss_mod._obs_lag_depths == {"Y": 3}

    d, Z = _eval_obs_pieces(ss_mod, _params_at_calib())

    y_orig = ss_mod._orig_state_names.index("Y")
    y_lag1 = ss_mod._obs_lag_column("Y", -1)
    y_lag2 = ss_mod._obs_lag_column("Y", -2)
    y_lag3 = ss_mod._obs_lag_column("Y", -3)

    # ``fast`` row: +1 on Y, -1 on lag1, zero on lag2, lag3.
    np.testing.assert_allclose(Z[0, y_orig], 1.0)
    np.testing.assert_allclose(Z[0, y_lag1], -1.0)
    np.testing.assert_allclose(Z[0, y_lag2], 0.0)
    np.testing.assert_allclose(Z[0, y_lag3], 0.0)
    # ``slow`` row: +1 on Y, zero on lag1, lag2, -1 on lag3.
    np.testing.assert_allclose(Z[1, y_orig], 1.0)
    np.testing.assert_allclose(Z[1, y_lag1], 0.0)
    np.testing.assert_allclose(Z[1, y_lag2], 0.0)
    np.testing.assert_allclose(Z[1, y_lag3], -1.0)
    np.testing.assert_allclose(d, [0.0, 0.0], atol=1e-12)


def test_multiple_obs_equations_different_variables_independent_chains():
    """Each variable referenced at a lag gets its own independent lag chain, with no cross-contamination."""
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["Y_diff", "C_diff"],
        measurement_error=["Y_diff", "C_diff"],
        constant_params="auto",
        observation_equations={
            "Y_diff": "log(Y[]) - log(Y[-1])",
            "C_diff": "log(C[]) - log(C[-2])",
        },
        verbose=False,
    )
    assert ss_mod._obs_lag_depths == {"Y": 1, "C": 2}

    d, Z = _eval_obs_pieces(ss_mod, _params_at_calib())
    np.testing.assert_allclose(d, [0.0, 0.0], atol=1e-12)

    y_orig = ss_mod._orig_state_names.index("Y")
    c_orig = ss_mod._orig_state_names.index("C")

    np.testing.assert_allclose(Z[0, y_orig], 1.0)
    np.testing.assert_allclose(Z[0, ss_mod._obs_lag_column("Y", -1)], -1.0)
    np.testing.assert_allclose(Z[1, c_orig], 1.0)
    np.testing.assert_allclose(Z[1, ss_mod._obs_lag_column("C", -2)], -1.0)
    np.testing.assert_allclose(Z[1, ss_mod._obs_lag_column("C", -1)], 0.0)

    # Y's lag slots do not appear in C's row, and C's do not appear in Y's.
    np.testing.assert_allclose(Z[0, ss_mod._obs_lag_column("C", -1)], 0.0)
    np.testing.assert_allclose(Z[0, ss_mod._obs_lag_column("C", -2)], 0.0)
    np.testing.assert_allclose(Z[1, ss_mod._obs_lag_column("Y", -1)], 0.0)


def test_obs_equation_alongside_pure_selector_observation():
    """Mixing an obs-equation series with a pure-selector observation gives independent rows."""
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["L", "Y_diff"],
        measurement_error=["L", "Y_diff"],
        constant_params="auto",
        observation_equations={"Y_diff": "log(Y[]) - log(Y[-1])"},
        verbose=False,
    )

    d, Z = _eval_obs_pieces(ss_mod, _params_at_calib())

    l_idx = ss_mod._orig_state_names.index("L")
    y_idx = ss_mod._orig_state_names.index("Y")
    y_lag1 = ss_mod._obs_lag_column("Y", -1)

    # Selector row for L: identity selector, zero intercept.
    np.testing.assert_allclose(Z[0, l_idx], 1.0)
    np.testing.assert_allclose(Z[0].sum(), 1.0)  # only L contributes
    np.testing.assert_allclose(d[0], 0.0, atol=1e-12)

    # Obs-equation row for Y_diff: +1 on Y, -1 on Y_lag1.
    np.testing.assert_allclose(Z[1, y_idx], 1.0)
    np.testing.assert_allclose(Z[1, y_lag1], -1.0)
    np.testing.assert_allclose(d[1], 0.0, atol=1e-12)


def test_observation_equations_lag_states_have_zero_selection_rows():
    """Obs-eq lag states are deterministic shifts, so their rows of R are all zero."""
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["Y_diff"],
        measurement_error=["Y_diff"],
        constant_params="auto",
        observation_equations={"Y_diff": "log(Y[]) - log(Y[-1])"},
        verbose=False,
    )
    R_aug = _evaluate_at(ss_mod, ss_mod.ssm["selection"], _params_at_calib())
    y_lag1 = ss_mod._obs_lag_starts["Y"]
    np.testing.assert_allclose(R_aug[y_lag1], 0.0)


def test_observation_equations_lag_chain_propagates_correctly():
    """
    After simulating the augmented dynamics, the lag-k slot holds the parent variable from k steps ago.

    A static inspection of the rows of T would miss an off-by-one error in the ``F_lag`` and ``C_lag`` blocks.
    """
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["dlog_Y"],
        measurement_error=["dlog_Y"],
        constant_params="auto",
        observation_equations={"dlog_Y": "log(Y[]) - log(Y[-3])"},
        verbose=False,
    )
    T_aug = _evaluate_at(ss_mod, ss_mod.ssm["transition"], _params_at_calib())

    rng = np.random.default_rng(0)
    x = np.zeros(T_aug.shape[0])
    x[: ss_mod._k_orig_states] = rng.normal(0, 0.1, size=ss_mod._k_orig_states)

    y_orig = ss_mod._orig_state_names.index("Y")
    history_Y = [x[y_orig]]
    for _ in range(4):
        x = T_aug @ x
        history_Y.append(x[y_orig])

    for k in range(1, 4):
        np.testing.assert_allclose(x[ss_mod._obs_lag_column("Y", -k)], history_Y[-1 - k], rtol=1e-10)


def test_observation_equation_simplifies_to_zero_produces_zero_row():
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["dummy"],
        measurement_error=["dummy"],
        constant_params="auto",
        observation_equations={"dummy": "log(Y[]) - log(Y[])"},
        verbose=False,
    )
    assert ss_mod._obs_lag_depths == {}

    d, Z = _eval_obs_pieces(ss_mod, _params_at_calib())
    np.testing.assert_allclose(d[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(Z[0], 0.0, atol=1e-12)


def test_observation_equations_carry_model_variable_assumptions():
    """
    Observation equations on a model whose variables carry ``positive`` assumptions linearize correctly.

    Sympy equality includes assumptions, so a parsed ``v[]`` without the model's assumptions would not match
    ``self.variables[i]`` and the linearization's ``xreplace`` would leave the raw time-t symbol in place. Evaluating
    the intercept then fails with ``MissingInputError``.
    """
    ss_mod = statespace_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y_obs"],
        measurement_error=["Y_obs"],
        constant_params="auto",
        observation_equations={"Y_obs": "log(Y[])"},
        verbose=False,
    )
    model = load_and_cache_model("open_rbc.gcn")
    d = _evaluate_at(ss_mod, ss_mod.ssm["obs_intercept"], model.parameters())
    Y_ss = float(model.steady_state(verbose=False, progressbar=False)["Y_ss"])
    np.testing.assert_allclose(d[0], np.log(Y_ss), rtol=1e-10)


def test_constant_params_baked_into_ss_obs_intercept():
    """
    Constant parameters are baked into an intercept that comes from the ``ss_obs_intercept`` branch.

    That branch reads ``steady_state_mapping`` directly, which is in the free parameter placeholders, so it needs
    its own constant substitution.
    """
    ss_mod = statespace_from_gcn(TEST_GCNS / "full_nk.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y", "C", "L"],
        measurement_error=["Y", "C", "L"],
        constant_params=[p for p in ss_mod.param_dict if p not in ss_mod.param_priors],
        ss_obs_intercept=["L"],
        verbose=False,
    )

    d = ss_mod.ssm["obs_intercept"]
    free_leaves = {a.name for a in ancestors([d]) if a.owner is None and a.name and not isinstance(a, TensorConstant)}
    leaked = free_leaves & set(ss_mod.constant_parameters)
    assert not leaked, f"Constant params leaked as free inputs of obs_intercept: {sorted(leaked)}"


def test_constant_params_baked_into_observation_equations():
    """
    A ``constant_params`` parameter inside an observation equation is baked in and not left a free input.

    A free input would have no PyMC variable to bind to when the Kalman likelihood is compiled. The battery's
    ``intercept_with_parameter`` case covers the baked value.
    """
    ss_mod = statespace_from_gcn(TEST_GCNS / NL_GCN, verbose=False)
    ss_mod.configure(
        observed_states=["Y"],
        measurement_error=["Y"],
        constant_params=["alpha"],
        observation_equations={"Y": "log(Y[]) + alpha"},
        verbose=False,
    )

    d = ss_mod.ssm["obs_intercept"]
    free_leaves = {a.name for a in ancestors([d]) if a.owner is None and a.name and not isinstance(a, TensorConstant)}
    assert "alpha" not in free_leaves


@pytest.fixture(scope="module")
def autocorrelation_setup():
    """Build a mixed-frequency RBC state space and a small fake posterior over its parameters."""
    ss_mod = statespace_from_gcn(TEST_GCNS / "rbc_linearized.gcn", verbose=False)
    ss_mod.configure(
        observed_states=["Y", "K"],
        measurement_error=["Y", "K"],
        temporal_aggregation={"Y": "sum"},
        solver="gensys",
        verbose=False,
    )
    with pm.Model():
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
        pm.Gamma("error_sigma_Y", alpha=2, beta=100)
        pm.Gamma("error_sigma_K", alpha=2, beta=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ss_mod.build_statespace_graph(np.full((40, 2), np.nan), add_norm_check=False)

    calib = load_and_cache_model("rbc_linearized.gcn").parameters()
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            name: (
                ("chain", "draw"),
                (float(calib[name]) if name in calib else 0.01) + 1e-4 * rng.standard_normal((2, 8)),
            )
            for name in ss_mod.param_names
        },
        coords={"chain": np.arange(2), "draw": np.arange(8)},
    )
    return ss_mod, posterior


def test_sample_autocorrelation_matrices_shape_and_normalization(autocorrelation_setup):
    ss_mod, posterior = autocorrelation_setup

    latent = ss_mod.sample_autocorrelation_matrices(posterior, n_lags=6)
    assert latent.dims == ("chain", "draw", "lag", "state", "state_aux")
    assert latent.sizes["lag"] == 7
    assert float(np.abs(latent).max()) <= 1.0 + 1e-6

    diag0 = latent.isel(lag=0).mean(["chain", "draw"]).values
    np.testing.assert_allclose(np.diag(diag0), 1.0, atol=1e-6)

    observed = ss_mod.sample_autocorrelation_matrices(posterior, n_lags=6, observed=True, lag_step=4)
    assert observed.dims == ("chain", "draw", "lag", "observed_state", "observed_state_aux")
    obs_diag0 = observed.isel(lag=0).mean(["chain", "draw"]).values
    np.testing.assert_allclose(np.diag(obs_diag0), 1.0, atol=1e-6)


def test_sample_autocorrelation_lag_step_subsamples_lags(autocorrelation_setup):
    ss_mod, posterior = autocorrelation_setup

    # Both lag k at lag_step=2 and lag 2k at lag_step=1 are T^(2k) @ Sigma.
    step_1 = ss_mod.sample_autocorrelation_matrices(posterior, n_lags=8).mean(["chain", "draw"])
    step_2 = ss_mod.sample_autocorrelation_matrices(posterior, n_lags=4, lag_step=2).mean(["chain", "draw"])
    np.testing.assert_allclose(step_2.values, step_1.isel(lag=slice(None, None, 2)).values, atol=1e-5)


def test_sample_autocorrelation_matches_analytical_acf(autocorrelation_setup):
    """The sampled latent ACF reproduces the analytical ``autocorrelation_matrix`` at a single calibrated draw."""
    ss_mod, _ = autocorrelation_setup
    model = load_and_cache_model("rbc_linearized.gcn")
    calib = model.parameters()

    # A degenerate posterior (one draw at the calibrated values) makes the comparison exact.
    degenerate = xr.Dataset(
        {
            name: (("chain", "draw"), np.full((1, 1), float(calib[name]) if name in calib else 0.01))
            for name in ss_mod.param_names
        },
        coords={"chain": [0], "draw": [0]},
    )

    analytical = autocorrelation_matrix(model, shock_cov_matrix=np.eye(1) * 0.01, return_xr=True, verbose=False)
    n_lags = analytical.sizes["lag"]
    sampled = ss_mod.sample_autocorrelation_matrices(degenerate, n_lags=n_lags).isel(chain=0, draw=0)

    for variable in analytical.coords["variable"].values:
        np.testing.assert_allclose(
            sampled.sel(state=variable, state_aux=variable).isel(lag=slice(0, n_lags)).values,
            analytical.sel(variable=variable, variable_aux=variable).values,
            atol=1e-4,
        )
