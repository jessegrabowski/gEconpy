import numpy as np
import pymc as pm
import pytensor
import pytest

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
    ],
)
def test_statespace_matrices_agree_with_model(gcn_file):
    ss_mod = load_and_cache_statespace(gcn_file)
    model = load_and_cache_model(gcn_file, "numpy", use_jax=False)

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


class TestConfigureStatespace:
    @pytest.mark.parametrize(
        "gcn_file",
        [
            "one_block_1_ss.gcn",
            "open_rbc.gcn",
            pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
            "rbc_linearized.gcn",
        ],
    )
    def test_minimal_valid_config(self, gcn_file):
        ss_mod = load_and_cache_statespace(gcn_file)
        obs = [ss_mod.state_names[0]]
        ss_mod.configure(observed_states=obs, verbose=False)

        assert ss_mod._configured is True
        assert ss_mod.observed_states == obs
        assert ss_mod.k_endog == len(obs)
        assert ss_mod.measurement_error is False
        assert ss_mod.full_covariance is False
        assert ss_mod._solver == "gensys"
        assert ss_mod.data_names == []

    def test_invalid_observed_state_raises(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        with pytest.raises(ValueError):
            ss_mod.configure(observed_states=["not_a_state"], verbose=False)

    def test_measurement_error_requires_observed(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        obs = [ss_mod.state_names[0]]
        meas = [ss_mod.state_names[1]]
        with pytest.raises(ValueError):
            ss_mod.configure(observed_states=obs, measurement_error=meas, verbose=False)

    def test_measurement_error_config(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        obs = ss_mod.state_names[:2]
        meas = [obs[0]]
        ss_mod.configure(observed_states=obs, measurement_error=meas, verbose=False)
        assert ss_mod.measurement_error is True
        assert ss_mod.error_states == meas

    def test_constant_params_validation_and_filtering(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        # Unknown constant param should raise
        with pytest.raises(ValueError):
            ss_mod.configure(
                observed_states=[ss_mod.state_names[0]], constant_params=["definitely_not_a_param"], verbose=False
            )

        # Known constant param should be removed from param_names
        input_param_names = [x.name for x in ss_mod.input_parameters]
        keep_out = input_param_names[0]
        pre = set(ss_mod.param_names)

        assert keep_out in pre
        ss_mod.configure(observed_states=[ss_mod.state_names[0]], constant_params=[keep_out], verbose=False)
        assert keep_out not in set(ss_mod.param_names)

    def test_full_shock_covariance_flag(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        ss_mod.configure(observed_states=[ss_mod.state_names[0]], full_shock_covaraince=True, verbose=False)
        names = ss_mod.param_names
        assert "state_cov" in names
        assert not any(f"sigma_{shock.base_name}" in names for shock in ss_mod.shocks)

    def test_solver_and_kwargs(self):
        # gensys
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        ss_mod.configure(observed_states=[ss_mod.state_names[0]], solver="gensys", tol=1e-5)
        assert ss_mod._solver == "gensys"
        assert ss_mod._solver_kwargs == {"tol": 1e-5}

        # cycle_reduction
        ss_mod = load_and_cache_statespace("open_rbc.gcn")
        ss_mod.configure(
            observed_states=[ss_mod.state_names[0]], solver="cycle_reduction", tol=1e-4, max_iter=123, verbose=False
        )
        assert ss_mod._solver == "cycle_reduction"
        assert ss_mod._solver_kwargs == {"tol": 1e-4, "max_iter": 123}

        # scan_cycle_reduction
        ss_mod = load_and_cache_statespace("rbc_linearized.gcn")
        ss_mod.configure(
            observed_states=[ss_mod.state_names[0]],
            solver="scan_cycle_reduction",
            tol=1e-3,
            max_iter=77,
            use_adjoint_gradients=False,
            verbose=False,
        )
        assert ss_mod._solver == "scan_cycle_reduction"
        assert ss_mod._solver_kwargs == {"tol": 1e-3, "max_iter": 77, "use_adjoint_gradients": False}

        # invalid solver
        ss_mod = load_and_cache_statespace("rbc_linearized.gcn")
        with pytest.raises(ValueError):
            ss_mod.configure(observed_states=[ss_mod.state_names[0]], solver="not_a_solver", verbose=False)

    def test_exog_list_with_k_exog_autonames(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        ss_mod.configure(observed_states=[ss_mod.state_names[0]], k_exog=2, verbose=False)
        assert ss_mod.k_exog == 2
        assert ss_mod.exog_state_names == ["exogenous_0", "exogenous_1"]
        assert ss_mod.data_names == ["exogenous_data"]

    def test_exog_list_infers_k_exog(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        names = ["x1", "x2", "x3"]
        ss_mod.configure(observed_states=[ss_mod.state_names[0]], exog_state_names=names, verbose=False)
        assert ss_mod.k_exog == len(names)
        assert ss_mod.exog_state_names == names
        assert ss_mod.data_names == ["exogenous_data"]

    def test_exog_dict_infers_k_exog_and_data_names(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        s0, s1 = ss_mod.state_names[:2]
        exog = {s0: ["z1"], s1: ["z1", "z2"]}
        ss_mod.configure(observed_states=[s0, s1], exog_state_names=exog, measurement_error=[s0, s1], verbose=False)
        assert isinstance(ss_mod.k_exog, dict)
        assert ss_mod.k_exog == {s0: 1, s1: 2}
        assert set(ss_mod.data_names) == {f"{s0}_exogenous_data", f"{s1}_exogenous_data"}

    def test_exog_mismatch_raises(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        with pytest.raises(ValueError):
            ss_mod.configure(
                observed_states=[ss_mod.state_names[0]], k_exog=1, exog_state_names=["x1", "x2"], verbose=False
            )

    def test_exog_dict_collapses_to_list_when_uniform(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        xs = ["u1", "u2"]
        exog = dict.fromkeys(ss_mod.state_names[:3], xs)
        ss_mod.configure(
            observed_states=ss_mod.state_names[:3],
            exog_state_names=exog,
            measurement_error=ss_mod.state_names[:3],
            verbose=False,
        )
        assert isinstance(ss_mod.exog_state_names, list)
        assert ss_mod.exog_state_names == xs
        assert ss_mod.k_exog == len(xs)
        assert ss_mod.data_names == ["exogenous_data"]

    def test_identification_error_when_too_many_observed(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        n_shocks = len(ss_mod.shocks)
        if len(ss_mod.state_names) <= n_shocks:
            pytest.skip("Not enough states to exceed shocks")
        obs = ss_mod.state_names[: n_shocks + 1]
        with pytest.raises(ValueError, match="Stochastic singularity"):
            ss_mod.configure(observed_states=obs, verbose=False)

    def test_mode_is_propagated(self):
        ss_mod = load_and_cache_statespace("one_block_1_ss.gcn")
        ss_mod.configure(observed_states=[ss_mod.state_names[0]], mode="JAX", verbose=False)
        assert ss_mod._mode == "JAX"
