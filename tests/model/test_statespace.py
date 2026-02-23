import numpy as np
import pandas as pd
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
        "sarima2_12.gcn",
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
