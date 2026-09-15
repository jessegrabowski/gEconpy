import numpy as np
import pandas as pd
import pytest

from gEconpy.exceptions import SteadyStateNotFoundError
from gEconpy.model.statistics import check_steady_state, prior_solvability_check, solvability_check
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.fixture(scope="module")
def model_with_priors():
    return load_and_cache_model("one_block_1_dist.gcn")


@pytest.fixture(scope="module")
def model_without_priors():
    return load_and_cache_model("one_block_1.gcn")


def _make_samples(model, n=5, **overrides):
    row = {k: float(v) for k, v in model._default_params.items()}
    row.update(overrides)
    return pd.DataFrame([row] * n)


def test_default_params_all_succeed_with_finite_norms(model_with_priors):
    result = solvability_check(model_with_priors, _make_samples(model_with_priors), progressbar=False)
    assert result["failure_step"].isna().all(), result["failure_step"].value_counts().to_dict()
    assert np.isfinite(result["norm_deterministic"]).all()
    assert np.isfinite(result["norm_stochastic"]).all()


def test_bad_params_labelled_as_steady_state_failure(model_with_priors):
    result = solvability_check(model_with_priors, _make_samples(model_with_priors, alpha=0.9999), progressbar=False)
    assert (result["failure_step"] == "steady_state").all()
    assert result["norm_deterministic"].isna().all()


def test_good_and_bad_rows_classified_independently(model_with_priors):
    good = _make_samples(model_with_priors)
    bad = _make_samples(model_with_priors, alpha=0.9999)
    samples = pd.concat([good, bad], ignore_index=True)

    result = solvability_check(model_with_priors, samples, progressbar=False)
    assert result["failure_step"].iloc[:5].isna().all()
    assert (result["failure_step"].iloc[5:] == "steady_state").all()


def test_partial_dataframe_fills_missing_params_from_model(model_with_priors):
    result = solvability_check(model_with_priors, pd.DataFrame({"alpha": [0.3, 0.4, 0.35]}), progressbar=False)
    assert result["failure_step"].isna().all()


@pytest.mark.parametrize("solver", ["cycle_reduction", "gensys"])
def test_both_solvers_succeed_at_defaults(model_with_priors, solver):
    result = solvability_check(model_with_priors, _make_samples(model_with_priors), solver=solver, progressbar=False)
    assert result["failure_step"].isna().all(), f"solver={solver!r}: {result['failure_step'].value_counts().to_dict()}"


def test_unknown_solver_raises_before_checking_any_draw(model_with_priors):
    with pytest.raises(ValueError, match="Unknown solver 'newton'"):
        solvability_check(model_with_priors, _make_samples(model_with_priors), solver="newton", progressbar=False)


def test_parallel_produces_same_outcomes_as_serial(model_with_priors):
    samples = _make_samples(model_with_priors)
    serial = solvability_check(model_with_priors, samples, cores=1, progressbar=False)
    parallel = solvability_check(model_with_priors, samples, cores=2, progressbar=False)

    assert serial["failure_step"].isna().all()
    assert parallel["failure_step"].isna().all()
    np.testing.assert_allclose(
        sorted(serial["norm_deterministic"]),
        sorted(parallel["norm_deterministic"]),
    )


def test_prior_solvability_check_returns_sampled_param_columns(model_with_priors):
    result = prior_solvability_check(model_with_priors, n_samples=5, progressbar=False)
    assert {"failure_step", "norm_deterministic", "norm_stochastic"}.issubset(result.columns)
    assert len(result) == 5
    assert all(name in result.columns for name in model_with_priors.param_priors)


def test_prior_solvability_check_param_subset_filters_columns(model_with_priors):
    result = prior_solvability_check(model_with_priors, n_samples=5, param_subset=["alpha", "rho"], progressbar=False)
    assert "alpha" in result.columns and "rho" in result.columns
    assert "gamma" not in result.columns


def test_prior_solvability_check_raises_without_priors(model_without_priors):
    with pytest.raises(ValueError, match="no param_priors"):
        prior_solvability_check(model_without_priors, n_samples=5, progressbar=False)


def test_prior_solvability_check_raises_on_unknown_param_subset(model_with_priors):
    with pytest.raises(ValueError, match="param_subset"):
        prior_solvability_check(model_with_priors, n_samples=5, param_subset=["not_a_param"])


def test_check_steady_state_logs_success_for_valid_steady_state(model_without_priors, caplog):
    steady_state = model_without_priors.steady_state(verbose=False, progressbar=False)

    with caplog.at_level("WARNING"):
        check_steady_state(model_without_priors, steady_state=steady_state)

    assert "successfully found" in caplog.text


def test_check_steady_state_solves_at_updated_parameters(model_without_priors, caplog):
    with caplog.at_level("WARNING"):
        check_steady_state(
            model_without_priors, steady_state_kwargs={"verbose": False, "progressbar": False}, beta=0.97, delta=0.05
        )

    assert caplog.messages == ["Steady state successfully found!"]


def test_check_steady_state_raises_on_wrong_steady_state(model_without_priors):
    steady_state = model_without_priors.steady_state(verbose=False, progressbar=False)
    first_variable = next(iter(steady_state))
    steady_state[first_variable] = steady_state[first_variable] + 1.0

    with pytest.raises(SteadyStateNotFoundError):
        check_steady_state(model_without_priors, steady_state=steady_state)


def test_check_steady_state_reports_violated_calibrating_equation(monkeypatch, caplog):
    model = load_and_cache_model("one_block_2_no_extra.gcn")
    steady_state = model.steady_state(verbose=False, progressbar=False)
    steady_state["L_ss"] = steady_state["L_ss"] * 2
    steady_state.success = False
    monkeypatch.setattr(model, "steady_state", lambda **_kwargs: steady_state)

    with caplog.at_level("WARNING"):
        check_steady_state(model)

    calibrating_equation = next(iter(model._calib_dict.to_sympy().values()))
    assert str(calibrating_equation) in caplog.text
