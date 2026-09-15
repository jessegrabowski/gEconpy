import numpy as np
import preliz as pz
import pytest

from gEconpy.model.sampling import (
    bounds_from_priors,
    sample_from_priors,
    sample_from_priors_qmc,
    sample_uniform,
    sample_uniform_from_priors,
)

PRIORS = {
    "alpha": pz.Beta(mu=0.5, sigma=0.1),
    "rho": pz.Beta(mu=0.95, sigma=0.04),
    "gamma": pz.HalfNormal(sigma=1.0),
    "sigma": pz.InverseGamma(mu=0.1, sigma=0.01),
}

BOUNDS: dict[str, tuple[float, float]] = {
    "alpha": (0.2, 0.8),
    "rho": (0.5, 0.99),
    "gamma": (0.1, 3.0),
}


def test_bounds_from_priors_always_finite_and_ordered():
    """Unbounded distributions (HalfNormal, InverseGamma) still produce finite bounds."""
    bounds = bounds_from_priors(PRIORS)
    for name, (lo, hi) in bounds.items():
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi, (
            f"{name}: expected finite ordered bounds, got [{lo}, {hi}]"
        )


@pytest.mark.parametrize(
    "method, n",
    [("random", 50), ("lhs", 50), ("sobol", 32), ("halton", 50), ("poisson_disk", 50)],
    ids=["random", "lhs", "sobol", "halton", "poisson_disk"],
)
def test_sample_uniform_all_values_within_bounds(method, n):
    df = sample_uniform(BOUNDS, n_samples=n, method=method)
    assert df.shape == (n, len(BOUNDS))
    for name, (lo, hi) in BOUNDS.items():
        assert (df[name] >= lo).all() and (df[name] <= hi).all(), f"{name} out of bounds with {method}"


def test_sample_uniform_sobol_rejects_non_power_of_two_and_suggests_next():
    with pytest.raises(ValueError, match=r"power of 2, got 10\. Try n_samples=16"):
        sample_uniform(BOUNDS, n_samples=10, method="sobol")


def test_sample_uniform_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown sampling method"):
        sample_uniform(BOUNDS, n_samples=10, method="not_a_method")


def test_sample_from_priors_qmc_rejects_engine_without_inverse_cdf():
    with pytest.raises(ValueError, match="Unknown method 'poisson_disk' for sample_from_priors_qmc"):
        sample_from_priors_qmc(PRIORS, n_samples=16, method="poisson_disk")


def test_sample_from_priors_returns_correct_shape_and_columns():
    df = sample_from_priors(PRIORS, n_samples=25, seed=0)
    assert df.shape == (25, len(PRIORS))
    assert list(df.columns) == list(PRIORS)


@pytest.mark.parametrize("seed", [3, np.random.default_rng(3)], ids=["int", "generator"])
def test_sample_from_priors_is_reproducible_under_seed(seed):
    first = sample_from_priors(PRIORS, n_samples=10, seed=seed)
    second = sample_from_priors(PRIORS, n_samples=10, seed=np.random.default_rng(3))
    np.testing.assert_allclose(first.values, second.values)


def test_sample_uniform_from_priors_respects_hdi_bounds():
    hdi_prob = 0.99
    bounds = bounds_from_priors(PRIORS, hdi_prob=hdi_prob)
    df = sample_uniform_from_priors(PRIORS, n_samples=64, method="sobol", hdi_prob=hdi_prob)
    for name, (lo, hi) in bounds.items():
        assert (df[name] >= lo).all() and (df[name] <= hi).all()


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
def test_sample_from_priors_qmc_produces_finite_values(method):
    """Clipping the unit draws to [eps, 1 - eps] keeps the ppf of unbounded priors finite."""
    df = sample_from_priors_qmc(PRIORS, n_samples=16, method=method, seed=0)
    assert df.shape == (16, len(PRIORS))
    assert np.isfinite(df.values).all()


def test_sample_from_priors_qmc_reproduces_prior_moments():
    """Passing a well-spread unit sample through each ppf gives draws with the prior's mean and variance."""
    df = sample_from_priors_qmc(PRIORS, n_samples=1024, method="sobol", seed=0)

    expected_mean = np.array([float(dist.mean()) for dist in PRIORS.values()])
    expected_var = np.array([float(dist.var()) for dist in PRIORS.values()])
    np.testing.assert_allclose(df.mean().values, expected_mean, rtol=1e-3)
    np.testing.assert_allclose(df.var().values, expected_var, rtol=1e-2)


@pytest.mark.parametrize("method", ["random", "sobol", "halton", "lhs", "poisson_disk"])
def test_sample_uniform_honors_generator_seed(method):
    first = sample_uniform(BOUNDS, n_samples=8, seed=np.random.default_rng(0), method=method)
    second = sample_uniform(BOUNDS, n_samples=8, seed=np.random.default_rng(0), method=method)
    np.testing.assert_allclose(first.values, second.values)
