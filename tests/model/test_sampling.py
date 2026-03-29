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
    """Unbounded distributions (HalfNormal, InverseGamma) must still produce finite bounds."""
    bounds = bounds_from_priors(PRIORS)
    for name, (lo, hi) in bounds.items():
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi, (
            f"{name}: expected finite ordered bounds, got [{lo}, {hi}]"
        )


@pytest.mark.parametrize("method", ["random", "lhs", "sobol", "halton", "poisson_disk"])
def test_sample_uniform_all_values_within_bounds(method):
    # sobol requires power-of-2 n_samples
    n = 32 if method == "sobol" else 50
    df = sample_uniform(BOUNDS, n_samples=n, method=method)
    assert df.shape == (n, len(BOUNDS))
    for name, (lo, hi) in BOUNDS.items():
        assert (df[name] >= lo).all() and (df[name] <= hi).all(), f"{name} out of bounds with {method}"


def test_sample_uniform_sobol_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        sample_uniform(BOUNDS, n_samples=10, method="sobol")


def test_sample_uniform_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown sampling method"):
        sample_uniform(BOUNDS, n_samples=10, method="not_a_method")


def test_sample_from_priors_returns_correct_shape_and_columns():
    df = sample_from_priors(PRIORS, n_samples=25)
    assert df.shape == (25, len(PRIORS))
    assert set(df.columns) == set(PRIORS)


def test_sample_uniform_from_priors_respects_hdi_bounds():
    hdi_prob = 0.99
    bounds = bounds_from_priors(PRIORS, hdi_prob=hdi_prob)
    df = sample_uniform_from_priors(PRIORS, n_samples=64, method="sobol", hdi_prob=hdi_prob)
    for name, (lo, hi) in bounds.items():
        assert (df[name] >= lo).all() and (df[name] <= hi).all()


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
def test_sample_from_priors_qmc_produces_finite_values(method):
    """Ppf clipping at [eps, 1-eps] must prevent ±inf from unbounded priors."""
    df = sample_from_priors_qmc(PRIORS, n_samples=16, method=method)
    assert df.shape == (16, len(PRIORS))
    assert np.isfinite(df.values).all()
