import numpy as np
import pandas as pd

from preliz.distributions.distributions import Distribution
from scipy.stats import qmc


def bounds_from_priors(
    priors: dict[str, Distribution],
    hdi_prob: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Derive finite parameter bounds from preliz priors via HDI intervals.

    Calls ``dist.hdi(hdi_prob)`` on each prior. This always returns finite
    bounds, even for unbounded distributions (Normal, HalfNormal,
    InverseGamma, etc.), making it safe to feed directly into QMC samplers
    that require explicit bounds.

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping of parameter name to preliz distribution.
    hdi_prob : float, default 0.99
        Probability mass covered by the HDI. Defaults to 0.99, which excludes
        extreme tails while keeping a generous range.

    Returns
    -------
    bounds : dict of str to tuple of float
        ``{name: (lower, upper)}`` for each prior.
    """
    results = {}

    for name, dist in priors.items():
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                low, upper = dist.hdi(hdi_prob)
        except ValueError:
            low = dist.ppf(1 - hdi_prob)
            upper = dist.ppf(hdi_prob)
        results[name] = (low, upper)

    return results


def sample_from_priors(
    priors: dict[str, Distribution],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw ``n_samples`` from a dict of preliz distributions using ``.rvs()``.

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping of parameter name to preliz distribution.
    n_samples : int
        Number of draws.
    seed : int, Generator, or None
        Random seed for reproducibility.

    Returns
    -------
    samples : pd.DataFrame
        Shape ``(n_samples, n_params)``. Column names match ``priors`` keys.
    """
    rng = np.random.default_rng(seed)
    data = {name: dist.rvs(n_samples, random_state=rng) for name, dist in priors.items()}
    return pd.DataFrame(data)


def sample_uniform(
    param_bounds: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
    method: str = "lhs",
) -> pd.DataFrame:
    """Generate parameter samples within explicit bounds using QMC or random sampling.

    Parameters
    ----------
    param_bounds : dict of str to tuple of float
        ``{name: (lower, upper)}`` for each parameter.
    n_samples : int
        Number of samples. Must be a power of 2 when ``method="sobol"``.
    seed : int, Generator, or None
        Random seed.
    method : str, default ``"lhs"``
        Sampling method. One of:

        - ``"random"`` — independent uniform draws.
        - ``"lhs"`` — Latin Hypercube Sampling.
        - ``"sobol"`` — Sobol sequence (scrambled). Requires ``n_samples`` to
          be a power of 2.
        - ``"halton"`` — Halton sequence.
        - ``"poisson_disk"`` — Poisson disk sampling (best spatial uniformity
          but slower).

    Returns
    -------
    samples : pd.DataFrame
        Shape ``(n_samples, n_params)``. Column names match ``param_bounds`` keys.

    Raises
    ------
    ValueError
        If ``method="sobol"`` and ``n_samples`` is not a power of 2.
    ValueError
        If ``method`` is not recognized.
    """
    names = list(param_bounds.keys())
    l_bounds = np.array([param_bounds[n][0] for n in names], dtype=float)
    u_bounds = np.array([param_bounds[n][1] for n in names], dtype=float)
    d = len(names)

    if method == "sobol" and (n_samples & (n_samples - 1)) != 0:
        raise ValueError(
            f"Sobol sequences require n_samples to be a power of 2, got {n_samples}. "
            f"Try n_samples={2 ** int(np.ceil(np.log2(n_samples)))}."
        )

    rng_seed = seed if isinstance(seed, (int, type(None))) else None

    if method == "random":
        rng = np.random.default_rng(seed)
        unit_samples = rng.uniform(size=(n_samples, d))
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=rng_seed)
        unit_samples = sampler.random(n_samples)
    elif method == "sobol":
        sampler = qmc.Sobol(d=d, scramble=True, seed=rng_seed)
        unit_samples = sampler.random(n_samples)
    elif method == "halton":
        sampler = qmc.Halton(d=d, scramble=True, seed=rng_seed)
        unit_samples = sampler.random(n_samples)
    elif method == "poisson_disk":
        sampler = qmc.PoissonDisk(d=d, seed=rng_seed)
        unit_samples = sampler.random(n_samples)
    else:
        raise ValueError(
            f"Unknown sampling method {method!r}. Choose from 'random', 'lhs', 'sobol', 'halton', 'poisson_disk'."
        )

    scaled = qmc.scale(unit_samples, l_bounds, u_bounds)
    return pd.DataFrame(scaled, columns=names)


def sample_uniform_from_priors(
    priors: dict[str, Distribution],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
    method: str = "lhs",
    hdi_prob: float = 0.99,
) -> pd.DataFrame:
    """QMC samples over prior HDI bounds — space-filling *and* prior-informed.

    Computes ``bounds_from_priors(priors, hdi_prob)``, then delegates to
    ``sample_uniform``.

    This is the recommended default for solvability checks. The goal is spatial
    coverage of the plausible parameter region, not density-weighted sampling.
    QMC on a uniform hypercube gives better coverage than Monte Carlo draws from
    the prior (which cluster near the mode).

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping of parameter name to preliz distribution.
    n_samples : int
        Number of samples. Must be a power of 2 when ``method="sobol"``.
    seed : int, Generator, or None
        Random seed.
    method : str, default ``"lhs"``
        Passed to :func:`sample_uniform`. See its docs for valid values.
    hdi_prob : float, default 0.99
        HDI probability used to compute bounds. See :func:`bounds_from_priors`.

    Returns
    -------
    samples : pd.DataFrame
        Shape ``(n_samples, n_params)``. Column names match ``priors`` keys.
    """
    bounds = bounds_from_priors(priors, hdi_prob=hdi_prob)
    return sample_uniform(bounds, n_samples, seed=seed, method=method)


def sample_from_priors_qmc(
    priors: dict[str, Distribution],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
    method: str = "sobol",
) -> pd.DataFrame:
    """Quasi-random samples that respect prior shapes via inverse-CDF (``ppf``).

    Generates unit-hypercube draws via the chosen QMC engine, then applies
    ``prior.ppf(u)`` column-by-column. This preserves the prior density shape
    but gives poorer spatial coverage than :func:`sample_uniform_from_priors`
    for the same ``n_samples``, because draws are still clustered near the
    prior mode.

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping of parameter name to preliz distribution.
    n_samples : int
        Number of samples. Must be a power of 2 when ``method="sobol"``.
    seed : int or None
        Random seed. Note: ``np.random.Generator`` objects are not accepted here
        because scipy QMC engines require integer seeds.
    method : str, default ``"sobol"``
        QMC engine. One of ``"sobol"``, ``"halton"``, ``"lhs"``.

    Returns
    -------
    samples : pd.DataFrame
        Shape ``(n_samples, n_params)``. Column names match ``priors`` keys.

    Raises
    ------
    ValueError
        If ``method="sobol"`` and ``n_samples`` is not a power of 2.
    """
    names = list(priors.keys())
    d = len(names)

    if method == "sobol" and (n_samples & (n_samples - 1)) != 0:
        raise ValueError(
            f"Sobol sequences require n_samples to be a power of 2, got {n_samples}. "
            f"Try n_samples={2 ** int(np.ceil(np.log2(n_samples)))}."
        )

    rng_seed = seed if isinstance(seed, (int, type(None))) else None

    if method == "sobol":
        sampler = qmc.Sobol(d=d, scramble=True, seed=rng_seed)
    elif method == "halton":
        sampler = qmc.Halton(d=d, scramble=True, seed=rng_seed)
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=rng_seed)
    else:
        raise ValueError(f"Unknown method {method!r} for sample_from_priors_qmc. Choose from 'sobol', 'halton', 'lhs'.")

    unit_samples = sampler.random(n_samples)

    # Apply inverse-CDF column by column, clipping to avoid ppf(0) or ppf(1) = ±inf
    eps = np.finfo(float).eps
    unit_samples = np.clip(unit_samples, eps, 1 - eps)

    data = {}
    for i, name in enumerate(names):
        data[name] = priors[name].ppf(unit_samples[:, i])

    return pd.DataFrame(data)
