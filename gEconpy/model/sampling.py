import numpy as np
import pandas as pd

from preliz.distributions.distributions import Distribution
from scipy.stats import qmc

QMC_ENGINES = {
    "lhs": qmc.LatinHypercube,
    "sobol": qmc.Sobol,
    "halton": qmc.Halton,
    "poisson_disk": qmc.PoissonDisk,
}
INVERSE_CDF_ENGINES = ("sobol", "halton", "lhs")


def bounds_from_priors(
    priors: dict[str, Distribution],
    hdi_prob: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """
    Derive finite parameter bounds from the highest density interval of each prior.

    Unbounded distributions such as Normal, HalfNormal, and InverseGamma still yield finite bounds, so the result
    can feed a quasi-Monte Carlo sampler directly.

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping from parameter name to preliz distribution.
    hdi_prob : float, optional
        Probability mass the interval covers. Defaults to 0.99.

    Returns
    -------
    bounds : dict of str to tuple of float
        ``(lower, upper)`` interval for each prior, keyed by parameter name.

    Examples
    --------
    The HalfNormal prior is unbounded above, and its 95 percent interval is still finite:

    .. code-block:: python

        import preliz as pz

        from gEconpy import bounds_from_priors

        priors = {"alpha": pz.Beta(mu=0.35, sigma=0.05), "sigma": pz.HalfNormal(sigma=1.0)}
        bounds = bounds_from_priors(priors, hdi_prob=0.95)
        print(bounds["alpha"], bounds["sigma"])
    """
    bounds = {}

    for name, dist in priors.items():
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                lower, upper = dist.hdi(hdi_prob)
        except ValueError:
            lower = dist.ppf(1 - hdi_prob)
            upper = dist.ppf(hdi_prob)
        bounds[name] = (lower, upper)

    return bounds


def sample_from_priors(
    priors: dict[str, Distribution],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Draw independent Monte Carlo samples from each prior.

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping from parameter name to preliz distribution.
    n_samples : int
        Number of draws.
    seed : int, Generator, or None, optional
        Seed or generator for the random draws. Defaults to None, which draws fresh entropy from the OS.

    Returns
    -------
    samples : DataFrame
        Draws of shape ``(n_samples, n_params)`` with one column per key of ``priors``.

    Examples
    --------
    Each column holds independent draws from the matching prior:

    .. code-block:: python

        import preliz as pz

        from gEconpy import sample_from_priors

        priors = {"alpha": pz.Beta(mu=0.35, sigma=0.05), "rho_A": pz.Beta(mu=0.95, sigma=0.02)}
        samples = sample_from_priors(priors, n_samples=100, seed=0)
        print(samples.describe())
    """
    rng = np.random.default_rng(seed)
    draws = {name: dist.rvs(n_samples, random_state=rng) for name, dist in priors.items()}
    return pd.DataFrame(draws)


def sample_uniform(
    param_bounds: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
    method: str = "lhs",
) -> pd.DataFrame:
    """
    Generate parameter samples that fill a box with uniform or quasi-Monte Carlo draws.

    Parameters
    ----------
    param_bounds : dict of str to tuple of float
        ``(lower, upper)`` interval for each parameter, keyed by parameter name.
    n_samples : int
        Number of samples. Must be a power of 2 when ``method="sobol"``.
    seed : int, Generator, or None, optional
        Seed for the sampler. Defaults to None.
    method : str, optional
        Sampling scheme, one of ``"random"`` (independent uniform draws), ``"lhs"`` (Latin hypercube),
        ``"sobol"`` (scrambled Sobol sequence), ``"halton"`` (scrambled Halton sequence), or ``"poisson_disk"``
        (Poisson disk sampling, the most even coverage and the slowest). Defaults to ``"lhs"``.

    Returns
    -------
    samples : DataFrame
        Draws of shape ``(n_samples, n_params)`` with one column per key of ``param_bounds``.

    Examples
    --------
    A Sobol sample of 64 points, a power of 2, stays inside the box:

    .. code-block:: python

        from gEconpy import sample_uniform

        bounds = {"alpha": (0.2, 0.5), "rho_A": (0.8, 0.99)}
        samples = sample_uniform(bounds, n_samples=64, method="sobol", seed=0)
        print(samples.min(), samples.max())
    """
    names = list(param_bounds.keys())
    lower_bounds = np.array([param_bounds[name][0] for name in names], dtype=float)
    upper_bounds = np.array([param_bounds[name][1] for name in names], dtype=float)

    if method == "random":
        rng = np.random.default_rng(seed)
        unit_samples = rng.uniform(size=(n_samples, len(names)))
    elif method in QMC_ENGINES:
        unit_samples = _unit_hypercube_samples(n_samples, n_dims=len(names), seed=seed, method=method)
    else:
        raise ValueError(
            f"Unknown sampling method {method!r}. Choose from 'random', 'lhs', 'sobol', 'halton', 'poisson_disk'."
        )

    scaled = qmc.scale(unit_samples, lower_bounds, upper_bounds)
    return pd.DataFrame(scaled, columns=names)


def sample_uniform_from_priors(
    priors: dict[str, Distribution],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
    method: str = "lhs",
    hdi_prob: float = 0.99,
) -> pd.DataFrame:
    """
    Fill the box spanned by the priors' highest density intervals with quasi-Monte Carlo draws.

    For a solvability check, uniform coverage of the plausible region finds failure regions that draws from the
    prior miss, because prior draws cluster near the mode.

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping from parameter name to preliz distribution.
    n_samples : int
        Number of samples. Must be a power of 2 when ``method="sobol"``.
    seed : int, Generator, or None, optional
        Seed for the sampler, forwarded to :func:`sample_uniform`. Defaults to None.
    method : str, optional
        Sampling scheme, forwarded to :func:`sample_uniform`. Defaults to ``"lhs"``.
    hdi_prob : float, optional
        Probability mass of the interval used as the box for each parameter, forwarded to
        :func:`bounds_from_priors`. Defaults to 0.99.

    Returns
    -------
    samples : DataFrame
        Draws of shape ``(n_samples, n_params)`` with one column per key of ``priors``.

    Examples
    --------
    The model's ``param_priors`` feed the sampler directly:

    .. code-block:: python

        from gEconpy import model_from_gcn, sample_uniform_from_priors
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        samples = sample_uniform_from_priors(model.param_priors, n_samples=50, seed=0)
        print(samples.head())
    """
    bounds = bounds_from_priors(priors, hdi_prob=hdi_prob)
    return sample_uniform(bounds, n_samples, seed=seed, method=method)


def sample_from_priors_qmc(
    priors: dict[str, Distribution],
    n_samples: int,
    seed: int | np.random.Generator | None = None,
    method: str = "sobol",
) -> pd.DataFrame:
    """
    Draw quasi-random samples that follow each prior's shape through its inverse CDF.

    A quasi-Monte Carlo engine fills the unit hypercube, then each column passes through the matching prior's
    ``ppf``. The draws keep the prior's density, so for the same ``n_samples`` they cover the tails less evenly
    than :func:`sample_uniform_from_priors`.

    Parameters
    ----------
    priors : dict of str to Distribution
        Mapping from parameter name to preliz distribution.
    n_samples : int
        Number of samples. Must be a power of 2 when ``method="sobol"``.
    seed : int, Generator, or None, optional
        Seed for the QMC engine. Defaults to None.
    method : str, optional
        QMC engine, one of ``"sobol"``, ``"halton"``, or ``"lhs"``. Defaults to ``"sobol"``.

    Returns
    -------
    samples : DataFrame
        Draws of shape ``(n_samples, n_params)`` with one column per key of ``priors``.

    Examples
    --------
    Each column follows its prior through the inverse CDF, so the sample means track the prior means:

    .. code-block:: python

        import preliz as pz

        from gEconpy import sample_from_priors_qmc

        priors = {"alpha": pz.Beta(mu=0.35, sigma=0.05), "sigma": pz.HalfNormal(sigma=1.0)}
        samples = sample_from_priors_qmc(priors, n_samples=32, seed=0)
        print(samples.mean())
    """
    names = list(priors.keys())
    if method not in INVERSE_CDF_ENGINES:
        raise ValueError(f"Unknown method {method!r} for sample_from_priors_qmc. Choose from 'sobol', 'halton', 'lhs'.")

    unit_samples = _unit_hypercube_samples(n_samples, n_dims=len(names), seed=seed, method=method)

    # ppf(0) and ppf(1) are infinite for unbounded priors, so keep the draws strictly inside the unit interval.
    eps = np.finfo(float).eps
    unit_samples = np.clip(unit_samples, eps, 1 - eps)

    draws = {name: priors[name].ppf(unit_samples[:, i]) for i, name in enumerate(names)}
    return pd.DataFrame(draws)


def _unit_hypercube_samples(
    n_samples: int,
    n_dims: int,
    seed: int | np.random.Generator | None,
    method: str,
) -> np.ndarray:
    if method == "sobol" and (n_samples & (n_samples - 1)) != 0:
        next_power_of_two = 2 ** int(np.ceil(np.log2(n_samples)))
        raise ValueError(
            f"Sobol sequences require n_samples to be a power of 2, got {n_samples}. Try n_samples={next_power_of_two}."
        )

    engine_kwargs = {"scramble": True} if method in ("sobol", "halton") else {}
    sampler = QMC_ENGINES[method](d=n_dims, seed=seed, **engine_kwargs)
    return sampler.random(n_samples)
