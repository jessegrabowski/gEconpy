from gEconpy.parser.dist_syntax import dist_syntax
import pytest


test_strings = [
    "Normal(mu=3, sigma=1)",
    "Gamma(alpha=2, beta=1) = 2.1",
    "Exponential(lam=1/100) = 0.01",
    "maxent(Normal())",
    "maxent(Normal(), lower=1, upper=3)",
    "maxent(Normal(), lower=1, upper=3) = 42",
    "Truncated(Normal(), lower=0, upper=5)",
    "Truncated(Normal(), lower=0) = 10",
    "Censored(Beta(alpha=2, beta=5), lower=0.1, upper=None)",
    "maxent(StudentT(nu=7), lower=3, upper=7) = 18 / 10",
]

expected_results = [
    {
        "wrapper_name": None,
        "wrapper_kwargs": {},
        "dist_name": "Normal",
        "dist_kwargs": {"mu": 3, "sigma": 1},
        "initial_value": None,
    },
    {
        "wrapper_name": None,
        "wrapper_kwargs": {},
        "dist_name": "Gamma",
        "dist_kwargs": {"alpha": 2, "beta": 1},
        "initial_value": 2.1,
    },
    {
        "wrapper_name": None,
        "wrapper_kwargs": {},
        "dist_name": "Exponential",
        "dist_kwargs": {"lam": [1, "/", 100]},
        "initial_value": 0.01,
    },
    {
        "wrapper_name": "maxent",
        "wrapper_kwargs": {},
        "dist_name": "Normal",
        "dist_kwargs": {},
        "initial_value": None,
    },
    {
        "wrapper_name": "maxent",
        "wrapper_kwargs": {"lower": 1, "upper": 3},
        "dist_name": "Normal",
        "dist_kwargs": {},
        "initial_value": None,
    },
    {
        "wrapper_name": "maxent",
        "wrapper_kwargs": {"lower": 1, "upper": 3},
        "dist_name": "Normal",
        "dist_kwargs": {},
        "initial_value": 42,
    },
    {
        "wrapper_name": "Truncated",
        "wrapper_kwargs": {"lower": 0, "upper": 5},
        "dist_name": "Normal",
        "dist_kwargs": {},
        "initial_value": None,
    },
    {
        "wrapper_name": "Truncated",
        "wrapper_kwargs": {"lower": 0},
        "dist_name": "Normal",
        "dist_kwargs": {},
        "initial_value": 10,
    },
    {
        "wrapper_name": "Censored",
        "wrapper_kwargs": {"lower": 0.1, "upper": "None"},
        "dist_name": "Beta",
        "dist_kwargs": {"alpha": 2, "beta": 5},
        "initial_value": None,
    },
    {
        "wrapper_name": "maxent",
        "wrapper_kwargs": {"lower": 3, "upper": 7},
        "dist_name": "StudentT",
        "dist_kwargs": {"nu": 7},
        "initial_value": [18, "/", 10],
    },
]


@pytest.mark.parametrize("case, expected_results", zip(test_strings, expected_results))
def test_distribution_parser(case, expected_results):
    [results] = dist_syntax.parse_string(case, parse_all=True)

    assert results["wrapper_name"] == expected_results["wrapper_name"]
    assert results["dist_name"] == expected_results["dist_name"]
    assert results["initial_value"] == expected_results["initial_value"]
    assert results["dist_kwargs"] == expected_results["dist_kwargs"]
    assert results["wrapper_kwargs"] == expected_results["wrapper_kwargs"]
