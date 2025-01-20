from string import Template

import numpy as np
import preliz as pz
import pytest

from gEconpy.exceptions import (
    InvalidDistributionException,
    MissingParameterValueException,
    RepeatedParameterException,
)
from gEconpy.parser.dist_syntax import dist_syntax
from gEconpy.parser.gEcon_parser import preprocess_gcn
from gEconpy.parser.parse_distributions import (
    create_prior_distribution_dictionary,
    preprocess_distribution_string,
)

test_strings = [
    "Normal(mu=3, sigma=1)",
    "Gamma(alpha=2, beta=1) = 2.1",
    "Exponential(lam=1/100) = 0.01",
    "maxent(Normal())",
    "maxent(Normal(), lower=1, upper=3)",
    "maxent(Normal(), lower=1, upper=3) = 42",
    "Truncated(Normal(), lower=0, upper=5)",
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


@pytest.fixture
def file():
    return """
            Block TEST
            {
                shocks
                {
                    epsilon[] ~ Normal(mu = 0, sigma = 1);
                };

                calibration
                {
                    alpha ~ Normal(mu = 0.5, sigma = 0.1) = 0.5;
                    gamma ~ maxent(Beta(), lower=0.1, upper=0.3) = 0.2;
                };
            };
    """


def test_extract_param_dist_simple(file):
    model, prior_dict = preprocess_gcn(file)
    assert list(prior_dict.keys()) == ["epsilon[]", "alpha", "gamma"]
    assert list(prior_dict.values()) == [
        "Normal(mu = 0, sigma = 1)",
        "Normal(mu = 0.5, sigma = 0.1) = 0.5",
        "maxent(Beta(), lower=0.1, upper=0.3) = 0.2",
    ]


def test_catch_no_initial_value(file):
    no_initial_value = """
            Block TEST
            {
                calibration
                {
                    alpha ~ N(mean = 0, sd = 1);
                };
            };
    """

    with pytest.raises(MissingParameterValueException):
        preprocess_gcn(no_initial_value)


extra_parenthesis_start = """
        Block TEST
        {
            calibration
            {
                alpha ~ Normal((mu = 0, sigma = 1) = 0.5;
            };
        };
"""

extra_parenthesis_end = """
        Block TEST
        {
            calibration
            {
                alpha ~ Normal(mu = 0, sigma = 1)) = 0.5;
            };
        };
"""

extra_equals = """
        Block TEST
        {
            calibration
            {
                alpha ~ Normal(mu == 0, sigma = 1) = 0.5;
            };
        };
"""

missing_common = """
        Block TEST
        {
            calibration
            {
                alpha ~ Normal(mu = 0 sigma = 1) = 0.5;
            };
        };
"""

typo_cases = [
    extra_parenthesis_start,
    extra_parenthesis_end,
    extra_equals,
    missing_common,
]

case_names = [
    "extra_parenthesis_start",
    "extra_parenthesis_end",
    "extra_equals",
    "missing_comma",
]


@pytest.mark.parametrize("case", typo_cases, ids=case_names)
def test_catch_distribution_typos(case):
    model, prior_dict = preprocess_gcn(case)
    for param_name, distribution_string in prior_dict.items():
        with pytest.raises(InvalidDistributionException):
            preprocess_distribution_string(
                variable_name=param_name, d_string=distribution_string
            )


def test_catch_repeated_parameter_definition(file):
    repeated_parameter = """
            Block TEST
            {
                calibration
                {
                    alpha ~ Normal(mu = 0, mu = 1) = 0.5;
                };
            };
    """
    model, prior_dict = preprocess_gcn(repeated_parameter)

    for param_name, distribution_string in prior_dict.items():
        with pytest.raises(RepeatedParameterException):
            preprocess_distribution_string(
                variable_name=param_name, d_string=distribution_string
            )


@pytest.mark.parametrize("mu", ["1.0", "mu_epsilon"], ids=["number", "hyper_param"])
def test_non_zero_shock_mean_raises(mu):
    test_case = Template("""
    Block TEST
    {
        shocks
        {
            epsilon[] ~ Normal(mu=$mu, sigma=3)
        };

        calibration
        {
            mu_epsilon ~ Normal(mu=0, sigma=1) = 0.5;
        };
    };
    """).safe_substitute(mu=mu)

    model, raw_prior_dict = preprocess_gcn(test_case)

    msg = (
        "Currently, the mean of all shocks must be zero"
        if mu == "1.0"
        else "Currently, only shock variance parameters can be assigned hyper-priors."
    )
    with pytest.raises(NotImplementedError, match=msg):
        create_prior_distribution_dictionary(raw_prior_dict)


@pytest.mark.parametrize("wrapper", ["maxent", "Truncated"], ids=str)
def test_wrapper_on_shock_raises(wrapper):
    test_case = Template("""
    Block TEST
    {
        shocks
        {
            epsilon[] ~ $wrapper(Normal(mu=1, sigma=3), lower=0.1, upper=1.0);
        };
    };
    """).safe_substitute(wrapper=wrapper)

    model, raw_prior_dict = preprocess_gcn(test_case)

    with pytest.raises(
        NotImplementedError,
        match="Wrapper functions are not allowed on shock distributions",
    ):
        create_prior_distribution_dictionary(raw_prior_dict)


@pytest.mark.parametrize("dist", ["Gamma", "InverseGamma"], ids=str)
def test_non_normal_shock_dist_raises(dist):
    test_case = Template("""
    Block TEST
    {
        shocks
        {
            epsilon[] ~ $dist(mu=1, sigma=3);
        };
    };
    """).safe_substitute(dist=dist)

    model, raw_prior_dict = preprocess_gcn(test_case)

    with pytest.raises(
        NotImplementedError,
        match="Only Normal distributions are currently allowed on shocks",
    ):
        create_prior_distribution_dictionary(raw_prior_dict)


def test_initial_value_on_shock_raises():
    test_case = """
           Block TEST
           {
               shocks
               {
                   epsilon[] ~ Normal(mu = 0, sigma = 1) = 0.5;
               };
           };
   """
    model, raw_prior_dict = preprocess_gcn(test_case)

    with pytest.raises(
        ValueError, match="Initial value not allowed on shock distributions"
    ):
        create_prior_distribution_dictionary(raw_prior_dict)


def test_parameter_parsing_simple(file):
    model, prior_dict = preprocess_gcn(file)

    dist_names = ["Normal", "Normal", "Beta"]
    wrapper_names = [None, None, "maxent"]

    param_dicts = [
        {"mu": 0.0, "sigma": 1.0, "initial_value": None},
        {"mu": 0.5, "sigma": 0.1, "initial_value": 0.5},
        {"initial_value": 0.2},
    ]
    wrapper_dicts = [{}, {}, {"lower": 0.1, "upper": 0.3}]

    for i, (param_name, distribution_string) in enumerate(prior_dict.items()):
        (dist_name, param_dict), (wrapper_name, wrapper_dict) = (
            preprocess_distribution_string(param_name, distribution_string)
        )

        assert dist_name == dist_names[i]
        assert param_dict == param_dicts[i]
        assert wrapper_name == wrapper_names[i]
        assert wrapper_dict == wrapper_dicts[i]


def test_multiple_shocks():
    compound_distribution = """Block TEST
            {
                identities
                {
                    log(A[]) = rho_A * log(A[-1]) + epsilon_A[];
                    log(B[]) = rho_B * log(B[-1]) + epsilon_B[];
                };

                shocks
                {
                    epsilon_A[] ~ Normal(mu=0, sigma=sigma_epsilon_A);
                    epsilon_B[] ~ Normal(mu=0, sigma=sigma_epsilon_B);
                    epsilon_C[] ~ Normal(mu=0, sigma=1);
                };

                calibration
                {
                    rho_A ~ Beta(mu=0.95, sigma=0.04) = 0.95;
                    rho_B ~ Beta(mu=0.95, sigma=0.04) = 0.95;

                    sigma_epsilon_A ~ Gamma(alpha=1, beta=0.1) = 0.01;
                    sigma_epsilon_B ~ Gamma(alpha=10, beta=7) = 0.01;
                };
            };"""

    _, raw_prior_dict = preprocess_gcn(compound_distribution)
    param_priors, shock_priors = create_prior_distribution_dictionary(raw_prior_dict)

    expected_names = ["rho_A", "rho_B"]
    expected_shock_names = ["epsilon_A", "epsilon_B", "epsilon_C"]

    expected_dists = {
        "rho_A": pz.Beta(mu=0.95, sigma=0.04),
        "rho_B": pz.Beta(mu=0.95, sigma=0.04),
        "sigma_epsilon_A": pz.Gamma(alpha=1, beta=0.1),
        "sigma_epsilon_B": pz.Gamma(alpha=10, beta=7),
    }

    assert list(param_priors.keys()) == expected_names
    assert list(shock_priors.keys()) == expected_shock_names

    for name in expected_names:
        dist = expected_dists[name]
        assert dist.mean() == expected_dists[name].mean()
        assert dist.std() == expected_dists[name].std()

    for name in expected_shock_names[:-1]:
        assert shock_priors[name].fixed_params == {"mu": 0.0}

        dists = shock_priors[name].hyper_param_dict.values()
        expected_shock_dists = [expected_dists[f"sigma_{name}"]]

        assert all(
            d.mean() == expected_d.mean()
            for d, expected_d in zip(dists, expected_shock_dists)
        )
        assert all(
            d.std() == expected_d.std()
            for d, expected_d in zip(dists, expected_shock_dists)
        )

    assert shock_priors["epsilon_C"].fixed_params == {"mu": 0.0, "sigma": 1.0}


def test_parse_distributions():
    file = """
        TEST_BLOCK
        {
            shocks
            {
                epsilon[] ~ Normal(mean=0, std=0.1);
            };

            calibration
            {
                alpha ~ beta(a=1, b=1) = 0.5;
                rho ~ gamma(mean=0.95, sd=1) = 0.95;
                sigma ~ inv_gamma(mean=0.01, sd=0.1) = 0.01;
                tau ~ halfnorm(MEAN=0.5, sd=1) = 1;
                psi ~ norm(mean=1.5, Sd=1.5, min=0) = 1;
            };
        };
    """

    # model, prior_dict = preprocess_gcn(file)
    # means = [0, 0.5, 0.95, 0.01, 0.5, 1.5]
    # stds = [0.1, 0.28867513459481287, 1, 0.1, 1, 1.5]
    #
    # for i, (variable_name, d_string) in enumerate(prior_dict.items()):
    #     d_name, param_dict = preprocess_distribution_string(variable_name, d_string)
    #     d = distribution_factory(
    #         variable_name=variable_name, d_name=d_name, param_dict=param_dict
    #     )
    #     np.testing.assert_allclose(d.mean(), means[i], atol=1e-3)
    #     np.testing.assert_allclose(d.std(), stds[i], atol=1e-3)
