import numpy as np
import pytest

from gEconpy.exceptions import (
    DistributionParsingError,
    InvalidDistributionException,
    MissingParameterValueException,
    RepeatedParameterException,
)
from gEconpy.parser.gEcon_parser import preprocess_gcn
from gEconpy.parser.parse_distributions import (
    create_prior_distribution_dictionary,
    distribution_factory,
    preprocess_distribution_string,
)


@pytest.fixture
def file():
    return """
            Block TEST
            {
                shocks
                {
                    epsilon[] ~ norm(mu = 0, sd = 1);
                };

                calibration
                {
                    alpha ~ N(mean = 0, sd = 1) = 0.5;
                };
            };
    """


def test_extract_param_dist_simple(file):
    model, prior_dict = preprocess_gcn(file)
    assert list(prior_dict.keys()) == ["epsilon[]", "alpha"]
    assert list(prior_dict.values()) == [
        "norm(mu = 0, sd = 1)",
        "N(mean = 0, sd = 1) = 0.5",
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
                alpha ~ N((mean = 0, sd = 1) = 0.5;
            };
        };
"""

extra_parenthesis_end = """
        Block TEST
        {
            calibration
            {
                alpha ~ N(mean = 0, sd = 1)) = 0.5;
            };
        };
"""

extra_equals = """
        Block TEST
        {
            calibration
            {
                alpha ~ N(mean == 0, sd = 1) = 0.5;
            };
        };
"""

missing_common = """
        Block TEST
        {
            calibration
            {
                alpha ~ N(mean = 0 sd = 1) = 0.5;
            };
        };
"""

shock_with_starting_value = """
        Block TEST
        {
            shocks
            {
                epsilon[] ~ N(mean = 0, sd = 1) = 0.5;
            };
        };
"""

typo_cases = [
    extra_parenthesis_start,
    extra_parenthesis_end,
    extra_equals,
    missing_common,
    shock_with_starting_value,
]

case_names = [
    "extra_parenthesis_start",
    "extra_parenthesis_end",
    "extra_equals",
    "missing_common",
    "shock_with_starting_value",
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
                    alpha ~ N(mean = 0, mean = 1) = 0.5;
                };
            };
    """
    model, prior_dict = preprocess_gcn(repeated_parameter)

    for param_name, distribution_string in prior_dict.items():
        with pytest.raises(RepeatedParameterException):
            preprocess_distribution_string(
                variable_name=param_name, d_string=distribution_string
            )


def test_parameter_parsing_simple(file):
    model, prior_dict = preprocess_gcn(file)
    dicts = [
        {"mu": 0.0, "sd": 1.0, "initial_value": None},
        {"mean": 0.0, "sd": 1.0, "initial_value": 0.5},
    ]

    for i, (param_name, distribution_string) in enumerate(prior_dict.items()):
        dist_name, param_dict = preprocess_distribution_string(
            param_name, distribution_string
        )

        assert dist_name == "normal"
        assert param_dict == dicts[i]


def test_parse_compound_distributions(file):
    compound_distribution = """Block TEST
            {
                calibration
                {
                    sigma_alpha ~ inv_gamma(a=20, scale=1) = 0.01;
                    mu_alpha ~ N(mean = 1, scale=1) = 0.01;
                    alpha ~ N(mean = mu_alpha, sd = sigma_alpha) = 0.5;
                };
            };"""

    model, raw_prior_dict = preprocess_gcn(compound_distribution)
    prior_dict, _ = create_prior_distribution_dictionary(raw_prior_dict)

    d = prior_dict["alpha"]

    assert d.rv_params["loc"].mean() == 1
    assert d.rv_params["loc"].std() == 1
    assert d.rv_params["scale"].mean() == 1 / (20 - 1)
    assert d.rv_params["scale"].var() == 1**2 / (20 - 1) ** 2 / (20 - 2)


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
                    epsilon_A[] ~ N(mean=0, sd=sigma_epsilon_A);
                    epsilon_B[] ~ N(mean=0, sd=sigma_epsilon_B);
                };

                calibration
                {
                    rho_A ~ Beta(mean=0.95, sd=0.04) = 0.95;
                    rho_B ~ Beta(mean=0.95, sd=0.04) = 0.95;

                    sigma_epsilon_A ~ Gamma(alpha=1, beta=0.1) = 0.01;
                    sigma_epsilon_B ~ Gamma(alpha=1, beta=0.1) = 0.01;
                };
            };"""

    model, raw_prior_dict = preprocess_gcn(compound_distribution)
    prior_dict, _ = create_prior_distribution_dictionary(raw_prior_dict)

    epsilon_A = prior_dict["epsilon_A[]"]
    epsilon_B = prior_dict["epsilon_B[]"]

    assert len(epsilon_A.rv_params) == 1
    assert len(epsilon_B.rv_params) == 1

    # self.assertEqual(d.rv_params['loc'].mean(), 1)
    # self.assertEqual(d.rv_params['loc'].std(), 1)
    # self.assertEqual(d.rv_params['scale'].mean(), 1 / (20 - 1))
    # self.assertEqual(d.rv_params['scale'].var(), 1 ** 2 / (20 - 1) ** 2 / (20 - 2))


def test_parse_distributions():
    file = """
        TEST_BLOCK
        {
            shocks
            {
                epsilon[] ~ N(mean=0, std=0.1);
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

    model, prior_dict = preprocess_gcn(file)
    means = [0, 0.5, 0.95, 0.01, 0.5, 1.5]
    stds = [0.1, 0.28867513459481287, 1, 0.1, 1, 1.5]

    for i, (variable_name, d_string) in enumerate(prior_dict.items()):
        d_name, param_dict = preprocess_distribution_string(variable_name, d_string)
        d = distribution_factory(
            variable_name=variable_name, d_name=d_name, param_dict=param_dict
        )
        np.testing.assert_allclose(d.mean(), means[i], atol=1e-3)
        np.testing.assert_allclose(d.std(), stds[i], atol=1e-3)
