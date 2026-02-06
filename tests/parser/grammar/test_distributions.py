import pytest

from pyparsing import ParseException

from gEconpy.parser.ast import GCNDistribution
from gEconpy.parser.grammar.distributions import parse_distribution


class TestSimpleDistributions:
    def test_beta_with_kwargs(self):
        d = parse_distribution("alpha ~ Beta(alpha=2, beta=5) = 0.35")
        assert d.parameter_name == "alpha"
        assert d.dist_name == "Beta"
        assert d.dist_kwargs == {"alpha": 2, "beta": 5}
        assert d.initial_value == 0.35
        assert not d.is_wrapped

    def test_normal_with_mu_sigma(self):
        d = parse_distribution("x ~ Normal(mu=0, sigma=1)")
        assert d.dist_name == "Normal"
        assert d.dist_kwargs == {"mu": 0, "sigma": 1}
        assert d.initial_value is None

    def test_gamma_distribution(self):
        d = parse_distribution("tau ~ Gamma(alpha=2, beta=1) = 2.1")
        assert d.dist_name == "Gamma"
        assert d.initial_value == 2.1

    def test_exponential_with_expression(self):
        d = parse_distribution("lam ~ Exponential(lam=1/100) = 0.01")
        assert d.dist_name == "Exponential"
        assert d.dist_kwargs["lam"] == 0.01
        assert d.initial_value == 0.01

    def test_empty_kwargs(self):
        d = parse_distribution("x ~ Normal()")
        assert d.dist_name == "Normal"
        assert d.dist_kwargs == {}

    def test_half_normal(self):
        d = parse_distribution("sigma ~ HalfNormal(sigma=5) = 1.0")
        assert d.dist_name == "HalfNormal"
        assert d.dist_kwargs == {"sigma": 5}

    def test_uniform(self):
        d = parse_distribution("x ~ Uniform(lower=0, upper=1) = 0.5")
        assert d.dist_name == "Uniform"
        assert d.dist_kwargs == {"lower": 0, "upper": 1}


class TestWrappedDistributions:
    def test_maxent_basic(self):
        d = parse_distribution("x ~ maxent(Normal())")
        assert d.is_wrapped
        assert d.wrapper_name == "maxent"
        assert d.dist_name == "Normal"

    def test_maxent_with_bounds(self):
        d = parse_distribution("alpha ~ maxent(Beta(), lower=0.2, upper=0.5, mass=0.99) = 0.35")
        assert d.wrapper_name == "maxent"
        assert d.dist_name == "Beta"
        assert d.wrapper_kwargs == {"lower": 0.2, "upper": 0.5, "mass": 0.99}
        assert d.initial_value == 0.35

    def test_truncated(self):
        d = parse_distribution("x ~ Truncated(Normal(), lower=0, upper=5) = 2.5")
        assert d.wrapper_name == "Truncated"
        assert d.dist_name == "Normal"
        assert d.wrapper_kwargs == {"lower": 0, "upper": 5}

    def test_truncated_one_bound(self):
        d = parse_distribution("x ~ Truncated(Normal(), lower=0) = 1.0")
        assert d.wrapper_kwargs == {"lower": 0}

    def test_censored_with_none(self):
        d = parse_distribution("x ~ Censored(Beta(alpha=2, beta=5), lower=0.1, upper=None)")
        assert d.wrapper_name == "Censored"
        assert d.wrapper_kwargs["lower"] == 0.1
        assert d.wrapper_kwargs["upper"] is None

    def test_maxent_with_inner_kwargs(self):
        d = parse_distribution("nu ~ maxent(StudentT(nu=7), lower=3, upper=7) = 5.0")
        assert d.dist_name == "StudentT"
        assert d.dist_kwargs == {"nu": 7}
        assert d.wrapper_kwargs == {"lower": 3, "upper": 7}


class TestRealWorldExamples:
    def test_rbc_beta(self):
        d = parse_distribution("beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99")
        assert d.parameter_name == "beta"
        assert d.initial_value == 0.99

    def test_rbc_delta(self):
        d = parse_distribution("delta ~ maxent(Beta(), lower=0.01, upper=0.05, mass=0.99) = 0.02")
        assert d.parameter_name == "delta"

    def test_rbc_sigma(self):
        d = parse_distribution("sigma_C ~ maxent(Gamma(), lower=1.01, upper=10.0, mass=0.99) = 1.5")
        assert d.parameter_name == "sigma_C"
        assert d.dist_name == "Gamma"

    def test_open_rbc_gamma(self):
        d = parse_distribution("gamma_rv ~ HalfNormal(sigma=5) = 1")
        assert d.parameter_name == "gamma_rv"
        assert d.initial_value == 1.0

    def test_open_rbc_alpha(self):
        d = parse_distribution("alpha ~ Beta(alpha=5, beta=5) = 0.32")
        assert d.dist_kwargs == {"alpha": 5, "beta": 5}

    def test_rho_parameter(self):
        d = parse_distribution("rho_A ~ Beta(alpha=3, beta=1) = 0.42")
        assert d.parameter_name == "rho_A"


class TestExpressionEvaluation:
    def test_division(self):
        d = parse_distribution("x ~ Exponential(lam=1/100) = 0.01")
        assert d.dist_kwargs["lam"] == pytest.approx(0.01)

    def test_multiplication(self):
        d = parse_distribution("x ~ Normal(mu=2*3, sigma=1)")
        assert d.dist_kwargs["mu"] == 6

    def test_complex_expression(self):
        d = parse_distribution("x ~ Normal(mu=18/10)")
        assert d.dist_kwargs["mu"] == pytest.approx(1.8)


class TestEdgeCases:
    def test_trailing_semicolon(self):
        d = parse_distribution("alpha ~ Beta(alpha=2, beta=5) = 0.35;")
        assert d.parameter_name == "alpha"

    def test_extra_whitespace(self):
        d = parse_distribution("  alpha   ~   Beta( alpha = 2 ,  beta = 5 )  =  0.35  ")
        assert d.parameter_name == "alpha"
        assert d.dist_kwargs == {"alpha": 2, "beta": 5}

    def test_underscore_in_param_name(self):
        d = parse_distribution("sigma_L ~ Gamma(alpha=2, beta=1) = 2.0")
        assert d.parameter_name == "sigma_L"

    def test_numeric_suffix_in_param_name(self):
        d = parse_distribution("rho_1 ~ Beta(alpha=1, beta=1) = 0.5")
        assert d.parameter_name == "rho_1"


class TestErrorCases:
    def test_missing_tilde_raises(self):
        with pytest.raises(ParseException):
            parse_distribution("alpha Beta(alpha=2)")

    def test_unknown_distribution_raises(self):
        with pytest.raises(ParseException):
            parse_distribution("x ~ UnknownDist()")

    def test_unknown_wrapper_raises(self):
        with pytest.raises(ParseException):
            parse_distribution("x ~ unknownwrap(Normal())")

    def test_missing_parens_raises(self):
        with pytest.raises(ParseException):
            parse_distribution("x ~ Normal")

    def test_unclosed_parens_raises(self):
        with pytest.raises(ParseException):
            parse_distribution("x ~ Normal(mu=0")

    def test_empty_string_raises(self):
        with pytest.raises(ParseException):
            parse_distribution("")


class TestSecurity:
    def test_cannot_execute_arbitrary_code_in_kwargs(self):
        # This should fail to parse - the expression parser only allows numbers and operators
        with pytest.raises((ParseException, ValueError)):
            parse_distribution("x ~ Normal(mu=__import__('os').system('ls'))")

    def test_cannot_execute_code_via_initial_value(self):
        with pytest.raises((ParseException, ValueError)):
            parse_distribution("x ~ Normal() = open('/etc/passwd')")

    def test_only_arithmetic_allowed(self):
        # This should work - simple arithmetic
        d = parse_distribution("x ~ Normal(mu=1+2)")
        assert d.dist_kwargs["mu"] == 3

    def test_division_is_safe(self):
        d = parse_distribution("x ~ Normal(mu=10/2)")
        assert d.dist_kwargs["mu"] == 5.0
