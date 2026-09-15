import pytest

from pyparsing import ParseBaseException

from gEconpy.parser.grammar import parse_distribution


class TestSimpleDistributions:
    def test_beta_with_kwargs(self):
        d = parse_distribution("alpha ~ Beta(alpha=2, beta=5) = 0.35;")
        assert d.parameter_name == "alpha"
        assert d.dist_name == "Beta"
        assert d.dist_kwargs == {"alpha": 2, "beta": 5}
        assert d.initial_value == 0.35
        assert not d.is_wrapped

    def test_without_initial_value(self):
        d = parse_distribution("x ~ Normal(mu=0, sigma=1);")
        assert d.dist_name == "Normal"
        assert d.dist_kwargs == {"mu": 0, "sigma": 1}
        assert d.initial_value is None

    def test_empty_kwargs(self):
        d = parse_distribution("x ~ Normal();")
        assert d.dist_name == "Normal"
        assert d.dist_kwargs == {}

    @pytest.mark.parametrize(
        "text,parameter_name,dist_name,initial_value",
        [
            ("tau ~ Gamma(alpha=2, beta=1) = 2.1;", "tau", "Gamma", 2.1),
            ("sigma ~ HalfNormal(sigma=5) = 1.0;", "sigma", "HalfNormal", 1.0),
            ("x ~ Uniform(lower=0, upper=1) = 0.5;", "x", "Uniform", 0.5),
            ("gamma_rv ~ HalfNormal(sigma=5) = 1;", "gamma_rv", "HalfNormal", 1.0),
            ("rho_A ~ Beta(alpha=3, beta=1) = 0.42;", "rho_A", "Beta", 0.42),
            ("sigma_L ~ Gamma(alpha=2, beta=1) = 2.0;", "sigma_L", "Gamma", 2.0),
            ("rho_1 ~ Beta(alpha=1, beta=1) = 0.5;", "rho_1", "Beta", 0.5),
        ],
    )
    def test_named_distributions(self, text, parameter_name, dist_name, initial_value):
        d = parse_distribution(text)
        assert d.parameter_name == parameter_name
        assert d.dist_name == dist_name
        assert d.initial_value == initial_value

    def test_extra_whitespace(self):
        d = parse_distribution("  alpha   ~   Beta( alpha = 2 ,  beta = 5 )  =  0.35  ;")
        assert d.parameter_name == "alpha"
        assert d.dist_kwargs == {"alpha": 2, "beta": 5}


class TestWrappedDistributions:
    def test_maxent_basic(self):
        d = parse_distribution("x ~ maxent(Normal());")
        assert d.is_wrapped
        assert d.wrapper_name == "maxent"
        assert d.dist_name == "Normal"
        assert d.wrapper_kwargs == {}

    def test_maxent_with_bounds(self):
        d = parse_distribution("alpha ~ maxent(Beta(), lower=0.2, upper=0.5, mass=0.99) = 0.35;")
        assert d.wrapper_name == "maxent"
        assert d.dist_name == "Beta"
        assert d.wrapper_kwargs == {"lower": 0.2, "upper": 0.5, "mass": 0.99}
        assert d.initial_value == 0.35

    def test_truncated(self):
        d = parse_distribution("x ~ Truncated(Normal(), lower=0, upper=5) = 2.5;")
        assert d.wrapper_name == "Truncated"
        assert d.dist_name == "Normal"
        assert d.wrapper_kwargs == {"lower": 0, "upper": 5}

    def test_truncated_one_bound(self):
        d = parse_distribution("x ~ Truncated(Normal(), lower=0) = 1.0;")
        assert d.wrapper_kwargs == {"lower": 0}

    def test_censored_with_none(self):
        d = parse_distribution("x ~ Censored(Beta(alpha=2, beta=5), lower=0.1, upper=None);")
        assert d.wrapper_name == "Censored"
        assert d.wrapper_kwargs == {"lower": 0.1, "upper": None}

    def test_maxent_with_inner_kwargs(self):
        d = parse_distribution("nu ~ maxent(StudentT(nu=7), lower=3, upper=7) = 5.0;")
        assert d.dist_name == "StudentT"
        assert d.dist_kwargs == {"nu": 7}
        assert d.wrapper_kwargs == {"lower": 3, "upper": 7}


class TestArgumentExpressions:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("1/100", 0.01),
            ("2*3", 6.0),
            ("18/10", 1.8),
            ("1+2", 3.0),
            ("10/2", 5.0),
            ("2*3+1", 7.0),
            ("1-2*3", -5.0),
        ],
    )
    def test_arithmetic_is_evaluated(self, expression, expected):
        d = parse_distribution(f"x ~ Normal(mu={expression}, sigma=1);")
        assert d.dist_kwargs["mu"] == pytest.approx(expected)

    def test_initial_value_expression(self):
        d = parse_distribution("lam ~ Exponential(lam=1/100) = 1/100;")
        assert d.initial_value == pytest.approx(0.01)

    def test_parameter_reference(self):
        d = parse_distribution("eps ~ Normal(mu=0, sigma=sigma_eps);")
        assert d.dist_kwargs["sigma"] == "sigma_eps"

    @pytest.mark.parametrize(
        "text",
        [
            "x ~ Normal(mu=__import__('os').system('ls'));",
            "x ~ Normal() = open('/etc/passwd');",
        ],
    )
    def test_arbitrary_code_does_not_parse(self, text):
        with pytest.raises(ParseBaseException):
            parse_distribution(text)


class TestErrorCases:
    @pytest.mark.parametrize(
        "text",
        [
            "alpha Beta(alpha=2);",
            "x ~ UnknownDist();",
            "x ~ unknownwrap(Normal());",
            "x ~ Normal;",
            "x ~ Normal(mu=0;",
            "",
        ],
        ids=["missing_tilde", "unknown_distribution", "unknown_wrapper", "missing_parens", "unclosed_parens", "empty"],
    )
    def test_invalid_declaration_raises(self, text):
        with pytest.raises(ParseBaseException):
            parse_distribution(text)
