import preliz as pz
import pytest

from gEconpy.exceptions import InvalidDistributionException
from gEconpy.parser.ast import GCNBlock, GCNDistribution, GCNEquation, GCNModel, Number, Parameter
from gEconpy.parser.grammar import parse_distribution
from gEconpy.parser.transform.to_distribution import (
    ast_to_distribution,
    ast_to_distribution_with_metadata,
    distributions_from_calibration,
    distributions_from_model,
)


class TestAstToDistribution:
    @pytest.mark.parametrize(
        ("dist_name", "dist_kwargs", "expected_type"),
        [
            ("Normal", {"mu": 0, "sigma": 1}, pz.Normal),
            ("Normal", {}, pz.Normal),
            ("Beta", {"alpha": 2, "beta": 5}, pz.Beta),
            ("Gamma", {"alpha": 2, "beta": 1}, pz.Gamma),
            ("Exponential", {"lam": 0.01}, pz.Exponential),
            ("HalfNormal", {"sigma": 5}, pz.HalfNormal),
            ("Uniform", {"lower": 0, "upper": 1}, pz.Uniform),
            ("StudentT", {"nu": 7}, pz.StudentT),
        ],
    )
    def test_builds_preliz_distribution(self, dist_name, dist_kwargs, expected_type):
        node = GCNDistribution(parameter_name="x", dist_name=dist_name, dist_kwargs=dist_kwargs)
        assert isinstance(ast_to_distribution(node), expected_type)

    def test_passes_kwargs_through(self):
        node = GCNDistribution(parameter_name="x", dist_name="Normal", dist_kwargs={"mu": 0, "sigma": 1})
        dist = ast_to_distribution(node)
        assert dist.mu == 0
        assert dist.sigma == 1

    @pytest.mark.parametrize(
        ("dist_name", "dist_kwargs", "wrapper_name", "wrapper_kwargs", "expected_type"),
        [
            ("Beta", {}, "maxent", {"lower": 0.95, "upper": 0.999, "mass": 0.99}, pz.Beta),
            ("Normal", {}, "Truncated", {"lower": 0, "upper": 5}, pz.Truncated),
            ("Normal", {}, "Truncated", {"lower": 0}, pz.Truncated),
            ("Beta", {"alpha": 2, "beta": 5}, "Censored", {"lower": 0.1, "upper": 0.9}, pz.Censored),
        ],
    )
    def test_applies_wrapper(self, dist_name, dist_kwargs, wrapper_name, wrapper_kwargs, expected_type):
        node = GCNDistribution(
            parameter_name="x",
            dist_name=dist_name,
            dist_kwargs=dist_kwargs,
            wrapper_name=wrapper_name,
            wrapper_kwargs=wrapper_kwargs,
        )
        assert isinstance(ast_to_distribution(node), expected_type)

    def test_invalid_distribution_name_raises(self):
        node = GCNDistribution(parameter_name="x", dist_name="NotARealDistribution", dist_kwargs={})
        with pytest.raises(InvalidDistributionException):
            ast_to_distribution(node)

    def test_invalid_wrapper_name_raises(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="Normal",
            dist_kwargs={},
            wrapper_name="NotARealWrapper",
            wrapper_kwargs={},
        )
        with pytest.raises(ValueError, match="Unknown distribution wrapper"):
            ast_to_distribution(node)


class TestDistributionWithMetadata:
    def test_metadata_extraction(self):
        node = GCNDistribution(
            parameter_name="alpha",
            dist_name="Beta",
            dist_kwargs={"alpha": 2, "beta": 5},
            initial_value=0.35,
        )
        _dist, metadata = ast_to_distribution_with_metadata(node)

        assert metadata == {
            "parameter_name": "alpha",
            "initial_value": 0.35,
            "is_wrapped": False,
            "wrapper_name": None,
        }

    def test_wrapped_metadata(self):
        node = GCNDistribution(
            parameter_name="beta",
            dist_name="Beta",
            dist_kwargs={},
            wrapper_name="maxent",
            wrapper_kwargs={"lower": 0.95, "upper": 0.999},
            initial_value=0.99,
        )
        _dist, metadata = ast_to_distribution_with_metadata(node)

        assert metadata == {
            "parameter_name": "beta",
            "initial_value": 0.99,
            "is_wrapped": True,
            "wrapper_name": "maxent",
        }


class TestFromParsedStrings:
    @pytest.mark.parametrize(
        ("source", "expected_type", "parameter_name", "initial_value"),
        [
            ("x ~ Normal(mu=0, sigma=1);", pz.Normal, "x", None),
            ("alpha ~ Beta(alpha=2, beta=5) = 0.35;", pz.Beta, "alpha", 0.35),
            ("beta ~ maxent(Beta(), lower=0.95, upper=0.999) = 0.99;", pz.Beta, "beta", 0.99),
            ("beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99;", pz.Beta, "beta", 0.99),
            ("delta ~ maxent(Beta(), lower=0.01, upper=0.05, mass=0.99) = 0.02;", pz.Beta, "delta", 0.02),
            ("sigma_C ~ maxent(Gamma(), lower=1.01, upper=10.0, mass=0.99) = 1.5;", pz.Gamma, "sigma_C", 1.5),
            ("tau ~ Gamma(alpha=2, beta=1) = 2.1;", pz.Gamma, "tau", 2.1),
            ("sigma ~ HalfNormal(sigma=5) = 1.0;", pz.HalfNormal, "sigma", 1.0),
            ("alpha ~ Beta(alpha=5, beta=5) = 0.32;", pz.Beta, "alpha", 0.32),
            ("rho_A ~ Beta(alpha=3, beta=1) = 0.42;", pz.Beta, "rho_A", 0.42),
        ],
    )
    def test_parsed_declaration_converts(self, source, expected_type, parameter_name, initial_value):
        dist, metadata = ast_to_distribution_with_metadata(parse_distribution(source))
        assert isinstance(dist, expected_type)
        assert metadata["parameter_name"] == parameter_name
        assert metadata["initial_value"] == initial_value


class TestDistributionsFromCalibration:
    def test_extracts_distributions_only(self):
        calibration = [
            GCNDistribution(
                parameter_name="alpha",
                dist_name="Beta",
                dist_kwargs={"alpha": 2, "beta": 5},
                initial_value=0.35,
            ),
            GCNEquation(lhs=Parameter(name="delta"), rhs=Number(value=0.025)),
            GCNDistribution(
                parameter_name="sigma",
                dist_name="HalfNormal",
                dist_kwargs={"sigma": 5},
                initial_value=1.0,
            ),
        ]

        result = distributions_from_calibration(calibration)

        assert set(result) == {"alpha", "sigma"}

    @pytest.mark.parametrize(
        "calibration",
        [
            [],
            [
                GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35)),
                GCNEquation(lhs=Parameter(name="beta"), rhs=Number(value=0.99)),
            ],
        ],
    )
    def test_no_distributions_gives_empty_dict(self, calibration):
        assert distributions_from_calibration(calibration) == {}


class TestDistributionsFromModel:
    def test_extracts_from_all_blocks(self):
        model = GCNModel(
            blocks=[
                GCNBlock(
                    name="HOUSEHOLD",
                    calibration=[
                        GCNDistribution(
                            parameter_name="beta",
                            dist_name="Beta",
                            dist_kwargs={"alpha": 2, "beta": 5},
                            initial_value=0.99,
                        )
                    ],
                ),
                GCNBlock(
                    name="FIRM",
                    calibration=[
                        GCNDistribution(
                            parameter_name="alpha",
                            dist_name="Beta",
                            dist_kwargs={"alpha": 5, "beta": 5},
                            initial_value=0.35,
                        )
                    ],
                ),
            ]
        )

        result = distributions_from_model(model)

        assert set(result) == {"beta", "alpha"}

    def test_later_block_overrides_earlier_declaration(self):
        model = GCNModel(
            blocks=[
                GCNBlock(
                    name="FIRST",
                    calibration=[GCNDistribution(parameter_name="alpha", dist_name="Beta", dist_kwargs={})],
                ),
                GCNBlock(
                    name="SECOND",
                    calibration=[GCNDistribution(parameter_name="alpha", dist_name="Gamma", dist_kwargs={})],
                ),
            ]
        )

        dist, _metadata = distributions_from_model(model)["alpha"]
        assert isinstance(dist, pz.Gamma)
