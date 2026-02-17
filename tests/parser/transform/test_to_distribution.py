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


class TestAstToDistributionBasic:
    def test_normal_distribution(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="Normal",
            dist_kwargs={"mu": 0, "sigma": 1},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Normal)
        assert dist.mu == 0
        assert dist.sigma == 1

    def test_beta_distribution(self):
        node = GCNDistribution(
            parameter_name="alpha",
            dist_name="Beta",
            dist_kwargs={"alpha": 2, "beta": 5},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Beta)

    def test_gamma_distribution(self):
        node = GCNDistribution(
            parameter_name="tau",
            dist_name="Gamma",
            dist_kwargs={"alpha": 2, "beta": 1},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Gamma)

    def test_exponential_distribution(self):
        node = GCNDistribution(
            parameter_name="lam",
            dist_name="Exponential",
            dist_kwargs={"lam": 0.01},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Exponential)

    def test_half_normal_distribution(self):
        node = GCNDistribution(
            parameter_name="sigma",
            dist_name="HalfNormal",
            dist_kwargs={"sigma": 5},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.HalfNormal)

    def test_uniform_distribution(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="Uniform",
            dist_kwargs={"lower": 0, "upper": 1},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Uniform)

    def test_student_t_distribution(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="StudentT",
            dist_kwargs={"nu": 7},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.StudentT)

    def test_empty_kwargs(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="Normal",
            dist_kwargs={},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Normal)


class TestWrappedDistributions:
    def test_maxent_wrapper(self):
        node = GCNDistribution(
            parameter_name="beta",
            dist_name="Beta",
            dist_kwargs={},
            wrapper_name="maxent",
            wrapper_kwargs={"lower": 0.95, "upper": 0.999, "mass": 0.99},
        )
        dist = ast_to_distribution(node)
        # maxent returns a Beta with fitted parameters
        assert isinstance(dist, pz.Beta)

    def test_truncated_wrapper(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="Normal",
            dist_kwargs={},
            wrapper_name="Truncated",
            wrapper_kwargs={"lower": 0, "upper": 5},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Truncated)

    def test_truncated_one_bound(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="Normal",
            dist_kwargs={},
            wrapper_name="Truncated",
            wrapper_kwargs={"lower": 0},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Truncated)

    def test_censored_wrapper(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="Beta",
            dist_kwargs={"alpha": 2, "beta": 5},
            wrapper_name="Censored",
            wrapper_kwargs={"lower": 0.1, "upper": 0.9},
        )
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Censored)


class TestDistributionWithMetadata:
    def test_metadata_extraction(self):
        node = GCNDistribution(
            parameter_name="alpha",
            dist_name="Beta",
            dist_kwargs={"alpha": 2, "beta": 5},
            initial_value=0.35,
        )
        _dist, metadata = ast_to_distribution_with_metadata(node)

        assert metadata["parameter_name"] == "alpha"
        assert metadata["initial_value"] == 0.35
        assert metadata["is_wrapped"] is False
        assert metadata["wrapper_name"] is None

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

        assert metadata["is_wrapped"] is True
        assert metadata["wrapper_name"] == "maxent"
        assert metadata["initial_value"] == 0.99


class TestFromParsedStrings:
    """Test conversion from parsed distribution strings."""

    def test_normal_from_string(self):
        node = parse_distribution("x ~ Normal(mu=0, sigma=1);")
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.Normal)

    def test_beta_with_initial_value(self):
        node = parse_distribution("alpha ~ Beta(alpha=2, beta=5) = 0.35;")
        dist, metadata = ast_to_distribution_with_metadata(node)
        assert isinstance(dist, pz.Beta)
        assert metadata["initial_value"] == 0.35

    def test_maxent_from_string(self):
        node = parse_distribution("beta ~ maxent(Beta(), lower=0.95, upper=0.999) = 0.99;")
        dist, metadata = ast_to_distribution_with_metadata(node)
        assert isinstance(dist, pz.Beta)
        assert metadata["initial_value"] == 0.99

    def test_gamma_from_string(self):
        node = parse_distribution("tau ~ Gamma(alpha=2, beta=1) = 2.1;")
        dist, metadata = ast_to_distribution_with_metadata(node)
        assert isinstance(dist, pz.Gamma)
        assert metadata["initial_value"] == 2.1

    def test_half_normal_from_string(self):
        node = parse_distribution("sigma ~ HalfNormal(sigma=5) = 1.0;")
        dist = ast_to_distribution(node)
        assert isinstance(dist, pz.HalfNormal)


class TestDistributionsFromCalibration:
    def test_extracts_distributions_only(self):
        calibration = [
            GCNDistribution(
                parameter_name="alpha",
                dist_name="Beta",
                dist_kwargs={"alpha": 2, "beta": 5},
                initial_value=0.35,
            ),
            GCNEquation(
                lhs=Parameter(name="delta"),
                rhs=Number(value=0.025),
            ),
            GCNDistribution(
                parameter_name="sigma",
                dist_name="HalfNormal",
                dist_kwargs={"sigma": 5},
                initial_value=1.0,
            ),
        ]

        result = distributions_from_calibration(calibration)

        assert len(result) == 2
        assert "alpha" in result
        assert "sigma" in result
        assert "delta" not in result

    def test_empty_calibration(self):
        result = distributions_from_calibration([])
        assert result == {}

    def test_no_distributions(self):
        calibration = [
            GCNEquation(lhs=Parameter(name="alpha"), rhs=Number(value=0.35)),
            GCNEquation(lhs=Parameter(name="beta"), rhs=Number(value=0.99)),
        ]
        result = distributions_from_calibration(calibration)
        assert result == {}


class TestDistributionsFromModel:
    def test_extracts_from_all_blocks(self):
        block1 = GCNBlock(name="HOUSEHOLD")
        block1.calibration = [
            GCNDistribution(
                parameter_name="beta",
                dist_name="Beta",
                dist_kwargs={"alpha": 2, "beta": 5},
                initial_value=0.99,
            ),
        ]

        block2 = GCNBlock(name="FIRM")
        block2.calibration = [
            GCNDistribution(
                parameter_name="alpha",
                dist_name="Beta",
                dist_kwargs={"alpha": 5, "beta": 5},
                initial_value=0.35,
            ),
        ]

        model = GCNModel(
            blocks=[block1, block2],
            options={},
            tryreduce=[],
            assumptions={},
        )

        result = distributions_from_model(model)

        assert len(result) == 2
        assert "beta" in result
        assert "alpha" in result


class TestErrorCases:
    def test_invalid_distribution_name_raises(self):
        node = GCNDistribution(
            parameter_name="x",
            dist_name="NotARealDistribution",
            dist_kwargs={},
        )
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


class TestRealWorldExamples:
    """Test patterns from actual GCN files."""

    def test_rbc_beta(self):
        node = parse_distribution("beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99;")
        _dist, metadata = ast_to_distribution_with_metadata(node)
        assert metadata["parameter_name"] == "beta"
        assert metadata["initial_value"] == 0.99

    def test_rbc_delta(self):
        node = parse_distribution("delta ~ maxent(Beta(), lower=0.01, upper=0.05, mass=0.99) = 0.02;")
        _dist, metadata = ast_to_distribution_with_metadata(node)
        assert metadata["parameter_name"] == "delta"

    def test_rbc_sigma_c(self):
        node = parse_distribution("sigma_C ~ maxent(Gamma(), lower=1.01, upper=10.0, mass=0.99) = 1.5;")
        _dist, metadata = ast_to_distribution_with_metadata(node)
        assert metadata["parameter_name"] == "sigma_C"

    def test_open_rbc_alpha(self):
        node = parse_distribution("alpha ~ Beta(alpha=5, beta=5) = 0.32;")
        dist, metadata = ast_to_distribution_with_metadata(node)
        assert isinstance(dist, pz.Beta)
        assert metadata["initial_value"] == 0.32

    def test_rho_parameter(self):
        node = parse_distribution("rho_A ~ Beta(alpha=3, beta=1) = 0.42;")
        _dist, metadata = ast_to_distribution_with_metadata(node)
        assert metadata["parameter_name"] == "rho_A"
