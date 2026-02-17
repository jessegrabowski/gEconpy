from pathlib import Path

import pytest

from gEconpy.parser.ast import GCNModel
from gEconpy.parser.errors import GCNGrammarError, GCNSemanticError
from gEconpy.parser.preprocessor import (
    ParseResult,
    preprocess,
    preprocess_file,
    quick_parse,
)


class TestQuickParse:
    def test_empty_raises_error(self):
        with pytest.raises(GCNGrammarError):
            quick_parse("")

    def test_simple_block(self):
        source = """
        block HOUSEHOLD
        {
            identities { Y[] = C[]; };
        };
        """
        result = quick_parse(source)
        assert len(result.blocks) == 1
        assert result.blocks[0].name == "HOUSEHOLD"

    def test_with_options(self):
        source = """
        options { output logfile = TRUE; };
        block TEST { identities { X[] = 1; }; };
        """
        result = quick_parse(source)
        assert result.options["output logfile"] is True


class TestPreprocess:
    def test_returns_parse_result(self):
        source = "block TEST { identities { X[] = 1; }; };"
        result = preprocess(source)
        assert isinstance(result, ParseResult)
        assert isinstance(result.ast, GCNModel)

    def test_source_preserved(self):
        source = "block TEST { identities { X[] = 1; }; };"
        result = preprocess(source)
        assert result.source == source

    def test_filename_preserved(self):
        source = "block TEST { };"
        result = preprocess(source, filename="test.gcn")
        assert result.filename == "test.gcn"

    def test_validation_runs_by_default(self):
        source = "block TEST { controls { C[]; }; };"  # Has controls but no objective
        result = preprocess(source)
        # Should have a warning about controls without objective
        assert len(list(result.validation_errors)) > 0

    def test_validation_can_be_disabled(self):
        source = "block TEST { controls { C[]; }; };"
        result = preprocess(source, validate=False)
        # _validation_errors should still be None
        assert result._validation_errors is None


class TestParseResult:
    @pytest.fixture
    def simple_model(self):
        source = """
        block HOUSEHOLD
        {
            controls { C[], K[]; };
            objective { U[] = log(C[]); };
            constraints { C[] = Y[] : lambda[]; };
            identities { Y[] = A[] * K[-1] ^ alpha; };
            calibration { alpha = 0.35; };
        };
        """
        return preprocess(source)

    def test_blocks_accessor(self, simple_model):
        assert len(simple_model.blocks) == 1
        assert simple_model.blocks[0].name == "HOUSEHOLD"

    def test_options_accessor(self, simple_model):
        assert simple_model.options == {}

    def test_tryreduce_accessor(self, simple_model):
        assert simple_model.tryreduce == []

    def test_assumptions_accessor(self, simple_model):
        assert isinstance(simple_model.assumptions, dict)

    def test_sympy_equations_lazy(self, simple_model):
        assert simple_model._sympy_equations is None
        _ = simple_model.sympy_equations
        assert simple_model._sympy_equations is not None

    def test_sympy_equations_content(self, simple_model):
        eqs = simple_model.sympy_equations
        assert "HOUSEHOLD" in eqs
        assert "identities" in eqs["HOUSEHOLD"]

    def test_distributions_lazy(self, simple_model):
        assert simple_model._distributions is None
        _ = simple_model.distributions
        assert simple_model._distributions is not None

    def test_has_errors_false_for_valid(self, simple_model):
        # May have warnings but not errors
        assert not simple_model.has_errors

    def test_validate_returns_errors(self, simple_model):
        errors = simple_model.validate(raise_on_error=False)
        assert errors is not None


class TestPreprocessWithDistributions:
    def test_distributions_extracted(self):
        source = """
        block TEST
        {
            calibration
            {
                alpha ~ Beta(alpha=2, beta=5) = 0.35;
                beta = 0.99;
            };
        };
        """
        result = preprocess(source)
        dists = result.distributions
        assert "alpha" in dists
        assert "beta" not in dists  # Not a distribution

    def test_multiple_distributions(self):
        source = """
        block TEST
        {
            calibration
            {
                alpha ~ Beta(alpha=2, beta=5) = 0.35;
                delta ~ Gamma(alpha=2, beta=1) = 0.025;
            };
        };
        """
        result = preprocess(source)
        dists = result.distributions
        assert len(dists) == 2


class TestPreprocessWithAllFeatures:
    def test_full_model(self):
        source = """
        options
        {
            output logfile = TRUE;
        };

        tryreduce
        {
            U[];
        };

        assumptions
        {
            positive { C[], K[]; };
        };

        block HOUSEHOLD
        {
            definitions
            {
                u[] = log(C[]);
            };

            controls
            {
                C[], K[];
            };

            objective
            {
                U[] = u[] + beta * E[][U[1]];
            };

            constraints
            {
                C[] + K[] = Y[] : lambda[];
            };

            identities
            {
                Y[] = A[] * K[-1] ^ alpha;
                log(A[]) = rho * log(A[-1]) + epsilon[];
            };

            shocks
            {
                epsilon[];
            };

            calibration
            {
                alpha = 0.35;
                beta = 0.99;
                rho = 0.95;
            };
        };
        """
        result = preprocess(source)

        assert result.options["output logfile"] is True
        assert "U" in result.tryreduce
        assert "C" in result.assumptions
        assert len(result.blocks) == 1

        block = result.blocks[0]
        assert len(block.definitions) == 1
        assert len(block.controls) == 2
        assert block.objective is not None
        assert len(block.constraints) == 1
        assert len(block.identities) == 2
        assert len(block.shocks) == 1
        assert len(block.calibration) == 3


class TestPreprocessFile:
    @pytest.fixture
    def gcn_dir(self):
        return Path(__file__).parent.parent / "_resources" / "test_gcns"

    def test_parse_existing_file(self, gcn_dir):
        gcn_path = gcn_dir / "one_block_1.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        result = preprocess_file(gcn_path)
        assert isinstance(result, ParseResult)
        assert len(result.blocks) >= 1
        assert result.filename == str(gcn_path)

    def test_parse_basic_rbc(self, gcn_dir):
        gcn_path = gcn_dir / "basic_rbc.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        result = preprocess_file(gcn_path)
        assert len(result.blocks) >= 1


class TestValidation:
    def test_duplicate_block_caught(self):
        source = """
        block TEST { identities { X[] = 1; }; };
        block TEST { identities { Y[] = 2; }; };
        """
        result = preprocess(source)
        assert result.has_errors
        assert any("Duplicate block" in str(e) for e in result.validation_errors)

    def test_duplicate_param_across_blocks_caught(self):
        source = """
        block A { calibration { alpha = 0.3; }; };
        block B { calibration { alpha = 0.4; }; };
        """
        result = preprocess(source)
        assert result.has_errors

    def test_validate_raises_on_error(self):
        source = """
        block TEST { };
        block TEST { };
        """
        result = preprocess(source, validate=False)
        with pytest.raises(GCNSemanticError):
            result.validate(raise_on_error=True)


class TestEdgeCases:
    def test_empty_source_raises(self):
        with pytest.raises(GCNGrammarError):
            preprocess("")

    def test_only_comments_raises(self):
        source = """
        # This is a comment
        # Another comment
        """
        with pytest.raises(GCNGrammarError):
            preprocess(source)

    def test_whitespace_only_raises(self):
        source = "   \n\n\t\t   \n   "
        with pytest.raises(GCNGrammarError):
            preprocess(source)

    def test_multiline_equation(self):
        source = """
        block TEST
        {
            calibration
            {
                Y_ss = (R_ss / (R_ss - delta * alpha)) ^ (sigma / (sigma + phi)) *
                       ((1 - alpha) ^ (-phi) * (W_ss) ^ (1 + phi)) ^ (1 / (sigma + phi));
            };
        };
        """
        result = preprocess(source)
        assert len(result.blocks) == 1
        assert len(result.blocks[0].calibration) == 1
