import pytest

from gEconpy.parser.ast import GCNModel
from gEconpy.parser.errors import GCNGrammarError, GCNSemanticError
from gEconpy.parser.preprocessor import (
    ParseResult,
    preprocess,
    preprocess_file,
    quick_parse,
)
from tests.conftest import TEST_GCNS


class TestQuickParse:
    def test_empty_raises_error(self):
        with pytest.raises(GCNGrammarError):
            quick_parse("")

    def test_simple_block(self):
        model = quick_parse("block HOUSEHOLD { identities { Y[] = C[]; }; };")
        assert len(model.blocks) == 1
        assert model.blocks[0].name == "HOUSEHOLD"

    def test_with_options(self):
        model = quick_parse("options { output logfile = TRUE; }; block TEST { identities { X[] = 1; }; };")
        assert model.options["output logfile"] is True


class TestPreprocess:
    def test_returns_parse_result_with_source(self):
        source = "block TEST { identities { X[] = 1; }; };"
        result = preprocess(source)
        assert isinstance(result, ParseResult)
        assert isinstance(result.ast, GCNModel)
        assert result.source == source

    @pytest.mark.parametrize("filename", [None, "test.gcn"])
    def test_filename_preserved(self, filename):
        result = preprocess("block TEST { };", filename=filename)
        assert result.filename == filename

    def test_validation_runs_by_default(self):
        result = preprocess("block TEST { controls { C[]; }; };")
        assert len(list(result.validation_errors)) > 0

    def test_validation_can_be_disabled(self):
        result = preprocess("block TEST { controls { C[]; }; };", validate=False)
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

    def test_ast_accessors(self, simple_model):
        assert [block.name for block in simple_model.blocks] == ["HOUSEHOLD"]
        assert simple_model.options == {}
        assert simple_model.tryreduce == []
        assert simple_model.assumptions == {}

    def test_sympy_equations_computed_on_first_access(self, simple_model):
        assert simple_model._sympy_equations is None
        equations = simple_model.sympy_equations
        assert simple_model._sympy_equations is equations
        assert "identities" in equations["HOUSEHOLD"]

    def test_distributions_computed_on_first_access(self, simple_model):
        assert simple_model._distributions is None
        distributions = simple_model.distributions
        assert simple_model._distributions is distributions
        assert distributions == {}

    def test_valid_model_has_no_errors(self, simple_model):
        assert not simple_model.has_errors
        assert not simple_model.validate(raise_on_error=False).has_errors


class TestPreprocessWithDistributions:
    def test_distributions_extracted(self):
        source = "block TEST { calibration { alpha ~ Beta(alpha=2, beta=5) = 0.35; beta = 0.99; }; };"
        assert set(preprocess(source).distributions) == {"alpha"}

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
        assert set(preprocess(source).distributions) == {"alpha", "delta"}


def test_preprocess_full_model():
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
    assert result.tryreduce == ["U"]
    assert set(result.assumptions) == {"C", "K"}
    assert len(result.blocks) == 1

    block = result.blocks[0]
    assert len(block.definitions) == 1
    assert len(block.controls) == 2
    assert len(block.objective) == 1
    assert len(block.constraints) == 1
    assert len(block.identities) == 2
    assert len(block.shocks) == 1
    assert len(block.calibration) == 3


@pytest.mark.parametrize("filename", ["one_block_1.gcn", "basic_rbc.gcn"])
def test_preprocess_file(filename):
    gcn_path = TEST_GCNS / filename
    result = preprocess_file(gcn_path)
    assert isinstance(result, ParseResult)
    assert len(result.blocks) >= 1
    assert result.filename == str(gcn_path)


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
        assert preprocess(source).has_errors

    def test_validate_raises_on_error(self):
        result = preprocess("block TEST { }; block TEST { };", validate=False)
        with pytest.raises(GCNSemanticError):
            result.validate(raise_on_error=True)


class TestEdgeCases:
    @pytest.mark.parametrize(
        "source",
        ["", "\n        # This is a comment\n        # Another comment\n        ", "   \n\n\t\t   \n   "],
        ids=["empty", "only_comments", "whitespace_only"],
    )
    def test_source_without_blocks_raises(self, source):
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
