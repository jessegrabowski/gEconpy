from pathlib import Path

import pytest

from gEconpy.parser.adapter import (
    block_to_legacy_dict,
    compare_parser_outputs,
    extract_prior_dict,
    model_to_legacy_block_dict,
    parse_gcn_legacy_format,
)
from gEconpy.parser.preprocessor import quick_parse


class TestBlockToLegacyDict:
    def test_identities_converted(self):
        source = """
        block TEST
        {
            identities
            {
                Y[] = C[] + I[];
            };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = block_to_legacy_dict(block)

        assert "identities" in result
        assert len(result["identities"]) == 1
        tokens = result["identities"][0]
        assert "Y[]" in tokens
        assert "=" in tokens
        assert "C[]" in tokens

    def test_calibration_converted(self):
        source = """
        block TEST
        {
            calibration
            {
                alpha = 0.35;
                beta = 0.99;
            };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = block_to_legacy_dict(block)

        assert "calibration" in result
        assert len(result["calibration"]) == 2

    def test_controls_converted(self):
        source = """
        block TEST
        {
            controls { C[], K[]; };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = block_to_legacy_dict(block)

        assert "controls" in result
        assert len(result["controls"]) == 1  # One list of controls
        assert "C[]" in result["controls"][0]
        assert "K[]" in result["controls"][0]

    def test_objective_converted(self):
        source = """
        block TEST
        {
            objective { U[] = log(C[]); };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = block_to_legacy_dict(block)

        assert "objective" in result
        assert len(result["objective"]) == 1

    def test_constraints_with_lagrange(self):
        source = """
        block TEST
        {
            constraints { C[] = Y[] : lambda[]; };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = block_to_legacy_dict(block)

        assert "constraints" in result
        tokens = result["constraints"][0]
        assert ":" in tokens
        assert "lambda[]" in tokens


class TestModelToLegacyBlockDict:
    def test_multiple_blocks(self):
        source = """
        block HOUSEHOLD { identities { C[] = Y[]; }; };
        block FIRM { identities { Y[] = A[] * K[-1]; }; };
        """
        model = quick_parse(source)
        result = model_to_legacy_block_dict(model)

        assert "HOUSEHOLD" in result
        assert "FIRM" in result


class TestExtractPriorDict:
    def test_extracts_distributions(self):
        source = """
        block TEST
        {
            calibration
            {
                alpha ~ Beta(alpha=2, beta=5) = 0.35;
            };
        };
        """
        model = quick_parse(source)
        result = extract_prior_dict(model)

        assert "alpha" in result
        assert "Beta" in result["alpha"]

    def test_extracts_wrapped_distributions(self):
        source = """
        block TEST
        {
            calibration
            {
                beta ~ maxent(Beta(), lower=0.95, upper=0.999) = 0.99;
            };
        };
        """
        model = quick_parse(source)
        result = extract_prior_dict(model)

        assert "beta" in result
        assert "maxent" in result["beta"]


class TestParseGcnLegacyFormat:
    def test_returns_tuple_of_correct_types(self):
        source = """
        block TEST { identities { X[] = 1; }; };
        """
        blocks, options, tryreduce, assumptions, prior_dict = parse_gcn_legacy_format(source)

        assert isinstance(blocks, dict)
        assert isinstance(options, dict)
        assert isinstance(tryreduce, list)
        assert isinstance(assumptions, dict)
        assert isinstance(prior_dict, dict)

    def test_extracts_options(self):
        source = """
        options { output logfile = TRUE; };
        block TEST { };
        """
        _blocks, options, _tryreduce, _assumptions, _prior_dict = parse_gcn_legacy_format(source)

        assert options.get("output logfile") is True

    def test_extracts_tryreduce(self):
        source = """
        tryreduce { U[]; };
        block TEST { };
        """
        _blocks, _options, tryreduce, _assumptions, _prior_dict = parse_gcn_legacy_format(source)

        assert "U[]" in tryreduce

    def test_extracts_block_names(self):
        source = """
        block HOUSEHOLD { };
        block FIRM { };
        """
        blocks, _options, _tryreduce, _assumptions, _prior_dict = parse_gcn_legacy_format(source)

        assert "HOUSEHOLD" in blocks
        assert "FIRM" in blocks


class TestCompareParserOutputs:
    def test_simple_model_no_differences(self):
        source = """
        block TEST
        {
            identities
            {
                Y[] = C[];
            };
        };
        """
        result = compare_parser_outputs(source)

        # Should not have errors
        assert "error" not in result["old"]
        assert "error" not in result["new"]

        # Block names should match
        assert "block_names" not in result["differences"]

    def test_with_options(self):
        source = """
        options { output logfile = TRUE; };
        block TEST { identities { X[] = 1; }; };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"]
        assert "error" not in result["new"]

    def test_with_tryreduce(self):
        source = """
        tryreduce { U[]; };
        block TEST { identities { U[] = log(C[]); }; };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"]
        assert "error" not in result["new"]


class TestIntegration:
    @pytest.fixture
    def gcn_dir(self):
        return Path(__file__).parent.parent / "_resources" / "test_gcns"

    def test_parse_one_block_1(self, gcn_dir):
        gcn_path = gcn_dir / "one_block_1.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        source = gcn_path.read_text()
        result = compare_parser_outputs(source)

        assert "error" not in result["old"]
        assert "error" not in result["new"]

    def test_parse_basic_rbc(self, gcn_dir):
        gcn_path = gcn_dir / "basic_rbc.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        source = gcn_path.read_text()
        result = compare_parser_outputs(source)

        assert "error" not in result["old"]
        assert "error" not in result["new"]

        # Block names should match
        old_blocks = set(result["old"]["blocks"].keys())
        new_blocks = set(result["new"]["blocks"].keys())
        assert old_blocks == new_blocks


class TestRealWorldFiles:
    @pytest.fixture
    def gcn_files_dir(self):
        return Path(__file__).parent.parent.parent / "GCN Files"

    def test_rbc_gcn(self, gcn_files_dir):
        gcn_path = gcn_files_dir / "RBC.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        source = gcn_path.read_text()
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser error: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser error: {result['new'].get('error')}"

        # Block names should match
        old_blocks = set(result["old"]["blocks"].keys())
        new_blocks = set(result["new"]["blocks"].keys())
        assert old_blocks == new_blocks, f"Block mismatch: old={old_blocks}, new={new_blocks}"
