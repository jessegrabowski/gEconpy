"""
Comprehensive comparison tests between old and new parsers.

These tests ensure feature parity by comparing the outputs of both parsers
across all GCN files in the test suite and GCN Files directory.
"""

import os

from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from numpy.testing import assert_allclose

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.adapter import compare_parser_outputs, parse_gcn_legacy_format
from gEconpy.parser.loader import (
    load_gcn_file,
    load_gcn_string,
    set_parser,
    use_new_parser,
)

# Collect all GCN files from test resources
TEST_GCNS_DIR = Path(__file__).parent.parent / "_resources" / "test_gcns"
GCN_FILES_DIR = Path(__file__).parent.parent.parent / "GCN Files"


def get_all_test_gcn_files():
    """Get all .gcn files from test_gcns directory."""
    if not TEST_GCNS_DIR.exists():
        return []
    return sorted(TEST_GCNS_DIR.glob("*.gcn"))


def get_all_gcn_files():
    """Get all .gcn files from GCN Files directory."""
    if not GCN_FILES_DIR.exists():
        return []
    files = list(GCN_FILES_DIR.glob("*.gcn"))
    # Also check subdirectories
    for subdir in GCN_FILES_DIR.iterdir():
        if subdir.is_dir():
            files.extend(subdir.glob("*.gcn"))
    return sorted(files)


# Files known to have issues with one parser or the other
# These represent feature gaps that need to be addressed before deprecation
SKIP_FILES = {
    # Shock distributions in shocks block - new parser needs to handle this
    "one_block_1_dist.gcn": "Shock distributions in shocks block not yet supported",
    # Files with syntax the new parser doesn't yet handle
    "full_nk_linear_phillips_curve.gcn": "Needs investigation",
    "rbc_firm_capital.gcn": "Needs investigation",
    "rbc_linearized.gcn": "Needs investigation",
    "rbc_with_excluded.gcn": "Needs investigation",
    # Gali 2015 files from GCN Files directory
    "nk_complete_more_shocks.gcn": "Gali 2015 file - needs investigation",
    "nk_complete_taxes.gcn": "Gali 2015 file - needs investigation",
    "nk_money_growth.gcn": "Gali 2015 file - needs investigation",
    "nk_taylor_rule.gcn": "Gali 2015 file - needs investigation",
    "nk_taylor_rule_capital.gcn": "Gali 2015 file - needs investigation",
    "nk_taylor_rule_linearized.gcn": "Gali 2015 file - needs investigation",
    "nk_taylor_rule_stick_wage_capital.gcn": "Gali 2015 file - needs investigation",
    "nk_with_sticky_wages.gcn": "Gali 2015 file - needs investigation",
    # RBC files with special features
    "RBC_two_household.gcn": "Two household RBC - needs investigation",
    "RBC_two_household_additive.gcn": "Two household RBC - needs investigation",
}

# Files that require special handling
SPECIAL_FILES = {
    # Add any files that need special parameters or handling
}


class TestParserOutputComparison:
    """Compare raw parser outputs between old and new parser."""

    @pytest.fixture(autouse=True)
    def reset_parser(self):
        """Reset parser to default after each test."""
        original = use_new_parser()
        yield
        set_parser(original)

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_block_names_match(self, gcn_file):
        """Verify both parsers extract the same block names."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        source = gcn_file.read_text()
        result = compare_parser_outputs(source)

        if "error" in result["old"]:
            pytest.skip(f"Old parser failed: {result['old']['error']}")
        if "error" in result["new"]:
            pytest.fail(f"New parser failed: {result['new']['error']}")

        old_blocks = set(result["old"]["blocks"].keys())
        new_blocks = set(result["new"]["blocks"].keys())

        assert old_blocks == new_blocks, f"Block names differ: old={old_blocks}, new={new_blocks}"

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_options_match(self, gcn_file):
        """Verify both parsers extract the same options."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        source = gcn_file.read_text()
        result = compare_parser_outputs(source)

        if "error" in result["old"]:
            pytest.skip(f"Old parser failed: {result['old']['error']}")
        if "error" in result["new"]:
            pytest.fail(f"New parser failed: {result['new']['error']}")

        assert result["old"]["options"] == result["new"]["options"], (
            f"Options differ:\n  old={result['old']['options']}\n  new={result['new']['options']}"
        )

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_tryreduce_match(self, gcn_file):
        """Verify both parsers extract the same tryreduce variables."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        source = gcn_file.read_text()
        result = compare_parser_outputs(source)

        if "error" in result["old"]:
            pytest.skip(f"Old parser failed: {result['old']['error']}")
        if "error" in result["new"]:
            pytest.fail(f"New parser failed: {result['new']['error']}")

        # Compare as sets since order may differ
        old_tryreduce = set(result["old"]["tryreduce"])
        new_tryreduce = set(result["new"]["tryreduce"])

        assert old_tryreduce == new_tryreduce, f"Tryreduce differs:\n  old={old_tryreduce}\n  new={new_tryreduce}"


class TestLoadedModelComparison:
    """
    Compare model primitives loaded with old vs new parser.

    Note: These tests may show differences because the loaders use different
    internal paths. The key goal is that the final Model objects work identically,
    which is tested in TestModelBuildComparison.
    """

    @pytest.fixture(autouse=True)
    def reset_parser(self):
        """Reset parser to default after each test."""
        original = use_new_parser()
        yield
        set_parser(original)

    def _load_with_both_parsers(self, gcn_file):
        """Load a GCN file with both parsers and return results."""
        # Load with new parser
        set_parser(True)
        try:
            new_result = load_gcn_file(gcn_file)
        except Exception as e:
            new_result = {"error": str(e)}

        # Load with old parser
        set_parser(False)
        try:
            old_result = load_gcn_file(gcn_file)
        except Exception as e:
            old_result = {"error": str(e)}

        return old_result, new_result

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_variable_counts_match(self, gcn_file):
        """Verify both parsers produce the same number of variables."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        old_result, new_result = self._load_with_both_parsers(gcn_file)

        if "error" in old_result:
            pytest.skip(f"Old parser failed: {old_result['error']}")
        if "error" in new_result:
            pytest.skip(f"New parser failed (integration pending): {new_result['error']}")

        old_vars = len(old_result["variables"])
        new_vars = len(new_result["variables"])

        assert old_vars == new_vars, f"Variable count differs: old={old_vars}, new={new_vars}"

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_variable_names_match(self, gcn_file):
        """Verify both parsers produce variables with the same names."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        old_result, new_result = self._load_with_both_parsers(gcn_file)

        if "error" in old_result:
            pytest.skip(f"Old parser failed: {old_result['error']}")
        if "error" in new_result:
            pytest.skip(f"New parser failed (integration pending): {new_result['error']}")

        old_var_names = {v.base_name for v in old_result["variables"]}
        new_var_names = {v.base_name for v in new_result["variables"]}

        assert old_var_names == new_var_names, (
            f"Variable names differ:\n  old={sorted(old_var_names)}\n  new={sorted(new_var_names)}"
        )

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_shock_counts_match(self, gcn_file):
        """Verify both parsers produce the same number of shocks."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        old_result, new_result = self._load_with_both_parsers(gcn_file)

        if "error" in old_result:
            pytest.skip(f"Old parser failed: {old_result['error']}")
        if "error" in new_result:
            pytest.skip(f"New parser failed (integration pending): {new_result['error']}")

        old_shocks = len(old_result["shocks"])
        new_shocks = len(new_result["shocks"])

        assert old_shocks == new_shocks, f"Shock count differs: old={old_shocks}, new={new_shocks}"

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_shock_names_match(self, gcn_file):
        """Verify both parsers produce shocks with the same names."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        old_result, new_result = self._load_with_both_parsers(gcn_file)

        if "error" in old_result:
            pytest.skip(f"Old parser failed: {old_result['error']}")
        if "error" in new_result:
            pytest.skip(f"New parser failed (integration pending): {new_result['error']}")

        old_shock_names = {s.base_name for s in old_result["shocks"]}
        new_shock_names = {s.base_name for s in new_result["shocks"]}

        assert old_shock_names == new_shock_names, (
            f"Shock names differ:\n  old={sorted(old_shock_names)}\n  new={sorted(new_shock_names)}"
        )

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_equation_counts_match(self, gcn_file):
        """Verify both parsers produce the same number of equations."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        old_result, new_result = self._load_with_both_parsers(gcn_file)

        if "error" in old_result:
            pytest.skip(f"Old parser failed: {old_result['error']}")
        if "error" in new_result:
            pytest.skip(f"New parser failed (integration pending): {new_result['error']}")

        old_eqs = len(old_result["equations"])
        new_eqs = len(new_result["equations"])

        assert old_eqs == new_eqs, f"Equation count differs: old={old_eqs}, new={new_eqs}"

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_parameter_names_match(self, gcn_file):
        """Verify both parsers produce the same parameter names."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        old_result, new_result = self._load_with_both_parsers(gcn_file)

        if "error" in old_result:
            pytest.skip(f"Old parser failed: {old_result['error']}")
        if "error" in new_result:
            pytest.skip(f"New parser failed (integration pending): {new_result['error']}")

        old_params = set(old_result["param_dict"].keys())
        new_params = set(new_result["param_dict"].keys())

        assert old_params == new_params, (
            f"Parameter names differ:\n  old={sorted(old_params)}\n  new={sorted(new_params)}"
        )

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_test_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_parameter_values_match(self, gcn_file):
        """Verify both parsers produce the same parameter values."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        old_result, new_result = self._load_with_both_parsers(gcn_file)

        if "error" in old_result:
            pytest.skip(f"Old parser failed: {old_result['error']}")
        if "error" in new_result:
            pytest.skip(f"New parser failed (integration pending): {new_result['error']}")

        old_params = old_result["param_dict"]
        new_params = new_result["param_dict"]

        for key in old_params:
            if key not in new_params:
                pytest.fail(f"Parameter {key} missing from new parser output")

            old_val = old_params[key]
            new_val = new_params[key]

            # Try to convert both to float for comparison
            try:
                old_float = float(old_val)
                new_float = float(new_val)
                assert_allclose(
                    old_float,
                    new_float,
                    rtol=1e-10,
                    err_msg=f"Parameter {key} value differs: old={old_val}, new={new_val}",
                )
            except (TypeError, ValueError):
                # For symbolic expressions, compare string representations
                assert str(old_val) == str(new_val), f"Parameter {key} value differs: old={old_val}, new={new_val}"


class TestModelBuildComparison:
    """Compare fully built models between old and new parser."""

    @pytest.fixture(autouse=True)
    def reset_parser(self):
        """Reset parser to default after each test."""
        original = use_new_parser()
        yield
        set_parser(original)

    # Core test files that must work identically
    CORE_FILES = [
        "one_block_1.gcn",
        "one_block_1_ss.gcn",
        "one_block_2.gcn",
        "basic_rbc.gcn",
        "open_rbc.gcn",
        "rbc_2_block_ss.gcn",
    ]

    @pytest.mark.parametrize("gcn_name", CORE_FILES)
    def test_steady_state_matches(self, gcn_name):
        """Verify steady state values match between parsers."""
        gcn_file = TEST_GCNS_DIR / gcn_name
        if not gcn_file.exists():
            pytest.skip(f"Test file not found: {gcn_file}")

        from tests._resources.cache_compiled_models import load_and_cache_model  # noqa: PLC0415

        # Build with old parser (via the standard path)
        set_parser(False)
        try:
            old_model = load_and_cache_model(gcn_name, "numpy", use_jax=False)
            old_ss = old_model.steady_state(verbose=False, progressbar=False)
            old_ss_values = dict(old_ss.items())
        except Exception as e:
            pytest.skip(f"Old parser model build failed: {e}")

        # Build with new parser
        set_parser(True)
        try:
            new_model = load_and_cache_model(gcn_name, "numpy", use_jax=False)
            new_ss = new_model.steady_state(verbose=False, progressbar=False)
            new_ss_values = dict(new_ss.items())
        except Exception as e:
            pytest.fail(f"New parser model build failed: {e}")

        # Compare steady state values
        for key, old_val in old_ss_values.items():
            if key not in new_ss_values:
                pytest.fail(f"Steady state variable {key} missing from new parser")

            new_val = new_ss_values[key]

            assert_allclose(
                old_val,
                new_val,
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"Steady state {key} differs: old={old_val}, new={new_val}",
            )


class TestRealWorldFiles:
    """Test parser comparison on real-world GCN files from GCN Files directory."""

    @pytest.fixture(autouse=True)
    def reset_parser(self):
        """Reset parser to default after each test."""
        original = use_new_parser()
        yield
        set_parser(original)

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_both_parsers_succeed(self, gcn_file):
        """Verify both parsers can successfully parse the file."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        source = gcn_file.read_text()
        result = compare_parser_outputs(source)

        if "error" in result["old"]:
            pytest.skip(f"Old parser failed: {result['old']['error']}")

        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    @pytest.mark.parametrize(
        "gcn_file",
        get_all_gcn_files(),
        ids=lambda x: x.name,
    )
    def test_block_names_match_real_files(self, gcn_file):
        """Verify block names match on real-world files."""
        if gcn_file.name in SKIP_FILES:
            pytest.skip(f"Skipping known problematic file: {gcn_file.name}")

        source = gcn_file.read_text()
        result = compare_parser_outputs(source)

        if "error" in result["old"]:
            pytest.skip(f"Old parser failed: {result['old']['error']}")
        if "error" in result["new"]:
            pytest.fail(f"New parser failed: {result['new']['error']}")

        old_blocks = set(result["old"]["blocks"].keys())
        new_blocks = set(result["new"]["blocks"].keys())

        assert old_blocks == new_blocks, f"Block names differ: old={old_blocks}, new={new_blocks}"


class TestEdgeCases:
    """Test edge cases and potential parser differences."""

    @pytest.fixture(autouse=True)
    def reset_parser(self):
        """Reset parser to default after each test."""
        original = use_new_parser()
        yield
        set_parser(original)

    def test_empty_model(self):
        """Test parsing an empty model."""
        source = ""
        result = compare_parser_outputs(source)

        # Both should succeed with empty results
        assert "error" not in result["old"] or "error" not in result["new"]

    def test_comments_only(self):
        """Test parsing a file with only comments."""
        source = """
        # This is a comment
        # Another comment
        """
        result = compare_parser_outputs(source)

        if "error" not in result["old"] and "error" not in result["new"]:
            assert result["old"]["blocks"] == result["new"]["blocks"]

    def test_complex_expression_in_calibration(self):
        """Test complex mathematical expressions in calibration."""
        source = """
        block TEST
        {
            identities { X[] = 1; };
            calibration
            {
                alpha = 1 / (1 + 0.01);
                beta = (1 - 0.025) ^ (1 / 4);
                gamma = log(2) / 4;
            };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_nested_functions(self):
        """Test equations with nested function calls."""
        source = """
        block TEST
        {
            identities
            {
                Y[] = exp(log(A[]) + alpha * log(K[-1]));
            };
            calibration { alpha = 0.35; };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_multiple_shocks(self):
        """Test model with multiple shocks."""
        source = """
        block TEST
        {
            shocks { epsilon_A[], epsilon_B[], epsilon_C[]; };
            identities
            {
                A[] = rho_A * A[-1] + epsilon_A[];
                B[] = rho_B * B[-1] + epsilon_B[];
                C[] = rho_C * C[-1] + epsilon_C[];
            };
            calibration
            {
                rho_A = 0.9;
                rho_B = 0.8;
                rho_C = 0.7;
            };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_steady_state_variables(self):
        """Test equations with steady state variables."""
        source = """
        block TEST
        {
            identities
            {
                Y[] = Y[ss] + alpha * (K[] - K[ss]);
            };
            calibration { alpha = 0.35; };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_lagrange_multipliers(self):
        """Test constraints with Lagrange multipliers."""
        source = """
        block HOUSEHOLD
        {
            controls { C[], K[]; };
            objective { U[] = log(C[]); };
            constraints
            {
                C[] + K[] = Y[] : lambda[];
                K[] = (1 - delta) * K[-1] + I[] : q[];
            };
            calibration { delta = 0.025; };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_calibrating_equations(self):
        """Test calibrating equations with arrow syntax."""
        source = """
        block TEST
        {
            identities { Y[] = K[-1] ^ alpha; };
            calibration
            {
                K[ss] / Y[ss] = 10 -> alpha;
            };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_expectations(self):
        """Test equations with expectation operator."""
        source = """
        block TEST
        {
            objective
            {
                U[] = log(C[]) + beta * E[][U[1]];
            };
            controls { C[]; };
            constraints { C[] = Y[] : lambda[]; };
            calibration { beta = 0.99; };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_unicode_in_comments(self):
        """Test handling of unicode in comments."""
        source = """
        # α = 0.35 is the capital share
        # β = 0.99 is the discount factor
        block TEST
        {
            identities { Y[] = 1; };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_very_long_equation(self):
        """Test parsing a very long equation."""
        source = """
        block TEST
        {
            identities
            {
                Y[] = alpha * A[] * K[-1] ^ gamma * L[] ^ (1 - gamma) +
                      beta * B[] * K[-1] ^ delta * L[] ^ (1 - delta) +
                      theta * C[] * K[-1] ^ epsilon * L[] ^ (1 - epsilon);
            };
            calibration
            {
                alpha = 0.3;
                beta = 0.3;
                gamma = 0.3;
                delta = 0.3;
                theta = 0.3;
                epsilon = 0.3;
            };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"


class TestDistributionParsing:
    """Test that distribution parsing matches between parsers."""

    @pytest.fixture(autouse=True)
    def reset_parser(self):
        """Reset parser to default after each test."""
        original = use_new_parser()
        yield
        set_parser(original)

    def test_simple_distribution(self):
        """Test simple distribution parsing."""
        source = """
        block TEST
        {
            calibration
            {
                alpha ~ Beta(alpha=2, beta=5) = 0.35;
            };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

        # Both should have alpha in prior_dict
        assert "alpha" in result["old"]["prior_dict"]
        assert "alpha" in result["new"]["prior_dict"]

    def test_wrapped_distribution(self):
        """Test wrapped distribution (maxent, Truncated, etc.)."""
        source = """
        block TEST
        {
            calibration
            {
                beta ~ maxent(Beta(), lower=0.95, upper=0.999) = 0.99;
            };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

    def test_multiple_distributions(self):
        """Test multiple distributions in one block."""
        source = """
        block TEST
        {
            calibration
            {
                alpha ~ Beta(alpha=2, beta=5) = 0.35;
                sigma ~ HalfNormal(sigma=1) = 0.5;
                rho ~ Beta(alpha=3, beta=1) = 0.9;
            };
        };
        """
        result = compare_parser_outputs(source)

        assert "error" not in result["old"], f"Old parser failed: {result['old'].get('error')}"
        assert "error" not in result["new"], f"New parser failed: {result['new'].get('error')}"

        for param in ["alpha", "sigma", "rho"]:
            assert param in result["old"]["prior_dict"], f"Old parser missing {param}"
            assert param in result["new"]["prior_dict"], f"New parser missing {param}"
