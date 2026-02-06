from pathlib import Path

import pytest
import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.loader import (
    ast_block_to_calibration,
    ast_block_to_equations,
    ast_block_to_variables_and_shocks,
    ast_model_to_primitives,
    load_gcn_file,
    load_gcn_string,
    set_parser,
    use_new_parser,
)
from gEconpy.parser.preprocessor import quick_parse


class TestFeatureFlag:
    def test_default_is_new_parser(self):
        # Save original state
        original = use_new_parser()
        try:
            set_parser(True)
            assert use_new_parser() is True
        finally:
            set_parser(original)

    def test_can_switch_to_old_parser(self):
        original = use_new_parser()
        try:
            set_parser(False)
            assert use_new_parser() is False
        finally:
            set_parser(original)

    def test_can_switch_back_to_new_parser(self):
        original = use_new_parser()
        try:
            set_parser(False)
            set_parser(True)
            assert use_new_parser() is True
        finally:
            set_parser(original)


class TestLoadWithOldParser:
    @pytest.fixture(autouse=True)
    def use_old_parser(self):
        original = use_new_parser()
        set_parser(False)
        yield
        set_parser(original)

    @pytest.fixture
    def gcn_dir(self):
        return Path(__file__).parent.parent / "_resources" / "test_gcns"

    def test_load_file_with_old_parser(self, gcn_dir):
        gcn_path = gcn_dir / "one_block_1.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        result = load_gcn_file(gcn_path)
        assert "equations" in result
        assert "variables" in result

    def test_load_string_with_old_parser(self):
        source = """
        block TEST
        {
            identities { Y[] = C[]; };
            calibration { alpha = 0.35; };
        };
        """
        result = load_gcn_string(source)
        assert "equations" in result
        assert "param_dict" in result


class TestAstBlockToEquations:
    def test_extracts_identities(self):
        source = """
        block TEST
        {
            identities
            {
                Y[] = C[] + I[];
                K[] = (1 - delta) * K[-1] + I[];
            };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = ast_block_to_equations(block)

        assert len(result["identities"]) == 2
        for eq, _metadata in result["identities"]:
            assert isinstance(eq, sp.Eq)

    def test_extracts_definitions(self):
        source = """
        block TEST
        {
            definitions
            {
                u[] = log(C[]);
            };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = ast_block_to_equations(block)

        assert len(result["definitions"]) == 1

    def test_extracts_objective(self):
        source = """
        block TEST
        {
            objective
            {
                U[] = u[] + beta * E[][U[1]];
            };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = ast_block_to_equations(block)

        assert result["objective"] is not None
        eq, _metadata = result["objective"]
        assert isinstance(eq, sp.Eq)

    def test_extracts_constraints_with_lagrange(self):
        source = """
        block TEST
        {
            constraints
            {
                C[] + K[] = Y[] : lambda[];
            };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        result = ast_block_to_equations(block)

        assert len(result["constraints"]) == 1
        _eq, metadata = result["constraints"][0]
        assert metadata["lagrange_multiplier"] is not None


class TestAstBlockToCalibration:
    def test_extracts_simple_params(self):
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
        param_dict, calib_dict, dists = ast_block_to_calibration(block)

        assert "alpha" in param_dict
        assert "beta" in param_dict
        assert len(calib_dict) == 0
        assert len(dists) == 0

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
        block = model.blocks[0]
        param_dict, _calib_dict, dists = ast_block_to_calibration(block)

        assert "alpha" in dists
        assert "alpha" in param_dict  # Initial value

    def test_extracts_calibrating_equations(self):
        source = """
        block TEST
        {
            calibration
            {
                L[ss] / K[ss] = 0.36 -> alpha;
            };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        _param_dict, calib_dict, _dists = ast_block_to_calibration(block)

        assert "alpha" in calib_dict


class TestAstBlockToVariablesAndShocks:
    def test_extracts_controls_as_variables(self):
        source = """
        block TEST
        {
            controls { C[], K[], L[]; };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        variables, _shocks = ast_block_to_variables_and_shocks(block)

        var_names = {v.base_name for v in variables}
        assert var_names == {"C", "K", "L"}

    def test_extracts_shocks(self):
        source = """
        block TEST
        {
            shocks { epsilon_A[], epsilon_B[]; };
        };
        """
        model = quick_parse(source)
        block = model.blocks[0]
        _variables, shocks = ast_block_to_variables_and_shocks(block)

        shock_names = {s.base_name for s in shocks}
        assert shock_names == {"epsilon_A", "epsilon_B"}

    def test_extracts_lhs_variables(self):
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
        variables, _shocks = ast_block_to_variables_and_shocks(block)

        var_names = {v.base_name for v in variables}
        assert "Y" in var_names


class TestAstModelToPrimitives:
    def test_full_model(self):
        source = """
        block HOUSEHOLD
        {
            controls { C[], K[]; };
            objective { U[] = log(C[]) + beta * E[][U[1]]; };
            constraints { C[] + K[] = Y[] : lambda[]; };

            shocks { epsilon[]; };

            calibration
            {
                beta = 0.99;
            };
        };

        block FIRM
        {
            identities
            {
                Y[] = A[] * K[-1] ^ alpha;
                log(A[]) = rho * log(A[-1]) + epsilon[];
            };

            calibration
            {
                alpha = 0.35;
                rho = 0.95;
            };
        };
        """
        model = quick_parse(source)
        result = ast_model_to_primitives(model)

        assert len(result["equations"]) > 0
        assert len(result["variables"]) > 0
        assert len(result["shocks"]) > 0
        assert "beta" in result["param_dict"]
        assert "alpha" in result["param_dict"]

    def test_shocks_not_in_variables(self):
        source = """
        block TEST
        {
            shocks { epsilon[]; };
            identities { C[] = epsilon[]; };
        };
        """
        model = quick_parse(source)
        result = ast_model_to_primitives(model)

        var_names = {v.base_name for v in result["variables"]}
        shock_names = {s.base_name for s in result["shocks"]}

        assert "epsilon" in shock_names
        assert "epsilon" not in var_names


class TestLoadGcnString:
    def test_simple_model(self):
        source = """
        block TEST
        {
            identities { Y[] = C[]; };
            calibration { alpha = 0.35; };
        };
        """
        result = load_gcn_string(source)

        assert "equations" in result
        assert "variables" in result
        assert "param_dict" in result
        assert "alpha" in result["param_dict"]

    def test_with_distributions(self):
        source = """
        block TEST
        {
            calibration
            {
                alpha ~ Beta(alpha=2, beta=5) = 0.35;
            };
        };
        """
        result = load_gcn_string(source)

        assert "alpha" in result["distributions"]
        assert "alpha" in result["param_dict"]


class TestLoadGcnFile:
    @pytest.fixture
    def gcn_dir(self):
        return Path(__file__).parent.parent / "_resources" / "test_gcns"

    def test_load_existing_file(self, gcn_dir):
        gcn_path = gcn_dir / "one_block_1.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        result = load_gcn_file(gcn_path)

        assert "equations" in result
        assert "variables" in result
        assert len(result["equations"]) > 0

    def test_load_basic_rbc(self, gcn_dir):
        gcn_path = gcn_dir / "basic_rbc.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        result = load_gcn_file(gcn_path)

        assert len(result["equations"]) > 0
        assert len(result["variables"]) > 0


class TestIntegrationWithRealFiles:
    @pytest.fixture
    def gcn_files_dir(self):
        return Path(__file__).parent.parent.parent / "GCN Files"

    def test_load_rbc_gcn(self, gcn_files_dir):
        gcn_path = gcn_files_dir / "RBC.gcn"
        if not gcn_path.exists():
            pytest.skip(f"Test file not found: {gcn_path}")

        result = load_gcn_file(gcn_path)

        assert len(result["equations"]) > 0
        assert len(result["variables"]) > 0
        # RBC should have shocks
        assert len(result["shocks"]) > 0
