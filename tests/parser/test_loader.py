import pytest
import sympy as sp

from gEconpy.data.examples import get_example_gcn
from gEconpy.parser.ast import GCNBlock
from gEconpy.parser.loader import (
    ast_block_to_calibration,
    ast_block_to_equations,
    ast_block_to_variables_and_shocks,
    ast_model_to_primitives,
    load_gcn_file,
    load_gcn_string,
)
from gEconpy.parser.preprocessor import quick_parse
from tests.conftest import TEST_GCNS


def parse_single_block(source: str) -> GCNBlock:
    return quick_parse(source).blocks[0]


class TestAstBlockToEquations:
    @pytest.mark.parametrize(
        "component, body, n_equations",
        [
            ("identities", "Y[] = C[] + I[]; K[] = (1 - delta) * K[-1] + I[];", 2),
            ("definitions", "u[] = log(C[]);", 1),
            ("objective", "U[] = u[] + beta * E[][U[1]];", 1),
            ("constraints", "C[] + K[] = Y[] : lambda[];", 1),
        ],
    )
    def test_extracts_component(self, component, body, n_equations):
        block = parse_single_block(f"block TEST {{ {component} {{ {body} }}; }};")
        equations = ast_block_to_equations(block)

        assert set(equations) == {"definitions", "objective", "constraints", "identities"}
        assert len(equations[component]) == n_equations
        assert all(isinstance(eq, sp.Eq) for eq, _metadata in equations[component])
        assert all(len(equations[other]) == 0 for other in equations if other != component)

    def test_constraint_metadata_records_lagrange_multiplier(self):
        block = parse_single_block("block TEST { constraints { C[] + K[] = Y[] : lambda[]; }; };")
        _eq, metadata = ast_block_to_equations(block)["constraints"][0]
        assert metadata["lagrange_multiplier"] is not None


class TestAstBlockToCalibration:
    def test_extracts_simple_params(self):
        block = parse_single_block("block TEST { calibration { alpha = 0.35; beta = 0.99; }; };")
        param_dict, calib_dict, dists = ast_block_to_calibration(block)

        assert set(param_dict) == {"alpha", "beta"}
        assert len(calib_dict) == 0
        assert len(dists) == 0

    def test_prior_initial_value_lands_in_param_dict(self):
        block = parse_single_block("block TEST { calibration { alpha ~ Beta(alpha=2, beta=5) = 0.35; }; };")
        param_dict, _calib_dict, dists = ast_block_to_calibration(block)

        assert "alpha" in dists
        assert param_dict["alpha"] == 0.35

    def test_extracts_calibrating_equations(self):
        block = parse_single_block("block TEST { calibration { L[ss] / K[ss] = 0.36 -> alpha; }; };")
        _param_dict, calib_dict, _dists = ast_block_to_calibration(block)

        assert "alpha" in calib_dict.to_string()


class TestAstBlockToVariablesAndShocks:
    def test_controls_and_equation_lhs_are_variables(self):
        block = parse_single_block("block TEST { controls { C[], K[], L[]; }; identities { Y[] = C[] + I[]; }; };")
        variables, shocks = ast_block_to_variables_and_shocks(block)

        assert {v.base_name for v in variables} == {"C", "K", "L", "Y"}
        assert shocks == []

    def test_extracts_shocks(self):
        block = parse_single_block("block TEST { shocks { epsilon_A[], epsilon_B[]; }; };")
        _variables, shocks = ast_block_to_variables_and_shocks(block)

        assert {s.base_name for s in shocks} == {"epsilon_A", "epsilon_B"}


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
        primitives = ast_model_to_primitives(quick_parse(source))

        assert len(primitives.equations) > 0
        assert {v.base_name for v in primitives.variables} >= {"C", "K", "Y", "A", "U"}
        assert [s.base_name for s in primitives.shocks] == ["epsilon"]
        assert set(primitives.param_dict) == {"alpha", "beta", "rho"}
        assert set(primitives.block_dict) == {"HOUSEHOLD", "FIRM"}

    def test_shocks_not_in_variables(self):
        source = "block TEST { shocks { epsilon[]; }; identities { C[] = epsilon[]; }; };"
        primitives = ast_model_to_primitives(quick_parse(source))

        assert [s.base_name for s in primitives.shocks] == ["epsilon"]
        assert [v.base_name for v in primitives.variables] == ["C"]

    def test_tryreduce_resolves_to_variables(self):
        source = "tryreduce { U[]; }; block TEST { identities { U[] = C[]; C[] = 1; }; };"
        primitives = ast_model_to_primitives(quick_parse(source))

        assert [v.base_name for v in primitives.tryreduce] == ["U"]

    @pytest.mark.parametrize("ss_name", ["STEADY_STATE", "STEADYSTATE", "SS", "STEADY"])
    def test_every_steady_state_block_name_is_consumed_not_solved(self, ss_name):
        source = f"""
        block {ss_name} {{ identities {{ C[ss] = 1; }}; }};
        block TEST {{ identities {{ C[] = 1; }}; }};
        """
        primitives = ast_model_to_primitives(quick_parse(source))

        assert set(primitives.block_dict) == {"TEST"}
        assert [str(k) for k in primitives.ss_solution_dict] == ["C_ss"]


class TestLoadGcnString:
    def test_simple_model(self):
        source = "block TEST { identities { Y[] = C[]; }; calibration { alpha = 0.35; }; };"
        primitives = load_gcn_string(source)

        assert len(primitives.equations) == 1
        assert {v.base_name for v in primitives.variables} == {"C", "Y"}
        assert primitives.param_dict["alpha"] == 0.35

    def test_with_distributions(self):
        source = "block TEST { calibration { alpha ~ Beta(alpha=2, beta=5) = 0.35; }; };"
        primitives = load_gcn_string(source)

        assert "alpha" in primitives.distributions
        assert primitives.param_dict["alpha"] == 0.35


@pytest.mark.parametrize(
    "gcn_path",
    [TEST_GCNS / "one_block_1.gcn", TEST_GCNS / "basic_rbc.gcn", get_example_gcn("RBC")],
    ids=["one_block_1", "basic_rbc", "example_rbc"],
)
def test_load_gcn_file(gcn_path):
    primitives = load_gcn_file(gcn_path)

    assert len(primitives.equations) > 0
    assert len(primitives.variables) > 0
    assert len(primitives.shocks) > 0
