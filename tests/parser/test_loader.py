import pytest
import sympy as sp

from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.data.examples import get_example_gcn
from gEconpy.exceptions import DuplicateParameterError
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

    def test_prior_without_initial_value_leaves_param_dict_empty(self):
        block = parse_single_block("block TEST { calibration { alpha ~ Beta(alpha=2, beta=5); }; };")
        param_dict, _calib_dict, dists = ast_block_to_calibration(block)

        assert list(dists) == ["alpha"]
        assert len(param_dict) == 0

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

        assert len(primitives.equations) == 6
        assert [v.base_name for v in primitives.variables] == ["A", "C", "K", "U", "Y", "lambda"]
        assert [s.base_name for s in primitives.shocks] == ["epsilon"]
        assert set(primitives.param_dict) == {"alpha", "beta", "rho"}
        assert set(primitives.block_dict) == {"HOUSEHOLD", "FIRM"}

    def test_same_parameter_calibrated_in_two_blocks_raises(self):
        source = "block A { calibration { alpha = 0.3; }; }; block B { calibration { alpha = 0.4; }; };"
        with pytest.raises(DuplicateParameterError):
            ast_model_to_primitives(quick_parse(source))

    def test_value_contradicting_an_assumption_names_the_conflict(self):
        source = (
            "assumptions { positive { alpha; }; }; "
            "block A { calibration { alpha = 0; }; identities { C[] = alpha; }; };"
        )
        with pytest.raises(ValueError, match=r"'alpha = 0' is impossible.*alpha \(.*positive.*drop the conflicting"):
            ast_model_to_primitives(quick_parse(source))

    def test_self_referential_calibration_is_rejected(self):
        source = "block A { calibration { alpha = alpha; }; identities { C[] = alpha; }; };"
        with pytest.raises(ValueError, match=r"'alpha = alpha' is always true.*correct the value"):
            ast_model_to_primitives(quick_parse(source))

    def test_shocks_not_in_variables(self):
        source = "block TEST { shocks { epsilon[]; }; identities { C[] = epsilon[]; }; };"
        primitives = ast_model_to_primitives(quick_parse(source))

        assert [s.base_name for s in primitives.shocks] == ["epsilon"]
        assert [v.base_name for v in primitives.variables] == ["C"]

    def test_tryreduce_resolves_to_variables_and_drops_unknown_names(self):
        source = "tryreduce { U[], NOT_A_VARIABLE[]; }; block TEST { identities { U[] = C[]; C[] = 1; }; };"
        primitives = ast_model_to_primitives(quick_parse(source))

        assert [v.base_name for v in primitives.tryreduce] == ["U"]

    def test_steady_state_definitions_are_substituted_into_identities(self):
        source = """
        block STEADY_STATE { definitions { a = 2; }; identities { C[ss] = a * 3; K[ss] = C[ss] + 1; }; };
        block TEST { identities { C[] = 1; K[] = 2; }; };
        """
        primitives = ast_model_to_primitives(quick_parse(source))

        assert primitives.ss_solution_dict == {"C_ss": 6.0, "K_ss": 7.0}

    def test_shock_distribution_links_to_parameter_prior(self):
        source = """
        block TEST
        {
            shocks { epsilon[] ~ Normal(mu=0, sigma=sigma_eps); eta[] ~ Normal(mu=0, sigma=0.01); };
            identities { U[] = epsilon[] + eta[]; };
            calibration { sigma_eps ~ HalfNormal(sigma=1) = 0.1; };
        };
        """
        primitives = ast_model_to_primitives(quick_parse(source))

        assert list(primitives.shock_distributions) == ["epsilon"]
        assert primitives.distribution_param_names == {"sigma_eps"}

        composite = primitives.shock_distributions["epsilon"]
        assert isinstance(composite, CompositeDistribution)
        assert composite.fixed_params == {"mu": 0.0}
        assert composite.param_name_to_hyper_name == {"sigma": "sigma_eps"}
        assert composite.hyper_param_dict["sigma"] is primitives.distributions["sigma_eps"]

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


RBC_VARIABLES = ["A", "C", "I", "K", "L", "TC", "U", "Y", "lambda", "mc", "r", "w"]
RBC_PARAMETERS = ["alpha", "beta", "delta", "rho_A", "sigma_C", "sigma_L"]


@pytest.mark.parametrize(
    "gcn_path, n_equations, variables, shocks, parameters, tryreduce",
    [
        (
            TEST_GCNS / "one_block_1.gcn",
            5,
            ["A", "C", "K", "U", "lambda"],
            ["epsilon"],
            ["alpha", "beta", "delta", "gamma", "rho"],
            [],
        ),
        (TEST_GCNS / "basic_rbc.gcn", 12, RBC_VARIABLES, ["epsilon_A"], RBC_PARAMETERS, ["U", "TC"]),
        (get_example_gcn("RBC"), 12, RBC_VARIABLES, ["epsilon_A"], RBC_PARAMETERS, ["U", "TC"]),
    ],
    ids=["one_block_1", "basic_rbc", "example_rbc"],
)
def test_load_gcn_file(gcn_path, n_equations, variables, shocks, parameters, tryreduce):
    primitives = load_gcn_file(gcn_path)

    assert len(primitives.equations) == n_equations
    assert [v.base_name for v in primitives.variables] == variables
    assert [s.base_name for s in primitives.shocks] == shocks
    assert list(primitives.param_dict) == parameters
    assert [v.base_name for v in primitives.tryreduce] == tryreduce
