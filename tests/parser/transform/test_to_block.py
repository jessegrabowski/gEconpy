import sympy as sp

from gEconpy.parser.preprocessor import quick_parse
from gEconpy.parser.transform.to_block import ast_block_to_block, ast_model_to_block_dict
from tests.conftest import parsed_symbol, parsed_var

SOURCE = """
block TEST
{
    definitions { u[] = log(C[]); };
    controls { C[], K[]; };
    objective { U[] = u[] + beta * E[][U[1]]; };
    constraints { C[] + K[] = Y[] : lambda[]; @exclude K[] = I[]; };
    identities { Y[] = A[] * K[-1] ^ alpha; };
    shocks { epsilon[]; };
    calibration
    {
        alpha ~ Beta(alpha=2, beta=5) = 0.35;
        sigma ~ Gamma(alpha=2, beta=1);
        beta = 0.99;
        Y[ss] / K[ss] = 0.36 -> delta;
    };
};
"""


def test_equations_are_numbered_consecutively_across_components():
    block = ast_block_to_block(quick_parse(SOURCE).blocks[0])

    assert list(block.definitions) == [0]
    assert list(block.objective) == [1]
    assert list(block.constraints) == [2, 3]
    assert list(block.identities) == [4]
    assert list(block.calibration) == [5, 6, 7]


def test_calibrating_equation_is_stored_as_parameter_equals_residual():
    block = ast_block_to_block(quick_parse(SOURCE).blocks[0])

    Y_ss, K_ss = parsed_var("Y", "ss"), parsed_var("K", "ss")
    assert block.calibration[7] == sp.Eq(parsed_symbol("delta"), sp.Float(0.36) - Y_ss / K_ss)
    assert block.equation_flags[7] == {"is_calibrating": True}


def test_prior_contributes_calibration_equation_only_with_initial_value():
    block = ast_block_to_block(quick_parse(SOURCE).blocks[0])

    assert block.calibration[5] == sp.Eq(parsed_symbol("alpha"), sp.Float(0.35))
    assert block.calibration[6] == sp.Eq(parsed_symbol("beta"), sp.Float(0.99))
    assert not any(eq.lhs == parsed_symbol("sigma") for eq in block.calibration.values())


def test_multipliers_and_tags_are_keyed_by_equation_number():
    block = ast_block_to_block(quick_parse(SOURCE).blocks[0])

    assert block.multipliers[2] == parsed_var("lambda", 0)
    assert [key for key, multiplier in block.multipliers.items() if multiplier is not None] == [2]
    assert block.equation_flags[3] == {"exclude": True, "is_calibrating": False}


def test_assumptions_apply_to_controls():
    block = ast_block_to_block(quick_parse(SOURCE).blocks[0], assumptions={"C": {"positive": True}})

    C, K = block.controls
    assert (C.base_name, C.is_positive) == ("C", True)
    assert (K.base_name, K.is_positive) == ("K", None)
    assert [shock.base_name for shock in block.shocks] == ["epsilon"]


def test_symbol_locations_point_at_declarations():
    block = ast_block_to_block(quick_parse(SOURCE).blocks[0], source=SOURCE)
    lines = SOURCE.split("\n")

    locations = {name: (loc.line, loc.column) for name, loc in block._symbol_locations.items()}
    assert set(locations) == {"C", "K", "epsilon", "alpha", "sigma", "beta", "delta"}
    assert lines[locations["C"][0] - 1][locations["C"][1] - 1 :].startswith("C[]")
    assert lines[locations["epsilon"][0] - 1][locations["epsilon"][1] - 1 :].startswith("epsilon[]")
    assert lines[locations["alpha"][0] - 1][locations["alpha"][1] - 1 :].startswith("alpha ~ Beta")
    assert lines[locations["sigma"][0] - 1][locations["sigma"][1] - 1 :].startswith("sigma ~ Gamma")
    assert lines[locations["beta"][0] - 1][locations["beta"][1] - 1 :].startswith("beta = 0.99")
    assert lines[locations["delta"][0] - 1][locations["delta"][1] - 1 :].startswith("Y[ss] / K[ss]")


def test_empty_components_are_none():
    block = ast_block_to_block(quick_parse("block TEST { identities { Y[] = C[]; }; };").blocks[0])

    assert block.definitions is None
    assert block.objective is None
    assert block.constraints is None
    assert block.calibration is None
    assert block.controls is None
    assert block.shocks is None


def test_model_conversion_skips_steady_state_block_and_expands_deep_lags():
    source = """
    block STEADY_STATE { identities { C[ss] = 1; }; };
    block TEST { identities { C[] = S[-3]; }; };
    """
    block_dict = ast_model_to_block_dict(quick_parse(source))

    assert list(block_dict) == ["TEST"]
    equations = list(block_dict["TEST"].identities.values())
    assert equations[0] == sp.Eq(parsed_var("C", 0), parsed_var("S__lag2", -1))
    assert len(equations) == 3
