from collections import defaultdict
from pathlib import Path

import pytest

from gEconpy.exceptions import DuplicateParameterError
from gEconpy.parser._legacy.file_loaders import (
    block_dict_to_model_primitives,
    block_dict_to_variables_and_shocks,
    gcn_to_block_dict,
    load_gcn,
    parsed_model_to_data,
)
from gEconpy.parser._legacy.gEcon_parser import preprocess_gcn
from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS

# Add tests for gEconpy.parser.file_loaders here

TEST_GCN_FILES = [
    "one_block_1.gcn",
    "one_block_1_ss.gcn",
    "one_block_2.gcn",
    "full_nk.gcn",
]
TEST_NAMES = ["one_block", "one_block_ss", "one_block_2", "full_nk"]
EXPECTED_BLOCKS = {
    "one_block": ["HOUSEHOLD"],
    "one_block_ss": ["HOUSEHOLD"],
    "one_block_2": ["HOUSEHOLD"],
    "full_nk": [
        "HOUSEHOLD",
        "WAGE_SETTING",
        "WAGE_EVOLUTION",
        "PREFERENCE_SHOCKS",
        "FIRM",
        "TECHNOLOGY_SHOCKS",
        "FIRM_PRICE_SETTING_PROBLEM",
        "PRICE_EVOLUTION",
        "MONETARY_POLICY",
        "EQUILIBRIUM",
    ],
}
nk_assumptions = defaultdict(lambda: DEFAULT_ASSUMPTIONS)
one_block_2_assumptions = defaultdict(lambda: DEFAULT_ASSUMPTIONS)

nk_other = {
    "TC": {"real": True, "negative": True},
    "delta": {"real": True, "positive": True},
    "beta": {"real": True, "positive": True},
    "sigma_C": {"real": True, "positive": True},
    "sigma_L": {"real": True, "positive": True},
    "gamma_I": {"real": True, "positive": True},
    "phi_H": {"real": True, "positive": True},
    "shock_technology": {"real": True, "positive": True},
    "shock_preference": {"real": True, "positive": True},
    "pi": {"real": True, "positive": True},
    "pi_star": {"real": True, "positive": True},
    "pi_obj": {"real": True, "positive": True},
    "r": {"real": True, "positive": True},
    "r_G": {"real": True, "positive": True},
    "mc": {"real": True, "positive": True},
    "w": {"real": True, "positive": True},
    "w_star": {"real": True, "positive": True},
    "Y": {"real": True, "positive": True},
    "C": {"real": True, "positive": True},
    "I": {"real": True, "positive": True},
    "K": {"real": True, "positive": True},
    "L": {"real": True, "positive": True},
}

nk_assumptions.update(nk_other)

one_2_other = {
    "Y": {"real": True, "positive": True},
    "C": {"real": True, "positive": True},
    "I": {"real": True, "positive": True},
    "K": {"real": True, "positive": True},
    "L": {"real": True, "positive": True},
    "A": {"real": True, "positive": True},
    "theta": {"real": True, "positive": True},
    "beta": {"real": True, "positive": True},
    "delta": {"real": True, "positive": True},
    "tau": {"real": True, "positive": True},
    "rho": {"real": True, "positive": True},
    "alpha": {"real": True, "positive": True},
}
one_block_2_assumptions.update(one_2_other)


EXPECTED_ASSUMPTIONS = {
    "one_block": defaultdict(lambda: DEFAULT_ASSUMPTIONS),
    "one_block_ss": defaultdict(lambda: DEFAULT_ASSUMPTIONS),
    "one_block_2": one_block_2_assumptions,
    "full_nk": nk_assumptions,
}
EXPECTED_OPTIONS = {
    "one_block": {},
    "one_block_ss": {"output logfile": False, "output LaTeX": False},
    "one_block_2": {"output logfile": False, "output LaTeX": False},
    "full_nk": {
        "output logfile": True,
        "output LaTeX": True,
        "output LaTeX landscape": True,
    },
}
EXPECTED_TRYREDUCE = {
    "one_block": [],
    "one_block_ss": ["C[]"],
    "one_block_2": ["C[]"],
    "full_nk": [],
}
EXPECTED_SS_LEN = {"one_block": 0, "one_block_ss": 9, "one_block_2": 0, "full_nk": 25}


@pytest.mark.parametrize("gcn_path, name", zip(TEST_GCN_FILES, TEST_NAMES, strict=False), ids=TEST_NAMES)
def test_build_model_blocks(gcn_path, name):
    raw_model = load_gcn(Path("tests") / "_resources" / "test_gcns" / gcn_path)
    parsed_model, _prior_dict = preprocess_gcn(raw_model)

    parse_result = parsed_model_to_data(parsed_model, False)
    blocks, assumptions, options, try_reduce_vars, steady_state_equations = parse_result

    assert list(blocks.keys()) == EXPECTED_BLOCKS[name]
    assert all(assumptions[var] == EXPECTED_ASSUMPTIONS[name][var] for var in assumptions)
    assert options.keys() == EXPECTED_OPTIONS[name].keys()
    assert all(options[k] == EXPECTED_OPTIONS[name][k] for k in options)
    assert try_reduce_vars == EXPECTED_TRYREDUCE[name]
    assert len(steady_state_equations) == EXPECTED_SS_LEN[name]


EXPECTED_VARIABLES = {
    "one_block": ["A", "C", "K", "U", "lambda"],
    "one_block_ss": ["C", "L", "I", "K", "Y", "U", "A", "lambda", "q", "lambda__H_1"],
    "one_block_2": ["Y", "C", "I", "K", "L", "A", "U", "lambda", "q", "lambda__H_1"],
    "full_nk": [
        "Y",
        "C",
        "I",
        "K",
        "B",
        "U",
        "L",
        "w",
        "r",
        "r_G",
        "pi",
        "Div",
        "lambda",
        "q",
        "w_star",
        "LHS_w",
        "RHS_w",
        "TC",
        "LHS",
        "RHS",
        "shock_preference",
        "shock_technology",
        "pi_star",
        "mc",
        "pi_obj",
    ],
}
EXPECTED_SHOCKS = {
    "one_block": ["epsilon"],
    "one_block_ss": ["epsilon"],
    "one_block_2": ["epsilon"],
    "full_nk": ["epsilon_preference", "epsilon_Y", "epsilon_pi", "epsilon_R"],
}


@pytest.mark.parametrize("gcn_path, name", zip(TEST_GCN_FILES, TEST_NAMES, strict=False), ids=TEST_NAMES)
def test_block_dict_to_variables_and_shocks(gcn_path, name):
    raw_model = load_gcn(Path("tests") / "_resources" / "test_gcns" / gcn_path)
    parsed_model, _prior_dict = preprocess_gcn(raw_model)

    parse_result = parsed_model_to_data(parsed_model, False)
    blocks, _assumptions, _options, _try_reduce_vars, _steady_state_equations = parse_result
    variables, shocks = block_dict_to_variables_and_shocks(blocks)

    expected_vars = set(EXPECTED_VARIABLES[name])
    var_names = {x.base_name for x in variables}
    assert var_names == expected_vars

    expected_shocks = set(EXPECTED_SHOCKS[name])
    shock_names = {x.base_name for x in shocks}
    assert shock_names == expected_shocks


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_duplicate_params.gcn",
        "one_block_1_duplicate_params_2.gcn",
    ],
    ids=["within_block", "between_blocks"],
)
def test_loading_fails_if_duplicate_parameters_in_two_blocks(gcn_file):
    with pytest.raises(DuplicateParameterError):
        outputs = gcn_to_block_dict(Path("tests") / "_resources" / "test_gcns" / gcn_file, True)
        (
            block_dict,
            assumptions,
            _options,
            tryreduce,
            _steady_state_equations,
            prior_dict,
        ) = outputs
        block_dict_to_model_primitives(block_dict, assumptions, tryreduce, prior_dict)
