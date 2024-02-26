import os

from collections import defaultdict

import numpy as np
import pytest

from gEconpy.exceptions.exceptions import (
    DuplicateParameterError,
    ExtraParameterError,
    OrphanParameterError,
)
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS
from gEconpy.parser.file_loaders import (
    block_dict_to_model_primitives,
    block_dict_to_param_dict,
    block_dict_to_variables_and_shocks,
    gcn_to_block_dict,
    load_gcn,
    model_from_gcn,
    parsed_model_to_data,
)
from gEconpy.parser.gEcon_parser import preprocess_gcn

GCN_ROOT = "tests/Test GCNs"

TEST_GCN_FILES = [
    "One_Block_Simple_1.gcn",
    "One_Block_Simple_1_w_Steady_State.gcn",
    "One_Block_Simple_2.gcn",
    "Full_New_Keyensian.gcn",
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
nk_other = {
    "TC": {"negative": True, "real": True},
    "delta": {"positive": True, "real": True},
    "beta": {"positive": True, "real": True},
    "sigma_C": {"positive": True, "real": True},
    "sigma_L": {"positive": True, "real": True},
    "gamma_I": {"positive": True, "real": True},
    "phi_H": {"positive": True, "real": True},
}
nk_assumptions.update(nk_other)

one_block_2_assumptions = defaultdict(lambda: DEFAULT_ASSUMPTIONS)
one_2_other = {
    "Y": {"positive": True, "real": True},
    "C": {"positive": True, "real": True},
    "I": {"positive": True, "real": True},
    "K": {"positive": True, "real": True},
    "L": {"positive": True, "real": True},
    "A": {"positive": True, "real": True},
    "theta": {"positive": True, "real": True},
    "beta": {"positive": True, "real": True},
    "delta": {"positive": True, "real": True},
    "tau": {"positive": True, "real": True},
    "rho": {"positive": True, "real": True},
    "alpha": {"positive": True, "real": True},
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
    "full_nk": {"output logfile": True, "output LaTeX": True, "output LaTeX landscape": True},
}

EXPECTED_TRYREDUCE = {
    "one_block": [],
    "one_block_ss": ["C[]"],
    "one_block_2": ["C[]"],
    "full_nk": ["Div[]", "TC[]"],
}

EXPECTED_SS_LEN = {"one_block": 1, "one_block_ss": 9, "one_block_2": 0, "full_nk": 0}

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


@pytest.mark.parametrize("gcn_path, name", zip(TEST_GCN_FILES, TEST_NAMES), ids=TEST_NAMES)
def test_build_model_blocks(gcn_path, name):
    raw_model = load_gcn(os.path.join(GCN_ROOT, gcn_path))
    parsed_model, prior_dict = preprocess_gcn(raw_model)

    parse_result = parsed_model_to_data(parsed_model, False)
    blocks, assumptions, options, try_reduce_vars, steady_state_equations = parse_result

    assert list(blocks.keys()) == EXPECTED_BLOCKS[name]
    assert all([assumptions[var] == EXPECTED_ASSUMPTIONS[name][var] for var in assumptions.keys()])
    assert options.keys() == EXPECTED_OPTIONS[name].keys()
    assert all([options[k] == EXPECTED_OPTIONS[name][k] for k in options.keys()])
    assert try_reduce_vars == EXPECTED_TRYREDUCE[name]
    assert len(steady_state_equations) == EXPECTED_SS_LEN[name]


@pytest.mark.parametrize("gcn_path, name", zip(TEST_GCN_FILES, TEST_NAMES), ids=TEST_NAMES)
def test_block_dict_to_variables_and_shocks(gcn_path, name):
    raw_model = load_gcn(os.path.join(GCN_ROOT, gcn_path))
    parsed_model, prior_dict = preprocess_gcn(raw_model)

    parse_result = parsed_model_to_data(parsed_model, False)
    blocks, assumptions, options, try_reduce_vars, steady_state_equations = parse_result
    variables, shocks = block_dict_to_variables_and_shocks(blocks)

    expected_vars = set(EXPECTED_VARIABLES[name])
    var_names = {x.base_name for x in variables}
    assert var_names == expected_vars

    expected_shocks = set(EXPECTED_SHOCKS[name])
    shock_names = {x.base_name for x in shocks}
    assert shock_names == expected_shocks


@pytest.mark.parametrize(
    "gcn_file",
    ["One_Block_Simple_with_duplicate_param_1.gcn", "One_Block_Simple_with_duplicate_param_2.gcn"],
    ids=["within_block", "between_blocks"],
)
def test_loading_fails_if_duplicate_parameters_in_two_blocks(gcn_file):
    with pytest.raises(DuplicateParameterError):
        outputs = gcn_to_block_dict(os.path.join(GCN_ROOT, gcn_file), True)
        block_dict, assumptions, options, tryreduce, steady_state_equations, prior_dict = outputs
        block_dict_to_model_primitives(block_dict, assumptions, tryreduce, prior_dict)


EXPECTED_PARAM_DICT = {
    "one_block_simple": dict(alpha=0.4, beta=0.99, delta=0.02, rho=0.95, gamma=1.5),
    "one_block_simple_2": dict(
        theta=0.357,
        beta=1 / 1.01,
        delta=0.02,
        tau=2,
        rho=0.95,
        Theta=0.95 * 1 / 1.01 + 3,
        zeta=-np.log(0.357),
    ),
}


@pytest.mark.parametrize(
    "gcn_path, name",
    [
        ("One_Block_Simple_1.gcn", "one_block_simple"),
        ("One_Block_Simple_2.gcn", "one_block_simple_2"),
    ],
    ids=["one_block_simple", "one_block_simple_2"],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_create_parameter_function(gcn_path, name, backend):
    expected = EXPECTED_PARAM_DICT[name]
    block_dict, *outputs = gcn_to_block_dict(os.path.join(GCN_ROOT, gcn_path), True)
    param_dict = block_dict_to_param_dict(block_dict, "param_dict")
    deterministic_dict = block_dict_to_param_dict(block_dict, "deterministic_dict")

    f, _ = compile_param_dict_func(param_dict, deterministic_dict, backend)

    inputs = list(param_dict.keys())
    np.random.shuffle(inputs)
    shuffled_input_dict = {k: param_dict[k] for k in inputs}
    output = f(**shuffled_input_dict)

    computed_param_dict = output.to_string().values_to_float()

    for k in expected.keys():
        np.testing.assert_allclose(
            computed_param_dict[k], expected[k], err_msg=f"{k} not close to tolerance"
        )


def test_load_gcn(gcn_path, name, simplify):
    model = model_from_gcn(
        os.path.join(GCN_ROOT, gcn_path), simplify_blocks=simplify, verbose=False
    )
    print(model.parameters(beta=0.5))
    assert False


def test_loading_fails_if_orphan_parameters():
    with pytest.raises(OrphanParameterError):
        model_from_gcn(os.path.join(GCN_ROOT, "Open_RBC_with_orphan_params.gcn"))


def test_loading_fails_if_extra_parameters():
    with pytest.raises(ExtraParameterError):
        model_from_gcn(os.path.join(GCN_ROOT, "Open_RBC_with_extra_params.gcn"))
