import os

from collections import defaultdict
from importlib.util import find_spec

import numpy as np
import pytest

from gEconpy.exceptions import (
    DuplicateParameterError,
    ExtraParameterError,
    OrphanParameterError,
)
from gEconpy.model.build import model_from_gcn
from gEconpy.model.compile import BACKENDS
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.steady_state import (
    compile_model_ss_functions,
    system_to_steady_state,
)
from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS
from gEconpy.parser.file_loaders import (
    block_dict_to_model_primitives,
    block_dict_to_param_dict,
    block_dict_to_variables_and_shocks,
    gcn_to_block_dict,
    load_gcn,
    parsed_model_to_data,
    simplify_provided_ss_equations,
    validate_results,
)
from gEconpy.parser.gEcon_parser import preprocess_gcn

JAX_INSTALLED = find_spec("jax") is not None

GCN_ROOT = "tests/Test GCNs"

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

one_block_2_assumptions = defaultdict(lambda: DEFAULT_ASSUMPTIONS)
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
    "full_nk": ["Div[]", "TC[]"],
}

EXPECTED_SS_LEN = {"one_block": 0, "one_block_ss": 9, "one_block_2": 0, "full_nk": 25}

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


@pytest.mark.parametrize(
    "gcn_path, name", zip(TEST_GCN_FILES, TEST_NAMES), ids=TEST_NAMES
)
def test_build_model_blocks(gcn_path, name):
    raw_model = load_gcn(os.path.join("tests/Test GCNs", gcn_path))
    parsed_model, prior_dict = preprocess_gcn(raw_model)

    parse_result = parsed_model_to_data(parsed_model, False)
    blocks, assumptions, options, try_reduce_vars, steady_state_equations = parse_result

    assert list(blocks.keys()) == EXPECTED_BLOCKS[name]
    assert all(
        [
            assumptions[var] == EXPECTED_ASSUMPTIONS[name][var]
            for var in assumptions.keys()
        ]
    )
    assert options.keys() == EXPECTED_OPTIONS[name].keys()
    assert all([options[k] == EXPECTED_OPTIONS[name][k] for k in options.keys()])
    assert try_reduce_vars == EXPECTED_TRYREDUCE[name]
    assert len(steady_state_equations) == EXPECTED_SS_LEN[name]


@pytest.mark.parametrize(
    "gcn_path, name", zip(TEST_GCN_FILES, TEST_NAMES), ids=TEST_NAMES
)
def test_block_dict_to_variables_and_shocks(gcn_path, name):
    raw_model = load_gcn(os.path.join("tests/Test GCNs", gcn_path))
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
    [
        "one_block_1_duplicate_params.gcn",
        "one_block_1_duplicate_params_2.gcn",
    ],
    ids=["within_block", "between_blocks"],
)
def test_loading_fails_if_duplicate_parameters_in_two_blocks(gcn_file):
    with pytest.raises(DuplicateParameterError):
        outputs = gcn_to_block_dict(os.path.join("tests/Test GCNs", gcn_file), True)
        (
            block_dict,
            assumptions,
            options,
            tryreduce,
            steady_state_equations,
            prior_dict,
        ) = outputs
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
        ("one_block_1.gcn", "one_block_simple"),
        ("one_block_2.gcn", "one_block_simple_2"),
    ],
    ids=["one_block_simple", "one_block_simple_2"],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_create_parameter_function(gcn_path, name, backend):
    expected = EXPECTED_PARAM_DICT[name]
    block_dict, *outputs = gcn_to_block_dict(
        os.path.join("tests/Test GCNs", gcn_path), True
    )
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


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_all_model_functions_return_arrays(backend: BACKENDS):
    outputs = gcn_to_block_dict(
        "tests/Test GCNs/one_block_1_ss.gcn", simplify_blocks=True
    )
    block_dict, assumptions, options, try_reduce, ss_solution_dict, prior_info = outputs

    (
        equations,
        param_dict,
        calib_dict,
        deterministic_dict,
        variables,
        shocks,
        param_priors,
        shock_priors,
        reduced_vars,
        singletons,
    ) = block_dict_to_model_primitives(
        block_dict,
        assumptions,
        try_reduce,
        prior_info,
        simplify_tryreduce=True,
        simplify_constants=True,
    )

    ss_solution_dict = simplify_provided_ss_equations(ss_solution_dict, variables)

    validate_results(equations, param_dict, calib_dict, deterministic_dict)
    steady_state_equations = system_to_steady_state(equations, shocks)

    kwargs = {}
    if backend == "pytensor":
        kwargs["mode"] = "JAX" if JAX_INSTALLED else "FAST_RUN"
    (f_params, f_ss, resid_funcs, error_funcs), cache = compile_model_ss_functions(
        steady_state_equations,
        ss_solution_dict,
        variables,
        param_dict,
        deterministic_dict,
        calib_dict,
        error_func="squared",
        backend=backend,
        **kwargs,
    )

    f_ss_resid, f_ss_jac = resid_funcs
    f_ss_error, f_ss_grad, f_ss_hess, f_ss_hessp = error_funcs

    parameters = f_params(**param_dict)
    ss = f_ss(**parameters)
    x0 = {var.to_ss().name: 0.8 for var in variables}
    x0.update(ss)
    for f in [f_ss_resid, f_ss_jac, f_ss_grad, f_ss_hess]:
        result = f(**x0, **parameters)
        assert isinstance(result, np.ndarray)

    result = f_ss_hessp(np.ones(len(variables)), **x0, **parameters)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        "full_nk.gcn",
    ],
    ids=["one_block_simple", "open_rbc", "full_nk"],
)
def test_load_gcn(gcn_file):
    from gEconpy.model.model import Model

    mod = model_from_gcn(
        os.path.join("tests/Test GCNs", gcn_file), simplify_blocks=True, verbose=False
    )
    assert isinstance(mod, Model)
    assert len(mod.shocks) > 0
    assert len(mod.variables) > 0
    assert len(mod.equations) > 0

    assert mod.f_params is not None

    assert mod.f_ss is not None
    assert mod.f_ss_jac is not None

    assert mod.f_ss_resid is not None
    assert mod.f_ss_error_grad is not None
    assert mod.f_ss_error_hess is not None


def test_loading_fails_if_orphan_parameters():
    with pytest.raises(OrphanParameterError):
        model_from_gcn(os.path.join("tests/Test GCNs", "open_rbc_orphan_params.gcn"))


def test_loading_fails_if_extra_parameters():
    with pytest.raises(ExtraParameterError):
        model_from_gcn(os.path.join("tests/Test GCNs", "open_rbc_extra_params.gcn"))


def test_build_report(caplog):
    model_from_gcn(
        "tests/Test GCNs/rbc_2_block.gcn",
        verbose=True,
        simplify_tryreduce=True,
        simplify_constants=True,
        simplify_blocks=True,
    )

    expected_report = r"""
                Model Building Complete.
                Found:
                    12 equations
                    12 variables
                    The following "variables" were defined as constants and have been substituted away:
                        P_t
                    1 stochastic shock
                        0 / 1 has a defined prior.
                    6 parameters
                        0 / 6 has a defined prior.
                    0 parameters to calibrate.
                    Model appears well defined and ready to proceed to solving."""

    expected_lines = [x.strip() for x in expected_report.strip().split("\n")]
    found_lines = [x.strip() for x in caplog.messages[-1].strip().split("\n")]

    for line1, line2 in zip(expected_lines, found_lines, strict=True):
        assert line1 == line2
